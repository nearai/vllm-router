import asyncio
import os

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from vllm_router.log import init_logger
from vllm_router.service_discovery import get_service_discovery
from vllm_router.quote import (
    ECDSA,
    ED25519,
    ecdsa_context,
    ed25519_context,
    generate_attestation,
)

router = APIRouter()
logger = init_logger(__name__)

TIMEOUT = 10  # Timeout for backend requests in seconds


@router.get("/v1/attestation/report")
async def attestation_report(
    request: Request,
    signing_algo: str | None = None,
    nonce: str | None = Query(None),
    signing_address: str | None = Query(None),
):
    signing_algo = ECDSA if signing_algo is None else signing_algo
    if signing_algo not in [ECDSA, ED25519]:
        raise HTTPException(status_code=400, detail="Invalid signing algorithm")

    context = ecdsa_context if signing_algo == ECDSA else ed25519_context

    # 1. Generate Router's own attestation
    # If signing_address is specified and matches Router's address, we return 200 with our attestation.
    # If it doesn't match, we still participate in aggregation (or return 404 if we were a single node, but we are a router/LB).

    router_attestation = None
    try:
        router_attestation = generate_attestation(context, nonce)
    except ValueError as exc:
        logger.error(f"Failed to generate router attestation: {exc}")

    if (
        signing_address
        and router_attestation
        and router_attestation["signing_address"] == signing_address
    ):
        # If requested signing_address matches router's, return only our attestation
        return JSONResponse(content=router_attestation)

    # 2. Broadcast to all backends
    service_discovery = get_service_discovery()
    endpoints = service_discovery.get_endpoint_info()

    # Deduplicate by URL or ID to avoid querying same instance multiple times if listed multiple times?
    # Usually endpoints are unique instances.
    unique_urls = set(endpoint.url for endpoint in endpoints if not endpoint.sleep)

    async def fetch_attestation(url: str):
        try:
            client = request.app.state.aiohttp_client_wrapper()
            # Pass query params
            params = {}
            if signing_algo:
                params["signing_algo"] = signing_algo
            if nonce:
                params["nonce"] = nonce
            if signing_address:
                params["signing_address"] = signing_address

            headers = {}
            headers["Authorization"] = f"Bearer {os.environ.get('OPENAI_API_KEY')}"

            async with client.get(
                f"{url}/v1/attestation/report",
                params=params,
                headers=headers,
                timeout=TIMEOUT,
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.warning(f"Failed to fetch attestation from {url}: {e}")
            return None

    tasks = [fetch_attestation(url) for url in unique_urls]
    results = await asyncio.gather(*tasks)

    all_attestations = []

    # Add backend attestations
    for res in results:
        if res:
            if "all_attestations" in res:
                all_attestations.extend(res["all_attestations"])
            else:
                # Fallback if backend is old version
                all_attestations.append(res)

    response = {}
    if router_attestation:
        response.update(router_attestation)
    response["all_attestations"] = all_attestations

    if signing_address:
        # Check if any attestation matches
        match = False
        for att in all_attestations:
            if att.get("signing_address", "").lower() == signing_address.lower():
                match = True
                break
        if not match:
            # If router doesn't match AND no backend matches -> 404
            raise HTTPException(status_code=404, detail="Signing address not found")

    return response


@router.get("/v1/signature/{chat_id}")
async def get_signature(
    request: Request,
    chat_id: str,
    signing_algo: str = None,
):
    # 1. Check Cache
    chat_cache = getattr(request.app.state, "chat_cache", None)
    backend_url = None
    if chat_cache:
        backend_url = chat_cache.get(chat_id)
        if backend_url:
            logger.debug(f"Cache hit for chat_id {chat_id}: {backend_url}")

    async def fetch_signature(url: str):
        try:
            client = request.app.state.aiohttp_client_wrapper()
            params = {}
            if signing_algo:
                params["signing_algo"] = signing_algo

            headers = {}
            headers["Authorization"] = f"Bearer {os.environ.get('OPENAI_API_KEY')}"

            async with client.get(
                f"{url}/v1/signature/{chat_id}",
                params=params,
                headers=headers,
                timeout=TIMEOUT,
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.warning(f"Failed to fetch signature from {url}: {e}")
            return None

    if backend_url:
        # Try specific backend
        result = await fetch_signature(backend_url)
        if result:
            return result
        else:
            logger.info(
                f"Cached backend {backend_url} failed for {chat_id}, falling back to broadcast"
            )

    # 2. Broadcast (fallback)
    service_discovery = get_service_discovery()
    endpoints = service_discovery.get_endpoint_info()
    unique_urls = set(endpoint.url for endpoint in endpoints if not endpoint.sleep)

    tasks = [fetch_signature(url) for url in unique_urls if url != backend_url]
    if not tasks and not backend_url:
        # No backends
        raise HTTPException(status_code=404, detail="No backends available")

    results = await asyncio.gather(*tasks)
    for res in results:
        if res:
            return res

    raise HTTPException(status_code=404, detail="Chat id not found or expired")
