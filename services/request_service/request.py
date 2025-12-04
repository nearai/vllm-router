# Copyright 2024-2025 The vLLM Production Stack Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --- Request Processing & Routing ---
import json
import os
import time
import uuid
from typing import Optional

import httpx
from fastapi import BackgroundTasks, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from requests import JSONDecodeError

from vllm_router.log import init_logger
from vllm_router.routers.routing_logic import (
    DisaggregatedPrefillRouter,
    KvawareRouter,
    PrefixAwareRouter,
    SessionRouter,
)
from vllm_router.service_discovery import get_service_discovery
from vllm_router.services.request_service.rewriter import (
    get_request_rewriter,
    is_request_rewriter_initialized,
)
from vllm_router.utils import replace_model_in_request_body, update_content_length

from vllm_router.services.metrics_service import num_incoming_requests_total

logger = init_logger(__name__)

_HOP_BY_HOP_HEADERS = {
    "host",
    "connection",
    "keep-alive",
    "proxy-connection",
    "transfer-encoding",
    "content-length",
    "upgrade",
    "te",  # codespell:ignore
    "trailer",
    "authorization",
}


# TODO: (Brian) check if request is json beforehand
async def process_request(
    request: Request,
    body,
    backend_url,
    request_id,
    endpoint,
    background_tasks: BackgroundTasks,
    is_streaming: bool = False,
):
    """
    Process a request by sending it to the chosen backend.

    Args:
        request(Request): Request object.
        body: The content of the request to send to the backend.
        backend_url: The URL of the backend to send the request to.
        request_id: A unique identifier for the request.
        endpoint: The endpoint to send the request to on the backend.
        is_streaming: Whether this is a streaming request.

    Yields:
        The response headers and status code, followed by the response content.

    Raises:
        HTTPError: If the backend returns a 4xx or 5xx status code.
    """
    first_token = False
    total_len = 0
    start_time = time.time()
    request.app.state.request_stats_monitor.on_new_request(
        backend_url, request_id, start_time
    )

    # sanitize the request headers
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS
    }
    # Add OPENAI_API_KEY if set
    if OPENAI_API_KEY := os.getenv("OPENAI_API_KEY"):
        logger.debug("Using OpenAI API key for backend authentication")
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"

    # For non-streaming requests, collect the full response to cache it properly
    full_response = bytearray()

    # Buffer for chat_id extraction in streaming mode
    stream_buffer = ""
    chat_id_found = False
    chat_cache = getattr(request.app.state, "chat_cache", None)

    client_wrapper = request.app.state.httpx_client_wrapper
    client = client_wrapper()
    # Track connection identities before the request to detect new connections.
    # This approach uses connection identity tracking rather than simple pool size
    # checking to reduce race conditions. However, it's still best-effort because:
    # - Other concurrent requests may create/close connections between our checks
    # - Connection pooling behavior depends on the underlying HTTP library
    # - In highly concurrent scenarios, false positives/negatives are still possible
    conn_ids_before = client_wrapper.get_connection_ids()
    pre_backend_time = time.time()

    logger.debug(
        f"[{request_id}] Connecting to backend {backend_url}{endpoint} "
        f"(pool_size={len(conn_ids_before)}, connect_timeout={client_wrapper.connect_timeout}s, "
        f"read_timeout={client_wrapper.read_timeout}s)"
    )

    try:
        async with client.stream(
            method=request.method,
            url=backend_url + endpoint,
            headers=headers,
            content=body,
        ) as backend_response:
            # This measures: connection acquisition + request send + backend TTFB
            time_to_response_headers = time.time() - pre_backend_time
            conn_ids_after = client_wrapper.get_connection_ids()
            http_version = backend_response.http_version
            # Detect if a new connection was created by checking for new connection IDs
            # This is more reliable than pool size checking but still best-effort in concurrent environments
            new_connections = conn_ids_after - conn_ids_before
            new_connection = len(new_connections) > 0
            if new_connection:
                logger.debug(
                    f"[{request_id}] Connected to {backend_url} (NEW connection): "
                    f"connect_time={time_to_response_headers * 1000:.2f}ms, "
                    f"http_version={http_version}, pool_size={len(conn_ids_after)}"
                )
            else:
                logger.debug(
                    f"[{request_id}] Connected to {backend_url} (reused connection): "
                    f"connect_time={time_to_response_headers * 1000:.2f}ms, "
                    f"http_version={http_version}, pool_size={len(conn_ids_after)}"
                )
            # Yield headers and status code first.
            yield dict(backend_response.headers), backend_response.status_code
            # Stream response content.
            async for chunk in backend_response.aiter_bytes():
                total_len += len(chunk)
                if not first_token:
                    first_token = True
                    request.app.state.request_stats_monitor.on_request_response(
                        backend_url, request_id, time.time()
                    )

                # Extract chat_id for streaming response
                if is_streaming and not chat_id_found and chat_cache is not None:
                    try:
                        text_chunk = chunk.decode("utf-8", errors="ignore")
                        stream_buffer += text_chunk
                        # Look for data: {...}
                        while "\n" in stream_buffer:
                            line, stream_buffer = stream_buffer.split("\n", 1)
                            line = line.strip()
                            if line.startswith("data: ") and line != "data: [DONE]":
                                try:
                                    data_str = line[6:].strip()
                                    data_json = json.loads(data_str)
                                    if "id" in data_json:
                                        chat_id_val = data_json["id"]
                                        chat_cache[chat_id_val] = backend_url
                                        chat_id_found = True
                                        break
                                except Exception:
                                    pass
                        if len(stream_buffer) > 4096:  # Stop buffering if too large
                            chat_id_found = True
                    except Exception:
                        pass

                # For non-streaming requests, collect the full response
                if full_response is not None:
                    full_response.extend(chunk)
                yield chunk

    except httpx.TimeoutException as e:
        # Specific handling for timeout errors (connect, read, write, pool)
        timeout_type = type(e).__name__
        logger.error(
            f"[{request_id}] Timeout ({timeout_type}) to backend {backend_url}: {str(e)}"
        )

        # Mark backend as unhealthy
        service_discovery = get_service_discovery()
        if hasattr(service_discovery, "mark_backend_unhealthy_during_request"):
            model_name = "unknown"
            try:
                if body:
                    request_json = json.loads(body)
                    model_name = request_json.get("model", "unknown")
            except:
                pass

            should_remove = service_discovery.mark_backend_unhealthy_during_request(
                backend_url, model_name
            )
            if should_remove:
                logger.warning(
                    f"Backend {backend_url} removed from pool due to timeout ({timeout_type})"
                )

        raise Exception(f"Backend {backend_url} timed out ({timeout_type}): {str(e)}")

    except httpx.RequestError as e:
        # Connection error or other request-level error
        logger.error(
            f"[{request_id}] Request failed to backend {backend_url}: {str(e)}"
        )

        # Mark backend as unhealthy
        service_discovery = get_service_discovery()
        if hasattr(service_discovery, "mark_backend_unhealthy_during_request"):
            # Extract model name from the request if possible
            model_name = "unknown"
            try:
                if body:
                    request_json = json.loads(body)
                    model_name = request_json.get("model", "unknown")
            except:
                pass

            should_remove = service_discovery.mark_backend_unhealthy_during_request(
                backend_url, model_name
            )
            if should_remove:
                logger.warning(
                    f"Backend {backend_url} removed from pool due to repeated failures"
                )

        # Re-raise the error with backend information
        raise Exception(f"Backend {backend_url} failed: {str(e)}")

    except httpx.HTTPStatusError as e:
        # HTTP error response (4xx, 5xx)
        logger.error(
            f"HTTP error from backend {backend_url}: {e.response.status_code} {e.response.reason_phrase}"
        )

        # For 5xx errors, mark backend as unhealthy
        if e.response.status_code >= 500:
            service_discovery = get_service_discovery()
            if hasattr(service_discovery, "mark_backend_unhealthy_during_request"):
                # Extract model name from the request if possible
                model_name = "unknown"
                try:
                    if body:
                        request_json = json.loads(body)
                        model_name = request_json.get("model", "unknown")
                except:
                    pass

                should_remove = service_discovery.mark_backend_unhealthy_during_request(
                    backend_url, model_name
                )
                if should_remove:
                    logger.warning(
                        f"Backend {backend_url} removed from pool due to HTTP {e.response.status_code} errors"
                    )

        # Re-raise the error with backend information
        raise Exception(
            f"Backend {backend_url} returned HTTP {e.response.status_code}: {e.response.reason_phrase}"
        )

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error with backend {backend_url}: {str(e)}")
        raise

    # Extract chat_id for non-streaming response
    if not is_streaming and chat_cache is not None and full_response:
        try:
            response_json = json.loads(full_response.decode("utf-8"))
            if "id" in response_json:
                chat_cache[response_json["id"]] = backend_url
        except Exception:
            pass

    end_time = time.time()
    request.app.state.request_stats_monitor.on_request_complete(
        backend_url, request_id, end_time
    )

    # Record success for circuit breaker recovery
    service_discovery = get_service_discovery()
    if hasattr(service_discovery, "record_request_success"):
        model_name = "unknown"
        try:
            if body:
                request_json_parsed = json.loads(body)
                model_name = request_json_parsed.get("model", "unknown")
        except Exception:
            pass
        service_discovery.record_request_success(backend_url, model_name)

    # Log timing breakdown
    backend_time = end_time - pre_backend_time
    total_time = end_time - start_time
    router_overhead = total_time - backend_time
    if total_time > 0:
        logger.debug(
            f"[{request_id}] timing: total={total_time * 1000:.2f}ms, "
            f"backend={backend_time * 1000:.2f}ms, "
            f"router_overhead={router_overhead * 1000:.2f}ms ({router_overhead / total_time * 100:.1f}%)"
        )

    # if debug_request:
    #    logger.debug(f"Finished the request with request id: {debug_request.headers.get('x-request-id', None)} at {time.time()}")
    # Store in semantic cache if applicable
    # Use the full response for non-streaming requests, or the last chunk for streaming
    if background_tasks and getattr(request.app.state, "callbacks", None):
        background_tasks.add_task(
            request.app.state.callbacks.post_request, request, full_response
        )


async def route_general_request(
    request: Request, endpoint: str, background_tasks: BackgroundTasks
):
    """
    Route the incoming request to the backend server and stream the response back to the client.

    This function extracts the requested model from the request body and retrieves the
    corresponding endpoints. It uses routing logic to determine the best server URL to handle
    the request, then streams the request to that server. If the requested model is not available,
    it returns an error response.

    Args:
        request (Request): The incoming HTTP request.
        endpoint (str): The endpoint to which the request should be routed.

    Returns:
        StreamingResponse: A response object that streams data from the backend server to the client.
    """
    if isinstance(request.app.state.router, DisaggregatedPrefillRouter):
        response = await route_disaggregated_prefill_request(
            request, endpoint, background_tasks
        )
        return response
    # Same as vllm, Get request_id from X-Request-Id header if available
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request_body = await request.body()
    request_json = json.loads(request_body) if request_body else {}

    if request.query_params:
        request_endpoint = request.query_params.get("id")
    else:
        request_endpoint = None

    if getattr(request.app.state, "callbacks", None) and (
        response_overwrite := request.app.state.callbacks.pre_request(
            request, request_body, request_json
        )
    ):
        response_overwrite.headers["X-Request-Id"] = request_id
        return response_overwrite

    requested_model = request_json.get("model", None)
    if requested_model is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request: missing 'model' in request body."},
            headers={"X-Request-Id": request_id},
        )

    # Apply request rewriting if enabled
    if is_request_rewriter_initialized():
        rewriter = get_request_rewriter()
        rewritten_body = rewriter.rewrite_request(
            request_body, requested_model, endpoint
        )
        logger.info(f"Request for model {requested_model} was rewritten")
        request_body = rewritten_body
        # Update request_json if the body was rewritten
        try:
            request_json = json.loads(request_body)
        except JSONDecodeError:
            logger.warning("Failed to parse rewritten request body as JSON")
            raise HTTPException(
                status_code=400, detail="Request body is not JSON parsable."
            )

    service_discovery = get_service_discovery()
    endpoints = service_discovery.get_endpoint_info()

    aliases = getattr(service_discovery, "aliases", None)
    if aliases and requested_model in aliases.keys():
        requested_model = aliases[requested_model]
        request_body = replace_model_in_request_body(request_json, requested_model)
        update_content_length(request, request_body)

    # Check if model has ever been seen (even if currently scaled to zero)
    model_ever_existed = False
    if hasattr(service_discovery, "has_ever_seen_model"):
        model_ever_existed = service_discovery.has_ever_seen_model(requested_model)

    if not request_endpoint:
        endpoints = list(
            filter(
                lambda x: requested_model in x.model_names and not x.sleep,
                endpoints,
            )
        )
        engine_stats = request.app.state.engine_stats_scraper.get_engine_stats()
        request_stats = request.app.state.request_stats_monitor.get_request_stats(
            time.time()
        )
    else:
        endpoints = list(
            filter(
                lambda x: requested_model in x.model_names
                and x.Id == request_endpoint
                and not x.sleep,
                endpoints,
            )
        )

    # Track all valid incoming requests
    num_incoming_requests_total.labels(model=requested_model).inc()

    if not endpoints:
        if not model_ever_existed:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"Model '{requested_model}' not found. Available models can be listed at /v1/models."
                },
                headers={"X-Request-Id": request_id},
            )
        else:
            # Model existed before but is now scaled to zero
            return JSONResponse(
                status_code=503,
                content={
                    "error": f"Model '{requested_model}' is temporarily unavailable. Please try again later."
                },
                headers={"X-Request-Id": request_id},
            )

    logger.debug(f"Routing request {request_id} for model: {requested_model}")

    # Extract actual session ID from request headers for logging
    session_id = request.app.state.router.extract_session_id(request, request_json)
    session_id_display = session_id if session_id is not None else "None"

    logger.debug(f"Routing request {request_id} for model: {requested_model}")

    # Track URLs we've already tried to avoid retrying the same backend
    tried_urls = set()

    # Implement retry logic for backend failures
    # Use a reasonable max retry count (not just len(endpoints) since endpoints can change)
    max_retries = 5
    last_error = None

    for attempt in range(max_retries):
        if attempt > 0:
            logger.info(
                f"Retry attempt {attempt + 1}/{max_retries} for request {request_id}"
            )
            # On retry, get fresh endpoint list (respects circuit breaker state)
            endpoints = service_discovery.get_endpoint_info()
            endpoints = list(
                filter(
                    lambda x: requested_model in x.model_names and not x.sleep,
                    endpoints,
                )
            )
            # Filter out already-tried endpoints
            endpoints = [ep for ep in endpoints if ep.url not in tried_urls]

            if not endpoints:
                logger.error(
                    f"No more backends available for request {request_id} "
                    f"(tried: {len(tried_urls)}, all circuit-broken or failed)"
                )
                break

            # Refresh stats
            engine_stats = request.app.state.engine_stats_scraper.get_engine_stats()
            request_stats = request.app.state.request_stats_monitor.get_request_stats(
                time.time()
            )

        if not endpoints:
            logger.error(f"No backends available for request {request_id}")
            break

        # Select backend using routing logic
        if request_endpoint:
            server_url = endpoints[0].url
            logger.debug(
                f"Routing request {request_id} to engine with Id: {endpoints[0].Id}"
            )
        elif isinstance(
            request.app.state.router, (KvawareRouter, PrefixAwareRouter, SessionRouter)
        ):
            server_url = await request.app.state.router.route_request(
                endpoints, engine_stats, request_stats, request, request_json
            )
        else:
            server_url = request.app.state.router.route_request(
                endpoints, engine_stats, request_stats, request
            )

        tried_urls.add(server_url)
        logger.info(
            f"Routing request {request_id} with session id {session_id_display} to {server_url}"
        )

        try:
            is_streaming = request_json.get("stream", False)
            stream_generator = process_request(
                request,
                request_body,
                server_url,
                request_id,
                endpoint,
                background_tasks,
                is_streaming=is_streaming,
            )
            headers, status = await anext(stream_generator)
            headers_dict = {key: value for key, value in headers.items()}
            headers_dict["X-Request-Id"] = request_id

            # Success! Return the streaming response
            return StreamingResponse(
                stream_generator,
                status_code=status,
                headers=headers_dict,
                media_type="text/event-stream",
            )

        except Exception as e:
            last_error = e
            error_msg = str(e)

            # Check if this is a backend failure that we should retry
            if "Backend" in error_msg and (
                "failed" in error_msg or "HTTP" in error_msg
            ):
                logger.warning(
                    f"Backend {server_url} failed for request {request_id}: {error_msg}"
                )
                # Continue to next iteration - circuit breaker already updated
                logger.info(
                    f"Will retry request {request_id}, tried backends: {tried_urls}"
                )
            else:
                # Not a backend failure, don't retry
                logger.error(
                    f"Non-recoverable error for request {request_id}: {error_msg}"
                )
                break

    # If we get here, all retries failed
    logger.error(
        f"All retries failed for request {request_id}. Last error: {last_error}"
    )

    if last_error and "Backend" in str(last_error):
        return JSONResponse(
            status_code=503,
            content={
                "error": f"All backends failed for model '{requested_model}'. Last error: {str(last_error)}"
            },
            headers={"X-Request-Id": request_id},
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Request failed for model '{requested_model}': {str(last_error)}"
            },
            headers={"X-Request-Id": request_id},
        )


async def send_request_to_prefiller(
    client: httpx.AsyncClient, endpoint: str, req_data: dict, request_id: str
):
    """Send a request to a prefiller service."""
    req_data = req_data.copy()
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    response = await client.post(endpoint, json=req_data, headers=headers)
    response.raise_for_status()
    return response.json()


async def send_request_to_decode(
    client: httpx.AsyncClient, endpoint: str, req_data: dict, request_id: str
):
    """Asynchronously stream the response from a service using a persistent client."""
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    async with client.stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def route_disaggregated_prefill_request(
    request: Request,
    endpoint: str,
    background_tasks: BackgroundTasks,
):
    in_router_time = time.time()
    # Same as vllm, Get request_id from X-Request-Id header if available
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request_json = await request.json()

    orig_max_tokens = request_json.get("max_tokens", 0)
    request_json["max_tokens"] = 1
    st = time.time()
    try:
        await send_request_to_prefiller(
            request.app.state.prefill_client, endpoint, request_json, request_id
        )
        et = time.time()
        logger.info(f"{request_id} prefill time (TTFT): {et - st:.4f}")
        logger.info(
            f"Routing request {request_id} with session id None to {request.app.state.prefill_client._base_url} at {et}, process time = {et - in_router_time:.4f}"
        )
        request_json["max_tokens"] = orig_max_tokens
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error in prefiller: {e}", exc_info=True)
        return JSONResponse(
            status_code=e.response.status_code,
            content={
                "error": {
                    "message": f"Prefiller error: {str(e)}",
                    "type": "prefiller_error",
                    "code": e.response.status_code,
                }
            },
            headers={"X-Request-Id": request_id},
        )
    except Exception as e:
        logger.error(f"Unexpected error in prefiller: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": f"Prefiller error: {str(e)}",
                    "type": "prefiller_error",
                    "code": 500,
                }
            },
            headers={"X-Request-Id": request_id},
        )

    async def generate_stream():
        try:
            async for chunk in send_request_to_decode(
                request.app.state.decode_client, endpoint, request_json, request_id
            ):
                yield chunk
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in decoder: {e}", exc_info=True)
            error_text = str(e)
            # Yield error as JSON response
            error_response = {
                "error": {
                    "message": f"Decoder error: {error_text}",
                    "type": "decoder_error",
                    "code": e.response.status_code,
                }
            }
            yield json.dumps(error_response).encode("utf-8")
        except Exception as e:
            logger.error(f"Unexpected error in decoder: {e}", exc_info=True)
            # Yield error as JSON response
            error_response = {
                "error": {
                    "message": f"Decoder error: {str(e)}",
                    "type": "decoder_error",
                    "code": 500,
                }
            }
            yield json.dumps(error_response).encode("utf-8")

    curr_time = time.time()
    logger.info(
        f"Routing request {request_id} with session id None to {request.app.state.decode_client._base_url} at {curr_time}, process time = {curr_time - et:.4f}"
    )

    return StreamingResponse(
        generate_stream(),
        media_type="application/json",
        headers={"X-Request-Id": request_id},
    )


async def route_sleep_wakeup_request(
    request: Request,
    endpoint: str,
    background_tasks: BackgroundTasks,
):
    in_router_time = time.time()
    # Same as vllm, Get request_id from X-Request-Id header if available
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())

    if request.query_params:
        request_endpoint = request.query_params.get("id")
    else:
        request_endpoint = None

    if request_endpoint is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid request: missing target Engine Id."},
            headers={"X-Request-Id": request_id},
        )

    service_discovery = get_service_discovery()
    endpoints = service_discovery.get_endpoint_info()

    endpoints = list(filter(lambda x: x.Id == request_endpoint, endpoints))
    if not endpoints:
        return JSONResponse(
            status_code=400,
            content={"error": f"Engine with Id {request_endpoint} not found."},
        )
    logger.debug(f"Routing request {request_id} to engine with Id: {endpoints[0].Id}")

    server_url = endpoints[0].url
    curr_time = time.time()
    logger.info(
        f"Routing request {request_id} to {server_url} at {curr_time}, process time = {curr_time - in_router_time:.4f}"
    )

    headers = {
        "X-Request-Id": request_id,
    }

    if VLLM_API_KEY := os.getenv("VLLM_API_KEY"):
        logger.info("Using vllm server authentication")
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    url = server_url + endpoint

    client = request.app.state.httpx_client_wrapper()
    if endpoint == "/is_sleeping":
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    else:
        request_body = await request.body()
        if request_body:
            req_data = json.loads(request_body)
            response = await client.post(url, json=req_data, headers=headers)
        else:
            response = await client.post(url, headers=headers)
        response.raise_for_status()

        pod_name = endpoints[0].pod_name
        if endpoint == "/sleep":
            service_discovery.add_sleep_label(pod_name)
        elif endpoint == "/wake_up":
            service_discovery.remove_sleep_label(pod_name)

        return JSONResponse(
            status_code=response.status_code,
            content={"status": "success"},
            headers={"X-Request-Id": request_id},
        )


async def route_general_transcriptions(
    request: Request,
    endpoint: str,  # "/v1/audio/transcriptions"
    background_tasks: BackgroundTasks,
):
    """Handles audio transcription requests by parsing form data and proxying to backend."""

    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))

    # --- 1. Form parsing ---
    try:
        form = await request.form()

        # Extract parameters from the form data
        file: UploadFile = form["file"]
        model: str = form["model"]
        prompt: Optional[str] = form.get("prompt", None)
        response_format: Optional[str] = form.get("response_format", "json")
        temperature_str: Optional[str] = form.get("temperature", None)
        temperature: Optional[float] = (
            float(temperature_str) if temperature_str is not None else None
        )
        language: Optional[str] = form.get("language", "en")
    except KeyError as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid request: missing '{e.args[0]}' in form data."},
        )

    logger.debug("==== Enter audio_transcriptions ====")
    logger.debug("Received upload: %s (%s)", file.filename, file.content_type)
    logger.debug(
        "Params: model=%s prompt=%r response_format=%r temperature=%r language=%s",
        model,
        prompt,
        response_format,
        temperature,
        language,
    )

    # --- 2. Service Discovery and Routing ---
    # Access singletons via request.app.state for consistent style
    service_discovery = (
        get_service_discovery()
    )  # This one is often still accessed directly via its get function
    router = request.app.state.router  # Access router from app.state
    engine_stats_scraper = (
        request.app.state.engine_stats_scraper
    )  # Access engine_stats_scraper from app.state
    request_stats_monitor = (
        request.app.state.request_stats_monitor
    )  # Access request_stats_monitor from app.state

    endpoints = service_discovery.get_endpoint_info()

    # filter the endpoints url by model name
    transcription_endpoints = []
    for ep in endpoints:
        for model_name in ep.model_names:
            if model == model_name and not ep.sleep:
                transcription_endpoints.append(ep)

    if not transcription_endpoints:
        logger.error("No transcription backend available for model %s", model)
        return JSONResponse(
            status_code=404,
            content={"error": f"No transcription backend for model {model}"},
        )

    # grab the current engine and request stats
    engine_stats = engine_stats_scraper.get_engine_stats()
    request_stats = request_stats_monitor.get_request_stats(time.time())

    # pick one using the router's configured logic (roundrobin, least-loaded, etc.)
    chosen_url = router.route_request(
        transcription_endpoints,
        engine_stats,
        request_stats,
        request,
    )

    logger.debug("Proxying transcription request to %s", chosen_url)

    # --- 3. Prepare and Proxy the Request ---
    payload_bytes = await file.read()
    files = {"file": (file.filename, payload_bytes, file.content_type)}

    data = {"model": model, "language": language}

    if prompt:
        data["prompt"] = prompt

    if response_format:
        data["response_format"] = response_format

    if temperature is not None:
        data["temperature"] = str(temperature)

    logger.info("Proxying transcription request for model %s to %s", model, chosen_url)

    try:
        client = request.app.state.httpx_client_wrapper()

        # httpx uses 'files' dict for multipart file upload
        httpx_files = {
            key: (filename, content, content_type)
            for key, (filename, content, content_type) in files.items()
        }

        backend_response = await client.post(
            f"{chosen_url}{endpoint}",
            files=httpx_files,
            data=data,
            timeout=300.0,
        )

        # --- 4. Return the response ---
        response_content = backend_response.json()
        headers = {
            k: v
            for k, v in backend_response.headers.items()
            if k.lower() not in ("content-encoding", "transfer-encoding", "connection")
        }

        headers["X-Request-Id"] = request_id

        return JSONResponse(
            content=response_content,
            status_code=backend_response.status_code,
            headers=headers,
        )
    except httpx.HTTPStatusError as response_error:
        try:
            error_content = response_error.response.json()
        except json.JSONDecodeError:
            # If JSON parsing fails, get text content
            text_content = response_error.response.text
            error_content = {"error": text_content}
        return JSONResponse(
            status_code=response_error.response.status_code, content=error_content
        )
    except httpx.RequestError as client_error:
        return JSONResponse(
            status_code=503,
            content={"error": f"Failed to connect to backend: {str(client_error)}"},
        )
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )
