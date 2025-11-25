from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from vllm_router.service_discovery import get_service_discovery, StaticServiceDiscovery
from vllm_router.log import init_logger

logger = init_logger(__name__)

config_router = APIRouter(prefix="/config", tags=["config"])


class AddBackendRequest(BaseModel):
    url: str


class RemoveBackendRequest(BaseModel):
    url: str


@config_router.post("/backends/add")
async def add_backend(request: AddBackendRequest):
    logger.info(f"received POST /config/backends/add request for url: {request.url}")
    service_discovery = get_service_discovery()
    logger.debug(f"service_discovery type: {type(service_discovery).__name__}")

    if not isinstance(service_discovery, StaticServiceDiscovery):
        logger.error(f"service discovery is not static, got {type(service_discovery).__name__}")
        raise HTTPException(status_code=400, detail="Service discovery is not static")

    logger.info(f"calling service_discovery.add_backend for {request.url}")
    await service_discovery.add_backend(url=request.url)
    logger.info(f"successfully completed add_backend for {request.url}")
    return {"status": "success", "message": f"Backend {request.url} added"}


@config_router.post("/backends/remove")
async def remove_backend(request: RemoveBackendRequest):
    service_discovery = get_service_discovery()
    if not isinstance(service_discovery, StaticServiceDiscovery):
        raise HTTPException(status_code=400, detail="Service discovery is not static")

    service_discovery.remove_backend(url=request.url)
    return {"status": "success", "message": f"Backend {request.url} removed"}
