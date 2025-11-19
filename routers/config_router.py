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
    service_discovery = get_service_discovery()
    if not isinstance(service_discovery, StaticServiceDiscovery):
        raise HTTPException(status_code=400, detail="Service discovery is not static")

    await service_discovery.add_backend(url=request.url)
    return {"status": "success", "message": f"Backend {request.url} added"}


@config_router.post("/backends/remove")
async def remove_backend(request: RemoveBackendRequest):
    service_discovery = get_service_discovery()
    if not isinstance(service_discovery, StaticServiceDiscovery):
        raise HTTPException(status_code=400, detail="Service discovery is not static")

    service_discovery.remove_backend(url=request.url)
    return {"status": "success", "message": f"Backend {request.url} removed"}
