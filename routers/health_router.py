import json
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from vllm_router.dynamic_config import get_dynamic_config_watcher
from vllm_router.service_discovery import get_service_discovery
from vllm_router.stats.engine_stats import get_engine_stats_scraper
from vllm_router.version import __version__

health_router = APIRouter()


@health_router.get("/version")
async def show_version():
    ver = {"version": __version__}
    return JSONResponse(content=ver)


@health_router.get("/health")
async def health() -> Response:
    """
    Endpoint to check the health status of various components.

    This function verifies the health of the service discovery module and
    the engine stats scraper. If either component is down, it returns a
    503 response with the appropriate status message. If both components
    are healthy, it returns a 200 OK response.

    Returns:
        Response: A JSONResponse with status code 503 if a component is
        down, or a plain Response with status code 200 if all components
        are healthy.
    """

    if not get_service_discovery().get_health():
        return JSONResponse(
            content={"status": "Service discovery module is down."}, status_code=503
        )
    if not get_engine_stats_scraper().get_health():
        return JSONResponse(
            content={"status": "Engine stats scraper is down."}, status_code=503
        )

    if get_dynamic_config_watcher() is not None:
        dynamic_config = get_dynamic_config_watcher().get_current_config()
        return JSONResponse(
            content={
                "status": "healthy",
                "dynamic_config": json.loads(dynamic_config.to_json_str()),
            },
            status_code=200,
        )
    else:
        return JSONResponse(content={"status": "healthy"}, status_code=200)


@health_router.get("/backends")
async def backends() -> Response:
    """
    Endpoint to retrieve the health status of all backends.

    Returns detailed information about each backend including:
    - URL and model name
    - Health status (healthy/unhealthy)
    - Consecutive failure count
    - Failure threshold before removal
    - Model label and type

    Returns:
        Response: A JSONResponse containing a list of backend health statuses
    """
    service_discovery = get_service_discovery()

    # Check if the service discovery has the method (only StaticServiceDiscovery has it)
    if hasattr(service_discovery, "get_backend_health_status"):
        backend_status = service_discovery.get_backend_health_status()
        return JSONResponse(
            content={
                "backends": backend_status,
                "total": len(backend_status),
                "healthy": sum(1 for b in backend_status if b["healthy"]),
                "unhealthy": sum(1 for b in backend_status if not b["healthy"]),
            },
            status_code=200,
        )
    else:
        return JSONResponse(
            content={"error": "Backend health status not available for this service discovery type"},
            status_code=501,
        )
