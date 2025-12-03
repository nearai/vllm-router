import json
import os
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from vllm_router.dynamic_config import get_dynamic_config_watcher
from vllm_router.graceful_shutdown import get_shutdown_manager
from vllm_router.service_discovery import get_service_discovery
from vllm_router.stats.engine_stats import get_engine_stats_scraper
from vllm_router.log import init_logger

logger = init_logger(__name__)

health_router = APIRouter()

GIT_REV_PATH = "/etc/.GIT_REV"


def _read_git_rev() -> str:
    """Read git revision from /etc/.GIT_REV file."""
    try:
        if os.path.exists(GIT_REV_PATH):
            with open(GIT_REV_PATH, "r") as f:
                return f.read().strip()
    except Exception:
        pass
    return "unknown"


GIT_REV = _read_git_rev()


@health_router.get("/version")
async def show_version():
    ver = {"version": GIT_REV, "type": "router"}
    return JSONResponse(content=ver)


@health_router.get("/health")
async def health() -> Response:
    """
    Endpoint to check the health status of various components.

    This function verifies the health of the service discovery module and
    the engine stats scraper. If either component is down, it returns a
    503 response with the appropriate status message. If both components
    are healthy, it returns a 200 OK response.

    During graceful shutdown, returns 503 to signal load balancers to stop
    sending traffic to this instance.

    Returns:
        Response: A JSONResponse with status code 503 if a component is
        down or the server is shutting down, or a plain Response with
        status code 200 if all components are healthy.
    """
    # Check if server is shutting down
    shutdown_manager = get_shutdown_manager()
    if shutdown_manager is not None and shutdown_manager.is_shutting_down:
        return JSONResponse(
            content={
                "status": "shutting_down",
                "in_flight_requests": shutdown_manager.in_flight_requests,
            },
            status_code=503,
            headers={"Connection": "close"},
        )

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
    - Circuit breaker state (closed/open/half_open)
    - Whether currently accepting requests
    - Cooldown remaining (if circuit is open)
    - Consecutive failure count
    - Failure threshold before removal
    - Model label and type

    Returns:
        Response: A JSONResponse containing a list of backend health statuses
    """
    logger.debug("received GET /backends request")
    service_discovery = get_service_discovery()

    # Check if the service discovery has the method (only StaticServiceDiscovery has it)
    if hasattr(service_discovery, "get_backend_health_status"):
        backend_status = service_discovery.get_backend_health_status()

        # Calculate summary statistics
        total = len(backend_status)
        healthy = sum(1 for b in backend_status if b.get("healthy", False))
        unhealthy = sum(1 for b in backend_status if not b.get("healthy", False))
        accepting = sum(1 for b in backend_status if b.get("accepting_requests", False))
        circuit_open = sum(
            1 for b in backend_status if b.get("circuit_state") == "open"
        )
        circuit_half_open = sum(
            1 for b in backend_status if b.get("circuit_state") == "half_open"
        )

        logger.info(
            f"returning {total} backends: {accepting} accepting, "
            f"{circuit_open} circuit-open, {circuit_half_open} half-open, "
            f"{unhealthy} permanently unhealthy"
        )
        return JSONResponse(
            content={
                "backends": backend_status,
                "total": total,
                "accepting_requests": accepting,
                "circuit_open": circuit_open,
                "circuit_half_open": circuit_half_open,
                "permanently_unhealthy": unhealthy,
            },
            status_code=200,
        )
    else:
        logger.warning(
            "backend health status not available for this service discovery type"
        )
        return JSONResponse(
            content={
                "error": "Backend health status not available for this service discovery type"
            },
            status_code=501,
        )
