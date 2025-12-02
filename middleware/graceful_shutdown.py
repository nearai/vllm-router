"""
Middleware for graceful shutdown handling.

This middleware:
1. Tracks all in-flight requests
2. Returns 503 Service Unavailable for new requests when shutdown is initiated
3. Allows health endpoints to continue responding during shutdown
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from vllm_router.graceful_shutdown import get_shutdown_manager
from vllm_router.log import init_logger

logger = init_logger(__name__)

# Endpoints that should continue to work during shutdown
# These are typically health check endpoints used by load balancers
SHUTDOWN_EXEMPT_PATHS = {
    "/health",
    "/version",
    "/backends",
    "/metrics",
}


class GracefulShutdownMiddleware(BaseHTTPMiddleware):
    """
    Middleware that enables graceful shutdown behavior.

    When shutdown is initiated:
    - New requests (except health checks) receive 503 Service Unavailable
    - In-flight requests are allowed to complete
    - Health endpoints continue to work but may indicate shutdown status
    """

    async def dispatch(self, request: Request, call_next):
        manager = get_shutdown_manager()

        # If no shutdown manager, just pass through
        if manager is None:
            return await call_next(request)

        path = request.url.path

        # Check if we're shutting down
        if manager.is_shutting_down:
            # Allow exempt paths (health checks, metrics)
            if path in SHUTDOWN_EXEMPT_PATHS:
                return await call_next(request)

            # Reject new requests with 503
            logger.info(f"Rejecting request to {path} - server is shutting down")
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Service is shutting down. Please retry with a different server.",
                    "detail": "The server is in graceful shutdown mode and not accepting new requests.",
                },
                headers={
                    "Retry-After": "5",
                    "Connection": "close",
                },
            )

        # Track this request
        manager.request_started()

        try:
            response = await call_next(request)
            return response
        finally:
            manager.request_completed()
