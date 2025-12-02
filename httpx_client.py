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
import time
from typing import Dict

import httpx

from vllm_router.log import init_logger

logger = init_logger(__name__)


# Track connection count per origin to detect new connections
_connection_counts: Dict[str, int] = {}


def _get_pool_size(transport) -> int:
    """Get the current connection pool size from transport."""
    try:
        if hasattr(transport, "_pool") and transport._pool:
            pool = transport._pool
            if hasattr(pool, "connections"):
                return len(pool.connections)
    except Exception:
        pass
    return -1


class HttpxClientWrapper:
    """
    Wrapper for httpx AsyncClient with HTTP/2 support.

    HTTP/2 enables connection multiplexing, allowing multiple concurrent
    streams over a single TCP connection. This dramatically improves
    performance for concurrent streaming requests compared to HTTP/1.1.
    """

    async_client = None

    def start(self):
        """Instantiate the client. Call from the FastAPI startup hook."""
        # Create custom transport with connection logging
        self._transport = httpx.AsyncHTTPTransport(
            http2=True,
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=100,
                keepalive_expiry=120,  # Keep connections alive for 2 minutes
            ),
        )
        self.async_client = httpx.AsyncClient(
            transport=self._transport,
            timeout=httpx.Timeout(None),  # No timeout for streaming requests
        )
        logger.info(
            f"httpx AsyncClient instantiated with HTTP/2 support. "
            f"Id {id(self.async_client)}"
        )

    def get_pool_size(self) -> int:
        """Get the current connection pool size."""
        return _get_pool_size(self._transport)

    def log_pool_status(self, context: str = ""):
        """Log the current connection pool status."""
        pool_size = self.get_pool_size()
        if pool_size >= 0:
            logger.debug(
                f"Connection pool status{' (' + context + ')' if context else ''}: "
                f"{pool_size} connections"
            )

    async def stop(self):
        """Gracefully shutdown. Call from FastAPI shutdown hook."""
        if self.async_client:
            logger.info(
                f"Closing httpx AsyncClient. Id: {id(self.async_client)}"
            )
            await self.async_client.aclose()
            self.async_client = None
            logger.info("httpx AsyncClient closed")

    def __call__(self):
        """Calling the instantiated HttpxClientWrapper returns the wrapped singleton."""
        assert self.async_client is not None
        return self.async_client
