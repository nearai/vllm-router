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
from typing import Dict, Set, Optional

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


def _get_connection_ids(transport) -> Set[str]:
    """
    Get a set of unique connection identifiers from the transport pool.

    This function creates unique identifiers for each connection in the pool
    based on the connection object's identity and remote address. This allows
    us to track individual connections rather than just counting pool size,
    which helps reduce (but not eliminate) race conditions in connection
    creation detection.

    Note: This is a best-effort approach. In highly concurrent environments,
    there may still be race conditions where:
    - Connection A is created by Request 1
    - Connection B is created by Request 2
    - Connection A is closed before we check connection IDs
    - Both requests might incorrectly think they created new connections

    Returns:
        Set[str]: A set of unique connection identifiers, or empty set if
                 connection tracking is not available.
    """
    try:
        if hasattr(transport, "_pool") and transport._pool:
            pool = transport._pool
            if hasattr(pool, "connections"):
                # Create unique IDs based on connection object identity and remote address
                conn_ids = set()
                for conn in pool.connections:
                    # Use a combination of object id and remote address for uniqueness
                    remote_addr = getattr(conn, "_remote_addr", None)
                    if remote_addr:
                        conn_id = f"{id(conn)}:{remote_addr}"
                    else:
                        conn_id = str(id(conn))
                    conn_ids.add(conn_id)
                return conn_ids
    except Exception as e:
        logger.debug(f"Failed to get connection IDs: {e}")
    return set()


class HttpxClientWrapper:
    """
    Wrapper for httpx AsyncClient with HTTP/2 support.

    HTTP/2 enables connection multiplexing, allowing multiple concurrent
    streams over a single TCP connection. This dramatically improves
    performance for concurrent streaming requests compared to HTTP/1.1.
    """

    async_client = None
    # Default timeout values (can be overridden via start())
    connect_timeout: float = 5.0
    read_timeout: Optional[float] = 300.0  # 5 minutes between chunks

    def start(
        self,
        connect_timeout: float = 30.0,
        read_timeout: Optional[float] = 300.0,
    ):
        """
        Instantiate the client. Call from the FastAPI startup hook.

        Args:
            connect_timeout: Timeout in seconds for establishing connections.
                This should be short to quickly detect dead backends.
            read_timeout: Timeout in seconds between receiving chunks.
                For streaming LLM requests, this is the max time allowed between tokens.
                Set to None for no read timeout (not recommended).
        """
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

        # Create custom transport with connection logging
        self._transport = httpx.AsyncHTTPTransport(
            http2=True,
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=100,
                keepalive_expiry=120,  # Keep connections alive for 2 minutes
            ),
        )

        # Configure granular timeouts:
        # - connect: time to establish TCP connection (short, to detect dead backends)
        # - read: time between receiving bytes (longer, for slow token generation)
        # - write: time to send data (generous default)
        # - pool: time to acquire connection from pool
        timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=30.0,
            pool=10.0,
        )

        self.async_client = httpx.AsyncClient(
            transport=self._transport,
            timeout=timeout,
        )
        logger.info(
            f"httpx AsyncClient instantiated with HTTP/2 support. "
            f"Timeouts: connect={connect_timeout}s, read={read_timeout}s. "
            f"Id {id(self.async_client)}"
        )

    def get_pool_size(self) -> int:
        """Get the current connection pool size."""
        return _get_pool_size(self._transport)

    def get_connection_ids(self) -> Set[str]:
        """Get a set of unique connection identifiers from the pool."""
        return _get_connection_ids(self._transport)

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
            logger.info(f"Closing httpx AsyncClient. Id: {id(self.async_client)}")
            await self.async_client.aclose()
            self.async_client = None
            logger.info("httpx AsyncClient closed")

    def __call__(self):
        """Calling the instantiated HttpxClientWrapper returns the wrapped singleton."""
        assert self.async_client is not None
        return self.async_client
