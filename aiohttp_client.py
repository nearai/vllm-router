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
import aiohttp
from aiohttp import TCPConnector
from aiohttp.resolver import AsyncResolver

from vllm_router.log import init_logger

logger = init_logger(__name__)

# DNS cache TTL in seconds
DNS_CACHE_TTL = 300


class AiohttpClientWrapper:

    async_client = None

    def start(self):
        """Instantiate the client. Call from the FastAPI startup hook."""
        # Use AsyncResolver (aiodns) for non-blocking DNS resolution
        # and configure DNS caching to avoid repeated lookups
        try:
            resolver = AsyncResolver()
            logger.info("Using AsyncResolver (aiodns) for DNS resolution")
        except Exception as e:
            logger.warning(f"Failed to create AsyncResolver: {e}, using default resolver")
            resolver = None

        connector = TCPConnector(
            limit=0,  # Unlimited connections for concurrency
            ttl_dns_cache=DNS_CACHE_TTL,  # Cache DNS lookups for 5 minutes
            keepalive_timeout=120,  # Keep connections alive for reuse (increased from 30)
            enable_cleanup_closed=True,  # Clean up closed connections
            resolver=resolver,
        )
        self.async_client = aiohttp.ClientSession(connector=connector)
        self.connector = connector  # Keep reference for stats logging
        logger.info(
            f"aiohttp ClientSession instantiated with DNS caching (TTL={DNS_CACHE_TTL}s), "
            f"keepalive_timeout=120s. Id {id(self.async_client)}"
        )

    async def stop(self):
        """Gracefully shutdown. Call from FastAPI shutdown hook."""
        logger.info(
            f"aiohttp async_client.closed: {self.async_client.closed} - Now close it. Id (will be unchanged): {id(self.async_client)}"
        )
        await self.async_client.close()
        logger.info(
            f"aiohttp async_client.closed: {self.async_client.closed}. Id (will be unchanged): {id(self.async_client)}"
        )
        self.async_client = None
        logger.info("aiohttp ClientSession closed")

    def __call__(self):
        """Calling the instantiated AiohttpClientWrapper returns the wrapped singleton."""
        # Ensure we don't use it if not started / running
        assert self.async_client is not None
        return self.async_client
