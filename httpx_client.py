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
import httpx

from vllm_router.log import init_logger

logger = init_logger(__name__)


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
        self.async_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(None),  # No timeout for streaming requests
            limits=httpx.Limits(
                max_connections=200,
                max_keepalive_connections=100,
            ),
        )
        logger.info(
            f"httpx AsyncClient instantiated with HTTP/2 support. "
            f"Id {id(self.async_client)}"
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
