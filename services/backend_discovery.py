import asyncio
import json
import os
import re
import threading
from typing import Dict, List, Optional, Set

import aiohttp

from vllm_router.log import init_logger
from vllm_router.service_discovery import get_service_discovery

logger = init_logger(__name__)

_global_backend_discovery: Optional["BackendDiscoveryService"] = None


class BackendDiscoveryService:
    """
    Service for automatically discovering and adding vLLM backends
    from Tailscale network status.
    """

    def __init__(
        self,
        tailscale_status_file: str,
        discovery_interval: int,
        port_range: str,
        timeout: int = 2,
    ):
        self.tailscale_status_file = tailscale_status_file
        self.discovery_interval = discovery_interval
        self.port_range = self._parse_port_range(port_range)
        self.timeout = timeout
        self._running = False
        self._discovery_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

        # Track discovered backends to avoid duplicate additions
        self._discovered_backends: Set[str] = set()

    def _parse_port_range(self, port_range: str) -> range:
        """Parse port range string like '8000-8010' into a range object."""
        match = re.match(r"^(\d+)-(\d+)$", port_range)
        if not match:
            raise ValueError(f"Invalid port range format: {port_range}")
        start, end = int(match.group(1)), int(match.group(2))
        if start > end:
            raise ValueError(f"Start port {start} cannot be greater than end port {end}")
        return range(start, end + 1)

    def parse_tailscale_status(self) -> List[Dict]:
        """
        Parse Tailscale status JSON file and extract online vllm-proxy peers.

        Returns:
            List of peer dictionaries with DNS names and other info
        """
        try:
            if not os.path.exists(self.tailscale_status_file):
                logger.warning(f"Tailscale status file not found: {self.tailscale_status_file}")
                return []

            with open(self.tailscale_status_file, "r", encoding="utf-8") as f:
                status_data = json.load(f)

            peers = []
            if "Peer" in status_data:
                for peer_data in status_data["Peer"].values():
                    if (
                        peer_data.get("Online", False)
                        and peer_data.get("HostName", "").startswith("vllm-proxy")
                    ):
                        dns_name = peer_data.get("DNSName", "")
                        if dns_name:
                            # Remove trailing dot if present
                            dns_name = dns_name.rstrip(".")
                            peers.append(peer_data)

            logger.info(f"Found {len(peers)} online vllm-proxy peers")
            return peers

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error parsing tailscale status file: {e}")
            return []

    async def test_backend_health(self, peer_dns: str, port: int) -> Optional[str]:
        """
        Test if a backend is healthy by checking the /v1/models endpoint.

        Args:
            peer_dns: DNS name of the peer
            port: Port to test

        Returns:
            Backend URL if healthy, None otherwise
        """
        backend_url = f"http://{peer_dns}:{port}"
        models_url = f"{backend_url}/v1/models"

        try:
            # Use the app's aiohttp client wrapper if available
            from vllm_router.aiohttp_client import AiohttpClientWrapper

            client_wrapper = AiohttpClientWrapper()
            if not client_wrapper._session:
                await client_wrapper.start()
            client = client_wrapper()

        except ImportError:
            # Fallback to creating our own client
            client = aiohttp.ClientSession()

        try:
            headers = {}
            if openai_api_key := os.getenv("OPENAI_API_KEY"):
                headers["Authorization"] = f"Bearer {openai_api_key}"
            logger.debug(f"Testing backend health at {models_url}")
            async with client.get(
                models_url, headers=headers, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    logger.debug(f"Backend health check SUCCESS: {backend_url}")
                    return backend_url
                else:
                    logger.debug(f"Backend health check FAILED: {backend_url} (HTTP {response.status})")
                    return None

        except asyncio.TimeoutError:
            logger.debug(f"Backend health check TIMEOUT: {backend_url}")
            return None
        except Exception as e:
            logger.debug(f"Backend health check ERROR: {backend_url} - {e}")
            return None
        finally:
            # Only close client if we created it ourselves
            if "client_wrapper" not in locals():
                await client.close()

    async def discover_backends(self) -> List[str]:
        """
        Discover healthy backends from Tailscale peers.

        Returns:
            List of healthy backend URLs
        """
        peers = self.parse_tailscale_status()
        if not peers:
            return []

        healthy_backends = []

        # Test all peers and ports in parallel
        tasks = []
        for peer in peers:
            peer_dns = peer.get("DNSName", "").rstrip(".")
            for port in self.port_range:
                tasks.append(self.test_backend_health(peer_dns, port))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, str):  # Healthy backend URL
                    healthy_backends.append(result)

        return healthy_backends

    async def add_healthy_backends(self, healthy_backends: List[str]) -> None:
        """
        Add healthy backends to the service discovery.

        Args:
            healthy_backends: List of healthy backend URLs
        """
        service_discovery = get_service_discovery()
        if not service_discovery:
            logger.error("Service discovery not initialized")
            return

        for backend_url in healthy_backends:
            if backend_url not in self._discovered_backends:
                try:
                    logger.info(f"Adding discovered backend: {backend_url}")
                    await service_discovery.add_backend(backend_url)
                    self._discovered_backends.add(backend_url)
                except Exception as e:
                    logger.error(f"Failed to add backend {backend_url}: {e}")

    async def discovery_loop(self) -> None:
        """Main discovery loop that runs periodically."""
        logger.info(f"Starting backend discovery loop (interval: {self.discovery_interval}s)")
        
        while self._running:
            try:
                logger.info("=== Starting backend discovery cycle ===")
                
                # Discover healthy backends
                healthy_backends = await self.discover_backends()
                
                # Add new healthy backends
                await self.add_healthy_backends(healthy_backends)
                
                logger.info(f"Discovery cycle completed. Found {len(healthy_backends)} healthy backends")
                logger.info("=== Backend discovery cycle completed ===")
                
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}", exc_info=True)
            
            # Wait for next cycle
            for _ in range(self.discovery_interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

    def start_discovery_loop(self) -> None:
        """Start the discovery loop in a separate thread with event loop."""
        if self._running:
            logger.warning("Backend discovery already running")
            return

        self._running = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        
        # Schedule the discovery loop
        self._discovery_task = asyncio.run_coroutine_threadsafe(
            self.discovery_loop(), self._loop
        )
        
        logger.info("Backend discovery service started")

    def _run_event_loop(self) -> None:
        """Run the event loop in the separate thread."""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    def stop(self) -> None:
        """Stop the backend discovery service."""
        if not self._running:
            return

        logger.info("Stopping backend discovery service")
        self._running = False

        # Cancel the discovery task
        if self._discovery_task and not self._discovery_task.done():
            self._discovery_task.cancel()

        # Stop the event loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        # Wait for the thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        logger.info("Backend discovery service stopped")


def initialize_backend_discovery(
    tailscale_status_file: str,
    discovery_interval: int,
    port_range: str,
    timeout: int = 2,
) -> BackendDiscoveryService:
    """
    Initialize the global backend discovery service.

    Args:
        tailscale_status_file: Path to Tailscale status JSON file
        discovery_interval: Discovery interval in seconds
        port_range: Port range to test (e.g., "8000-8010")
        timeout: Health check timeout in seconds

    Returns:
        Initialized BackendDiscoveryService instance
    """
    global _global_backend_discovery
    
    if _global_backend_discovery is not None:
        logger.warning("Backend discovery service already initialized")
        return _global_backend_discovery

    _global_backend_discovery = BackendDiscoveryService(
        tailscale_status_file=tailscale_status_file,
        discovery_interval=discovery_interval,
        port_range=port_range,
        timeout=timeout,
    )
    
    _global_backend_discovery.start_discovery_loop()
    return _global_backend_discovery


def get_backend_discovery() -> Optional[BackendDiscoveryService]:
    """
    Get the global backend discovery service.

    Returns:
        BackendDiscoveryService instance if initialized, None otherwise
    """
    global _global_backend_discovery
    return _global_backend_discovery


def cleanup_backend_discovery() -> None:
    """Clean up the global backend discovery service."""
    global _global_backend_discovery
    
    if _global_backend_discovery is not None:
        _global_backend_discovery.stop()
        _global_backend_discovery = None
