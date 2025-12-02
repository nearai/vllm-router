import asyncio
import json
import os
import re
import threading
from typing import Dict, List, Optional, Set

import aiohttp

from vllm_router.log import init_logger
from vllm_router.service_discovery import get_service_discovery
from vllm_router import utils

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
            raise ValueError(
                f"Start port {start} cannot be greater than end port {end}"
            )
        return range(start, end + 1)

    def parse_tailscale_status(self) -> List[Dict]:
        """
        Parse Tailscale status JSON file and extract all online peers.
        Excludes vllm-router peers to prevent infinite routing loops.

        Returns:
            List of peer dictionaries with DNS names and other info
        """
        logger.debug(f"Parsing Tailscale status file: {self.tailscale_status_file}")
        try:
            if not os.path.exists(self.tailscale_status_file):
                logger.warning(
                    f"Tailscale status file not found: {self.tailscale_status_file}"
                )
                return []

            logger.debug(
                f"Reading Tailscale status file from {self.tailscale_status_file}"
            )
            with open(self.tailscale_status_file, "r", encoding="utf-8") as f:
                status_data = json.load(f)

            peers = []
            if "Peer" in status_data:
                peer_count = 0
                online_peer_count = 0
                candidate_peer_count = 0

                for peer_id, peer_data in status_data["Peer"].items():
                    peer_count += 1
                    is_online = peer_data.get("Online", False)
                    hostname = peer_data.get("HostName", "")

                    logger.debug(
                        f"Processing peer {peer_id}: {hostname} (online: {is_online})"
                    )

                    if not is_online:
                        logger.debug(f"Skipping offline peer: {hostname}")
                        continue

                    online_peer_count += 1

                    # Skip vllm-router peers to prevent infinite routing loops
                    if hostname.startswith("vllm-router"):
                        logger.debug(
                            f"Skipping vllm-router peer to prevent routing loops: {hostname}"
                        )
                        continue

                    dns_name = peer_data.get("DNSName", "")
                    if dns_name:
                        # Remove trailing dot if present
                        dns_name = dns_name.rstrip(".")
                        peers.append(peer_data)
                        candidate_peer_count += 1
                        logger.debug(
                            f"Found online candidate peer: {dns_name} (hostname: {hostname})"
                        )
                    else:
                        logger.debug(f"Skipping online peer {hostname} - no DNS name")

                logger.info(
                    f"Tailscale status parsed: {candidate_peer_count}/{online_peer_count}/{peer_count} "
                    f"(candidates/online/total) peers found"
                )
            else:
                logger.warning("No 'Peer' section found in Tailscale status file")

            logger.info(f"Found {len(peers)} online candidate peers to test")
            return peers

        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error parsing tailscale status file: {e}")
            logger.debug(
                f"Error details for {self.tailscale_status_file}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return []

    def _validate_vllm_models_response(self, data: dict, backend_url: str) -> bool:
        """
        Validate that the /v1/models response is from a vLLM backend.

        A valid vLLM response should have:
        - "object": "list"
        - "data": array of models where each model has:
          - "id": model identifier
          - "owned_by": "vllm"

        Args:
            data: The JSON response data
            backend_url: The backend URL (for logging)

        Returns:
            True if this is a valid vLLM backend response, False otherwise
        """
        # Check top-level structure
        if data.get("object") != "list":
            logger.debug(
                f"Backend {backend_url} response missing 'object': 'list' field"
            )
            return False

        models_data = data.get("data")
        if not isinstance(models_data, list):
            logger.debug(f"Backend {backend_url} response 'data' is not a list")
            return False

        if len(models_data) == 0:
            logger.debug(f"Backend {backend_url} has no models")
            return False

        # Validate each model has required fields and is owned by vllm
        for model in models_data:
            if not isinstance(model, dict):
                logger.debug(
                    f"Backend {backend_url} has invalid model entry (not a dict)"
                )
                return False

            model_id = model.get("id")
            if not model_id:
                logger.debug(f"Backend {backend_url} has model without 'id' field")
                return False

            owned_by = model.get("owned_by")
            if owned_by != "vllm":
                logger.debug(
                    f"Backend {backend_url} model '{model_id}' has owned_by='{owned_by}' (expected 'vllm')"
                )
                return False

        model_ids = [m.get("id") for m in models_data]
        logger.debug(
            f"Backend {backend_url} validated as vLLM backend with models: {model_ids}"
        )
        return True

    async def test_backend_health(self, peer_dns: str, port: int) -> Optional[str]:
        """
        Test if a backend is healthy by checking:
        1. /v1/models endpoint returns valid vLLM model data (owned_by: vllm)
        2. /v1/attestation/report endpoint is available

        Args:
            peer_dns: DNS name of the peer
            port: Port to test

        Returns:
            Backend URL if healthy and attestation is available, None otherwise
        """
        backend_url = f"http://{peer_dns}:{port}"
        models_url = f"{backend_url}/v1/models"

        logger.debug(f"Starting health test for backend at {backend_url}")
        logger.debug(f"Testing endpoint: {models_url}")

        # Create a new session for each health check since we run in a separate event loop
        async with aiohttp.ClientSession() as client:
            try:
                headers = {}
                if openai_api_key := os.getenv("OPENAI_API_KEY"):
                    headers["Authorization"] = f"Bearer {openai_api_key}"
                    logger.debug(
                        f"Using OpenAI API key for authentication with {backend_url}"
                    )
                else:
                    logger.debug(
                        f"No OpenAI API key found, proceeding without authentication for {backend_url}"
                    )

                logger.debug(
                    f"Sending GET request to {models_url} with timeout {self.timeout}s"
                )
                async with client.get(
                    models_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    logger.debug(
                        f"Received response from {models_url}: HTTP {response.status}"
                    )

                    if response.status != 200:
                        logger.debug(
                            f"Backend health check FAILED: {backend_url} (HTTP {response.status})"
                        )
                        logger.debug(
                            f"Response headers for {backend_url}: {dict(response.headers)}"
                        )
                        return None

                    # Parse and validate the models response
                    try:
                        data = await response.json()
                    except Exception as e:
                        logger.debug(
                            f"Backend {backend_url} returned invalid JSON: {e}"
                        )
                        return None

                    # Validate this is a vLLM backend (owned_by: vllm)
                    if not self._validate_vllm_models_response(data, backend_url):
                        logger.debug(
                            f"Backend {backend_url} is not a valid vLLM backend. Skipping."
                        )
                        return None

                    logger.debug(f"Backend models check SUCCESS: {backend_url}")

                    # Check attestation endpoint availability
                    logger.debug(f"Checking attestation endpoint for {backend_url}")
                    attestation_available = utils.check_attestation_available(
                        backend_url, self.timeout
                    )

                    if attestation_available:
                        logger.info(
                            f"Discovered healthy vLLM backend with attestation: {backend_url}"
                        )
                        return backend_url
                    else:
                        logger.warning(
                            f"Backend {backend_url} is a valid vLLM backend but attestation is not available. Skipping."
                        )
                        return None

            except asyncio.TimeoutError:
                logger.debug(
                    f"Backend health check TIMEOUT: {backend_url} (timeout: {self.timeout}s)"
                )
                return None
            except aiohttp.ClientError as e:
                logger.debug(
                    f"Backend health check CLIENT ERROR: {backend_url} - {type(e).__name__}: {e}"
                )
                return None
            except Exception as e:
                logger.debug(
                    f"Backend health check UNEXPECTED ERROR: {backend_url} - {type(e).__name__}: {e}"
                )
                return None

    async def discover_backends(self) -> List[str]:
        """
        Discover healthy backends from Tailscale peers.

        Returns:
            List of healthy backend URLs
        """
        logger.debug("Starting backend discovery process")
        peers = self.parse_tailscale_status()
        if not peers:
            logger.debug("No peers found from Tailscale status")
            return []

        logger.info(f"Found {len(peers)} online candidate peers to test")
        logger.debug(
            f"Candidate peer details: {[{k: v for k, v in peer.items() if k in ['DNSName', 'HostName', 'Online']} for peer in peers]}"
        )

        healthy_backends = []
        total_tests = len(peers) * len(self.port_range)
        logger.debug(
            f"Planning to test {total_tests} endpoint combinations ({len(peers)} peers × {len(self.port_range)} ports)"
        )

        # Test all peers and ports in parallel
        tasks = []
        for peer in peers:
            peer_dns = peer.get("DNSName", "").rstrip(".")
            logger.debug(f"Creating health check tasks for peer {peer_dns}")
            for port in self.port_range:
                task = self.test_backend_health(peer_dns, port)
                tasks.append(task)
                logger.debug(f"Scheduled health check for {peer_dns}:{port}")

        if tasks:
            logger.debug(f"Executing {len(tasks)} health check tasks in parallel")
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_results = 0
            failed_results = 0
            for i, result in enumerate(results):
                if isinstance(result, str):  # Healthy backend URL
                    healthy_backends.append(result)
                    successful_results += 1
                    logger.debug(
                        f"Health check result {i+1}/{len(results)}: SUCCESS - {result}"
                    )
                elif isinstance(result, Exception):
                    failed_results += 1
                    logger.debug(
                        f"Health check result {i+1}/{len(results)}: EXCEPTION - {result}"
                    )
                else:
                    failed_results += 1
                    logger.debug(f"Health check result {i+1}/{len(results)}: FAILED")

            logger.info(
                f"Health check completed: {successful_results} healthy, {failed_results} unhealthy backends found"
            )

        logger.debug(
            f"Backend discovery process completed, found {len(healthy_backends)} healthy backends: {healthy_backends}"
        )
        return healthy_backends

    async def add_healthy_backends(self, healthy_backends: List[str]) -> None:
        """
        Add healthy backends to the service discovery.

        Args:
            healthy_backends: List of healthy backend URLs
        """
        logger.debug(
            f"Starting to add {len(healthy_backends)} healthy backends to service discovery"
        )
        service_discovery = get_service_discovery()
        if not service_discovery:
            logger.error("Service discovery not initialized - cannot add backends")
            return

        logger.debug(
            f"Service discovery available, currently tracking {len(self._discovered_backends)} backends"
        )

        new_backends_added = 0
        for backend_url in healthy_backends:
            if backend_url not in self._discovered_backends:
                try:
                    logger.info(f"Adding newly discovered backend: {backend_url}")
                    await service_discovery.add_backend(backend_url)
                    self._discovered_backends.add(backend_url)
                    new_backends_added += 1
                    logger.debug(
                        f"Successfully added backend {backend_url} to service discovery"
                    )
                except Exception as e:
                    logger.error(f"Failed to add backend {backend_url}: {e}")
                    logger.debug(
                        f"Error details for backend {backend_url}: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
            else:
                logger.debug(f"Backend {backend_url} already discovered, skipping")

        logger.info(
            f"Backend addition completed: {new_backends_added} new backends added, {len(self._discovered_backends)} total tracked"
        )

    async def discovery_loop(self) -> None:
        """Main discovery loop that runs periodically."""
        logger.info(
            f"Starting backend discovery loop (interval: {self.discovery_interval}s)"
        )

        while self._running:
            try:
                logger.info("=== Starting backend discovery cycle ===")

                # Discover healthy backends
                healthy_backends = await self.discover_backends()

                # Add new healthy backends
                await self.add_healthy_backends(healthy_backends)

                logger.info(
                    f"Discovery cycle completed. Found {len(healthy_backends)} healthy backends"
                )
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
