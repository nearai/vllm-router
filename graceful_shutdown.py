"""
Graceful shutdown module for vllm-router.

This module provides functionality to gracefully shutdown the router by:
1. Tracking all in-flight requests
2. Rejecting new requests when shutdown is initiated
3. Waiting for ongoing requests to complete before fully shutting down
"""

import asyncio
import signal
import threading
from typing import Optional

from vllm_router.log import init_logger

logger = init_logger(__name__)


class GracefulShutdownManager:
    """
    Manages graceful shutdown for the vllm-router.

    Tracks in-flight requests and provides mechanisms to:
    - Reject new requests when shutdown is initiated
    - Wait for ongoing requests to complete
    """

    def __init__(self, timeout: float = 30.0):
        """
        Initialize the graceful shutdown manager.

        Args:
            timeout: Maximum time to wait for in-flight requests to complete (in seconds)
        """
        self._timeout = timeout
        self._is_shutting_down = False
        self._in_flight_requests = 0
        self._lock = threading.Lock()
        self._shutdown_event: Optional[asyncio.Event] = None
        self._all_requests_done_event: Optional[asyncio.Event] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for async operations."""
        self._loop = loop
        self._shutdown_event = asyncio.Event()
        self._all_requests_done_event = asyncio.Event()
        # Initially, no requests means "all done"
        self._all_requests_done_event.set()

    @property
    def is_shutting_down(self) -> bool:
        """Check if shutdown has been initiated."""
        return self._is_shutting_down

    @property
    def in_flight_requests(self) -> int:
        """Get the current number of in-flight requests."""
        with self._lock:
            return self._in_flight_requests

    def request_started(self):
        """Called when a new request starts processing."""
        with self._lock:
            self._in_flight_requests += 1
            if self._all_requests_done_event is not None and self._loop is not None:
                # Clear the event since we now have requests in flight
                self._loop.call_soon_threadsafe(self._all_requests_done_event.clear)
            logger.debug(
                f"Request started. In-flight requests: {self._in_flight_requests}"
            )

    def request_completed(self):
        """Called when a request completes (successfully or with error)."""
        with self._lock:
            self._in_flight_requests = max(0, self._in_flight_requests - 1)
            current_count = self._in_flight_requests
            logger.debug(f"Request completed. In-flight requests: {current_count}")

            if (
                current_count == 0
                and self._all_requests_done_event is not None
                and self._loop is not None
            ):
                # All requests done, signal the event
                self._loop.call_soon_threadsafe(self._all_requests_done_event.set)

    def initiate_shutdown(self):
        """Initiate graceful shutdown. New requests will be rejected."""
        if self._is_shutting_down:
            logger.warning("Shutdown already initiated")
            return

        self._is_shutting_down = True
        logger.info(
            f"Graceful shutdown initiated. Waiting for {self.in_flight_requests} in-flight requests to complete..."
        )

        if self._shutdown_event is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)

    async def wait_for_requests(self) -> bool:
        """
        Wait for all in-flight requests to complete.

        Returns:
            True if all requests completed within timeout, False otherwise.
        """
        if self._all_requests_done_event is None:
            logger.warning("Event loop not set, cannot wait for requests")
            return True

        try:
            # Wait for either all requests to complete or timeout
            await asyncio.wait_for(
                self._all_requests_done_event.wait(), timeout=self._timeout
            )
            logger.info("All in-flight requests completed successfully")
            return True
        except asyncio.TimeoutError:
            remaining = self.in_flight_requests
            logger.warning(
                f"Graceful shutdown timeout ({self._timeout}s) reached with {remaining} "
                f"requests still in flight. Forcing shutdown."
            )
            return False


# Global instance
_shutdown_manager: Optional[GracefulShutdownManager] = None


def get_shutdown_manager() -> Optional[GracefulShutdownManager]:
    """Get the global shutdown manager instance."""
    return _shutdown_manager


def initialize_shutdown_manager(timeout: float = 30.0) -> GracefulShutdownManager:
    """
    Initialize the global shutdown manager.

    Args:
        timeout: Maximum time to wait for in-flight requests to complete (in seconds)

    Returns:
        The initialized shutdown manager instance.
    """
    global _shutdown_manager
    _shutdown_manager = GracefulShutdownManager(timeout=timeout)
    return _shutdown_manager


def setup_signal_handlers(loop: asyncio.AbstractEventLoop):
    """
    Set up signal handlers for graceful shutdown.

    Args:
        loop: The asyncio event loop to use for signal handling.
    """
    manager = get_shutdown_manager()
    if manager is None:
        logger.warning(
            "Shutdown manager not initialized, skipping signal handler setup"
        )
        return

    manager.set_event_loop(loop)

    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        manager.initiate_shutdown()

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Signal handlers registered for graceful shutdown (SIGTERM, SIGINT)")
