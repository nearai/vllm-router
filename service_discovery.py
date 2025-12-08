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
import abc
import asyncio
import enum
import hashlib
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import aiohttp

from vllm_router import utils
from vllm_router.log import init_logger

logger = init_logger(__name__)


class CircuitState(enum.Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if backend recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for a single backend endpoint."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    consecutive_failures: int = 0  # Consecutive failures (resets on success)
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    consecutive_successes: int = 0
    failure_threshold: int = 3  # Failures required to open circuit
    cooldown_seconds: float = 5.0  # Initial cooldown when circuit opens
    max_cooldown_seconds: float = 30.0  # Maximum cooldown with backoff
    current_cooldown: float = 5.0  # Current cooldown (increases with backoff)
    endpoint_id: str = ""  # For logging

    def record_failure(self) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        self.consecutive_successes = 0

        if self.state == CircuitState.CLOSED:
            # Only open circuit after reaching failure threshold
            if self.consecutive_failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
                self.current_cooldown = self.cooldown_seconds
                logger.warning(
                    f"Circuit OPENED for {self.endpoint_id} after {self.consecutive_failures} consecutive failures "
                    f"(cooldown: {self.current_cooldown}s)"
                )
            else:
                logger.debug(
                    f"Circuit failure {self.consecutive_failures}/{self.failure_threshold} for {self.endpoint_id} "
                    f"(still CLOSED)"
                )
        elif self.state == CircuitState.HALF_OPEN:
            # Failed during test, reopen with increased cooldown
            self.state = CircuitState.OPEN
            self.current_cooldown = min(
                self.current_cooldown * 1.5, self.max_cooldown_seconds
            )
            logger.warning(
                f"Circuit reopened from HALF_OPEN for {self.endpoint_id}, "
                f"backoff cooldown: {self.current_cooldown:.1f}s"
            )

    def record_success(self) -> None:
        """Record a success and potentially close the circuit."""
        self.last_success_time = time.time()
        self.consecutive_successes += 1
        self.consecutive_failures = 0  # Reset consecutive failures on success

        if self.state == CircuitState.HALF_OPEN:
            # Success during test, close the circuit
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.current_cooldown = self.cooldown_seconds  # Reset cooldown
            logger.info(f"Circuit CLOSED for {self.endpoint_id} after successful test")
        elif self.state == CircuitState.CLOSED:
            # Ongoing success, reset failure count periodically
            if self.consecutive_successes >= 3:
                self.failure_count = max(0, self.failure_count - 1)

    def should_allow_request(self) -> bool:
        """Check if a request should be allowed through this circuit."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if cooldown period has elapsed
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.current_cooldown:
                self.state = CircuitState.HALF_OPEN
                logger.info(
                    f"Circuit transitioning to HALF_OPEN for {self.endpoint_id} "
                    f"after {elapsed:.1f}s cooldown"
                )
                return True
            logger.debug(
                f"Circuit OPEN for {self.endpoint_id}, cooldown remaining: "
                f"{self.current_cooldown - elapsed:.1f}s"
            )
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited requests to test the backend
            return True

        return False

    def is_healthy(self) -> bool:
        """Check if circuit considers the backend healthy."""
        return self.state == CircuitState.CLOSED


_global_service_discovery: "Optional[ServiceDiscovery]" = None


class ServiceDiscoveryType(enum.Enum):
    STATIC = "static"


@dataclass
class ModelInfo:
    """Information about a model including its relationships and metadata."""

    id: str
    object: str
    created: int = 0
    owned_by: str = "vllm"
    root: Optional[str] = None
    parent: Optional[str] = None
    is_adapter: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelInfo":
        """Create a ModelInfo instance from a dictionary."""
        return cls(
            id=data.get("id"),
            object=data.get("object", "model"),
            created=data.get("created", int(time.time())),
            owned_by=data.get("owned_by", "vllm"),
            root=data.get("root", None),
            parent=data.get("parent", None),
            is_adapter=data.get("parent") is not None,
        )

    def to_dict(self) -> Dict:
        """Convert the ModelInfo instance to a dictionary."""
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
            "root": self.root,
            "parent": self.parent,
            "is_adapter": self.is_adapter,
        }


@dataclass
class EndpointInfo:
    # Endpoint's url
    url: str

    # Model names
    model_names: List[str]

    # Endpoint Id
    Id: str

    # Added timestamp
    added_timestamp: float

    # Model label
    model_label: str

    # Endpoint's sleep status
    sleep: bool

    # Pod name
    pod_name: Optional[str] = None

    # Service name
    service_name: Optional[str] = None

    # Namespace
    namespace: Optional[str] = None

    # Model information including relationships
    model_info: Dict[str, ModelInfo] = None

    def __str__(self):
        return f"EndpointInfo(url={self.url}, model_names={self.model_names}, added_timestamp={self.added_timestamp}, model_label={self.model_label}, service_name={self.service_name},pod_name={self.pod_name}, namespace={self.namespace})"

    def get_base_models(self) -> List[str]:
        """
        Get the list of base models (models without parents) available on this endpoint.
        """
        if not self.model_info:
            return []
        return [
            model_id for model_id, info in self.model_info.items() if not info.parent
        ]

    def get_adapters(self) -> List[str]:
        """
        Get the list of adapters (models with parents) available on this endpoint.
        """
        if not self.model_info:
            return []
        return [model_id for model_id, info in self.model_info.items() if info.parent]

    def get_adapters_for_model(self, base_model: str) -> List[str]:
        """
        Get the list of adapters available for a specific base model.

        Args:
            base_model: The ID of the base model

        Returns:
            List of adapter IDs that are based on the specified model
        """
        if not self.model_info:
            return []
        return [
            model_id
            for model_id, info in self.model_info.items()
            if info.parent == base_model
        ]

    def has_model(self, model_id: str) -> bool:
        """
        Check if a specific model (base model or adapter) is available on this endpoint.

        Args:
            model_id: The ID of the model to check

        Returns:
            True if the model is available, False otherwise
        """
        return model_id in self.model_names

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get detailed information about a specific model.

        Args:
            model_id: The ID of the model to get information for

        Returns:
            ModelInfo object containing model information if available, None otherwise
        """
        if not self.model_info:
            return None
        return self.model_info.get(model_id)


class ServiceDiscovery(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_endpoint_info(
        self, include_circuit_broken: bool = False
    ) -> List[EndpointInfo]:
        """
        Get the URLs of the serving engines that are available for
        querying.

        Args:
            include_circuit_broken: If True, include endpoints with open circuits.
                                   Default False excludes them for normal routing.

        Returns:
            a list of engine URLs
        """
        pass

    def get_health(self) -> bool:
        """
        Check if the service discovery module is healthy.

        Returns:
            True if the service discovery module is healthy, False otherwise
        """
        return True

    def close(self) -> None:
        """
        Close the service discovery module.
        """
        pass


class StaticServiceDiscovery(ServiceDiscovery):
    def __init__(
        self,
        app,
        urls: List[str] | None = None,
        models: List[str] | None = None,
        aliases: List[str] | None = None,
        model_labels: List[str] | None = None,
        model_types: List[str] | None = None,
        static_backend_health_checks: bool = False,
        prefill_model_labels: List[str] | None = None,
        decode_model_labels: List[str] | None = None,
        health_check_include_models_endpoint: bool = True,
        health_check_include_attestation: bool = True,
        health_check_removal_threshold: int = 3,
        backend_health_check_timeout_seconds: int = 10,
        circuit_breaker_threshold: int = 3,
        circuit_breaker_cooldown_seconds: float = 5.0,
        circuit_breaker_max_cooldown_seconds: float = 30.0,
        health_check_interval: int = 60,
        health_check_max_concurrent: int = 5,
    ):
        self.app = app

        # Handle empty backend configuration
        if urls is None:
            urls = []
        if models is None:
            models = []
        if model_types is None:
            model_types = []

        # Validate that if both are provided, they have the same length
        if urls and models:
            assert len(urls) == len(
                models
            ), "URLs and models should have the same length"
        elif urls and not models:
            raise ValueError("Models must be provided when URLs are specified")
        elif models and not urls:
            raise ValueError("URLs must be provided when models are specified")

        self.urls = urls
        self.models = models
        self.aliases = aliases
        self.model_labels = model_labels
        self.model_types = model_types
        self.engines_id = [str(uuid.uuid4()) for i in range(0, len(urls))]
        self.added_timestamp = int(time.time())
        self.unhealthy_endpoint_hashes = []
        self._running = True
        self._lock = threading.Lock()

        # Health check configuration
        self.health_check_include_models_endpoint = health_check_include_models_endpoint
        self.health_check_include_attestation = health_check_include_attestation
        self.health_check_removal_threshold = health_check_removal_threshold
        self.backend_health_check_timeout = backend_health_check_timeout_seconds
        self.health_check_interval = health_check_interval
        self.health_check_max_concurrent = health_check_max_concurrent

        # Track consecutive failures for each backend (key: hash, value: failure count)
        self.backend_failure_counts: Dict[str, int] = {}

        # Circuit breaker configuration and state
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown = circuit_breaker_cooldown_seconds
        self.circuit_breaker_max_cooldown = circuit_breaker_max_cooldown_seconds
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        logger.info(
            f"Circuit breaker config: threshold={circuit_breaker_threshold} failures, "
            f"cooldown={circuit_breaker_cooldown_seconds}s, max_cooldown={circuit_breaker_max_cooldown_seconds}s"
        )
        logger.info(
            f"Health check config: interval={health_check_interval}s, "
            f"max_concurrent={health_check_max_concurrent}"
        )

        self.start_health_check_task()
        self.prefill_model_labels = prefill_model_labels
        self.decode_model_labels = decode_model_labels

    def _get_or_create_circuit_breaker(self, endpoint_hash: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an endpoint."""
        if endpoint_hash not in self._circuit_breakers:
            self._circuit_breakers[endpoint_hash] = CircuitBreaker(
                failure_threshold=self.circuit_breaker_threshold,
                cooldown_seconds=self.circuit_breaker_cooldown,
                max_cooldown_seconds=self.circuit_breaker_max_cooldown,
                current_cooldown=self.circuit_breaker_cooldown,
                endpoint_id=endpoint_hash,
            )
        return self._circuit_breakers[endpoint_hash]

    def record_request_success(self, url: str, model: str) -> None:
        """
        Record a successful request to a backend.
        This helps the circuit breaker close after successful recovery.

        Args:
            url: The backend URL
            model: The model name
        """
        endpoint_hash = self.get_model_endpoint_hash(url, model)
        with self._lock:
            circuit = self._get_or_create_circuit_breaker(endpoint_hash)
            circuit.record_success()

    def is_backend_available(self, url: str, model: str) -> bool:
        """
        Check if a backend is available for requests (circuit not open).

        Args:
            url: The backend URL
            model: The model name

        Returns:
            True if the backend should receive requests, False otherwise
        """
        endpoint_hash = self.get_model_endpoint_hash(url, model)
        with self._lock:
            # Check if permanently unhealthy
            if endpoint_hash in self.unhealthy_endpoint_hashes:
                return False
            # Check circuit breaker state
            circuit = self._get_or_create_circuit_breaker(endpoint_hash)
            return circuit.should_allow_request()

    def get_available_endpoints(
        self, endpoints: List["EndpointInfo"]
    ) -> List["EndpointInfo"]:
        """
        Filter endpoints to only include those with open circuits.

        Args:
            endpoints: List of endpoint info objects

        Returns:
            Filtered list of available endpoints
        """
        available = []
        for endpoint in endpoints:
            for model in endpoint.model_names:
                if self.is_backend_available(endpoint.url, model):
                    available.append(endpoint)
                    break  # Only need to check one model per endpoint
        return available

    async def add_backend(self, url: str):
        logger.info(f"add_backend called with url: {url}")
        try:
            logger.debug(
                f"creating aiohttp session to fetch models from {url}/v1/models"
            )
            models_data = []
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{url}/v1/models") as response:
                    logger.debug(
                        f"received response from {url}/v1/models with status: {response.status}"
                    )
                    if response.status != 200:
                        logger.error(
                            f"failed to fetch models from {url}: HTTP {response.status}"
                        )
                        return

                    data = await response.json()
                    models_data = data.get("data", [])
                    logger.info(
                        f"fetched {len(models_data)} models from {url}: {[m.get('id') for m in models_data]}"
                    )
            if not models_data:
                logger.error(f"no models found at {url}/v1/models")
                return

            with self._lock:
                logger.debug(f"acquired lock, current backends count: {len(self.urls)}")
                logger.debug(f"current backends: {list(zip(self.urls, self.models))}")

                models_to_add = []
                for model_data in models_data:
                    model_id = model_data.get("id")
                    if not model_id:
                        logger.warning(
                            f"skipping model from {url} with no id: {model_data}"
                        )
                        continue

                    # Check if specific combination exists
                    exists = False
                    for u, m in zip(self.urls, self.models):
                        if u == url and m == model_id:
                            exists = True
                            break

                    if exists:
                        logger.info(
                            f"backend {url} with model {model_id} already exists, skipping"
                        )
                        continue

                    models_to_add.append(model_id)

                if not models_to_add:
                    logger.info(
                        f"no new models to add from {url}, all models already registered"
                    )
                    return

                logger.info(
                    f"adding {len(models_to_add)} new models from {url}: {models_to_add}"
                )

                for model_id in models_to_add:
                    self.urls.append(url)
                    self.models.append(model_id)
                    self.engines_id.append(str(uuid.uuid4()))

                    if self.model_labels is not None:
                        self.model_labels.append("default")
                        logger.debug(f"appended default model_label for {model_id}")

                    if self.model_types is not None:
                        self.model_types.append("default")
                        logger.debug(f"appended default model_type for {model_id}")

                    logger.info(
                        f"successfully added backend {url} with model {model_id}"
                    )

                logger.info(
                    f"backend registration complete, total backends: {len(self.urls)}"
                )

        except aiohttp.ClientError as e:
            logger.error(f"network error adding backend {url}: {type(e).__name__}: {e}")
        except asyncio.TimeoutError:
            logger.error(f"timeout error adding backend {url}")
        except Exception as e:
            logger.error(
                f"unexpected error adding backend {url}: {type(e).__name__}: {e}",
                exc_info=True,
            )

    def remove_backend(self, url: str):
        with self._lock:
            try:
                # Find all indices matching the URL (iterate backwards to avoid index shifting issues)
                indices = [i for i, u in enumerate(self.urls) if u == url]

                if not indices:
                    logger.warning(f"Backend {url} not found.")
                    return

                # Remove in reverse order
                for idx in sorted(indices, reverse=True):
                    model = self.models[idx]
                    self.urls.pop(idx)
                    self.models.pop(idx)
                    self.engines_id.pop(idx)

                    if self.model_labels is not None:
                        self.model_labels.pop(idx)

                    if self.model_types is not None:
                        self.model_types.pop(idx)

                    logger.info(f"Removed backend {url} with model {model}")
            except Exception as e:
                logger.error(f"Error removing backend: {e}")

    def _check_single_backend_health(
        self, url: str, model: str, model_type: str
    ) -> Tuple[str, str, str, bool, List[Tuple[str, bool]]]:
        """
        Check health of a single backend.

        Returns:
            Tuple of (url, model, endpoint_hash, is_healthy, health_check_results)
        """
        endpoint_hash = self.get_model_endpoint_hash(url, model)
        is_healthy = True
        health_check_results = []

        # Check 1: Model inference health (existing check)
        if self.model_types:
            model_healthy = utils.is_model_healthy(url, model, model_type)
            health_check_results.append(("inference", model_healthy))
            if not model_healthy:
                is_healthy = False
                logger.debug(f"Model inference check failed for {model} at {url}")

        # Check 2: Models endpoint availability
        if self.health_check_include_models_endpoint:
            models_list = utils.fetch_models_list(
                url, self.backend_health_check_timeout
            )
            models_available = models_list is not None
            health_check_results.append(("models_endpoint", models_available))
            if not models_available:
                is_healthy = False
                logger.debug(f"Models endpoint check failed for {url}")
            elif models_list and model not in models_list:
                logger.warning(
                    f"Model {model} no longer available at {url}. Available models: {models_list}"
                )
                is_healthy = False

        # Check 3: Attestation endpoint availability
        if self.health_check_include_attestation:
            attestation_available = utils.check_attestation_available(
                url, self.backend_health_check_timeout
            )
            health_check_results.append(("attestation", attestation_available))
            if not attestation_available:
                is_healthy = False
                logger.debug(f"Attestation endpoint check failed for {url}")

        return (url, model, endpoint_hash, is_healthy, health_check_results)

    def get_unhealthy_endpoint_hashes(self) -> list[str]:
        unhealthy_endpoints = []
        backends_to_remove = []

        try:
            # Build list of backends to check
            backends_to_check = list(
                zip(self.urls, self.models, self.model_types, strict=True)
            )

            if not backends_to_check:
                return unhealthy_endpoints

            logger.debug(
                f"Starting health checks for {len(backends_to_check)} backends "
                f"(max_concurrent={self.health_check_max_concurrent})"
            )

            # Use ThreadPoolExecutor to limit concurrent health checks
            with ThreadPoolExecutor(
                max_workers=self.health_check_max_concurrent
            ) as executor:
                # Submit all health check tasks
                futures = {
                    executor.submit(
                        self._check_single_backend_health, url, model, model_type
                    ): (url, model)
                    for url, model, model_type in backends_to_check
                }

                # Process results as they complete
                for future in as_completed(futures):
                    url, model = futures[future]
                    try:
                        (
                            url,
                            model,
                            endpoint_hash,
                            is_healthy,
                            health_check_results,
                        ) = future.result()

                        # Update failure counts and determine if backend should be removed
                        if is_healthy:
                            # Reset failure count on success
                            if endpoint_hash in self.backend_failure_counts:
                                del self.backend_failure_counts[endpoint_hash]
                            logger.debug(f"{model} at {url} is healthy")
                        else:
                            # Increment failure count
                            self.backend_failure_counts[endpoint_hash] = (
                                self.backend_failure_counts.get(endpoint_hash, 0) + 1
                            )
                            failure_count = self.backend_failure_counts[endpoint_hash]

                            logger.warning(
                                f"{model} at {url} not healthy! Failure count: {failure_count}/{self.health_check_removal_threshold}. "
                                f"Failed checks: {[check for check, result in health_check_results if not result]}"
                            )

                            unhealthy_endpoints.append(endpoint_hash)

                            # Mark for removal if threshold exceeded
                            if failure_count >= self.health_check_removal_threshold:
                                logger.error(
                                    f"Backend {url} with model {model} exceeded failure threshold "
                                    f"({failure_count} >= {self.health_check_removal_threshold}). Marking for removal."
                                )
                                backends_to_remove.append(url)

                    except Exception as e:
                        logger.error(f"Error checking health of {model} at {url}: {e}")

        except ValueError:
            logger.error(
                "To perform health check, each model has to define a static_model_type and at least one static_backend. "
                "Skipping health checks for now."
            )

        # Remove backends that exceeded the failure threshold
        for url in set(backends_to_remove):
            logger.info(
                f"Permanently removing backend {url} due to repeated health check failures"
            )
            self.remove_backend(url)
            # Increment Prometheus counter for removed backends
            try:
                from vllm_router.services.metrics_service import (
                    vllm_router_backends_removed_total,
                )

                vllm_router_backends_removed_total.inc()
            except ImportError:
                pass  # Metrics service may not be available in all contexts

        return unhealthy_endpoints

    async def check_model_health(self):
        while self._running:
            try:
                if len(self.urls) == 0:
                    logger.warning("No backends found, skipping health check")
                    await asyncio.sleep(self.health_check_interval)
                    continue
                self.unhealthy_endpoint_hashes = self.get_unhealthy_endpoint_hashes()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                logger.debug("Health check task cancelled")
                break
            except Exception as e:
                logger.error(e)

    def start_health_check_task(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.check_model_health(), self.loop)
        logger.info("Health check thread started")

    def get_model_endpoint_hash(self, url: str, model: str) -> str:
        return hashlib.md5(f"{url}{model}".encode()).hexdigest()

    def _get_model_info(self, model: str) -> Dict[str, ModelInfo]:
        """
        Get detailed model information. For static serving engines, we don't query the engine, instead we use predefined
        static model info.

        Args:
            model: the model name

        Returns:
            Dictionary mapping model IDs to their information, including parent-child relationships
        """
        return {
            model: ModelInfo(
                id=model,
                object="model",
                owned_by="vllm",
                parent=None,
                is_adapter=False,
                root=None,
                created=int(time.time()),
            )
        }

    def get_endpoint_info(
        self, include_circuit_broken: bool = False
    ) -> List[EndpointInfo]:
        """
        Get the URLs of the serving engines that are available for
        querying.

        Args:
            include_circuit_broken: If True, include endpoints with open circuits.
                                   Default False excludes them for normal routing.

        Returns:
            a list of engine URLs
        """
        with self._lock:
            endpoint_infos = []
            for i, (url, model) in enumerate(zip(self.urls, self.models)):
                endpoint_hash = self.get_model_endpoint_hash(url, model)

                # Skip permanently unhealthy backends
                if endpoint_hash in self.unhealthy_endpoint_hashes:
                    continue

                # Skip circuit-broken backends unless explicitly requested
                if not include_circuit_broken:
                    circuit = self._get_or_create_circuit_breaker(endpoint_hash)
                    if not circuit.should_allow_request():
                        logger.debug(
                            f"Skipping circuit-broken backend {url} for model {model} "
                            f"(state: {circuit.state.value}, cooldown remaining: "
                            f"{max(0, circuit.current_cooldown - (time.time() - circuit.last_failure_time)):.1f}s)"
                        )
                        continue

                model_label = self.model_labels[i] if self.model_labels else "default"
                endpoint_info = EndpointInfo(
                    url=url,
                    model_names=[model],  # Convert single model to list
                    Id=self.engines_id[i],
                    sleep=False,
                    added_timestamp=self.added_timestamp,
                    model_label=model_label,
                    model_info=self._get_model_info(model),
                )
                endpoint_infos.append(endpoint_info)
            return endpoint_infos

    async def initialize_client_sessions(self) -> None:
        """
        Initialize aiohttp ClientSession objects for prefill and decode endpoints.
        This must be called from an async context during app startup.
        """
        if (
            self.prefill_model_labels is not None
            and self.decode_model_labels is not None
        ):
            endpoint_infos = self.get_endpoint_info()
            for endpoint_info in endpoint_infos:
                if endpoint_info.model_label in self.prefill_model_labels:
                    self.app.state.prefill_client = aiohttp.ClientSession(
                        base_url=endpoint_info.url,
                        timeout=aiohttp.ClientTimeout(total=None),
                    )
                elif endpoint_info.model_label in self.decode_model_labels:
                    self.app.state.decode_client = aiohttp.ClientSession(
                        base_url=endpoint_info.url,
                        timeout=aiohttp.ClientTimeout(total=None),
                    )

    def get_backend_health_status(self) -> List[Dict]:
        """
        Get the health status of all backends including circuit breaker state.

        Returns:
            List of dictionaries containing backend health information
        """
        with self._lock:
            backend_status = []
            current_time = time.time()

            for i, (url, model) in enumerate(zip(self.urls, self.models)):
                endpoint_hash = self.get_model_endpoint_hash(url, model)
                failure_count = self.backend_failure_counts.get(endpoint_hash, 0)
                is_permanently_unhealthy = (
                    endpoint_hash in self.unhealthy_endpoint_hashes
                )

                # Get circuit breaker status
                circuit = self._get_or_create_circuit_breaker(endpoint_hash)
                circuit_state = circuit.state.value
                is_accepting_requests = (
                    circuit.should_allow_request() and not is_permanently_unhealthy
                )

                # Calculate cooldown remaining if circuit is open
                cooldown_remaining = 0.0
                if circuit.state == CircuitState.OPEN:
                    cooldown_remaining = max(
                        0,
                        circuit.current_cooldown
                        - (current_time - circuit.last_failure_time),
                    )

                status = {
                    "url": url,
                    "model": model,
                    "engine_id": self.engines_id[i],
                    "healthy": not is_permanently_unhealthy,
                    "accepting_requests": is_accepting_requests,
                    "circuit_state": circuit_state,
                    "cooldown_remaining_seconds": round(cooldown_remaining, 1),
                    "failure_count": failure_count,
                    "failure_threshold": self.health_check_removal_threshold,
                    "model_label": (
                        self.model_labels[i] if self.model_labels else "default"
                    ),
                }

                if self.model_types:
                    status["model_type"] = self.model_types[i]

                backend_status.append(status)

            return backend_status

    def close(self):
        """
        Close the service discovery module and clean up health check resources.
        """
        self._running = False
        if hasattr(self, "loop") and self.loop.is_running():
            # Schedule a coroutine to gracefully shut down the event loop
            async def shutdown():
                tasks = [
                    t
                    for t in asyncio.all_tasks(self.loop)
                    if t is not asyncio.current_task()
                ]
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                self.loop.stop()

            future = asyncio.run_coroutine_threadsafe(shutdown(), self.loop)
            try:
                future.result(timeout=15.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out waiting for shutdown(loop might already be closed)"
                )
            except Exception as e:
                logger.warning(f"Error during health check shutdown: {e}")

        if hasattr(self, "thread") and self.thread.is_alive():
            self.thread.join(timeout=5.0)

        if hasattr(self, "loop") and not self.loop.is_closed():
            self.loop.close()

    def mark_backend_unhealthy_during_request(self, url: str, model: str) -> bool:
        """
        Mark a backend as unhealthy during request processing.
        This is called when a backend fails to respond to a request.

        Uses circuit breaker pattern:
        - First failure: Opens circuit (backend excluded for cooldown period)
        - Circuit auto-recovers after cooldown to test if backend is back
        - Repeated failures increase cooldown (exponential backoff)
        - After threshold failures, backend is permanently removed

        Args:
            url: The backend URL that failed
            model: The model name that was being requested

        Returns:
            bool: True if the backend should be permanently removed from the pool, False otherwise
        """
        endpoint_hash = self.get_model_endpoint_hash(url, model)

        with self._lock:
            # Get or create circuit breaker
            circuit = self._get_or_create_circuit_breaker(endpoint_hash)

            # Record the failure in circuit breaker (this opens the circuit)
            circuit.record_failure()

            # Increment total failure count for permanent removal tracking
            self.backend_failure_counts[endpoint_hash] = (
                self.backend_failure_counts.get(endpoint_hash, 0) + 1
            )
            failure_count = self.backend_failure_counts[endpoint_hash]

            logger.warning(
                f"Backend {url} with model {model} failed during request processing. "
                f"Circuit state: {circuit.state.value}, "
                f"Total failures: {failure_count}/{self.health_check_removal_threshold}, "
                f"Cooldown: {circuit.current_cooldown}s"
            )

            # Check if we should permanently remove this backend
            if failure_count >= self.health_check_removal_threshold:
                logger.error(
                    f"Backend {url} with model {model} exceeded failure threshold "
                    f"({failure_count} >= {self.health_check_removal_threshold}). "
                    f"Permanently removing from pool."
                )

                # Add to unhealthy endpoints list (permanent removal)
                if endpoint_hash not in self.unhealthy_endpoint_hashes:
                    self.unhealthy_endpoint_hashes.append(endpoint_hash)

                # Clean up tracking data
                del self.backend_failure_counts[endpoint_hash]
                if endpoint_hash in self._circuit_breakers:
                    del self._circuit_breakers[endpoint_hash]

                # Increment Prometheus counter for removed backends
                try:
                    from vllm_router.services.metrics_service import (
                        vllm_router_backends_removed_total,
                    )

                    vllm_router_backends_removed_total.inc()
                except ImportError:
                    pass  # Metrics service may not be available in all contexts

                return True

            return False


def _create_service_discovery(
    service_discovery_type: ServiceDiscoveryType, *args, **kwargs
) -> ServiceDiscovery:
    """
    Create a service discovery module with the given type and arguments.

    Args:
        service_discovery_type: the type of service discovery module
        *args: positional arguments for the service discovery module
        **kwargs: keyword arguments for the service discovery module

    Returns:
        the created service discovery module
    """

    if service_discovery_type == ServiceDiscoveryType.STATIC:
        return StaticServiceDiscovery(*args, **kwargs)
    else:
        raise ValueError("Invalid service discovery type")


def initialize_service_discovery(
    service_discovery_type: ServiceDiscoveryType, *args, **kwargs
) -> ServiceDiscovery:
    """
    Initialize the service discovery module with the given type and arguments.

    Args:
        service_discovery_type: the type of service discovery module
        *args: positional arguments for the service discovery module
        **kwargs: keyword arguments for the service discovery module

    Returns:
        the initialized service discovery module

    Raises:
        ValueError: if the service discovery module is already initialized
        ValueError: if the service discovery type is invalid
    """
    global _global_service_discovery
    if _global_service_discovery is not None:
        raise ValueError("Service discovery module already initialized")

    _global_service_discovery = _create_service_discovery(
        service_discovery_type, *args, **kwargs
    )
    return _global_service_discovery


def reconfigure_service_discovery(
    service_discovery_type: ServiceDiscoveryType, *args, **kwargs
) -> ServiceDiscovery:
    """
    Reconfigure the service discovery module with the given type and arguments.
    """
    global _global_service_discovery
    if _global_service_discovery is None:
        raise ValueError("Service discovery module not initialized")

    new_service_discovery = _create_service_discovery(
        service_discovery_type, *args, **kwargs
    )

    _global_service_discovery.close()
    _global_service_discovery = new_service_discovery
    return _global_service_discovery


def get_service_discovery() -> ServiceDiscovery:
    """
    Get the initialized service discovery module.

    Returns:
        the initialized service discovery module

    Raises:
        ValueError: if the service discovery module is not initialized
    """
    global _global_service_discovery
    if _global_service_discovery is None:
        raise ValueError("Service discovery module not initialized")

    return _global_service_discovery
