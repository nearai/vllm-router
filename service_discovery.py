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
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import aiohttp

from vllm_router import utils
from vllm_router.log import init_logger

logger = init_logger(__name__)

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
    def get_endpoint_info(self) -> List[EndpointInfo]:
        """
        Get the URLs of the serving engines that are available for
        querying.

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
        urls: List[str],
        models: List[str],
        aliases: List[str] | None = None,
        model_labels: List[str] | None = None,
        model_types: List[str] | None = None,
        static_backend_health_checks: bool = False,
        prefill_model_labels: List[str] | None = None,
        decode_model_labels: List[str] | None = None,
    ):
        self.app = app
        assert len(urls) == len(models), "URLs and models should have the same length"
        self.urls = urls
        self.models = models
        self.aliases = aliases
        self.model_labels = model_labels
        self.model_types = model_types
        self.engines_id = [str(uuid.uuid4()) for i in range(0, len(urls))]
        self.added_timestamp = int(time.time())
        self.unhealthy_endpoint_hashes = []
        self._running = True
        if static_backend_health_checks:
            self.start_health_check_task()
        self.prefill_model_labels = prefill_model_labels
        self.decode_model_labels = decode_model_labels

    def get_unhealthy_endpoint_hashes(self) -> list[str]:
        unhealthy_endpoints = []
        try:
            for url, model, model_type in zip(
                self.urls, self.models, self.model_types, strict=True
            ):
                if utils.is_model_healthy(url, model, model_type):
                    logger.debug(f"{model} at {url} is healthy")
                else:
                    logger.warning(f"{model} at {url} not healthy!")
                    unhealthy_endpoints.append(self.get_model_endpoint_hash(url, model))
        except ValueError:
            logger.error(
                "To perform health check, each model has to define a static_model_type and at least one static_backend. "
                "Skipping health checks for now."
            )
        return unhealthy_endpoints

    async def check_model_health(self):
        while self._running:
            try:
                self.unhealthy_endpoint_hashes = self.get_unhealthy_endpoint_hashes()
                await asyncio.sleep(60)
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

    def get_endpoint_info(self) -> List[EndpointInfo]:
        """
        Get the URLs of the serving engines that are available for
        querying.

        Returns:
            a list of engine URLs
        """
        endpoint_infos = []
        for i, (url, model) in enumerate(zip(self.urls, self.models)):
            if (
                self.get_model_endpoint_hash(url, model)
                in self.unhealthy_endpoint_hashes
            ):
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
