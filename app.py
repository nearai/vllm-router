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
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

import sentry_sdk
import uvicorn
from cachetools import TTLCache
from fastapi import Depends, FastAPI

from vllm_router.aiohttp_client import AiohttpClientWrapper
from vllm_router.httpx_client import HttpxClientWrapper
from vllm_router.auth import verify_admin_access, verify_user_access
from vllm_router.dynamic_config import (
    DynamicRouterConfig,
    get_dynamic_config_watcher,
    initialize_dynamic_config_watcher,
)
from vllm_router.graceful_shutdown import (
    get_shutdown_manager,
    initialize_shutdown_manager,
    setup_signal_handlers,
)
from vllm_router.middleware import GracefulShutdownMiddleware
from vllm_router.parsers.parser import parse_args
from vllm_router.routers.batches_router import batches_router
from vllm_router.routers.config_router import config_router
from vllm_router.routers.files_router import files_router
from vllm_router.routers.health_router import health_router
from vllm_router.routers.main_router import main_router
from vllm_router.routers.metrics_router import metrics_router
from vllm_router.routers.proxy_router import router as proxy_router
from vllm_router.routers.routing_logic import (
    cleanup_routing_logic,
    get_routing_logic,
    initialize_routing_logic,
)
from vllm_router.service_discovery import (
    ServiceDiscoveryType,
    get_service_discovery,
    initialize_service_discovery,
)
from vllm_router.services.batch_service import initialize_batch_processor
from vllm_router.services.callbacks_service.callbacks import configure_custom_callbacks
from vllm_router.services.files_service import initialize_storage
from vllm_router.services.request_service.rewriter import (
    get_request_rewriter,
)
from vllm_router.stats.engine_stats import (
    get_engine_stats_scraper,
    initialize_engine_stats_scraper,
)
from vllm_router.stats.log_stats import log_stats
from vllm_router.stats.request_stats import (
    get_request_stats_monitor,
    initialize_request_stats_monitor,
)
from vllm_router.utils import (
    parse_comma_separated_args,
    parse_static_aliases,
    parse_static_urls,
    set_ulimit,
)
from vllm_router.services.backend_discovery import (
    cleanup_backend_discovery,
    get_backend_discovery,
    initialize_backend_discovery,
)

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.aiohttp_client_wrapper.start()
    app.state.httpx_client_wrapper.start()
    if hasattr(app.state, "batch_processor"):
        await app.state.batch_processor.initialize()

    service_discovery = get_service_discovery()
    if hasattr(service_discovery, "initialize_client_sessions"):
        await service_discovery.initialize_client_sessions()

    app.state.event_loop = asyncio.get_event_loop()
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(app.state.event_loop)

    yield
    
    # Wait for in-flight requests to complete during graceful shutdown
    shutdown_manager = get_shutdown_manager()
    if shutdown_manager is not None and shutdown_manager.is_shutting_down:
        logger.info(
            f"Waiting for {shutdown_manager.in_flight_requests} in-flight requests to complete..."
        )
        await shutdown_manager.wait_for_requests()
    
    await app.state.httpx_client_wrapper.stop()
    await app.state.aiohttp_client_wrapper.stop()

    # Close the threaded-components
    logger.info("Closing engine stats scraper")
    engine_stats_scraper = get_engine_stats_scraper()
    engine_stats_scraper.close()

    logger.info("Closing service discovery module")
    service_discovery = get_service_discovery()
    service_discovery.close()

    # Close the optional dynamic config watcher
    dyn_cfg_watcher = get_dynamic_config_watcher()
    if dyn_cfg_watcher is not None:
        logger.info("Closing dynamic config watcher")
        dyn_cfg_watcher.close()

    # Close routing logic instances
    logger.info("Closing routing logic instances")
    cleanup_routing_logic()

    # Close backend discovery service
    logger.info("Closing backend discovery service")
    cleanup_backend_discovery()


def initialize_all(app: FastAPI, args):
    """
    Initialize all the components of the router with the given arguments.

    Args:
        app (FastAPI): FastAPI application
        args: the parsed command-line arguments

    Raises:
        ValueError: if the service discovery type is invalid
    """
    # Initialize the root logger with the specified format
    from vllm_router.log import init_logger
    import logging
    
    # Convert string log level to logging constant
    log_level_map = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "trace": logging.DEBUG,  # Map trace to debug
    }
    
    log_level = log_level_map.get(args.log_level.lower(), logging.INFO)
    root_logger = init_logger("vllm_router", log_level=log_level, log_format=args.log_format)

    # Reconfigure all existing vllm_router loggers to use the same level and format
    # (they were created at module import time with default values)
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("vllm_router"):
            init_logger(name, log_level=log_level, log_format=args.log_format)

    # Configure uvicorn logger to use the same format
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers.clear()
    uvicorn_logger.setLevel(log_level)
    
    # Add handlers to uvicorn logger
    for handler in root_logger.handlers:
        uvicorn_logger.addHandler(handler)
    uvicorn_logger.propagate = False
    if sentry_dsn := args.sentry_dsn:
        sentry_sdk.init(
            dsn=sentry_dsn,
            send_default_pii=True,
            profile_lifecycle="trace",
            traces_sample_rate=args.sentry_traces_sample_rate,
            profile_session_sample_rate=args.sentry_profile_session_sample_rate,
        )

    if args.service_discovery == "static":
        initialize_service_discovery(
            ServiceDiscoveryType.STATIC,
            app=app,
            urls=parse_static_urls(args.static_backends) if args.static_backends else None,
            models=parse_comma_separated_args(args.static_models) if args.static_models else None,
            aliases=(
                parse_static_aliases(args.static_aliases)
                if args.static_aliases
                else None
            ),
            model_types=(
                parse_comma_separated_args(args.static_model_types)
                if args.static_model_types
                else None
            ),
            model_labels=(
                parse_comma_separated_args(args.static_model_labels)
                if args.static_model_labels
                else None
            ),
            static_backend_health_checks=args.static_backend_health_checks,
            prefill_model_labels=args.prefill_model_labels,
            decode_model_labels=args.decode_model_labels,
            health_check_include_models_endpoint=args.health_check_include_models_endpoint,
            health_check_include_attestation=args.health_check_include_attestation,
            health_check_removal_threshold=args.health_check_removal_threshold,
            backend_health_check_timeout_seconds=args.backend_health_check_timeout_seconds,
        )
    else:
        raise ValueError(f"Invalid service discovery type: {args.service_discovery}")

    # Initialize singletons via custom functions.
    initialize_engine_stats_scraper(args.engine_stats_interval)
    initialize_request_stats_monitor(args.request_stats_window)

    if args.enable_batch_api:
        logger.info("Initializing batch API")
        app.state.batch_storage = initialize_storage(
            args.file_storage_class, args.file_storage_path
        )
        app.state.batch_processor = initialize_batch_processor(
            args.batch_processor, args.file_storage_path, app.state.batch_storage
        )

    # Initialize dynamic config watcher
    if args.dynamic_config_yaml or args.dynamic_config_json:
        init_config = DynamicRouterConfig.from_args(args)
        if args.dynamic_config_yaml:
            initialize_dynamic_config_watcher(
                args.dynamic_config_yaml, "YAML", 10, init_config, app
            )
        elif args.dynamic_config_json:
            initialize_dynamic_config_watcher(
                args.dynamic_config_json, "JSON", 10, init_config, app
            )

    if args.callbacks:
        configure_custom_callbacks(args.callbacks, app)

    initialize_routing_logic(
        args.routing_logic,
        session_key=args.session_key,
        lmcache_controller_port=args.lmcache_controller_port,
        prefill_model_labels=args.prefill_model_labels,
        decode_model_labels=args.decode_model_labels,
        kv_aware_threshold=args.kv_aware_threshold,
    )

    # --- Hybrid addition: attach singletons to FastAPI state ---
    app.state.engine_stats_scraper = get_engine_stats_scraper()
    app.state.request_stats_monitor = get_request_stats_monitor()
    app.state.router = get_routing_logic()
    app.state.request_rewriter = get_request_rewriter()

    # Initialize chat cache (maxsize=10000, ttl=1 hour)
    app.state.chat_cache = TTLCache(maxsize=10000, ttl=3600)

    # Initialize backend discovery if enabled
    if args.enable_backend_discovery:
        logger.info("Initializing backend discovery service")
        initialize_backend_discovery(
            tailscale_status_file=args.tailscale_status_file,
            discovery_interval=args.discovery_interval,
            port_range=args.discovery_port_range,
            timeout=args.discovery_timeout,
        )


app = FastAPI(lifespan=lifespan)

# Add graceful shutdown middleware (must be added before routers)
app.add_middleware(GracefulShutdownMiddleware)

app.include_router(health_router)
app.include_router(main_router, dependencies=[Depends(verify_user_access)])
app.include_router(proxy_router, dependencies=[Depends(verify_user_access)])
app.include_router(files_router, dependencies=[Depends(verify_user_access)])
app.include_router(batches_router, dependencies=[Depends(verify_user_access)])
app.include_router(metrics_router, dependencies=[Depends(verify_admin_access)])
app.include_router(config_router, dependencies=[Depends(verify_admin_access)])
app.state.aiohttp_client_wrapper = AiohttpClientWrapper()
app.state.httpx_client_wrapper = HttpxClientWrapper()


def main():
    args = parse_args()
    
    # Initialize graceful shutdown manager
    initialize_shutdown_manager(timeout=args.graceful_shutdown_timeout)
    logger.info(f"Graceful shutdown enabled with timeout: {args.graceful_shutdown_timeout}s")
    
    initialize_all(app, args)
    if args.log_stats:
        threading.Thread(
            target=log_stats,
            args=(
                app,
                args.log_stats_interval,
            ),
            daemon=True,
        ).start()

    # Workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active.
    set_ulimit()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
