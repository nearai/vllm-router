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
import argparse
import json
import logging
import sys

from vllm_router import utils
from vllm_router.parsers.yaml_utils import (
    read_and_process_yaml_config_file,
)
from vllm_router.version import __version__

logger = logging.getLogger(__name__)


def verify_required_args_provided(args: argparse.Namespace) -> None:
    if not args.routing_logic:
        logger.error("--routing-logic must be provided.")
        sys.exit(1)
    if not args.service_discovery:
        logger.error("--service-discovery must be provided.")
        sys.exit(1)


def load_initial_config_from_config_file_if_required(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    dynamic_config_yaml = args.dynamic_config_yaml
    dynamic_config_json = args.dynamic_config_json

    if dynamic_config_yaml:
        logger.info(
            f"Initial loading of dynamic YAML config file at {dynamic_config_yaml}"
        )
        yaml_config = read_and_process_yaml_config_file(dynamic_config_yaml)
        parser.set_defaults(**yaml_config)
        args = parser.parse_args()
    elif dynamic_config_json:
        logger.info(
            f"Initial loading of dynamic JSON config file at {dynamic_config_json}"
        )
        with open(dynamic_config_json, encoding="utf-8") as f:
            parser.set_defaults(**json.load(f))
            args = parser.parse_args()

    return args


def validate_static_model_types(model_types: str | None) -> None:
    if model_types is None:
        raise ValueError(
            "Static model types must be provided when using the backend healthcheck."
        )
    all_models = utils.ModelType.get_all_fields()
    for model_type in utils.parse_comma_separated_args(model_types):
        if model_type not in all_models:
            raise ValueError(
                f"The model type '{model_type}' is not supported. Supported model types are '{','.join(all_models)}'"
            )


# --- Argument Parsing and Initialization ---
def validate_args(args):
    verify_required_args_provided(args)
    if args.service_discovery == "static":
        # Allow starting without backends, but if one is provided, the other must be too
        if args.static_backends is not None and args.static_models is None:
            raise ValueError(
                "Static models must be provided when static backends are specified."
            )
        if args.static_models is not None and args.static_backends is None:
            raise ValueError(
                "Static backends must be provided when static models are specified."
            )
        # If both are provided, validate they have the same number of items
        if args.static_backends is not None and args.static_models is not None:
            backend_count = len(utils.parse_comma_separated_args(args.static_backends))
            model_count = len(utils.parse_comma_separated_args(args.static_models))
            if backend_count != model_count:
                raise ValueError(
                    f"Number of static backends ({backend_count}) must match number of static models ({model_count})."
                )
        if args.static_backend_health_checks:
            validate_static_model_types(args.static_model_types)
    if args.routing_logic == "session" and args.session_key is None:
        raise ValueError(
            "Session key must be provided when using session routing logic."
        )
    if args.log_stats and args.log_stats_interval <= 0:
        raise ValueError("Log stats interval must be greater than 0.")
    if args.engine_stats_interval <= 0:
        raise ValueError("Engine stats interval must be greater than 0.")
    if args.request_stats_window <= 0:
        raise ValueError("Request stats window must be greater than 0.")
    if not (0.0 <= args.sentry_traces_sample_rate <= 1.0):
        raise ValueError("Sentry traces sample rate must be between 0.0 and 1.0.")
    if not (0.0 <= args.sentry_profile_session_sample_rate <= 1.0):
        raise ValueError(
            "Sentry profile session sample rate must be between 0.0 and 1.0."
        )
    if args.graceful_shutdown_timeout <= 0:
        raise ValueError("Graceful shutdown timeout must be greater than 0.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FastAPI app.")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="The host to run the server on."
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="The port to run the server on."
    )
    parser.add_argument(
        "--service-discovery",
        type=str,
        choices=["static"],
        help="The service discovery type.",
    )
    parser.add_argument(
        "--static-backends",
        type=str,
        default=None,
        help="The URLs of static backends, separated by commas. E.g., http://localhost:8000,http://localhost:8001",
    )
    parser.add_argument(
        "--static-models",
        type=str,
        default=None,
        help="The models of static backends, separated by commas. E.g., model1,model2",
    )
    parser.add_argument(
        "--static-aliases",
        type=str,
        default=None,
        help="The aliases of static backends, separated by commas. E.g., your-custom-model:llama3",
    )
    parser.add_argument(
        "--static-model-types",
        type=str,
        default=None,
        help="Specify the static model types of each model. This is used for the backend health check, separated by commas. E.g. chat,embeddings,rerank",
    )
    parser.add_argument(
        "--static-model-labels",
        type=str,
        default=None,
        help="The model labels of static backends, separated by commas. E.g., model1,model2",
    )
    parser.add_argument(
        "--static-backend-health-checks",
        action="store_true",
        help="Enable this flag to make vllm-router check periodically if the models work by sending dummy requests to their endpoints.",
    )
    parser.add_argument(
        "--backend-health-check-timeout-seconds",
        type=int,
        default=10,
        help="Timeout in seconds for backend health check requests (default: 10).",
    )
    parser.add_argument(
        "--health-check-include-models-endpoint",
        action="store_true",
        default=True,
        help="Enable periodic checks of /v1/models endpoint to verify backend availability and update model lists.",
    )
    parser.add_argument(
        "--health-check-include-attestation",
        action="store_true",
        default=True,
        help="Enable periodic checks of /v1/attestation/report endpoint. Backends without this endpoint will be removed.",
    )
    parser.add_argument(
        "--health-check-removal-threshold",
        type=int,
        default=3,
        help="Number of consecutive health check failures before permanently removing a backend (default: 3).",
    )
    parser.add_argument(
        "--routing-logic",
        type=str,
        choices=[
            "roundrobin",
            "session",
            "kvaware",
            "prefixaware",
            "disaggregated_prefill",
        ],
        help="The routing logic to use",
    )
    parser.add_argument(
        "--lmcache-controller-port",
        type=int,
        default=9000,
        help="The port of the LMCache controller.",
    )
    parser.add_argument(
        "--session-key",
        type=str,
        default=None,
        help="The key (in the header) to identify a session.",
    )
    parser.add_argument(
        "--callbacks",
        type=str,
        default=None,
        help="Path to the callback instance extending CustomCallbackHandler. Consists of <file path without .py ending>.<instance variable name>.",
    )

    # Request rewriter arguments
    parser.add_argument(
        "--request-rewriter",
        type=str,
        default="noop",
        choices=["noop"],
        help="The request rewriter to use. Default is 'noop' (no rewriting).",
    )

    # Batch API
    # TODO(gaocegege): Make these batch api related arguments to a separate config.
    parser.add_argument(
        "--enable-batch-api",
        action="store_true",
        help="Enable the batch API for processing files.",
    )
    parser.add_argument(
        "--file-storage-class",
        type=str,
        default="local_file",
        choices=["local_file"],
        help="The file storage class to use.",
    )
    parser.add_argument(
        "--file-storage-path",
        type=str,
        default="/tmp/vllm_files",
        help="The path to store files.",
    )
    parser.add_argument(
        "--batch-processor",
        type=str,
        default="local",
        choices=["local"],
        help="The batch processor to use.",
    )

    # Monitoring
    parser.add_argument(
        "--engine-stats-interval",
        type=int,
        default=30,
        help="The interval in seconds to scrape engine statistics.",
    )
    parser.add_argument(
        "--request-stats-window",
        type=int,
        default=60,
        help="The sliding window in seconds to compute request statistics.",
    )
    parser.add_argument(
        "--log-stats", action="store_true", help="Log statistics periodically."
    )
    parser.add_argument(
        "--log-stats-interval",
        type=int,
        default=10,
        help="The interval in seconds to log statistics.",
    )

    # Config files
    group = parser.add_argument_group(
        "Dynamic config file",
        "Only one dynamic config file (YAML or JSON) can be provided",
    )
    exclusive_group = group.add_mutually_exclusive_group()
    exclusive_group.add_argument(
        "--dynamic-config-yaml",
        type=str,
        default=None,
        help="The path to the YAML file containing the dynamic configuration, cannot be used with --dynamic-config-json.",
    )
    exclusive_group.add_argument(
        "--dynamic-config-json",
        type=str,
        default=None,
        help="The path to the JSON file containing the dynamic configuration, cannot be used with --dynamic-config-yaml.",
    )

    # Add --version argument
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version and exit",
    )

    # Add feature gates argument
    parser.add_argument(
        "--feature-gates",
        type=str,
        default="",
        help="Comma-separated list of feature gates (e.g., 'SemanticCache=true')",
    )

    # Add log level argument
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Log level for uvicorn. Default is 'info'.",
    )

    # Add log format argument
    parser.add_argument(
        "--log-format",
        type=str,
        default="json",
        choices=["text", "json"],
        help="Log format for the router. Default is 'json'. Use 'json' for structured logging.",
    )

    parser.add_argument(
        "--sentry-dsn",
        type=str,
        help="Enables Sentry Error Reporting to the specified Data Source Name",
    )

    parser.add_argument(
        "--sentry-traces-sample-rate",
        type=float,
        default=0.1,
        help="The sample rate for Sentry traces. Default is 0.1 (10%)",
    )

    parser.add_argument(
        "--sentry-profile-session-sample-rate",
        type=float,
        default=1.0,
        help="The sample rate for Sentry profiling sessions. Default is 1.0 (100%)",
    )

    parser.add_argument(
        "--prefill-model-labels",
        type=str,
        default=None,
        help="The model labels of prefill backends, separated by commas. E.g., model1,model2",
    )

    parser.add_argument(
        "--decode-model-labels",
        type=str,
        default=None,
        help="The model labels of decode backends, separated by commas. E.g., model1,model2",
    )

    parser.add_argument(
        "--kv-aware-threshold",
        type=int,
        default=2000,
        help="The threshold for kv-aware routing.",
    )

    # Backend discovery arguments
    parser.add_argument(
        "--enable-backend-discovery",
        action="store_true",
        help="Enable automatic backend discovery from Tailscale status",
    )
    parser.add_argument(
        "--tailscale-status-file",
        type=str,
        default="/shared/tailscale_status.json",
        help="Path to Tailscale status JSON file",
    )
    parser.add_argument(
        "--discovery-interval",
        type=int,
        default=30,
        help="Backend discovery interval in seconds",
    )
    parser.add_argument(
        "--discovery-port-range",
        type=str,
        default="8000-8010",
        help="Port range to test for backends (e.g., 8000-8010)",
    )
    parser.add_argument(
        "--discovery-timeout",
        type=int,
        default=2,
        help="Timeout in seconds for backend health check requests",
    )

    # Graceful shutdown
    parser.add_argument(
        "--graceful-shutdown-timeout",
        type=float,
        default=30.0,
        help="Maximum time in seconds to wait for in-flight requests to complete during shutdown (default: 30.0)",
    )

    # Backend request timeouts
    parser.add_argument(
        "--backend-connect-timeout",
        type=float,
        default=5.0,
        help="Timeout in seconds for establishing a connection to a backend (default: 5.0). "
        "This is a short timeout to quickly detect dead backends.",
    )
    parser.add_argument(
        "--backend-read-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds between receiving chunks from a backend (default: 300.0 / 5 minutes). "
        "For streaming requests, this is the max time allowed between tokens. "
        "Set to 0 for no read timeout (not recommended).",
    )

    # Circuit breaker configuration
    parser.add_argument(
        "--circuit-breaker-threshold",
        type=int,
        default=3,
        help="Number of consecutive failures required to open the circuit breaker (default: 3). "
        "Higher values make the circuit breaker less aggressive.",
    )
    parser.add_argument(
        "--circuit-breaker-cooldown",
        type=float,
        default=5.0,
        help="Initial cooldown in seconds when circuit breaker opens (default: 5.0).",
    )
    parser.add_argument(
        "--circuit-breaker-max-cooldown",
        type=float,
        default=30.0,
        help="Maximum cooldown in seconds with exponential backoff (default: 30.0).",
    )

    args = parser.parse_args()
    args = load_initial_config_from_config_file_if_required(parser, args)

    validate_args(args)
    return args
