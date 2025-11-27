import abc
import enum
import io
import json
import re
import resource
import wave
from typing import Optional

import requests
from fastapi.requests import Request
from starlette.datastructures import MutableHeaders

from vllm_router.log import init_logger

logger = init_logger(__name__)

# prepare a WAV byte to prevent repeatedly generating it
# Generate a 0.1 second silent audio file
# This will be used for the /v1/audio/transcriptions endpoint
_SILENT_WAV_BYTES = None
with io.BytesIO() as wav_buffer:
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)  # mono audio channel, standard configuration
        wf.setsampwidth(2)  # 16 bit audio, common bit depth for wav file
        wf.setframerate(16000)  # 16 kHz sample rate
        wf.writeframes(b"\x00\x00" * 1600)  # 0.1 second of silence

    # retrieves the generated wav bytes, return
    _SILENT_WAV_BYTES = wav_buffer.getvalue()
    logger.debug(
        "======A default silent WAV file has been stored in memory within py application process===="
    )


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Note: if the class is called with _create=False, it will return None
        if the instance does not exist.
        """
        if cls not in cls._instances:
            if kwargs.get("_create") is False:
                return None
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class SingletonABCMeta(abc.ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Note: if the class is called with _create=False, it will return None
        if the instance does not exist.
        """
        if cls not in cls._instances:
            if kwargs.get("create") is False:
                return None
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ModelType(enum.Enum):
    chat = "chat"
    completion = "completion"
    embeddings = "embeddings"
    rerank = "rerank"
    score = "score"
    transcription = "transcription"
    vision = "vision"

    @staticmethod
    def get_url(model_type: str):
        match ModelType[model_type]:
            case ModelType.chat | ModelType.vision:
                return "/v1/chat/completions"
            case ModelType.completion:
                return "/v1/completions"
            case ModelType.embeddings:
                return "/v1/embeddings"
            case ModelType.rerank:
                return "/v1/rerank"
            case ModelType.score:
                return "/v1/score"
            case ModelType.transcription:
                return "/v1/audio/transcriptions"

    @staticmethod
    def get_test_payload(model_type: str):
        match ModelType[model_type]:
            case ModelType.chat:
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello",
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 3,
                }
            case ModelType.completion:
                return {"prompt": "Hello", "max_tokens": 3}
            case ModelType.embeddings:
                return {"input": "Hello"}
            case ModelType.rerank:
                return {"query": "Hello", "documents": ["Test"]}
            case ModelType.score:
                return {"encoding_format": "float", "text_1": "Test", "test_2": "Test2"}
            case ModelType.transcription:
                if _SILENT_WAV_BYTES is not None:
                    logger.debug("=====Silent WAV Bytes is being used=====")
                    return {
                        "file": ("empty.wav", _SILENT_WAV_BYTES, "audio/wav"),
                    }
            case ModelType.vision:
                return {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "This is a test. Just reply with yes",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAG0lEQVR4nGLinfJq851wJn69udZSvIAAAAD//yf3BLKCfW8HAAAAAElFTkSuQmCC"
                                    },
                                },
                            ],
                        }
                    ]
                }

    @staticmethod
    def get_all_fields():
        return [model_type.name for model_type in ModelType]


def validate_url(url: str) -> bool:
    """
    Validates the format of the given URL.

    Args:
        url (str): The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    regex = re.compile(
        r"^(http|https)://"  # Protocol
        r"(([a-zA-Z0-9_-]+\.)+[a-zA-Z]{2,}|"  # Domain name
        r"localhost|"  # Or localhost
        r"\d{1,3}(\.\d{1,3}){3})"  # Or IPv4 address
        r"(:\d+)?"  # Optional port
        r"(/.*)?$"  # Optional path
    )
    return bool(regex.match(url))


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630 # noqa: E501
def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase"
                "with error %s. This can cause fd limit errors like"
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )


def parse_static_urls(static_backends: str):
    urls = static_backends.split(",")
    backend_urls = []
    for url in urls:
        if validate_url(url):
            backend_urls.append(url)
        else:
            logger.warning(f"Skipping invalid URL: {url}")
    return backend_urls


def parse_comma_separated_args(comma_separated_string: Optional[str]):
    if comma_separated_string is None:
        return None
    return comma_separated_string.split(",")


def parse_static_aliases(static_aliases: str):
    aliases = {}
    for alias_and_model in static_aliases.split(","):
        alias, model = alias_and_model.split(":")
        aliases[alias] = model
    logger.info(f"Loaded aliases {aliases}")
    return aliases


def replace_model_in_request_body(request_json: dict, model: str):
    request_json["model"] = model
    request_body = json.dumps(request_json)
    return request_body


def update_content_length(request: Request, request_body: str):
    headers = MutableHeaders(request.headers)
    headers["Content-Length"] = str(len(request_body))
    request._headers = headers


def is_model_healthy(url: str, model: str, model_type: str) -> bool:
    model_url = ModelType.get_url(model_type)
    full_url = f"{url}{model_url}"
    
    logger.debug(f"Starting health check for {model_type} model {model} at {full_url}")

    try:
        if model_type == "transcription":
            # for transcription, the backend expects multipart/form-data with a file
            # we will use pre-generated silent wav bytes
            logger.debug(f"Testing transcription model {model} with multipart/form-data request to {full_url}")
            response = requests.post(
                f"{url}{model_url}",
                files=ModelType.get_test_payload(model_type),  # multipart/form-data
                data={"model": model},
                timeout=10,
            )
        else:
            # for other model types (chat, completion, etc.)
            test_payload = {"model": model} | ModelType.get_test_payload(model_type)
            logger.debug(f"Testing {model_type} model {model} with JSON payload to {full_url}")
            logger.debug(f"Request payload for {model}: {test_payload}")
            response = requests.post(
                f"{url}{model_url}",
                headers={"Content-Type": "application/json"},
                json=test_payload,
                timeout=10,
            )

        logger.debug(f"Health check response for {model} at {full_url}: HTTP {response.status_code}")
        logger.debug(f"Response headers for {model}: {dict(response.headers)}")

        response.raise_for_status()

        if model_type == "transcription":
            logger.debug(f"Transcription model {model} health check passed (HTTP {response.status_code})")
            return True
        else:
            logger.debug(f"Parsing JSON response for {model_type} model {model}")
            response_json = response.json()  # verify it's valid json for other model types
            logger.debug(f"Response JSON for {model}: {response_json}")
            logger.info(f"{model_type} model {model} at {url} is healthy (HTTP {response.status_code})")
            return True  # validation passed

    except requests.exceptions.Timeout as e:
        logger.debug(f"{model_type} Model {model} at {url} health check TIMEOUT: {e}")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.debug(f"{model_type} Model {model} at {url} health check CONNECTION ERROR: {e}")
        return False
    except requests.exceptions.HTTPError as e:
        logger.debug(f"{model_type} Model {model} at {url} health check HTTP ERROR: {e}")
        logger.debug(f"Response content for {model}: {e.response.text if e.response else 'No response'}")
        return False
    except requests.exceptions.RequestException as e:
        logger.debug(f"{model_type} Model {model} at {url} health check REQUEST ERROR: {e}")
        return False
    except json.JSONDecodeError as e:
        logger.debug(f"{model_type} Model {model} at {url} health check JSON DECODE ERROR: {e}")
        return False
    except Exception as e:
        logger.debug(f"{model_type} Model {model} at {url} health check UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return False


def fetch_models_list(url: str, timeout: int = 10) -> Optional[list[str]]:
    """
    Fetch the list of models from a backend's /v1/models endpoint.

    Args:
        url: The base URL of the backend
        timeout: Request timeout in seconds

    Returns:
        List of model IDs if successful, None if the endpoint is not available
    """
    models_url = f"{url}/v1/models"
    logger.debug(f"Fetching models list from {models_url} (timeout: {timeout}s)")
    
    try:
        logger.debug(f"Sending GET request to {models_url}")
        response = requests.get(models_url, timeout=timeout)
        logger.debug(f"Models endpoint response from {url}: HTTP {response.status_code}")
        
        response.raise_for_status()
        
        logger.debug(f"Parsing JSON response from {models_url}")
        data = response.json()
        models_data = data.get("data", [])
        model_ids = [model.get("id") for model in models_data if model.get("id")]
        
        logger.debug(f"Raw models data from {url}: {models_data}")
        logger.info(f"Successfully fetched {len(model_ids)} models from {url}: {model_ids}")
        
        return model_ids
        
    except requests.exceptions.Timeout as e:
        logger.debug(f"Timeout fetching models from {url}: {e}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.debug(f"Connection error fetching models from {url}: {e}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.debug(f"HTTP error fetching models from {url}: {e}")
        logger.debug(f"Response content from {url}: {e.response.text if e.response else 'No response'}")
        return None
    except requests.exceptions.RequestException as e:
        logger.debug(f"Request error fetching models from {url}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode error fetching models from {url}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Unexpected error fetching models from {url}: {type(e).__name__}: {e}")
        return None


def check_attestation_available(url: str, timeout: int = 10) -> bool:
    """
    Check if the /v1/attestation/report endpoint is available on a backend.

    Args:
        url: The base URL of the backend
        timeout: Request timeout in seconds

    Returns:
        True if the endpoint responds successfully, False otherwise
    """
    attestation_url = f"{url}/v1/attestation/report"
    logger.debug(f"Checking attestation endpoint availability at {attestation_url} (timeout: {timeout}s)")
    
    try:
        logger.debug(f"Sending GET request to attestation endpoint {attestation_url}")
        response = requests.get(attestation_url, timeout=timeout)
        logger.debug(f"Attestation endpoint response from {url}: HTTP {response.status_code}")
        
        response.raise_for_status()
        
        # Verify it returns valid JSON
        logger.debug(f"Parsing JSON response from attestation endpoint {attestation_url}")
        response_json = response.json()
        logger.debug(f"Attestation response JSON from {url}: {response_json}")
        logger.info(f"Attestation endpoint available at {url} (HTTP {response.status_code})")
        
        return True
        
    except requests.exceptions.Timeout as e:
        logger.debug(f"Timeout checking attestation endpoint at {url}: {e}")
        return False
    except requests.exceptions.ConnectionError as e:
        logger.debug(f"Connection error checking attestation endpoint at {url}: {e}")
        return False
    except requests.exceptions.HTTPError as e:
        logger.debug(f"HTTP error checking attestation endpoint at {url}: {e}")
        logger.debug(f"Attestation response content from {url}: {e.response.text if e.response else 'No response'}")
        return False
    except requests.exceptions.RequestException as e:
        logger.debug(f"Request error checking attestation endpoint at {url}: {e}")
        return False
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode error checking attestation endpoint at {url}: {e}")
        return False
    except Exception as e:
        logger.debug(f"Unexpected error checking attestation endpoint at {url}: {type(e).__name__}: {e}")
        return False
