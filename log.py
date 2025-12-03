import json
import logging
import sys
from datetime import datetime
from logging import Logger
from typing import Any, Dict


def build_format(color):
    reset = "\x1b[0m"
    underline = "\x1b[3m"
    return f"{color}[%(asctime)s] %(levelname)s:{reset} %(message)s {underline}(%(filename)s:%(lineno)d:%(name)s){reset}"


class CustomFormatter(logging.Formatter):
    grey = "\x1b[1m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: build_format(grey),
        logging.INFO: build_format(green),
        logging.WARNING: build_format(yellow),
        logging.ERROR: build_format(red),
        logging.CRITICAL: build_format(bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Create the base log entry
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "filename": record.filename,
            "line_number": record.lineno,
            "function": record.funcName,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        else:
            log_entry["exception"] = None

        # Add stack trace if present
        if record.stack_info:
            log_entry["stack_trace"] = self.formatStack(record.stack_info)
        else:
            log_entry["stack_trace"] = None

        # Add any extra fields that might be present
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "message",
            }:
                log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def init_logger(name: str, log_level=logging.DEBUG, log_format: str = "text") -> Logger:
    """
    Initialize a logger with specified format and level.

    Args:
        name: Logger name
        log_level: Logging level (default: DEBUG)
        log_format: Log format - "text" or "json" (default: "text")

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Choose formatter based on format
    if log_format.lower() == "json":
        formatter = JsonFormatter()
    else:
        formatter = CustomFormatter()

    # Setup stdout handler for DEBUG to INFO
    stdout_stream = logging.StreamHandler(sys.stdout)
    stdout_stream.setLevel(log_level)
    stdout_stream.setFormatter(formatter)
    stdout_stream.addFilter(MaxLevelFilter(logging.INFO))
    logger.addHandler(stdout_stream)

    # Setup stderr handler for WARNING and above
    error_stream = logging.StreamHandler(sys.stderr)
    error_stream.setLevel(logging.WARNING)
    error_stream.setFormatter(formatter)
    logger.addHandler(error_stream)

    logger.propagate = False

    return logger
