"""
Structured Logging with Trace Correlation

Phase 15: Observability

Configures logging to include trace_id and correlation_id.
"""

import logging
import json
import sys
from datetime import datetime
from typing import Optional

from .tracing import get_trace_id, get_current_span


class StructuredFormatter(logging.Formatter):
    """
    JSON structured log formatter with trace context.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Get trace context
        trace_id = get_trace_id()
        span = get_current_span()
        span_id = None
        if span and span.get_span_context().is_valid:
            span_id = format(span.get_span_context().span_id, '016x')

        # Build log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": trace_id,
            "span_id": span_id,
        }

        # Add correlation_id if available
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = record.correlation_id

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "correlation_id", "message", "taskName"
            ):
                if not key.startswith("_"):
                    try:
                        json.dumps(value)  # Test if serializable
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)

        return json.dumps(log_entry)


class TraceContextFilter(logging.Filter):
    """
    Filter that adds trace context to log records.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = get_trace_id() or "no-trace"
        return True


def configure_logging(
    level: str = "INFO",
    structured: bool = True,
    service_name: str = "zakops-backend"
):
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        structured: Use JSON structured format
        service_name: Service name for logs
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    # Set formatter
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s] %(message)s"
        ))

    # Add trace context filter
    handler.addFilter(TraceContextFilter())

    root_logger.addHandler(handler)

    # Set levels for noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)

    logging.info(f"Logging configured: {service_name}, level={level}, structured={structured}")
