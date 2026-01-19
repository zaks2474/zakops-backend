"""
Observability Module

Phase 15: OpenTelemetry

Provides distributed tracing, metrics collection, and structured logging.
"""

from .tracing import (
    init_tracing,
    get_tracer,
    get_current_span,
    get_trace_id,
    create_span,
    inject_trace_context,
    extract_trace_context,
    traced,
    add_correlation_id_to_span,
    add_event_to_span,
)
from .metrics import (
    init_metrics,
    get_meter,
    record_counter,
    record_histogram,
)
from .logging import configure_logging

__all__ = [
    # Tracing
    "init_tracing",
    "get_tracer",
    "get_current_span",
    "get_trace_id",
    "create_span",
    "inject_trace_context",
    "extract_trace_context",
    "traced",
    "add_correlation_id_to_span",
    "add_event_to_span",
    # Metrics
    "init_metrics",
    "get_meter",
    "record_counter",
    "record_histogram",
    # Logging
    "configure_logging",
]
