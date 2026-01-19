"""
OpenTelemetry Tracing

Phase 15: Observability

Provides distributed tracing with trace_id propagation.
"""

import logging
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from functools import wraps

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode, Span
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.propagate import set_global_textmap, inject, extract

logger = logging.getLogger(__name__)

# Global tracer
_tracer: Optional[trace.Tracer] = None
_propagator = TraceContextTextMapPropagator()


def init_tracing(
    service_name: str = "zakops-backend",
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing.

    Args:
        service_name: Name of the service
        service_version: Version of the service
        otlp_endpoint: OTLP exporter endpoint (e.g., "http://localhost:4317")
        console_export: Enable console export for debugging

    Returns:
        Configured tracer
    """
    global _tracer

    # Create resource
    resource = Resource.create({
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add exporters
    if otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info(f"OTel tracing: OTLP exporter configured -> {otlp_endpoint}")

    if console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OTel tracing: Console exporter enabled")

    # Set global tracer provider
    trace.set_tracer_provider(provider)

    # Set global propagator
    set_global_textmap(_propagator)

    # Get tracer
    _tracer = trace.get_tracer(service_name, service_version)

    logger.info(f"OTel tracing initialized: {service_name} v{service_version}")

    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("zakops-backend")
    return _tracer


def get_current_span() -> Optional[Span]:
    """Get the current active span."""
    return trace.get_current_span()


def get_trace_id() -> Optional[str]:
    """Get the current trace ID as hex string."""
    span = get_current_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, '032x')
    return None


@contextmanager
def create_span(
    name: str,
    attributes: Dict[str, Any] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
):
    """
    Create a new span as context manager.

    Usage:
        with create_span("my_operation", {"key": "value"}) as span:
            # do work
            span.set_attribute("result", "success")
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=attributes or {}
    ) as span:
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def traced(
    name: Optional[str] = None,
    attributes: Dict[str, Any] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
) -> Callable:
    """
    Decorator to trace a function.

    Usage:
        @traced("my_function")
        async def my_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with create_span(span_name, attributes, kind) as span:
                # Add function arguments as attributes
                span.set_attribute("function.name", func.__name__)
                return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with create_span(span_name, attributes, kind) as span:
                span.set_attribute("function.name", func.__name__)
                return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def inject_trace_context(carrier: Dict[str, str]) -> Dict[str, str]:
    """
    Inject trace context into a carrier (e.g., HTTP headers).

    Args:
        carrier: Dictionary to inject context into

    Returns:
        Carrier with trace context
    """
    inject(carrier)
    return carrier


def extract_trace_context(carrier: Dict[str, str]) -> trace.Context:
    """
    Extract trace context from a carrier (e.g., HTTP headers).

    Args:
        carrier: Dictionary containing trace context

    Returns:
        Extracted context
    """
    return extract(carrier)


def add_correlation_id_to_span(correlation_id: str, span: Optional[Span] = None):
    """Add correlation_id (deal_id) to current span."""
    span = span or get_current_span()
    if span:
        span.set_attribute("correlation_id", correlation_id)
        span.set_attribute("deal_id", correlation_id)


def add_event_to_span(
    name: str,
    attributes: Dict[str, Any] = None,
    span: Optional[Span] = None
):
    """Add an event to the current span."""
    span = span or get_current_span()
    if span:
        span.add_event(name, attributes or {})
