"""
OpenTelemetry Tracing Middleware

Phase 15: Observability

FastAPI middleware for automatic request tracing with OpenTelemetry.
"""

import time
import logging
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ....core.observability.tracing import (
    get_tracer, extract_trace_context,
    add_correlation_id_to_span, get_trace_id
)
from ....core.observability.metrics import record_counter, record_histogram
from opentelemetry import trace

logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that creates OpenTelemetry spans for HTTP requests.

    Features:
    - Extracts trace context from incoming headers
    - Creates request span
    - Adds standard HTTP attributes
    - Records request metrics
    - Propagates trace_id to response
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip tracing for health endpoints
        if request.url.path in ["/health", "/health/live", "/health/ready", "/health/startup"]:
            return await call_next(request)

        # Extract trace context from headers
        carrier = dict(request.headers)
        context = extract_trace_context(carrier)

        # Get or generate trace_id
        trace_id = request.headers.get("X-Trace-ID")
        if not trace_id:
            trace_id = uuid4().hex

        # Get correlation_id (deal_id) if present
        correlation_id = request.headers.get("X-Correlation-ID")

        tracer = get_tracer()
        start_time = time.time()

        # Create span for request
        with tracer.start_as_current_span(
            f"{request.method} {request.url.path}",
            context=context,
            kind=trace.SpanKind.SERVER,
            attributes={
                "http.method": request.method,
                "http.url": str(request.url),
                "http.route": request.url.path,
                "http.scheme": request.url.scheme,
                "http.host": request.url.hostname or "",
                "http.user_agent": request.headers.get("user-agent", ""),
                "trace_id": trace_id,
            }
        ) as span:
            # Add correlation_id if present
            if correlation_id:
                add_correlation_id_to_span(correlation_id, span)

            # Store trace_id in request state for handlers
            request.state.trace_id = trace_id
            request.state.correlation_id = correlation_id

            try:
                response = await call_next(request)

                # Add response attributes
                span.set_attribute("http.status_code", response.status_code)

                # Record metrics
                duration = time.time() - start_time
                record_counter("http_requests_total", 1, {
                    "method": request.method,
                    "path": request.url.path,
                    "status": str(response.status_code)
                })
                record_histogram("http_request_duration_seconds", duration, {
                    "method": request.method,
                    "path": request.url.path
                })

                # Add trace_id to response headers
                otel_trace_id = get_trace_id()
                response.headers["X-Trace-ID"] = otel_trace_id or trace_id

                return response

            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                span.record_exception(e)

                # Record error metrics
                record_counter("http_requests_total", 1, {
                    "method": request.method,
                    "path": request.url.path,
                    "status": "500"
                })

                raise


def get_trace_id_from_request(request: Request) -> str:
    """Get trace_id from request state."""
    return getattr(request.state, "trace_id", None)


def get_correlation_id_from_request(request: Request) -> str:
    """Get correlation_id from request state."""
    return getattr(request.state, "correlation_id", None)
