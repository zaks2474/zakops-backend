"""
Trace ID Middleware

Phase 5: API Stabilization

Adds trace_id and correlation_id to all requests for observability.
"""

import contextvars
from uuid import uuid4
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


# Context variables for request-scoped values
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")


def get_trace_id() -> str:
    """
    Get the current trace ID.

    Returns the trace ID from the current request context,
    or generates a new one if not set.
    """
    return trace_id_var.get() or str(uuid4())


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID.

    Returns the correlation ID from the current request context,
    or None if not set.
    """
    return correlation_id_var.get() or None


def set_trace_id(trace_id: str) -> None:
    """Set the trace ID for the current context."""
    trace_id_var.set(trace_id)


def set_correlation_id(correlation_id: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_var.set(correlation_id)


class TraceMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts or generates trace/correlation IDs.

    Headers:
    - X-Trace-ID: Unique ID for this request (generated if not provided)
    - X-Correlation-ID: ID linking related requests (e.g., deal_id)

    The trace ID is used for distributed tracing and log correlation.
    The correlation ID links related business operations (e.g., all
    requests for a specific deal).
    """

    async def dispatch(self, request: Request, call_next):
        # Extract or generate trace ID
        trace_id = request.headers.get("X-Trace-ID") or str(uuid4())
        trace_id_var.set(trace_id)

        # Extract correlation ID if provided
        correlation_id = request.headers.get("X-Correlation-ID") or ""
        correlation_id_var.set(correlation_id)

        # Add to request state for easy access in route handlers
        request.state.trace_id = trace_id
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add trace ID to response headers
        response.headers["X-Trace-ID"] = trace_id
        if correlation_id:
            response.headers["X-Correlation-ID"] = correlation_id

        return response
