"""
Shared API Middleware

Phase 5: API Stabilization

Provides cross-cutting concerns for all API endpoints:
- Error handling with standardized responses
- Trace ID propagation for observability
- Correlation ID support for related requests
"""

from .error_handler import register_error_handlers
from .trace import (
    TraceMiddleware,
    get_trace_id,
    get_correlation_id,
    set_trace_id,
    set_correlation_id,
)

__all__ = [
    "register_error_handlers",
    "TraceMiddleware",
    "get_trace_id",
    "get_correlation_id",
    "set_trace_id",
    "set_correlation_id",
]
