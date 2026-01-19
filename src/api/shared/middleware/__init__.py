"""
Shared API Middleware

Phase 5: API Stabilization
Phase 7: Authentication & Security
Phase 15: Observability

Provides cross-cutting concerns for all API endpoints:
- Error handling with standardized responses
- Trace ID propagation for observability
- OpenTelemetry distributed tracing
- Correlation ID support for related requests
- Authentication and session management
"""

from .error_handler import register_error_handlers
from .trace import (
    TraceMiddleware,
    get_trace_id,
    get_correlation_id,
    set_trace_id,
    set_correlation_id,
)
from .tracing import (
    TracingMiddleware,
    get_trace_id_from_request,
    get_correlation_id_from_request,
)
from .auth import (
    AuthMiddleware,
    get_current_operator,
    require_auth,
    is_auth_required,
)

__all__ = [
    # Error handling
    "register_error_handlers",
    # Trace (legacy)
    "TraceMiddleware",
    "get_trace_id",
    "get_correlation_id",
    "set_trace_id",
    "set_correlation_id",
    # OpenTelemetry Tracing
    "TracingMiddleware",
    "get_trace_id_from_request",
    "get_correlation_id_from_request",
    # Auth
    "AuthMiddleware",
    "get_current_operator",
    "require_auth",
    "is_auth_required",
]
