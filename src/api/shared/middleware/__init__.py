"""
Shared API Middleware

Phase 5: API Stabilization
Phase 7: Authentication & Security

Provides cross-cutting concerns for all API endpoints:
- Error handling with standardized responses
- Trace ID propagation for observability
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
from .auth import (
    AuthMiddleware,
    get_current_operator,
    require_auth,
    is_auth_required,
)

__all__ = [
    # Error handling
    "register_error_handlers",
    # Trace
    "TraceMiddleware",
    "get_trace_id",
    "get_correlation_id",
    "set_trace_id",
    "set_correlation_id",
    # Auth
    "AuthMiddleware",
    "get_current_operator",
    "require_auth",
    "is_auth_required",
]
