"""
Shared API Utilities

Phase 5: API Stabilization

Common utilities, responses, and middleware for all API endpoints.
"""

from .responses import (
    ResponseMeta,
    SuccessResponse,
    ListMeta,
    ListResponse,
    ErrorDetail,
    ErrorBody,
    ErrorResponse,
)

from .error_codes import (
    ErrorCode,
    get_status_code,
    is_client_error,
    is_server_error,
)

from .exceptions import (
    APIException,
    ValidationError,
    NotFoundError,
    ConflictError,
    UnauthorizedError,
    ForbiddenError,
    BusinessLogicError,
    DatabaseError,
    ExternalServiceError,
    AgentError,
)

from .middleware import (
    register_error_handlers,
    TraceMiddleware,
    get_trace_id,
    get_correlation_id,
    set_trace_id,
    set_correlation_id,
)

__all__ = [
    # Responses
    "ResponseMeta",
    "SuccessResponse",
    "ListMeta",
    "ListResponse",
    "ErrorDetail",
    "ErrorBody",
    "ErrorResponse",
    # Error codes
    "ErrorCode",
    "get_status_code",
    "is_client_error",
    "is_server_error",
    # Exceptions
    "APIException",
    "ValidationError",
    "NotFoundError",
    "ConflictError",
    "UnauthorizedError",
    "ForbiddenError",
    "BusinessLogicError",
    "DatabaseError",
    "ExternalServiceError",
    "AgentError",
    # Middleware
    "register_error_handlers",
    "TraceMiddleware",
    "get_trace_id",
    "get_correlation_id",
    "set_trace_id",
    "set_correlation_id",
]
