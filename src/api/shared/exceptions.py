"""
API Exception Classes

Phase 5: API Stabilization

Custom exceptions that map to standard error responses.
"""

from typing import Optional, List

from .error_codes import ErrorCode, get_status_code
from .responses import ErrorDetail


class APIException(Exception):
    """
    Base exception for API errors.

    All custom API exceptions should inherit from this class.
    The error handler middleware will catch these and return
    standardized error responses.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        trace_id: Optional[str] = None
    ):
        self.code = code
        self.message = message
        self.details = details
        self.trace_id = trace_id
        self.status_code = get_status_code(code)
        super().__init__(message)


class ValidationError(APIException):
    """
    Validation error for invalid request data.

    HTTP Status: 400
    """

    def __init__(
        self,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        trace_id: Optional[str] = None
    ):
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            details=details,
            trace_id=trace_id
        )


class NotFoundError(APIException):
    """
    Resource not found error.

    HTTP Status: 404
    """

    def __init__(
        self,
        resource: str,
        resource_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ):
        message = f"{resource} not found"
        if resource_id:
            message = f"{resource} with ID '{resource_id}' not found"

        # Use specific error code if available
        code_map = {
            "Deal": ErrorCode.DEAL_NOT_FOUND,
            "Action": ErrorCode.ACTION_NOT_FOUND,
            "Artifact": ErrorCode.ARTIFACT_NOT_FOUND,
            "Quarantine item": ErrorCode.QUARANTINE_ITEM_NOT_FOUND,
            "QuarantineItem": ErrorCode.QUARANTINE_ITEM_NOT_FOUND,
        }
        code = code_map.get(resource, ErrorCode.NOT_FOUND)

        super().__init__(code=code, message=message, trace_id=trace_id)
        self.resource = resource
        self.resource_id = resource_id


class ConflictError(APIException):
    """
    Conflict error (e.g., duplicate, already exists, invalid state).

    HTTP Status: 409
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CONFLICT,
        trace_id: Optional[str] = None
    ):
        super().__init__(code=code, message=message, trace_id=trace_id)


class UnauthorizedError(APIException):
    """
    Authentication required error.

    HTTP Status: 401
    """

    def __init__(
        self,
        message: str = "Authentication required",
        trace_id: Optional[str] = None
    ):
        super().__init__(
            code=ErrorCode.UNAUTHORIZED,
            message=message,
            trace_id=trace_id
        )


class ForbiddenError(APIException):
    """
    Permission denied error.

    HTTP Status: 403
    """

    def __init__(
        self,
        message: str = "Permission denied",
        trace_id: Optional[str] = None
    ):
        super().__init__(
            code=ErrorCode.FORBIDDEN,
            message=message,
            trace_id=trace_id
        )


class BusinessLogicError(APIException):
    """
    Business logic violation error.

    Use this for domain-specific errors like invalid state transitions.

    HTTP Status: Varies by error code (typically 422)
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        trace_id: Optional[str] = None
    ):
        super().__init__(
            code=code,
            message=message,
            details=details,
            trace_id=trace_id
        )


class DatabaseError(APIException):
    """
    Database operation error.

    HTTP Status: 500
    """

    def __init__(
        self,
        message: str = "A database error occurred",
        trace_id: Optional[str] = None
    ):
        super().__init__(
            code=ErrorCode.DATABASE_ERROR,
            message=message,
            trace_id=trace_id
        )


class ExternalServiceError(APIException):
    """
    External service error.

    HTTP Status: 502
    """

    def __init__(
        self,
        service: str,
        message: Optional[str] = None,
        trace_id: Optional[str] = None
    ):
        msg = message or f"External service '{service}' is unavailable"
        super().__init__(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message=msg,
            trace_id=trace_id
        )
        self.service = service


class AgentError(APIException):
    """
    Agent execution error.

    HTTP Status: 500
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.AGENT_ERROR,
        trace_id: Optional[str] = None
    ):
        super().__init__(
            code=code,
            message=message,
            trace_id=trace_id
        )
