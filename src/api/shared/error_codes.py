"""
Standard Error Codes

Phase 5: API Stabilization

Consistent error codes across all API endpoints with HTTP status mapping.
"""

from enum import Enum


class ErrorCode(str, Enum):
    """API error codes."""

    # Client errors (4xx)
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    BAD_REQUEST = "BAD_REQUEST"

    # Business logic errors
    DEAL_NOT_FOUND = "DEAL_NOT_FOUND"
    ACTION_NOT_FOUND = "ACTION_NOT_FOUND"
    ARTIFACT_NOT_FOUND = "ARTIFACT_NOT_FOUND"
    INVALID_STAGE_TRANSITION = "INVALID_STAGE_TRANSITION"
    ACTION_ALREADY_APPROVED = "ACTION_ALREADY_APPROVED"
    ACTION_ALREADY_REJECTED = "ACTION_ALREADY_REJECTED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    QUARANTINE_ITEM_NOT_FOUND = "QUARANTINE_ITEM_NOT_FOUND"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"

    # Agent errors
    AGENT_ERROR = "AGENT_ERROR"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"


# HTTP status code mapping
ERROR_STATUS_CODES = {
    ErrorCode.VALIDATION_ERROR: 400,
    ErrorCode.BAD_REQUEST: 400,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.DEAL_NOT_FOUND: 404,
    ErrorCode.ACTION_NOT_FOUND: 404,
    ErrorCode.ARTIFACT_NOT_FOUND: 404,
    ErrorCode.QUARANTINE_ITEM_NOT_FOUND: 404,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.CONFLICT: 409,
    ErrorCode.ACTION_ALREADY_APPROVED: 409,
    ErrorCode.ACTION_ALREADY_REJECTED: 409,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.INVALID_STAGE_TRANSITION: 422,
    ErrorCode.APPROVAL_REQUIRED: 422,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.DATABASE_ERROR: 500,
    ErrorCode.EXTERNAL_SERVICE_ERROR: 502,
    ErrorCode.AGENT_ERROR: 500,
    ErrorCode.AGENT_TIMEOUT: 504,
    ErrorCode.TOOL_EXECUTION_ERROR: 500,
}


def get_status_code(error_code: ErrorCode) -> int:
    """
    Get HTTP status code for an error code.

    Args:
        error_code: The error code

    Returns:
        HTTP status code (defaults to 500 if not mapped)
    """
    return ERROR_STATUS_CODES.get(error_code, 500)


def is_client_error(error_code: ErrorCode) -> bool:
    """Check if the error code represents a client error (4xx)."""
    status = get_status_code(error_code)
    return 400 <= status < 500


def is_server_error(error_code: ErrorCode) -> bool:
    """Check if the error code represents a server error (5xx)."""
    status = get_status_code(error_code)
    return status >= 500
