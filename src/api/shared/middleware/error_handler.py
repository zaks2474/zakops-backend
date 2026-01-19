"""
Global Error Handler Middleware

Phase 5: API Stabilization

Catches exceptions and returns standardized error responses.
"""

import logging
import traceback
from uuid import uuid4

from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError

from ..exceptions import APIException
from ..responses import ErrorResponse, ErrorBody, ErrorDetail
from ..error_codes import ErrorCode

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI):
    """
    Register all error handlers on the FastAPI app.

    This function sets up exception handlers for:
    - APIException (custom API errors)
    - RequestValidationError (FastAPI validation)
    - Generic Exception (catch-all for unexpected errors)
    """

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        """Handle custom API exceptions."""
        trace_id = exc.trace_id or str(uuid4())

        logger.warning(
            f"API Error: {exc.code} - {exc.message}",
            extra={
                "trace_id": trace_id,
                "error_code": str(exc.code),
                "path": request.url.path
            }
        )

        error_body = ErrorBody(
            code=exc.code.value if isinstance(exc.code, ErrorCode) else str(exc.code),
            message=exc.message,
            details=exc.details,
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={"error": error_body.model_dump(mode="json")}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle FastAPI validation errors."""
        trace_id = str(uuid4())

        # Convert Pydantic errors to our format
        details = []
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            details.append(ErrorDetail(
                field=field,
                message=error["msg"],
                code=error["type"]
            ))

        logger.warning(
            f"Validation Error: {len(details)} field(s)",
            extra={
                "trace_id": trace_id,
                "path": request.url.path,
                "errors": [d.model_dump() for d in details]
            }
        )

        error_body = ErrorBody(
            code=ErrorCode.VALIDATION_ERROR.value,
            message="Request validation failed",
            details=details,
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=400,
            content={"error": error_body.model_dump(mode="json")}
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions."""
        trace_id = str(uuid4())

        logger.error(
            f"Unhandled Exception: {type(exc).__name__}: {exc}",
            extra={
                "trace_id": trace_id,
                "path": request.url.path,
                "traceback": traceback.format_exc()
            }
        )

        # Don't expose internal details in production
        error_body = ErrorBody(
            code=ErrorCode.INTERNAL_ERROR.value,
            message="An internal error occurred",
            trace_id=trace_id
        )

        return JSONResponse(
            status_code=500,
            content={"error": error_body.model_dump(mode="json")}
        )
