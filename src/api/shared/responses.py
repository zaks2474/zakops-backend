"""
Standard API Response Models

Phase 5: API Stabilization
Spec Reference: Master Architecture Specification §4

Provides consistent response shapes across all endpoints.
"""

from datetime import datetime, timezone
from typing import TypeVar, Generic, Optional, List, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field


T = TypeVar('T')


def _utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class ResponseMeta(BaseModel):
    """Metadata included in all responses."""

    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=_utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat().replace("+00:00", "Z")
        }


class SuccessResponse(BaseModel, Generic[T]):
    """
    Standard success response wrapper.

    Response shape:
    {
        "data": { ... },
        "meta": {
            "trace_id": "abc-123",
            "correlation_id": "deal-456",
            "timestamp": "2026-01-19T12:00:00Z"
        }
    }
    """

    data: T
    meta: ResponseMeta = Field(default_factory=ResponseMeta)

    @classmethod
    def create(
        cls,
        data: T,
        correlation_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> "SuccessResponse[T]":
        meta = ResponseMeta(
            trace_id=trace_id or str(uuid4()),
            correlation_id=correlation_id
        )
        return cls(data=data, meta=meta)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat().replace("+00:00", "Z")
        }


class ListMeta(ResponseMeta):
    """Metadata for list responses with pagination."""

    total: int = 0
    limit: int = 20
    offset: int = 0
    has_more: bool = False


class ListResponse(BaseModel, Generic[T]):
    """
    Standard list response with pagination.

    Response shape:
    {
        "data": [ ... ],
        "meta": {
            "total": 100,
            "limit": 20,
            "offset": 0,
            "has_more": true,
            "trace_id": "abc-123"
        }
    }
    """

    data: List[T]
    meta: ListMeta

    @classmethod
    def create(
        cls,
        data: List[T],
        total: int,
        limit: int = 20,
        offset: int = 0,
        correlation_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> "ListResponse[T]":
        meta = ListMeta(
            trace_id=trace_id or str(uuid4()),
            correlation_id=correlation_id,
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + len(data)) < total
        )
        return cls(data=data, meta=meta)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat().replace("+00:00", "Z")
        }


class ErrorDetail(BaseModel):
    """Detailed error information for validation errors."""

    field: Optional[str] = None
    message: str
    code: Optional[str] = None


class ErrorBody(BaseModel):
    """Error body with code, message, and details."""

    code: str
    message: str
    details: Optional[List[ErrorDetail]] = None
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=_utcnow)

    @classmethod
    def validation_error(
        cls,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        trace_id: Optional[str] = None
    ) -> "ErrorBody":
        return cls(
            code="VALIDATION_ERROR",
            message=message,
            details=details,
            trace_id=trace_id or str(uuid4())
        )

    @classmethod
    def not_found(
        cls,
        resource: str,
        resource_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> "ErrorBody":
        message = f"{resource} not found"
        if resource_id:
            message = f"{resource} with ID '{resource_id}' not found"
        return cls(
            code="NOT_FOUND",
            message=message,
            trace_id=trace_id or str(uuid4())
        )

    @classmethod
    def internal_error(
        cls,
        message: str = "An internal error occurred",
        trace_id: Optional[str] = None
    ) -> "ErrorBody":
        return cls(
            code="INTERNAL_ERROR",
            message=message,
            trace_id=trace_id or str(uuid4())
        )

    @classmethod
    def unauthorized(
        cls,
        message: str = "Authentication required",
        trace_id: Optional[str] = None
    ) -> "ErrorBody":
        return cls(
            code="UNAUTHORIZED",
            message=message,
            trace_id=trace_id or str(uuid4())
        )

    @classmethod
    def forbidden(
        cls,
        message: str = "Permission denied",
        trace_id: Optional[str] = None
    ) -> "ErrorBody":
        return cls(
            code="FORBIDDEN",
            message=message,
            trace_id=trace_id or str(uuid4())
        )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat().replace("+00:00", "Z")
        }


class ErrorResponse(BaseModel):
    """
    Standard error response.

    Response shape:
    {
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Human-readable error message",
            "details": [...],
            "trace_id": "abc-123",
            "timestamp": "2026-01-19T12:00:00Z"
        }
    }
    """

    error: ErrorBody

    @classmethod
    def create(
        cls,
        code: str,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        trace_id: Optional[str] = None
    ) -> "ErrorResponse":
        return cls(
            error=ErrorBody(
                code=code,
                message=message,
                details=details,
                trace_id=trace_id or str(uuid4())
            )
        )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat().replace("+00:00", "Z")
        }
