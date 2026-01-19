"""
Outbox Models

Phase 3: Execution Hardening
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class OutboxStatus(str, Enum):
    """Status of an outbox entry."""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    DEAD = "dead"  # Exceeded max attempts


class OutboxEntry(BaseModel):
    """An entry in the outbox table."""

    id: UUID = Field(default_factory=uuid4)
    correlation_id: UUID
    aggregate_type: str = "event"
    aggregate_id: str = ""
    event_type: str
    schema_version: int = 1
    event_data: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[UUID] = None

    status: OutboxStatus = OutboxStatus.PENDING
    attempts: int = 0
    max_attempts: int = 5

    last_attempt_at: Optional[datetime] = None
    next_attempt_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None

    created_at: datetime = Field(default_factory=_utcnow)

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
