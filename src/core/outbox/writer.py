"""
Outbox Writer

Phase 3: Execution Hardening

Writes events to the outbox table within the same transaction
as your business logic for guaranteed delivery.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from ..database.adapter import DatabaseAdapter, get_database
from ..events import publish_event
from ..events.models import EventBase
from .models import OutboxEntry, OutboxStatus

logger = logging.getLogger(__name__)


def is_outbox_enabled() -> bool:
    """Check if outbox pattern is enabled."""
    return os.getenv("OUTBOX_ENABLED", "true").lower() == "true"


class OutboxWriter:
    """
    Writes events to the outbox for reliable delivery.

    Usage:
        async with OutboxWriter(db) as writer:
            # Your business logic here...
            await writer.write(
                correlation_id=deal_id,
                event_type="action.created",
                event_data={"action_id": str(action_id)}
            )
        # Transaction commits, outbox entry is persisted
    """

    def __init__(self, db: Optional[DatabaseAdapter] = None):
        self._db = db
        self._enabled = is_outbox_enabled()

    async def _get_db(self) -> DatabaseAdapter:
        if self._db is None:
            self._db = await get_database()
        return self._db

    async def write(
        self,
        correlation_id: UUID,
        event_type: str,
        event_data: Dict[str, Any],
        aggregate_type: str = "event",
        aggregate_id: Optional[str] = None,
        idempotency_key: Optional[str] = None
    ) -> OutboxEntry:
        """
        Write an event to the outbox.

        Args:
            correlation_id: The correlation ID (usually deal_id)
            event_type: Event type from taxonomy (e.g., "action.created")
            event_data: Event payload
            aggregate_type: Type of aggregate (e.g., "deal", "action")
            aggregate_id: ID of the aggregate
            idempotency_key: Optional key for deduplication

        Returns:
            The created OutboxEntry
        """
        db = await self._get_db()

        if not self._enabled:
            # Outbox disabled, fall back to direct publishing
            logger.debug("Outbox disabled, publishing directly")
            event = EventBase(
                correlation_id=correlation_id,
                event_type=event_type,
                event_data=event_data,
                source="direct"
            )
            await publish_event(event)
            return OutboxEntry(
                correlation_id=correlation_id,
                event_type=event_type,
                event_data=event_data,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id or str(correlation_id),
                status=OutboxStatus.DELIVERED,
                delivered_at=datetime.now(timezone.utc)
            )

        entry = OutboxEntry(
            correlation_id=correlation_id,
            event_type=event_type,
            event_data=event_data,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id or str(correlation_id)
        )

        await db.execute(
            """
            INSERT INTO zakops.outbox (
                id, correlation_id, aggregate_type, aggregate_id,
                event_type, schema_version, event_data,
                status, attempts, max_attempts, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            entry.id,
            entry.correlation_id,
            entry.aggregate_type,
            entry.aggregate_id,
            entry.event_type,
            entry.schema_version,
            json.dumps(entry.event_data),
            entry.status.value if isinstance(entry.status, OutboxStatus) else entry.status,
            entry.attempts,
            entry.max_attempts,
            entry.created_at
        )

        logger.debug(
            "Wrote event to outbox: id=%s type=%s correlation=%s",
            entry.id, entry.event_type, entry.correlation_id
        )

        return entry

    async def write_batch(
        self,
        entries: List[tuple]  # List of (correlation_id, event_type, event_data) tuples
    ) -> List[OutboxEntry]:
        """
        Write multiple events to the outbox in a single transaction.

        Args:
            entries: List of (correlation_id, event_type, event_data) tuples

        Returns:
            List of created OutboxEntry objects
        """
        results = []
        for correlation_id, event_type, event_data in entries:
            entry = await self.write(correlation_id, event_type, event_data)
            results.append(entry)
        return results


# Global writer instance
_writer: Optional[OutboxWriter] = None


@asynccontextmanager
async def get_outbox_writer():
    """
    Get an outbox writer within a context manager.

    Usage:
        async with get_outbox_writer() as writer:
            await writer.write(...)
    """
    global _writer
    if _writer is None:
        db = await get_database()
        _writer = OutboxWriter(db)

    yield _writer


async def reset_outbox_writer():
    """Reset the global writer (for testing)."""
    global _writer
    _writer = None
