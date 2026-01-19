"""
Transactional Event Publisher

Phase 3: Execution Hardening

Combines business operations with event publishing in a single transaction
to guarantee atomicity: either both succeed or both fail.
"""

from typing import Optional, Dict, Any, List
from uuid import UUID
from contextlib import asynccontextmanager

from ..database.adapter import get_database, DatabaseAdapter
from .writer import OutboxWriter
from .models import OutboxEntry


class TransactionalPublisher:
    """
    Publishes events transactionally with business operations.

    This ensures that events are written to the outbox in the same
    transaction as your business data changes.

    Usage:
        async with TransactionalPublisher() as txn:
            # Your business logic
            await txn.db.execute("INSERT INTO actions ...")

            # Emit event (same transaction)
            await txn.emit(
                correlation_id=deal_id,
                event_type="action.created",
                event_data={"action_id": str(action.id)}
            )
        # Both commit together or both rollback
    """

    def __init__(self):
        self.db: Optional[DatabaseAdapter] = None
        self._writer: Optional[OutboxWriter] = None
        self._events: List[OutboxEntry] = []

    async def __aenter__(self):
        self.db = await get_database()
        self._writer = OutboxWriter(self.db)
        self._events = []
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Success - events already written to outbox in same transaction
            pass
        else:
            # Failure - transaction will rollback, outbox entries too
            self._events = []
        return False

    async def emit(
        self,
        correlation_id: UUID,
        event_type: str,
        event_data: Dict[str, Any],
        aggregate_type: str = "event",
        aggregate_id: Optional[str] = None
    ) -> OutboxEntry:
        """
        Emit an event (writes to outbox in current transaction).

        Args:
            correlation_id: The correlation ID (usually deal_id)
            event_type: Event type from taxonomy
            event_data: Event payload
            aggregate_type: Type of aggregate
            aggregate_id: ID of the aggregate

        Returns:
            The created OutboxEntry
        """
        entry = await self._writer.write(
            correlation_id=correlation_id,
            event_type=event_type,
            event_data=event_data,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id
        )
        self._events.append(entry)
        return entry

    async def emit_batch(
        self,
        events: List[tuple]  # List of (correlation_id, event_type, event_data)
    ) -> List[OutboxEntry]:
        """
        Emit multiple events in the same transaction.

        Args:
            events: List of (correlation_id, event_type, event_data) tuples

        Returns:
            List of created OutboxEntry objects
        """
        entries = []
        for correlation_id, event_type, event_data in events:
            entry = await self.emit(correlation_id, event_type, event_data)
            entries.append(entry)
        return entries

    @property
    def emitted_events(self) -> List[OutboxEntry]:
        """Get list of events emitted in this transaction."""
        return self._events.copy()


@asynccontextmanager
async def transactional_publish():
    """
    Context manager for transactional event publishing.

    Usage:
        async with transactional_publish() as txn:
            await txn.db.execute("INSERT INTO deals ...")
            await txn.emit(deal_id, "deal.created", {...})
    """
    publisher = TransactionalPublisher()
    async with publisher:
        yield publisher
