"""
Inbox Guard

Phase 3: Execution Hardening

Provides exactly-once processing guarantees for event consumers
via the inbox deduplication table.
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from ..database.adapter import get_database, DatabaseAdapter

logger = logging.getLogger(__name__)


class InboxGuard:
    """
    Guards against duplicate event processing.

    Uses the inbox table to track which events have been processed
    by which consumers, ensuring exactly-once semantics.

    Usage:
        async with InboxGuard(event_id, "my-consumer") as guard:
            if guard.should_process:
                await process_event(event)
            else:
                logger.info("Event already processed, skipping")

    If processing fails (exception raised), the inbox entry is removed
    to allow retry.
    """

    def __init__(self, event_id: UUID, consumer_id: str):
        self.event_id = event_id
        self.consumer_id = consumer_id
        self.should_process = False
        self._db: Optional[DatabaseAdapter] = None

    async def __aenter__(self):
        self._db = await get_database()

        # Try to insert into inbox (will fail if duplicate due to UNIQUE constraint)
        try:
            await self._db.execute(
                """
                INSERT INTO zakops.inbox (event_id, consumer_id, processed_at)
                VALUES ($1, $2, $3)
                """,
                self.event_id,
                self.consumer_id,
                datetime.now(timezone.utc)
            )
            self.should_process = True
            logger.debug(
                f"InboxGuard: event {self.event_id} marked for processing by {self.consumer_id}"
            )
        except Exception as e:
            # Already processed (unique constraint violation)
            self.should_process = False
            logger.debug(
                f"InboxGuard: event {self.event_id} already processed by {self.consumer_id}"
            )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.should_process:
            # Processing failed, remove from inbox to allow retry
            try:
                await self._db.execute(
                    """
                    DELETE FROM zakops.inbox
                    WHERE event_id = $1 AND consumer_id = $2
                    """,
                    self.event_id,
                    self.consumer_id
                )
                logger.warning(
                    f"InboxGuard: removed entry for failed processing of event {self.event_id}"
                )
            except Exception as delete_error:
                logger.error(
                    f"InboxGuard: failed to remove inbox entry after error: {delete_error}"
                )
        return False


async def is_processed(event_id: UUID, consumer_id: str) -> bool:
    """
    Check if an event has been processed by a consumer.

    Args:
        event_id: The event ID to check
        consumer_id: The consumer identifier

    Returns:
        True if already processed, False otherwise
    """
    db = await get_database()

    result = await db.fetchrow(
        """
        SELECT 1 FROM zakops.inbox
        WHERE event_id = $1 AND consumer_id = $2
        """,
        event_id,
        consumer_id
    )

    return result is not None


async def mark_processed(event_id: UUID, consumer_id: str) -> bool:
    """
    Mark an event as processed.

    Args:
        event_id: The event ID to mark
        consumer_id: The consumer identifier

    Returns:
        True if marked successfully, False if already processed
    """
    db = await get_database()

    try:
        await db.execute(
            """
            INSERT INTO zakops.inbox (event_id, consumer_id, processed_at)
            VALUES ($1, $2, $3)
            """,
            event_id,
            consumer_id,
            datetime.now(timezone.utc)
        )
        return True
    except Exception:
        # Already processed (unique constraint violation)
        return False


async def remove_processed(event_id: UUID, consumer_id: str) -> bool:
    """
    Remove a processed event record (for retry scenarios).

    Args:
        event_id: The event ID to remove
        consumer_id: The consumer identifier

    Returns:
        True if removed, False if not found
    """
    db = await get_database()

    await db.execute(
        """
        DELETE FROM zakops.inbox
        WHERE event_id = $1 AND consumer_id = $2
        """,
        event_id,
        consumer_id
    )

    return True
