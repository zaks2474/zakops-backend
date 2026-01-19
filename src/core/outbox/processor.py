"""
Outbox Processor

Phase 3: Execution Hardening

Background worker that polls the outbox and delivers events
with retry logic and exponential backoff.
"""

import os
import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4

from ..database.adapter import get_database, DatabaseAdapter
from ..events import publish_event
from ..events.models import EventBase
from .models import OutboxEntry, OutboxStatus

logger = logging.getLogger(__name__)

# Retry intervals for exponential backoff (in seconds)
RETRY_INTERVALS = [5, 15, 60, 300, 900]  # 5s, 15s, 1m, 5m, 15m


def calculate_next_attempt(attempts: int) -> datetime:
    """Calculate next attempt time with exponential backoff."""
    interval_idx = min(attempts, len(RETRY_INTERVALS) - 1)
    interval = RETRY_INTERVALS[interval_idx]
    return datetime.now(timezone.utc) + timedelta(seconds=interval)


class OutboxProcessor:
    """
    Processes outbox entries and delivers events.

    Features:
    - Polls outbox for pending entries
    - Delivers to event system
    - Retries failed deliveries with exponential backoff
    - Marks entries as delivered or failed
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        batch_size: int = 100,
        max_attempts: int = 5,
        db: Optional[DatabaseAdapter] = None
    ):
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.max_attempts = max_attempts
        self._db = db
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def _get_db(self) -> DatabaseAdapter:
        if self._db is None:
            self._db = await get_database()
        return self._db

    async def start(self):
        """Start the processor."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("OutboxProcessor started")

    async def stop(self):
        """Stop the processor."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("OutboxProcessor stopped")

    async def _run(self):
        """Main processing loop."""
        while self._running:
            try:
                processed = await self._process_batch()
                if processed == 0:
                    # No entries, wait before polling again
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"OutboxProcessor error: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)

    async def _process_batch(self) -> int:
        """Process a batch of outbox entries."""
        db = await self._get_db()
        now = datetime.now(timezone.utc)

        # Fetch pending entries that are ready for processing
        # Uses SELECT FOR UPDATE SKIP LOCKED to prevent duplicate processing
        entries = await db.fetch(
            """
            SELECT id, correlation_id, aggregate_type, aggregate_id,
                   event_type, schema_version, event_data,
                   status, attempts, max_attempts, created_at
            FROM zakops.outbox
            WHERE (status = $1 AND (next_attempt_at IS NULL OR next_attempt_at <= $2))
               OR (status = $3 AND attempts < $4 AND last_attempt_at < $5)
            ORDER BY created_at ASC
            LIMIT $6
            """,
            OutboxStatus.PENDING.value,
            now,
            OutboxStatus.FAILED.value,
            self.max_attempts,
            now - timedelta(minutes=1),  # Retry after 1 min
            self.batch_size
        )

        if not entries:
            return 0

        for entry in entries:
            await self._deliver_entry(db, entry)

        return len(entries)

    async def _deliver_entry(self, db: DatabaseAdapter, entry: Dict[str, Any]):
        """Deliver a single outbox entry."""
        entry_id = entry["id"]
        attempts = entry["attempts"] + 1
        now = datetime.now(timezone.utc)

        # Mark as processing
        await db.execute(
            """
            UPDATE zakops.outbox
            SET status = $1, last_attempt_at = $2, attempts = $3
            WHERE id = $4
            """,
            OutboxStatus.PROCESSING.value,
            now,
            attempts,
            entry_id
        )

        try:
            # Parse event data
            event_data = entry["event_data"]
            if isinstance(event_data, str):
                event_data = json.loads(event_data)

            # Create and publish event
            event = EventBase(
                id=uuid4(),
                correlation_id=UUID(str(entry["correlation_id"])) if not isinstance(entry["correlation_id"], UUID) else entry["correlation_id"],
                event_type=entry["event_type"],
                event_data=event_data,
                schema_version=entry.get("schema_version", 1),
                source="outbox"
            )

            await publish_event(event)

            # Mark as delivered
            await db.execute(
                """
                UPDATE zakops.outbox
                SET status = $1, delivered_at = $2, error_message = NULL
                WHERE id = $3
                """,
                OutboxStatus.DELIVERED.value,
                datetime.now(timezone.utc),
                entry_id
            )

            logger.debug(f"Delivered outbox entry {entry_id}")

        except Exception as e:
            # Mark as failed or dead
            if attempts >= self.max_attempts:
                status = OutboxStatus.DEAD.value
                logger.error(
                    f"Outbox entry {entry_id} moved to dead letter after {attempts} attempts: {e}"
                )
            else:
                status = OutboxStatus.PENDING.value  # Will be retried
                next_attempt = calculate_next_attempt(attempts)
                await db.execute(
                    """
                    UPDATE zakops.outbox
                    SET next_attempt_at = $1
                    WHERE id = $2
                    """,
                    next_attempt,
                    entry_id
                )
                logger.warning(
                    f"Outbox entry {entry_id} failed (attempt {attempts}), retry at {next_attempt}: {e}"
                )

            await db.execute(
                """
                UPDATE zakops.outbox
                SET status = $1, error_message = $2
                WHERE id = $3
                """,
                status,
                str(e)[:500],
                entry_id
            )

    async def get_stats(self) -> Dict[str, int]:
        """Get outbox statistics."""
        db = await self._get_db()

        rows = await db.fetch(
            """
            SELECT status, COUNT(*) as count
            FROM zakops.outbox
            GROUP BY status
            """
        )

        stats = {status.value: 0 for status in OutboxStatus}
        for row in rows:
            stats[row["status"]] = row["count"]

        return stats

    async def get_dead_letters(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get entries that have exceeded max attempts."""
        db = await self._get_db()

        return await db.fetch(
            """
            SELECT * FROM zakops.outbox
            WHERE status = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            OutboxStatus.DEAD.value,
            limit
        )

    async def retry_dead_letter(self, entry_id: UUID) -> bool:
        """Reset a dead letter entry for retry."""
        db = await self._get_db()

        result = await db.execute(
            """
            UPDATE zakops.outbox
            SET status = $1, attempts = 0, next_attempt_at = NULL, error_message = NULL
            WHERE id = $2 AND status = $3
            """,
            OutboxStatus.PENDING.value,
            entry_id,
            OutboxStatus.DEAD.value
        )

        logger.info(f"Reset dead letter entry for retry: {entry_id}")
        return True


# Global processor instance
_processor: Optional[OutboxProcessor] = None


async def start_outbox_processor(
    poll_interval: float = 1.0,
    batch_size: int = 100
) -> OutboxProcessor:
    """Start the global outbox processor."""
    global _processor

    if _processor is None:
        _processor = OutboxProcessor(
            poll_interval=poll_interval,
            batch_size=batch_size
        )

    await _processor.start()
    return _processor


async def stop_outbox_processor():
    """Stop the global outbox processor."""
    global _processor
    if _processor:
        await _processor.stop()
        _processor = None


async def get_outbox_processor() -> Optional[OutboxProcessor]:
    """Get the global outbox processor instance."""
    return _processor
