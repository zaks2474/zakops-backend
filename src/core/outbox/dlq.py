"""
Dead Letter Queue (DLQ) Management

Handles failed outbox entries that exceed max retry attempts.

Phase 13: Production Hardening
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID
from dataclasses import dataclass
from enum import Enum

from ..database.adapter import get_database
from .models import OutboxStatus

logger = logging.getLogger(__name__)


class DLQAction(str, Enum):
    """Actions that can be taken on DLQ entries."""
    RETRY = "retry"
    PURGE = "purge"
    ARCHIVE = "archive"


@dataclass
class DLQEntry:
    """A dead letter queue entry."""
    id: UUID
    correlation_id: str
    event_type: str
    event_data: Dict[str, Any]
    attempts: int
    max_attempts: int
    last_error: Optional[str]
    created_at: datetime
    failed_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "correlation_id": str(self.correlation_id),
            "event_type": self.event_type,
            "event_data": self.event_data,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "failed_at": self.failed_at.isoformat() if self.failed_at else None
        }


class DLQManager:
    """
    Manages the Dead Letter Queue.

    Responsibilities:
    - Query DLQ entries
    - Retry failed entries
    - Purge old entries
    - Generate DLQ reports
    """

    async def get_entries(
        self,
        limit: int = 100,
        offset: int = 0,
        correlation_id: Optional[str] = None
    ) -> List[DLQEntry]:
        """Get DLQ entries."""
        db = await get_database()

        if correlation_id:
            rows = await db.fetch(
                """
                SELECT id, correlation_id, event_type, event_data, attempts,
                       max_attempts, error_message as last_error, created_at,
                       COALESCE(last_attempt_at, created_at) as failed_at
                FROM zakops.outbox
                WHERE status = $1 AND correlation_id = $2
                ORDER BY created_at DESC
                LIMIT $3 OFFSET $4
                """,
                OutboxStatus.DEAD.value, correlation_id, limit, offset
            )
        else:
            rows = await db.fetch(
                """
                SELECT id, correlation_id, event_type, event_data, attempts,
                       max_attempts, error_message as last_error, created_at,
                       COALESCE(last_attempt_at, created_at) as failed_at
                FROM zakops.outbox
                WHERE status = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                OutboxStatus.DEAD.value, limit, offset
            )

        return [
            DLQEntry(
                id=row["id"],
                correlation_id=str(row["correlation_id"]) if row["correlation_id"] else "",
                event_type=row["event_type"],
                event_data=row["event_data"] if isinstance(row["event_data"], dict) else {},
                attempts=row["attempts"],
                max_attempts=row.get("max_attempts", 5),
                last_error=row.get("last_error"),
                created_at=row["created_at"],
                failed_at=row["failed_at"]
            )
            for row in rows
        ]

    async def get_count(self, correlation_id: Optional[str] = None) -> int:
        """Get total DLQ entry count."""
        db = await get_database()

        if correlation_id:
            result = await db.fetchrow(
                "SELECT COUNT(*) as count FROM zakops.outbox WHERE status = $1 AND correlation_id = $2",
                OutboxStatus.DEAD.value, correlation_id
            )
        else:
            result = await db.fetchrow(
                "SELECT COUNT(*) as count FROM zakops.outbox WHERE status = $1",
                OutboxStatus.DEAD.value
            )

        return result["count"] if result else 0

    async def retry_entry(self, entry_id: UUID, operator_id: Optional[str] = None) -> bool:
        """
        Retry a DLQ entry by resetting its status.

        Args:
            entry_id: The outbox entry ID
            operator_id: ID of operator performing the action

        Returns:
            True if entry was reset for retry
        """
        db = await get_database()

        # Reset to pending with attempts reset
        result = await db.execute(
            """
            UPDATE zakops.outbox
            SET status = $1,
                attempts = 0,
                error_message = NULL,
                next_attempt_at = NULL,
                last_attempt_at = NOW()
            WHERE id = $2 AND status = $3
            RETURNING id
            """,
            OutboxStatus.PENDING.value, entry_id, OutboxStatus.DEAD.value
        )

        success = result is not None and "UPDATE" in str(result)

        if success:
            logger.info(f"DLQ entry {entry_id} reset for retry by {operator_id}")
            await self._log_action(entry_id, DLQAction.RETRY, operator_id)

        return success

    async def retry_all(
        self,
        correlation_id: Optional[str] = None,
        operator_id: Optional[str] = None
    ) -> int:
        """Retry all DLQ entries (optionally filtered by correlation_id)."""
        db = await get_database()

        # Get count first
        count = await self.get_count(correlation_id)

        if correlation_id:
            await db.execute(
                """
                UPDATE zakops.outbox
                SET status = $1, attempts = 0, error_message = NULL,
                    next_attempt_at = NULL, last_attempt_at = NOW()
                WHERE status = $2 AND correlation_id = $3
                """,
                OutboxStatus.PENDING.value, OutboxStatus.DEAD.value, correlation_id
            )
        else:
            await db.execute(
                """
                UPDATE zakops.outbox
                SET status = $1, attempts = 0, error_message = NULL,
                    next_attempt_at = NULL, last_attempt_at = NOW()
                WHERE status = $2
                """,
                OutboxStatus.PENDING.value, OutboxStatus.DEAD.value
            )

        logger.info(f"DLQ retry all: reset {count} entries by {operator_id}")

        return count

    async def purge_entry(self, entry_id: UUID, operator_id: Optional[str] = None) -> bool:
        """
        Permanently delete a DLQ entry.

        Args:
            entry_id: The outbox entry ID
            operator_id: ID of operator performing the action

        Returns:
            True if entry was deleted
        """
        db = await get_database()

        # Log before delete
        await self._log_action(entry_id, DLQAction.PURGE, operator_id)

        result = await db.execute(
            "DELETE FROM zakops.outbox WHERE id = $1 AND status = $2",
            entry_id, OutboxStatus.DEAD.value
        )

        success = result is not None and "DELETE" in str(result)
        logger.info(f"DLQ entry {entry_id} purged by {operator_id}")

        return success

    async def purge_old(self, days: int = 30, operator_id: Optional[str] = None) -> int:
        """Purge DLQ entries older than specified days."""
        db = await get_database()

        # Get count first
        result = await db.fetchrow(
            """
            SELECT COUNT(*) as count FROM zakops.outbox
            WHERE status = $1
            AND created_at < NOW() - INTERVAL '%s days'
            """ % days,
            OutboxStatus.DEAD.value
        )
        count = result["count"] if result else 0

        await db.execute(
            """
            DELETE FROM zakops.outbox
            WHERE status = $1
            AND created_at < NOW() - INTERVAL '%s days'
            """ % days,
            OutboxStatus.DEAD.value
        )

        logger.info(f"DLQ purged {count} entries older than {days} days by {operator_id}")

        return count

    async def get_stats(self) -> Dict[str, Any]:
        """Get DLQ statistics."""
        db = await get_database()

        total = await db.fetchrow(
            "SELECT COUNT(*) as count FROM zakops.outbox WHERE status = $1",
            OutboxStatus.DEAD.value
        )

        by_type = await db.fetch(
            """
            SELECT event_type, COUNT(*) as count
            FROM zakops.outbox
            WHERE status = $1
            GROUP BY event_type
            ORDER BY count DESC
            """,
            OutboxStatus.DEAD.value
        )

        oldest = await db.fetchrow(
            """
            SELECT MIN(created_at) as oldest
            FROM zakops.outbox
            WHERE status = $1
            """,
            OutboxStatus.DEAD.value
        )

        return {
            "total_count": total["count"] if total else 0,
            "by_event_type": {row["event_type"]: row["count"] for row in by_type},
            "oldest_entry": oldest["oldest"].isoformat() if oldest and oldest["oldest"] else None
        }

    async def _log_action(self, entry_id: UUID, action: DLQAction, operator_id: Optional[str]):
        """Log DLQ action for audit."""
        # This could write to an audit table
        logger.info(f"DLQ action: {action.value} on {entry_id} by {operator_id}")


# Convenience function
async def get_dlq_manager() -> DLQManager:
    """Get DLQ manager instance."""
    return DLQManager()
