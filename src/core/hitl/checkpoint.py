"""
Checkpoint Store

Phase 6: HITL & Checkpoints
Spec Reference: Durable Execution section

Provides durable execution checkpoints for:
- Resumable workflows (restart from last checkpoint)
- State persistence across failures
- Long-running operation recovery

Uses the zakops.execution_checkpoints table from Phase 1.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..database.adapter import get_database, DatabaseAdapter

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class CheckpointStatus(str, Enum):
    """Status of a checkpoint."""
    ACTIVE = "active"
    RESUMED = "resumed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class CheckpointType(str, Enum):
    """Type of checkpoint."""
    STATE = "state"  # General state checkpoint
    APPROVAL_GATE = "approval_gate"  # Waiting for approval
    EXTERNAL_CALL = "external_call"  # Waiting for external response
    TIMER = "timer"  # Scheduled continuation
    ERROR_RECOVERY = "error_recovery"  # After error for retry


@dataclass
class Checkpoint:
    """A durable execution checkpoint."""
    checkpoint_id: str
    correlation_id: str
    action_id: Optional[str] = None
    run_id: Optional[str] = None

    # Checkpoint info
    checkpoint_name: str = ""
    checkpoint_type: CheckpointType = CheckpointType.STATE
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)

    # Sequence
    sequence_number: int = 0

    # Status
    status: CheckpointStatus = CheckpointStatus.ACTIVE
    expires_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None
    resumed_by: Optional[str] = None

    # Timestamps
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "correlation_id": self.correlation_id,
            "action_id": self.action_id,
            "run_id": self.run_id,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_type": self.checkpoint_type.value,
            "checkpoint_data": self.checkpoint_data,
            "sequence_number": self.sequence_number,
            "status": self.status.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "resumed_at": self.resumed_at.isoformat() if self.resumed_at else None,
            "resumed_by": self.resumed_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class CheckpointStore:
    """
    Durable checkpoint store for execution state.

    Uses PostgreSQL (zakops.execution_checkpoints) for durability.
    Enables:
    - Resumable workflows
    - State persistence across failures
    - Long-running operation recovery

    Checkpoint Types:
    - state: General state checkpoint
    - approval_gate: Waiting for human approval
    - external_call: Waiting for external response
    - timer: Scheduled continuation
    - error_recovery: Checkpoint before retry

    Usage:
        store = CheckpointStore()

        # Save checkpoint
        checkpoint = await store.save_checkpoint(
            correlation_id=deal_id,
            action_id=action_id,
            checkpoint_name="after_document_generation",
            checkpoint_data={"document_id": doc.id, "step": 3}
        )

        # Resume from checkpoint
        checkpoint = await store.get_latest_checkpoint(action_id=action_id)
        if checkpoint:
            step = checkpoint.checkpoint_data.get("step", 0)
            # Resume from step...

        # Mark as resumed
        await store.mark_resumed(checkpoint.checkpoint_id, resumed_by="operator@example.com")
    """

    def __init__(self, db: Optional[DatabaseAdapter] = None):
        self._db = db

    async def _get_db(self) -> DatabaseAdapter:
        if self._db is None:
            self._db = await get_database()
        return self._db

    async def save_checkpoint(
        self,
        correlation_id: str,
        checkpoint_name: str,
        checkpoint_data: Dict[str, Any],
        *,
        action_id: Optional[str] = None,
        run_id: Optional[str] = None,
        checkpoint_type: CheckpointType = CheckpointType.STATE,
        expires_in_hours: Optional[int] = None,
    ) -> Checkpoint:
        """
        Save a durable checkpoint.

        Args:
            correlation_id: Correlation ID (usually deal_id)
            checkpoint_name: Name of the checkpoint (e.g., "after_step_3")
            checkpoint_data: State to persist
            action_id: Associated action ID
            run_id: Associated agent run ID
            checkpoint_type: Type of checkpoint
            expires_in_hours: Optional expiration time

        Returns:
            The saved Checkpoint
        """
        db = await self._get_db()
        now = _utcnow()

        # Get next sequence number for this action
        sequence_number = 0
        if action_id:
            row = await db.fetchrow(
                """
                SELECT COALESCE(MAX(sequence_number), -1) + 1 as next_seq
                FROM zakops.execution_checkpoints
                WHERE action_id = $1
                """,
                action_id,
            )
            if row:
                sequence_number = row["next_seq"]

        checkpoint_id = str(uuid4())
        expires_at = None
        if expires_in_hours:
            expires_at = now + timedelta(hours=expires_in_hours)

        # Insert checkpoint
        await db.execute(
            """
            INSERT INTO zakops.execution_checkpoints (
                id, correlation_id, action_id, run_id,
                checkpoint_name, checkpoint_type, checkpoint_data,
                sequence_number, status, expires_at, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            """,
            checkpoint_id,
            correlation_id,
            action_id,
            str(run_id) if run_id else None,
            checkpoint_name,
            checkpoint_type.value,
            json.dumps(checkpoint_data),
            sequence_number,
            CheckpointStatus.ACTIVE.value,
            expires_at,
            now,
        )

        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            correlation_id=correlation_id,
            action_id=action_id,
            run_id=str(run_id) if run_id else None,
            checkpoint_name=checkpoint_name,
            checkpoint_type=checkpoint_type,
            checkpoint_data=checkpoint_data,
            sequence_number=sequence_number,
            status=CheckpointStatus.ACTIVE,
            expires_at=expires_at,
            created_at=now,
        )

        logger.info(
            "Saved checkpoint: id=%s action_id=%s name=%s seq=%d",
            checkpoint_id,
            action_id,
            checkpoint_name,
            sequence_number,
        )

        return checkpoint

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        db = await self._get_db()

        row = await db.fetchrow(
            "SELECT * FROM zakops.execution_checkpoints WHERE id = $1",
            checkpoint_id,
        )

        if not row:
            return None

        return self._row_to_checkpoint(row)

    async def get_latest_checkpoint(
        self,
        *,
        action_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
        active_only: bool = True,
    ) -> Optional[Checkpoint]:
        """
        Get the latest checkpoint for an action or correlation.

        Args:
            action_id: Filter by action ID
            correlation_id: Filter by correlation ID
            checkpoint_name: Filter by checkpoint name
            active_only: Only return active checkpoints

        Returns:
            The latest matching checkpoint, or None
        """
        db = await self._get_db()

        conditions = []
        params = []
        param_idx = 1

        if action_id:
            conditions.append(f"action_id = ${param_idx}")
            params.append(action_id)
            param_idx += 1
        if correlation_id:
            conditions.append(f"correlation_id = ${param_idx}")
            params.append(correlation_id)
            param_idx += 1
        if checkpoint_name:
            conditions.append(f"checkpoint_name = ${param_idx}")
            params.append(checkpoint_name)
            param_idx += 1
        if active_only:
            conditions.append(f"status = ${param_idx}")
            params.append(CheckpointStatus.ACTIVE.value)
            param_idx += 1

        if not conditions:
            return None

        query = f"""
            SELECT * FROM zakops.execution_checkpoints
            WHERE {" AND ".join(conditions)}
            ORDER BY sequence_number DESC, created_at DESC
            LIMIT 1
        """

        row = await db.fetchrow(query, *params)

        if not row:
            return None

        return self._row_to_checkpoint(row)

    async def get_checkpoints_for_action(
        self,
        action_id: str,
        *,
        include_resumed: bool = False,
        limit: int = 100,
    ) -> List[Checkpoint]:
        """Get all checkpoints for an action."""
        db = await self._get_db()

        if include_resumed:
            query = """
                SELECT * FROM zakops.execution_checkpoints
                WHERE action_id = $1
                ORDER BY sequence_number ASC
                LIMIT $2
            """
            rows = await db.fetch(query, action_id, limit)
        else:
            query = """
                SELECT * FROM zakops.execution_checkpoints
                WHERE action_id = $1 AND status = $2
                ORDER BY sequence_number ASC
                LIMIT $3
            """
            rows = await db.fetch(query, action_id, CheckpointStatus.ACTIVE.value, limit)

        return [self._row_to_checkpoint(row) for row in rows]

    async def mark_resumed(
        self,
        checkpoint_id: str,
        *,
        resumed_by: Optional[str] = None,
    ) -> bool:
        """
        Mark a checkpoint as resumed.

        Args:
            checkpoint_id: The checkpoint ID
            resumed_by: Who/what resumed execution

        Returns:
            True if updated, False if not found
        """
        db = await self._get_db()
        now = _utcnow()

        result = await db.execute(
            """
            UPDATE zakops.execution_checkpoints
            SET status = $1, resumed_at = $2, resumed_by = $3
            WHERE id = $4 AND status = $5
            """,
            CheckpointStatus.RESUMED.value,
            now,
            resumed_by,
            checkpoint_id,
            CheckpointStatus.ACTIVE.value,
        )

        # Check if row was updated
        updated = "UPDATE 1" in str(result) if result else False

        if updated:
            logger.info(
                "Checkpoint resumed: id=%s by=%s",
                checkpoint_id,
                resumed_by,
            )

        return updated

    async def cancel_checkpoint(
        self,
        checkpoint_id: str,
    ) -> bool:
        """
        Cancel a checkpoint (e.g., when action is cancelled).

        Returns True if updated, False if not found.
        """
        db = await self._get_db()

        result = await db.execute(
            """
            UPDATE zakops.execution_checkpoints
            SET status = $1
            WHERE id = $2 AND status = $3
            """,
            CheckpointStatus.CANCELLED.value,
            checkpoint_id,
            CheckpointStatus.ACTIVE.value,
        )

        return "UPDATE 1" in str(result) if result else False

    async def cancel_checkpoints_for_action(
        self,
        action_id: str,
    ) -> int:
        """
        Cancel all active checkpoints for an action.

        Returns the number of checkpoints cancelled.
        """
        db = await self._get_db()

        result = await db.execute(
            """
            UPDATE zakops.execution_checkpoints
            SET status = $1
            WHERE action_id = $2 AND status = $3
            """,
            CheckpointStatus.CANCELLED.value,
            action_id,
            CheckpointStatus.ACTIVE.value,
        )

        # Parse count from result
        if result and "UPDATE" in str(result):
            try:
                count = int(str(result).split()[1])
                logger.info("Cancelled %d checkpoints for action %s", count, action_id)
                return count
            except (ValueError, IndexError):
                pass

        return 0

    async def expire_old_checkpoints(
        self,
        *,
        batch_size: int = 100,
    ) -> int:
        """
        Expire checkpoints that have passed their expiration time.

        This should be called periodically by a background job.

        Returns the number of checkpoints expired.
        """
        db = await self._get_db()
        now = _utcnow()

        result = await db.execute(
            """
            UPDATE zakops.execution_checkpoints
            SET status = $1
            WHERE status = $2
              AND expires_at IS NOT NULL
              AND expires_at < $3
            """,
            CheckpointStatus.EXPIRED.value,
            CheckpointStatus.ACTIVE.value,
            now,
        )

        if result and "UPDATE" in str(result):
            try:
                count = int(str(result).split()[1])
                if count > 0:
                    logger.info("Expired %d checkpoints", count)
                return count
            except (ValueError, IndexError):
                pass

        return 0

    async def cleanup_old_checkpoints(
        self,
        *,
        older_than_days: int = 30,
        batch_size: int = 1000,
    ) -> int:
        """
        Delete old non-active checkpoints.

        Keeps active checkpoints regardless of age.

        Returns the number of checkpoints deleted.
        """
        db = await self._get_db()
        cutoff = _utcnow() - timedelta(days=older_than_days)

        result = await db.execute(
            """
            DELETE FROM zakops.execution_checkpoints
            WHERE status != $1
              AND created_at < $2
            """,
            CheckpointStatus.ACTIVE.value,
            cutoff,
        )

        if result and "DELETE" in str(result):
            try:
                count = int(str(result).split()[1])
                if count > 0:
                    logger.info("Cleaned up %d old checkpoints", count)
                return count
            except (ValueError, IndexError):
                pass

        return 0

    def _row_to_checkpoint(self, row: Dict[str, Any]) -> Checkpoint:
        """Convert a database row to a Checkpoint."""
        checkpoint_data = row.get("checkpoint_data") or {}
        if isinstance(checkpoint_data, str):
            try:
                checkpoint_data = json.loads(checkpoint_data)
            except (json.JSONDecodeError, TypeError):
                checkpoint_data = {}

        return Checkpoint(
            checkpoint_id=str(row.get("id", "")),
            correlation_id=str(row.get("correlation_id", "")),
            action_id=row.get("action_id"),
            run_id=str(row.get("run_id")) if row.get("run_id") else None,
            checkpoint_name=row.get("checkpoint_name", ""),
            checkpoint_type=CheckpointType(row.get("checkpoint_type", "state")),
            checkpoint_data=checkpoint_data,
            sequence_number=int(row.get("sequence_number", 0)),
            status=CheckpointStatus(row.get("status", "active")),
            expires_at=row.get("expires_at"),
            resumed_at=row.get("resumed_at"),
            resumed_by=row.get("resumed_by"),
            created_at=row.get("created_at"),
        )


# Global store instance
_store: Optional[CheckpointStore] = None


async def get_checkpoint_store() -> CheckpointStore:
    """Get the global checkpoint store instance."""
    global _store
    if _store is None:
        _store = CheckpointStore()
    return _store
