"""
Deal Workflow Engine

Manages deal stage transitions with validation, events, and idempotency.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from uuid import UUID
from enum import Enum
from dataclasses import dataclass, field
import json
import logging

from ..database.adapter import get_database
from ..events import publish_deal_event
from ..events.taxonomy import DealEventType

logger = logging.getLogger(__name__)


class DealStage(str, Enum):
    """Valid deal stages."""
    INBOUND = "inbound"
    INITIAL_REVIEW = "initial_review"
    DUE_DILIGENCE = "due_diligence"
    NEGOTIATION = "negotiation"
    DOCUMENTATION = "documentation"
    CLOSING = "closing"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"
    ARCHIVED = "archived"


# Valid stage transitions
STAGE_TRANSITIONS: Dict[DealStage, List[DealStage]] = {
    DealStage.INBOUND: [DealStage.INITIAL_REVIEW, DealStage.CLOSED_LOST, DealStage.ARCHIVED],
    DealStage.INITIAL_REVIEW: [DealStage.DUE_DILIGENCE, DealStage.INBOUND, DealStage.CLOSED_LOST],
    DealStage.DUE_DILIGENCE: [DealStage.NEGOTIATION, DealStage.INITIAL_REVIEW, DealStage.CLOSED_LOST],
    DealStage.NEGOTIATION: [DealStage.DOCUMENTATION, DealStage.DUE_DILIGENCE, DealStage.CLOSED_LOST],
    DealStage.DOCUMENTATION: [DealStage.CLOSING, DealStage.NEGOTIATION, DealStage.CLOSED_LOST],
    DealStage.CLOSING: [DealStage.CLOSED_WON, DealStage.DOCUMENTATION, DealStage.CLOSED_LOST],
    DealStage.CLOSED_WON: [DealStage.ARCHIVED],  # Terminal state
    DealStage.CLOSED_LOST: [DealStage.ARCHIVED],  # Terminal state
    DealStage.ARCHIVED: [],  # Final state
}


@dataclass
class StageTransition:
    """Record of a stage transition."""
    deal_id: str
    from_stage: str
    to_stage: str
    transitioned_by: Optional[str] = None
    reason: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    idempotent_hit: bool = False  # True if this was a duplicate request


class DealWorkflowEngine:
    """
    Manages deal lifecycle and stage transitions.
    """

    async def get_valid_transitions(self, deal_id: str) -> List[str]:
        """Get valid next stages for a deal."""
        db = await get_database()

        deal = await db.fetchrow(
            "SELECT stage FROM zakops.deals WHERE deal_id = $1",
            deal_id
        )

        if not deal:
            raise ValueError(f"Deal not found: {deal_id}")

        current_stage = deal["stage"]

        try:
            stage_enum = DealStage(current_stage)
            return [s.value for s in STAGE_TRANSITIONS.get(stage_enum, [])]
        except ValueError:
            # Unknown stage - allow transition to any non-terminal state
            return [s.value for s in DealStage if s not in (DealStage.CLOSED_WON, DealStage.CLOSED_LOST, DealStage.ARCHIVED)]

    async def transition_stage(
        self,
        deal_id: str,
        new_stage: str,
        idempotency_key: Optional[str] = None,
        transitioned_by: Optional[str] = None,
        reason: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> StageTransition:
        """
        Transition a deal to a new stage (idempotent).

        Args:
            deal_id: Deal ID
            new_stage: Target stage
            idempotency_key: Unique key for safe retries (recommended)
            transitioned_by: User/agent performing transition
            reason: Reason for transition
            trace_id: Trace ID for correlation

        Returns:
            StageTransition record

        Raises:
            ValueError: If transition is invalid

        Idempotency:
            If idempotency_key matches a recent transition (24h),
            returns the existing result without making changes.
        """
        db = await get_database()

        # Check idempotency first (before any locks)
        if idempotency_key:
            existing = await db.fetchrow(
                """
                SELECT details, actor, created_at
                FROM zakops.deal_events
                WHERE deal_id = $1
                  AND idempotency_key = $2
                  AND event_type = 'stage_changed'
                  AND created_at > NOW() - INTERVAL '24 hours'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                deal_id, idempotency_key
            )

            if existing:
                details = existing.get("details", {})
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except:
                        details = {}

                logger.info(
                    f"Idempotent hit for deal {deal_id}, "
                    f"key={idempotency_key}, transition={details.get('from_stage')}->{details.get('to_stage')}"
                )
                return StageTransition(
                    deal_id=deal_id,
                    from_stage=details.get("from_stage", ""),
                    to_stage=details.get("to_stage", ""),
                    transitioned_by=existing.get("actor"),
                    reason=details.get("reason"),
                    timestamp=existing["created_at"],
                    idempotent_hit=True
                )

        # Get current stage
        deal = await db.fetchrow(
            "SELECT stage FROM zakops.deals WHERE deal_id = $1",
            deal_id
        )

        if not deal:
            raise ValueError(f"Deal not found: {deal_id}")

        current_stage = deal["stage"]

        # Already in target stage? Return success (idempotent no-op)
        if current_stage == new_stage:
            logger.info(f"Deal {deal_id} already in stage {new_stage} (no-op)")
            return StageTransition(
                deal_id=deal_id,
                from_stage=current_stage,
                to_stage=new_stage,
                transitioned_by=transitioned_by,
                reason="Already in target stage (no-op)",
                timestamp=datetime.now(timezone.utc),
                idempotent_hit=True
            )

        # Validate transition
        try:
            current_enum = DealStage(current_stage)
            new_enum = DealStage(new_stage)
        except ValueError as e:
            raise ValueError(f"Invalid stage: {e}")

        valid_transitions = STAGE_TRANSITIONS.get(current_enum, [])
        if new_enum not in valid_transitions:
            raise ValueError(
                f"Invalid transition: {current_stage} -> {new_stage}. "
                f"Valid transitions: {[s.value for s in valid_transitions]}"
            )

        # Perform transition
        now = datetime.now(timezone.utc)

        await db.execute(
            """
            UPDATE zakops.deals
            SET stage = $2, updated_at = $3
            WHERE deal_id = $1
            """,
            deal_id, new_stage, now
        )

        # Record in deal_events as stage change (with idempotency_key)
        details_json = json.dumps({
            "from_stage": current_stage,
            "to_stage": new_stage,
            "reason": reason or "",
            "trace_id": trace_id or ""
        })

        await db.execute(
            """
            INSERT INTO zakops.deal_events
            (deal_id, event_type, source, actor, actor_type, details, idempotency_key, created_at)
            VALUES ($1, 'stage_changed', 'workflow', $2, $3, $4::jsonb, $5, $6)
            """,
            deal_id,
            transitioned_by or "system",
            "user" if transitioned_by else "system",
            details_json,
            idempotency_key,
            now
        )

        # Emit event via event system
        try:
            await publish_deal_event(
                deal_id=UUID(deal_id) if isinstance(deal_id, str) else deal_id,
                event_type=DealEventType.STAGE_CHANGED.value,
                event_data={
                    "from_stage": current_stage,
                    "to_stage": new_stage,
                    "transitioned_by": transitioned_by,
                    "reason": reason,
                    "trace_id": trace_id,
                    "timestamp": now.isoformat()
                }
            )
        except Exception as e:
            logger.warning(f"Failed to publish stage change event: {e}")

        logger.info(f"Deal {deal_id} transitioned: {current_stage} -> {new_stage}")

        return StageTransition(
            deal_id=deal_id,
            from_stage=current_stage,
            to_stage=new_stage,
            transitioned_by=transitioned_by,
            reason=reason,
            timestamp=now,
            idempotent_hit=False
        )

    async def get_stage_history(self, deal_id: str) -> List[Dict[str, Any]]:
        """Get stage transition history for a deal."""
        db = await get_database()

        history = await db.fetch(
            """
            SELECT event_type, actor as transitioned_by, details, created_at
            FROM zakops.deal_events
            WHERE deal_id = $1 AND event_type = 'stage_changed'
            ORDER BY created_at ASC
            """,
            deal_id
        )

        result = []
        for h in history:
            details = h.get("details", {})
            if isinstance(details, str):
                import json
                try:
                    details = json.loads(details)
                except:
                    details = {}

            result.append({
                "from_stage": details.get("from_stage"),
                "to_stage": details.get("to_stage"),
                "transitioned_by": h.get("transitioned_by"),
                "reason": details.get("reason"),
                "timestamp": h["created_at"].isoformat() if h.get("created_at") else None
            })

        return result

    async def get_deal_stages_summary(self) -> Dict[str, int]:
        """Get count of deals in each stage."""
        db = await get_database()

        results = await db.fetch(
            """
            SELECT stage, COUNT(*) as count
            FROM zakops.deals
            WHERE deleted = FALSE AND status = 'active'
            GROUP BY stage
            ORDER BY stage
            """
        )

        return {r["stage"]: r["count"] for r in results}


# Singleton instance
_workflow_engine: Optional[DealWorkflowEngine] = None


async def get_workflow_engine() -> DealWorkflowEngine:
    """Get workflow engine instance."""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = DealWorkflowEngine()
    return _workflow_engine
