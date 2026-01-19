"""
Approval Workflow

Phase 6: HITL & Checkpoints
Spec Reference: Human-in-the-Loop section

Manages the approval workflow for actions:
- Request approval for actions
- Track approval status
- Handle approval/rejection with audit trail
- Escalation and timeout handling

Compatible with existing action approval flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from ..database.adapter import get_database, DatabaseAdapter
from .risk import RiskAssessment, RiskLevel, assess_risk

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """An approval request for an action."""
    request_id: str
    action_id: str
    action_type: str
    correlation_id: Optional[str] = None
    deal_id: Optional[str] = None

    # Risk info
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_reasons: List[str] = field(default_factory=list)

    # Approval info
    status: ApprovalStatus = ApprovalStatus.PENDING
    requested_by: Optional[str] = None
    requested_at: Optional[datetime] = None
    approver_roles: List[str] = field(default_factory=list)

    # Decision info
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None
    decision_reason: Optional[str] = None

    # Escalation
    escalated_to: Optional[str] = None
    escalated_at: Optional[datetime] = None
    escalation_reason: Optional[str] = None

    # Timeout
    expires_at: Optional[datetime] = None

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "correlation_id": self.correlation_id,
            "deal_id": self.deal_id,
            "risk_level": self.risk_level.value,
            "risk_reasons": self.risk_reasons,
            "status": self.status.value,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat() if self.requested_at else None,
            "approver_roles": self.approver_roles,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at.isoformat() if self.decided_at else None,
            "decision_reason": self.decision_reason,
            "escalated_to": self.escalated_to,
            "escalated_at": self.escalated_at.isoformat() if self.escalated_at else None,
            "escalation_reason": self.escalation_reason,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "context": self.context,
        }


class ApprovalWorkflow:
    """
    Manages approval workflows for actions.

    This service:
    - Creates approval requests based on risk assessment
    - Tracks approval status
    - Handles approval/rejection decisions
    - Manages escalation and timeouts
    - Maintains audit trail

    Compatible with existing action engine approval flow.

    Usage:
        workflow = ApprovalWorkflow()

        # Check if approval needed
        if workflow.needs_approval(action_type, inputs):
            request = await workflow.request_approval(action_id, action_type, ...)
            # Wait for decision...

        # Approve/reject
        await workflow.approve(request_id, approver_id, reason)
        await workflow.reject(request_id, approver_id, reason)
    """

    def __init__(self, db: Optional[DatabaseAdapter] = None):
        self._db = db

    async def _get_db(self) -> DatabaseAdapter:
        if self._db is None:
            self._db = await get_database()
        return self._db

    def needs_approval(
        self,
        action_type: str,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if an action needs approval.

        Uses the RiskAssessor to determine if approval is required.
        """
        assessment = assess_risk(action_type, inputs, context)
        return assessment.requires_approval

    def get_risk_assessment(
        self,
        action_type: str,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """Get full risk assessment for an action."""
        return assess_risk(action_type, inputs, context)

    async def request_approval(
        self,
        action_id: str,
        action_type: str,
        *,
        correlation_id: Optional[str] = None,
        deal_id: Optional[str] = None,
        requested_by: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        timeout_hours: int = 24,
    ) -> ApprovalRequest:
        """
        Create an approval request for an action.

        Args:
            action_id: The action requiring approval
            action_type: Type of action
            correlation_id: Correlation ID for tracing
            deal_id: Associated deal ID
            requested_by: Who requested the action
            inputs: Action inputs (for risk assessment)
            context: Additional context
            timeout_hours: Hours until auto-expiration

        Returns:
            ApprovalRequest with request details
        """
        # Assess risk
        assessment = assess_risk(action_type, inputs, context)

        # Determine approver roles based on risk level
        approver_roles = ["operator"]
        if assessment.risk_level == RiskLevel.CRITICAL:
            approver_roles = ["admin"]
        elif assessment.risk_level == RiskLevel.HIGH:
            approver_roles = ["operator", "admin"]

        now = _utcnow()
        request = ApprovalRequest(
            request_id=str(uuid4()),
            action_id=action_id,
            action_type=action_type,
            correlation_id=correlation_id,
            deal_id=deal_id,
            risk_level=assessment.risk_level,
            risk_reasons=assessment.reasons,
            status=ApprovalStatus.PENDING,
            requested_by=requested_by,
            requested_at=now,
            approver_roles=approver_roles,
            expires_at=now + timedelta(hours=timeout_hours),
            context=context or {},
        )

        # Store in database (using deal_events for now)
        db = await self._get_db()
        await db.execute(
            """
            INSERT INTO zakops.deal_events (
                deal_id, event_type, source, actor, details, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            str(deal_id or correlation_id or action_id)[:20],
            "hitl.approval_requested",
            "approval_workflow",
            requested_by or "system",
            {
                "request_id": request.request_id,
                "action_id": action_id,
                "action_type": action_type,
                "risk_level": assessment.risk_level.value,
                "risk_reasons": assessment.reasons,
                "approver_roles": approver_roles,
                "expires_at": request.expires_at.isoformat() if request.expires_at else None,
            },
            now,
        )

        logger.info(
            "Created approval request: request_id=%s action_id=%s risk=%s",
            request.request_id,
            action_id,
            assessment.risk_level.value,
        )

        return request

    async def approve(
        self,
        action_id: str,
        *,
        approver_id: str,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Approve an action.

        This method is compatible with the existing action engine.
        It records the approval in the event system.

        Args:
            action_id: The action to approve
            approver_id: ID of the approver
            reason: Optional reason for approval

        Returns:
            Dict with approval result
        """
        now = _utcnow()
        db = await self._get_db()

        # Record approval event
        await db.execute(
            """
            INSERT INTO zakops.deal_events (
                deal_id, event_type, source, actor, details, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            str(action_id)[:20],
            "hitl.approved",
            "approval_workflow",
            approver_id,
            {
                "action_id": action_id,
                "approver_id": approver_id,
                "reason": reason,
                "decided_at": now.isoformat(),
            },
            now,
        )

        logger.info(
            "Action approved: action_id=%s approver=%s",
            action_id,
            approver_id,
        )

        return {
            "action_id": action_id,
            "status": "approved",
            "approver_id": approver_id,
            "reason": reason,
            "decided_at": now.isoformat(),
        }

    async def reject(
        self,
        action_id: str,
        *,
        rejector_id: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Reject an action.

        Args:
            action_id: The action to reject
            rejector_id: ID of the rejector
            reason: Reason for rejection (required)

        Returns:
            Dict with rejection result
        """
        now = _utcnow()
        db = await self._get_db()

        # Record rejection event
        await db.execute(
            """
            INSERT INTO zakops.deal_events (
                deal_id, event_type, source, actor, details, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            str(action_id)[:20],
            "hitl.rejected",
            "approval_workflow",
            rejector_id,
            {
                "action_id": action_id,
                "rejector_id": rejector_id,
                "reason": reason,
                "decided_at": now.isoformat(),
            },
            now,
        )

        logger.info(
            "Action rejected: action_id=%s rejector=%s reason=%s",
            action_id,
            rejector_id,
            reason,
        )

        return {
            "action_id": action_id,
            "status": "rejected",
            "rejector_id": rejector_id,
            "reason": reason,
            "decided_at": now.isoformat(),
        }

    async def escalate(
        self,
        action_id: str,
        *,
        escalated_by: str,
        escalate_to: str,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Escalate an action to a higher authority.

        Args:
            action_id: The action to escalate
            escalated_by: ID of who is escalating
            escalate_to: Role or ID to escalate to
            reason: Reason for escalation

        Returns:
            Dict with escalation result
        """
        now = _utcnow()
        db = await self._get_db()

        # Record escalation event
        await db.execute(
            """
            INSERT INTO zakops.deal_events (
                deal_id, event_type, source, actor, details, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            str(action_id)[:20],
            "hitl.escalated",
            "approval_workflow",
            escalated_by,
            {
                "action_id": action_id,
                "escalated_by": escalated_by,
                "escalate_to": escalate_to,
                "reason": reason,
                "escalated_at": now.isoformat(),
            },
            now,
        )

        logger.info(
            "Action escalated: action_id=%s by=%s to=%s",
            action_id,
            escalated_by,
            escalate_to,
        )

        return {
            "action_id": action_id,
            "status": "escalated",
            "escalated_by": escalated_by,
            "escalate_to": escalate_to,
            "reason": reason,
            "escalated_at": now.isoformat(),
        }

    async def get_pending_approvals(
        self,
        *,
        deal_id: Optional[str] = None,
        approver_role: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get pending approval requests.

        Args:
            deal_id: Filter by deal ID
            approver_role: Filter by approver role
            limit: Maximum results

        Returns:
            List of pending approval events
        """
        db = await self._get_db()

        # Query recent approval requests that haven't been decided
        query = """
            SELECT * FROM zakops.deal_events
            WHERE event_type = 'hitl.approval_requested'
            AND created_at > NOW() - INTERVAL '7 days'
            ORDER BY created_at DESC
            LIMIT $1
        """
        rows = await db.fetch(query, limit)

        # Filter out those that have been decided
        decided_action_ids = set()
        decided_query = """
            SELECT DISTINCT details->>'action_id' as action_id
            FROM zakops.deal_events
            WHERE event_type IN ('hitl.approved', 'hitl.rejected', 'hitl.escalated')
            AND created_at > NOW() - INTERVAL '7 days'
        """
        decided_rows = await db.fetch(decided_query)
        for row in decided_rows:
            if row.get("action_id"):
                decided_action_ids.add(row["action_id"])

        # Build result
        results = []
        for row in rows:
            details = row.get("details") or {}
            action_id = details.get("action_id")
            if action_id in decided_action_ids:
                continue

            if deal_id and row.get("deal_id") != deal_id:
                continue

            if approver_role:
                roles = details.get("approver_roles", [])
                if approver_role not in roles:
                    continue

            results.append({
                "action_id": action_id,
                "action_type": details.get("action_type"),
                "risk_level": details.get("risk_level"),
                "risk_reasons": details.get("risk_reasons", []),
                "approver_roles": details.get("approver_roles", []),
                "requested_at": row.get("created_at"),
                "expires_at": details.get("expires_at"),
                "deal_id": row.get("deal_id"),
            })

        return results[:limit]


# Global workflow instance
_workflow: Optional[ApprovalWorkflow] = None


async def get_approval_workflow() -> ApprovalWorkflow:
    """Get the global approval workflow instance."""
    global _workflow
    if _workflow is None:
        _workflow = ApprovalWorkflow()
    return _workflow
