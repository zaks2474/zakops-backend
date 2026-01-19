"""
HITL API Endpoints

Phase 6: HITL & Checkpoints
Provides endpoints for human-in-the-loop approval workflows.

Endpoints:
- POST /api/hitl/assess-risk - Assess risk for an action
- GET /api/hitl/approval-queue - Get actions awaiting approval
- POST /api/hitl/actions/{action_id}/approve - Approve an action
- POST /api/hitl/actions/{action_id}/reject - Reject an action
- POST /api/hitl/actions/{action_id}/quarantine - Quarantine an action
- GET /api/hitl/actions/{action_id}/checkpoints - Get checkpoints for action
- GET /api/hitl/actions/{action_id}/checkpoints/{name} - Get specific checkpoint
"""

from typing import Optional, List
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

from ....core.hitl import (
    assess_risk,
    get_approval_workflow,
    get_checkpoint_store,
    RiskLevel,
)
from ....core.database.adapter import get_database


router = APIRouter(prefix="/api/hitl", tags=["hitl"])


# ============================================================================
# Request/Response Models
# ============================================================================

class RiskAssessmentRequest(BaseModel):
    """Request to assess risk for an action."""
    action_type: str = Field(..., description="Action type (e.g., 'send_email')")
    action_data: Optional[dict] = Field(None, description="Action payload/inputs")
    context: Optional[dict] = Field(None, description="Additional context")


class RiskAssessmentResponse(BaseModel):
    """Risk assessment result."""
    level: str
    requires_approval: bool
    requires_quarantine: bool = False
    reasons: List[str]
    recommendations: List[str] = []


class ApprovalDecisionRequest(BaseModel):
    """Request to approve/reject/quarantine an action."""
    reason: Optional[str] = Field(None, description="Reason for decision")


class ActionQueueItem(BaseModel):
    """An action in the approval queue."""
    action_id: str
    action_type: str
    title: str
    risk_level: str
    status: str
    created_at: str
    deal_id: Optional[str] = None


class CheckpointItem(BaseModel):
    """A checkpoint summary."""
    checkpoint_id: str
    checkpoint_name: str
    sequence_number: int
    status: str
    created_at: str


class CheckpointDetailResponse(BaseModel):
    """Full checkpoint data."""
    checkpoint_id: str
    checkpoint_name: str
    checkpoint_type: str
    checkpoint_data: dict
    sequence_number: int
    status: str
    created_at: str


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/assess-risk", response_model=RiskAssessmentResponse)
async def assess_action_risk(request: RiskAssessmentRequest):
    """
    Assess risk level for an action.

    Used by: Agent before creating actions, UI for previewing risk

    Returns risk level, whether approval is required, and factors.
    """
    assessment = assess_risk(
        action_type=request.action_type,
        inputs=request.action_data,
        context=request.context,
    )

    return RiskAssessmentResponse(
        level=assessment.risk_level.value,
        requires_approval=assessment.requires_approval,
        requires_quarantine=assessment.risk_level == RiskLevel.CRITICAL,
        reasons=assessment.reasons,
        recommendations=assessment.recommendations,
    )


@router.get("/approval-queue", response_model=List[ActionQueueItem])
async def get_approval_queue(
    status: Optional[str] = Query("PENDING_APPROVAL", description="Filter by status"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    deal_id: Optional[str] = Query(None, description="Filter by deal ID"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
):
    """
    Get actions awaiting approval.

    Used by: Actions page, Quarantine page

    Returns actions sorted by risk level (critical first) then by creation time.
    """
    db = await get_database()

    conditions = ["status = $1"]
    params: List = [status]
    param_idx = 2

    if risk_level:
        conditions.append(f"risk_level = ${param_idx}")
        params.append(risk_level.lower())
        param_idx += 1

    if deal_id:
        conditions.append(f"deal_id = ${param_idx}")
        params.append(deal_id)
        param_idx += 1

    # Query from the actions engine's SQLite store is separate
    # For now, query deal_events for approval requests
    query = f"""
        SELECT DISTINCT ON (details->>'action_id')
            details->>'action_id' as action_id,
            details->>'action_type' as action_type,
            COALESCE(details->>'title', details->>'action_type') as title,
            COALESCE(details->>'risk_level', 'medium') as risk_level,
            'PENDING_APPROVAL' as status,
            created_at,
            deal_id
        FROM zakops.deal_events
        WHERE event_type = 'hitl.approval_requested'
        AND created_at > NOW() - INTERVAL '7 days'
        AND NOT EXISTS (
            SELECT 1 FROM zakops.deal_events de2
            WHERE de2.event_type IN ('hitl.approved', 'hitl.rejected')
            AND de2.details->>'action_id' = zakops.deal_events.details->>'action_id'
        )
        ORDER BY details->>'action_id', created_at DESC
        LIMIT ${param_idx}
    """
    params.append(limit)

    try:
        rows = await db.fetch(query, *params)
    except Exception:
        # Fallback to empty if table doesn't have expected data
        rows = []

    return [
        ActionQueueItem(
            action_id=str(row.get("action_id", "")),
            action_type=row.get("action_type", "unknown"),
            title=row.get("title", "Untitled"),
            risk_level=row.get("risk_level", "medium"),
            status=row.get("status", "PENDING_APPROVAL"),
            created_at=row["created_at"].isoformat() if row.get("created_at") else "",
            deal_id=str(row.get("deal_id")) if row.get("deal_id") else None,
        )
        for row in rows
    ]


@router.post("/actions/{action_id}/approve")
async def approve_action(
    action_id: str,
    request: ApprovalDecisionRequest,
):
    """
    Approve an action.

    Used by: Actions page approve button

    Note: In Phase 7, operator_id will come from authentication.
    """
    # TODO: Get operator_id from auth session (Phase 7)
    operator_id = "system"

    workflow = await get_approval_workflow()
    result = await workflow.approve(
        action_id=action_id,
        approver_id=operator_id,
        reason=request.reason,
    )

    return {
        "status": "approved",
        "action_id": action_id,
        "approved_by": operator_id,
        "reason": request.reason,
    }


@router.post("/actions/{action_id}/reject")
async def reject_action(
    action_id: str,
    request: ApprovalDecisionRequest,
):
    """
    Reject an action.

    Used by: Actions page reject button
    """
    if not request.reason:
        raise HTTPException(
            status_code=400,
            detail="Reason is required when rejecting an action"
        )

    operator_id = "system"

    workflow = await get_approval_workflow()
    result = await workflow.reject(
        action_id=action_id,
        rejector_id=operator_id,
        reason=request.reason,
    )

    return {
        "status": "rejected",
        "action_id": action_id,
        "rejected_by": operator_id,
        "reason": request.reason,
    }


@router.post("/actions/{action_id}/quarantine")
async def quarantine_action(
    action_id: str,
    request: ApprovalDecisionRequest,
):
    """
    Quarantine an action for further review.

    Used by: Actions page, automated risk detection
    """
    if not request.reason:
        raise HTTPException(
            status_code=400,
            detail="Reason is required for quarantine"
        )

    operator_id = "system"

    workflow = await get_approval_workflow()
    result = await workflow.escalate(
        action_id=action_id,
        escalated_by=operator_id,
        escalate_to="admin",
        reason=request.reason,
    )

    return {
        "status": "quarantined",
        "action_id": action_id,
        "quarantined_by": operator_id,
        "reason": request.reason,
    }


@router.get("/actions/{action_id}/checkpoints", response_model=List[CheckpointItem])
async def get_action_checkpoints(
    action_id: str,
    include_resumed: bool = Query(False, description="Include resumed checkpoints"),
):
    """
    Get checkpoints for an action.

    Used by: Action detail view, debugging
    """
    store = await get_checkpoint_store()
    checkpoints = await store.get_checkpoints_for_action(
        action_id=action_id,
        include_resumed=include_resumed,
    )

    return [
        CheckpointItem(
            checkpoint_id=cp.checkpoint_id,
            checkpoint_name=cp.checkpoint_name,
            sequence_number=cp.sequence_number,
            status=cp.status.value,
            created_at=cp.created_at.isoformat() if cp.created_at else "",
        )
        for cp in checkpoints
    ]


@router.get("/actions/{action_id}/checkpoints/{checkpoint_name}")
async def get_checkpoint_data(
    action_id: str,
    checkpoint_name: str,
):
    """
    Get checkpoint data by name.

    Used by: Action detail view, debugging, resume operations
    """
    store = await get_checkpoint_store()
    checkpoint = await store.get_latest_checkpoint(
        action_id=action_id,
        checkpoint_name=checkpoint_name,
    )

    if checkpoint is None:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint '{checkpoint_name}' not found for action {action_id}"
        )

    return CheckpointDetailResponse(
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_name=checkpoint.checkpoint_name,
        checkpoint_type=checkpoint.checkpoint_type.value,
        checkpoint_data=checkpoint.checkpoint_data,
        sequence_number=checkpoint.sequence_number,
        status=checkpoint.status.value,
        created_at=checkpoint.created_at.isoformat() if checkpoint.created_at else "",
    )


@router.post("/actions/{action_id}/checkpoints/{checkpoint_id}/resume")
async def resume_from_checkpoint(
    action_id: str,
    checkpoint_id: str,
):
    """
    Mark a checkpoint as resumed.

    Used by: Resume operation when restarting execution
    """
    operator_id = "system"

    store = await get_checkpoint_store()
    success = await store.mark_resumed(
        checkpoint_id=checkpoint_id,
        resumed_by=operator_id,
    )

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Checkpoint {checkpoint_id} not found or already resumed"
        )

    return {
        "status": "resumed",
        "checkpoint_id": checkpoint_id,
        "resumed_by": operator_id,
    }


@router.get("/risk-rules")
async def get_risk_rules():
    """
    Get configured risk rules.

    Used by: Admin UI for viewing/editing risk configuration
    """
    from ....core.hitl import get_risk_assessor

    assessor = get_risk_assessor()
    rules = []

    for rule in assessor.rules:
        rules.append({
            "name": rule.name,
            "description": rule.description,
            "risk_level": rule.risk_level.value,
            "requires_approval": rule.requires_approval,
            "action_types": list(rule.action_types),
            "action_type_patterns": rule.action_type_patterns,
        })

    return {
        "rules": rules,
        "default_risk_level": assessor.default_risk_level.value,
        "default_requires_approval": assessor.default_requires_approval,
    }
