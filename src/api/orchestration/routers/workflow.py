"""
Deal Workflow API

Endpoints for managing deal stage transitions.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ....core.deals.workflow import get_workflow_engine, DealStage

router = APIRouter(prefix="/api/deals", tags=["workflow"])


class TransitionRequest(BaseModel):
    """Request to transition deal stage."""
    new_stage: str
    reason: Optional[str] = None


class TransitionResponse(BaseModel):
    """Response from stage transition."""
    deal_id: str
    from_stage: str
    to_stage: str
    success: bool
    timestamp: str


class ValidTransitionsResponse(BaseModel):
    """Response with valid transitions."""
    deal_id: str
    current_stage: Optional[str] = None
    valid_transitions: List[str]


@router.get("/{deal_id}/valid-transitions", response_model=ValidTransitionsResponse)
async def get_valid_transitions(deal_id: str):
    """Get valid stage transitions for a deal."""
    engine = await get_workflow_engine()

    try:
        transitions = await engine.get_valid_transitions(deal_id)
        return ValidTransitionsResponse(
            deal_id=deal_id,
            valid_transitions=transitions
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{deal_id}/transition", response_model=TransitionResponse)
async def transition_deal_stage(
    deal_id: str,
    body: TransitionRequest,
    transitioned_by: Optional[str] = Query(None, description="User performing the transition")
):
    """Transition a deal to a new stage."""
    engine = await get_workflow_engine()

    try:
        transition = await engine.transition_stage(
            deal_id=deal_id,
            new_stage=body.new_stage,
            transitioned_by=transitioned_by,
            reason=body.reason
        )

        return TransitionResponse(
            deal_id=transition.deal_id,
            from_stage=transition.from_stage,
            to_stage=transition.to_stage,
            success=True,
            timestamp=transition.timestamp.isoformat()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{deal_id}/stage-history")
async def get_stage_history(deal_id: str):
    """Get stage transition history for a deal."""
    engine = await get_workflow_engine()

    try:
        # Verify deal exists first
        valid_transitions = await engine.get_valid_transitions(deal_id)
        history = await engine.get_stage_history(deal_id)
        return {"deal_id": deal_id, "history": history}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/stages/summary")
async def get_stages_summary():
    """Get count of deals in each stage."""
    engine = await get_workflow_engine()
    summary = await engine.get_deal_stages_summary()
    return {
        "stages": summary,
        "available_stages": [s.value for s in DealStage]
    }
