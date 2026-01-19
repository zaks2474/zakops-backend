"""
Human-in-the-Loop (HITL) Module

Phase 6: HITL & Checkpoints
Spec Reference: Human-in-the-Loop section

Provides risk assessment, approval workflows, and checkpointing.

Usage:
    from src.core.hitl import assess_risk, require_approval, checkpoint

    # Assess risk for an action
    risk = await assess_risk(action_type, action_data)

    # Check if approval is required
    if risk.requires_approval:
        await workflow.request_approval(action_id, action_type, ...)

    # Save checkpoint during execution
    store = await get_checkpoint_store()
    await store.save_checkpoint(correlation_id, "step_1", {"data": "..."})
"""

from .risk import (
    RiskLevel,
    RiskAssessment,
    RiskRule,
    RiskAssessor,
    get_risk_assessor,
    assess_risk,
)

from .approval import (
    ApprovalStatus,
    ApprovalRequest,
    ApprovalWorkflow,
    get_approval_workflow,
)

from .checkpoint import (
    CheckpointStatus,
    CheckpointType,
    Checkpoint,
    CheckpointStore,
    get_checkpoint_store,
)


# Convenience function matching spec
async def require_approval(
    action_type: str,
    action_data: dict = None,
) -> bool:
    """Check if an action requires approval."""
    assessment = assess_risk(action_type, action_data)
    return assessment.requires_approval


# Context manager for checkpointing
async def checkpoint(action_id, checkpoint_name: str, correlation_id=None):
    """Get a checkpoint store for the given action."""
    store = await get_checkpoint_store()
    store._action_id = action_id
    store._correlation_id = correlation_id or action_id
    return store


async def restore_checkpoint(action_id, checkpoint_name: str = None):
    """Restore checkpoint data for an action."""
    store = await get_checkpoint_store()
    return await store.get_latest_checkpoint(
        action_id=str(action_id),
        checkpoint_name=checkpoint_name,
    )


__all__ = [
    # Risk
    "RiskLevel",
    "RiskAssessment",
    "RiskRule",
    "RiskAssessor",
    "get_risk_assessor",
    "assess_risk",
    "require_approval",
    # Approval
    "ApprovalStatus",
    "ApprovalRequest",
    "ApprovalWorkflow",
    "get_approval_workflow",
    # Checkpoint
    "CheckpointStatus",
    "CheckpointType",
    "Checkpoint",
    "CheckpointStore",
    "get_checkpoint_store",
    "checkpoint",
    "restore_checkpoint",
]
