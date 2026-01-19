"""
ZakOps Deal Management

Deal workflow engine for stage transitions and lifecycle management.
"""

from .workflow import (
    DealStage,
    STAGE_TRANSITIONS,
    StageTransition,
    DealWorkflowEngine,
    get_workflow_engine,
)

__all__ = [
    "DealStage",
    "STAGE_TRANSITIONS",
    "StageTransition",
    "DealWorkflowEngine",
    "get_workflow_engine",
]
