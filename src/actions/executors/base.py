from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from actions.engine.models import ActionError, ActionPayload, ArtifactMetadata


@dataclass(frozen=True)
class ExecutionContext:
    """
    Executor context.

    Keep this intentionally small and explicit: executors should not reach out to
    global singletons directly.
    """

    action: ActionPayload
    deal: Optional[Dict[str, Any]] = None
    case_file: Optional[Dict[str, Any]] = None
    tool_gateway: Any = None
    cloud_allowed: bool = False
    registry: Any = None


@dataclass(frozen=True)
class ExecutionResult:
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[ArtifactMetadata] = field(default_factory=list)


class ActionExecutionError(RuntimeError):
    """Structured executor error that maps directly to ActionError."""

    def __init__(self, error: ActionError):
        super().__init__(error.message)
        self.error = error


class ActionExecutor:
    """Interface for action execution plugins."""

    action_type: str = ""

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        return True, None

    def dry_run(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        raise ActionExecutionError(
            ActionError(
                code="dry_run_not_supported",
                message=f"Dry-run not supported for action type {payload.type}",
                category="validation",
                retryable=False,
            )
        )

    def estimate_cost(self, payload: ActionPayload, ctx: ExecutionContext) -> Dict[str, Any]:
        return {"estimated_cost_usd": 0.0}

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        raise NotImplementedError
