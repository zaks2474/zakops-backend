from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ActionCreationValidationError(Exception):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


def validate_action_creation(*, action_type: str, capability_id: Optional[str] = None) -> None:
    """
    Validate that an action can run *before* writing it to the Actions store.

    This prevents "missing executor" / "capability mismatch" broken actions.
    """
    at = (action_type or "").strip()
    if not at:
        raise ActionCreationValidationError(
            code="action_type_missing",
            message="Missing action_type",
            details={},
        )

    from actions.executors.registry import get_executor

    executor = get_executor(at)
    if executor is None:
        raise ActionCreationValidationError(
            code="executor_not_found",
            message=f"No executor registered for action type: {at}",
            details={"action_type": at},
        )

    # Tool invocations: ensure tool exists (executor exists for TOOL.* via ToolInvokeExecutor).
    if at.upper().startswith("TOOL."):
        tool_id = at.split(".", 1)[1].strip() if "." in at else ""
        if not tool_id:
            raise ActionCreationValidationError(
                code="tool_id_missing",
                message="TOOL action_type missing tool_id",
                details={"action_type": at},
            )
        from tools.registry import get_tool_registry

        tool = get_tool_registry().get_tool(tool_id)
        if tool is None:
            raise ActionCreationValidationError(
                code="tool_not_found",
                message=f"Tool not found for tool action: {tool_id}",
                details={"tool_id": tool_id, "action_type": at},
            )

    cap_id = (capability_id or "").strip() or None
    if not cap_id:
        return

    from actions.capabilities.registry import get_registry as get_capability_registry
    from tools.registry import get_tool_registry

    cap_reg = get_capability_registry()
    # Ensure tool capabilities are discoverable.
    try:
        cap_reg.index_tools(get_tool_registry())
    except Exception:
        pass

    manifest = cap_reg.get_capability(cap_id)
    if manifest is None:
        raise ActionCreationValidationError(
            code="capability_not_found",
            message=f"Capability not found: {cap_id}",
            details={"capability_id": cap_id, "action_type": at},
        )

    if str(getattr(manifest, "action_type", "") or "").strip() != at:
        raise ActionCreationValidationError(
            code="capability_action_type_mismatch",
            message="capability.action_type does not match action_type",
            details={
                "capability_id": cap_id,
                "capability_action_type": str(getattr(manifest, "action_type", "") or "").strip(),
                "action_type": at,
            },
        )

