from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Optional

from actions.engine.models import ActionError, ActionPayload
from tools.gateway import ToolErrorCode, ToolInvocationContext, ToolResult, get_tool_gateway
from tools.registry import get_tool_registry

from .base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult


def _run_coro_blocking(coro):
    """
    Run a coroutine from sync code.

    - If no loop is running: uses asyncio.run()
    - If a loop is already running (e.g., unit tests): runs in a dedicated thread
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}
    error: Dict[str, BaseException] = {}

    def _thread_main() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as e:  # pragma: no cover
            error["exc"] = e

    t = threading.Thread(target=_thread_main, name="tool-invoke-loop", daemon=True)
    t.start()
    t.join()
    if error:
        raise error["exc"]
    return result.get("value")


def _tool_result_to_action_error(*, tool_name: str, result: ToolResult) -> ActionError:
    retryable = bool(result.should_retry())
    category = "cloud_transient" if retryable else "dependency"
    return ActionError(
        code=f"tool_{result.error_code.lower()}",
        message=result.error_message or f"Tool invocation failed: {tool_name}",
        category=category,  # type: ignore[arg-type]
        retryable=retryable,
        details={
            "tool_name": tool_name,
            "tool_error_code": result.error_code,
            "retry_after_seconds": result.retry_after_seconds,
        },
    )


class ToolInvokeExecutor(ActionExecutor):
    """
    Generic executor for TOOL.<tool_id> action types.

    Inputs:
      action.inputs: tool arguments

    Outputs:
      outputs.tool_result: raw tool output
    """

    action_type = "TOOL.__dynamic__"

    def _tool_name_from_action_type(self, action_type: str) -> str:
        at = (action_type or "").strip()
        if not at.upper().startswith("TOOL."):
            return ""
        return at.split(".", 1)[1].strip()

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        tool_name = self._tool_name_from_action_type(payload.type)
        if not tool_name:
            return False, "Invalid tool action_type; expected TOOL.<tool_id>"
        tool = get_tool_registry().get_tool(tool_name)
        if not tool:
            return False, f"Tool not found in registry: {tool_name}"

        required = []
        schema = tool.input_schema or {}
        if isinstance(schema, dict):
            required = schema.get("required") or []
        if isinstance(required, list):
            missing = [k for k in required if k not in (payload.inputs or {})]
            if missing:
                return False, f"Missing required tool inputs: {missing}"

        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        tool_name = self._tool_name_from_action_type(payload.type)
        if not tool_name:
            raise ActionExecutionError(
                ActionError(
                    code="invalid_tool_action_type",
                    message="Invalid tool action_type; expected TOOL.<tool_id>",
                    category="validation",
                    retryable=False,
                )
            )

        gateway = ctx.tool_gateway or get_tool_gateway()

        context = ToolInvocationContext(
            action_id=payload.action_id,
            action_status=payload.status,
            deal_id=payload.deal_id,
            user_id=payload.created_by,
            session_id=None,
            approved=False,
            bypass_db_approval=False,
        )

        result = _run_coro_blocking(gateway.invoke(tool_name=tool_name, args=payload.inputs or {}, context=context))
        if not isinstance(result, ToolResult):
            raise ActionExecutionError(
                ActionError(
                    code="tool_gateway_invalid_response",
                    message="Tool gateway returned invalid response type",
                    category="dependency",
                    retryable=False,
                    details={"tool_name": tool_name, "type": str(type(result))},
                )
            )

        if not result.success:
            # If gateway is disabled, treat as a clear dependency/safety failure.
            if result.error_code == ToolErrorCode.GATEWAY_DISABLED:
                raise ActionExecutionError(
                    ActionError(
                        code="tool_gateway_disabled",
                        message=result.error_message or "Tool gateway disabled",
                        category="dependency",
                        retryable=False,
                        details={"tool_name": tool_name},
                    )
                )
            raise ActionExecutionError(_tool_result_to_action_error(tool_name=tool_name, result=result))

        return ExecutionResult(
            outputs={
                "tool_name": tool_name,
                "tool_output": result.output or {},
                "duration_ms": int(result.duration_ms),
            },
            artifacts=[],
        )

