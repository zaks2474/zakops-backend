from __future__ import annotations

import asyncio
import os
import re
import threading
from typing import Any, Dict, List, Optional

from actions.engine.models import ActionError, ActionPayload
from actions.executors.base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult
from actions.memory.triage_feedback import append_feedback, build_feedback_entry
from integrations.n8n_webhook import emit_quarantine_rejected
from tools.gateway import ToolErrorCode, ToolInvocationContext, ToolResult, get_tool_gateway


_LABEL_ID_RE = re.compile(r"\bID:\s*([A-Za-z0-9_-]+)\b")


def _run_coro_blocking(coro):
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

    t = threading.Thread(target=_thread_main, name="email-triage-reject-loop", daemon=True)
    t.start()
    t.join()
    if error:
        raise error["exc"]
    return result.get("value")


def _extract_first_text(output: Any) -> str:
    if isinstance(output, dict):
        content = output.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                text = first.get("text")
                if isinstance(text, str):
                    return text
    if isinstance(output, str):
        return output
    return ""


def _parse_label_id(output: Any) -> Optional[str]:
    text = _extract_first_text(output)
    m = _LABEL_ID_RE.search(text or "")
    return m.group(1) if m else None


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


class EmailTriageRejectEmailExecutor(ActionExecutor):
    """
    EMAIL_TRIAGE.REJECT_EMAIL (approval-gated)

    - Applies Gmail labels via ToolGateway (reversible).
    - Records thread_to_non_deal mapping for deterministic future routing.
    """

    action_type = "EMAIL_TRIAGE.REJECT_EMAIL"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        inputs = payload.inputs or {}
        message_id = str(inputs.get("message_id") or "").strip()
        if not message_id:
            return False, "Missing required inputs.message_id"
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}
        message_id = str(inputs.get("message_id") or "").strip()
        thread_id = str(inputs.get("thread_id") or "").strip()
        reason = str(inputs.get("reason") or "operator_rejected").strip()[:500] or "operator_rejected"

        labels_to_add = inputs.get("labels_to_add") or ["ZakOps/NonDeal", "ZakOps/Processed"]
        labels_to_remove = inputs.get("labels_to_remove") or ["ZakOps/Deal", "ZakOps/Quarantine"]

        if not message_id:
            raise ActionExecutionError(
                ActionError(
                    code="invalid_inputs",
                    message="inputs.message_id is required",
                    category="validation",
                    retryable=False,
                )
            )

        registry = getattr(ctx, "registry", None)
        if registry and thread_id:
            existing = registry.get_thread_deal_mapping(thread_id)
            if existing:
                raise ActionExecutionError(
                    ActionError(
                        code="thread_already_mapped_to_deal",
                        message="Thread is already mapped to a deal; refusing to mark non-deal",
                        category="validation",
                        retryable=False,
                        details={"thread_id": thread_id, "deal_id": existing},
                    )
                )

        gateway = ctx.tool_gateway or get_tool_gateway()
        invocation_context = ToolInvocationContext(
            action_id=payload.action_id,
            action_status=payload.status,
            deal_id=payload.deal_id,
            user_id=payload.created_by,
            session_id=None,
            approved=True,
            bypass_db_approval=False,
        )

        def _ensure_label_id(name: str) -> str:
            result = _run_coro_blocking(
                gateway.invoke(
                    tool_name="gmail__get_or_create_label",
                    args={"name": name},
                    context=invocation_context,
                )
            )
            if not isinstance(result, ToolResult):
                raise ActionExecutionError(
                    ActionError(
                        code="tool_gateway_invalid_response",
                        message="Tool gateway returned invalid response type",
                        category="dependency",
                        retryable=False,
                        details={"tool_name": "gmail__get_or_create_label", "type": str(type(result))},
                    )
                )
            if not result.success:
                if result.error_code == ToolErrorCode.GATEWAY_DISABLED:
                    raise ActionExecutionError(
                        ActionError(
                            code="tool_gateway_disabled",
                            message=result.error_message or "Tool gateway disabled",
                            category="dependency",
                            retryable=False,
                            details={"tool_name": "gmail__get_or_create_label"},
                        )
                    )
                raise ActionExecutionError(_tool_result_to_action_error(tool_name="gmail__get_or_create_label", result=result))

            label_id = _parse_label_id(result.output)
            if not label_id:
                raise ActionExecutionError(
                    ActionError(
                        code="label_id_parse_failed",
                        message="Could not parse label id from tool output",
                        category="dependency",
                        retryable=False,
                        details={"label_name": name},
                    )
                )
            return label_id

        add_ids: List[str] = []
        for name in labels_to_add if isinstance(labels_to_add, list) else []:
            n = str(name or "").strip()
            if n:
                add_ids.append(_ensure_label_id(n))

        remove_ids: List[str] = []
        for name in labels_to_remove if isinstance(labels_to_remove, list) else []:
            n = str(name or "").strip()
            if n:
                try:
                    remove_ids.append(_ensure_label_id(n))
                except Exception:
                    # Best-effort removal; missing labels should not block rejection.
                    continue

        modify_args: Dict[str, Any] = {"messageId": message_id}
        if add_ids:
            modify_args["addLabelIds"] = add_ids
        if remove_ids:
            modify_args["removeLabelIds"] = remove_ids

        modify = _run_coro_blocking(
            gateway.invoke(
                tool_name="gmail__modify_email",
                args=modify_args,
                context=invocation_context,
            )
        )
        if not isinstance(modify, ToolResult):
            raise ActionExecutionError(
                ActionError(
                    code="tool_gateway_invalid_response",
                    message="Tool gateway returned invalid response type",
                    category="dependency",
                    retryable=False,
                    details={"tool_name": "gmail__modify_email", "type": str(type(modify))},
                )
            )
        if not modify.success:
            if modify.error_code == ToolErrorCode.GATEWAY_DISABLED:
                raise ActionExecutionError(
                    ActionError(
                        code="tool_gateway_disabled",
                        message=modify.error_message or "Tool gateway disabled",
                        category="dependency",
                        retryable=False,
                        details={"tool_name": "gmail__modify_email"},
                    )
                )
            raise ActionExecutionError(_tool_result_to_action_error(tool_name="gmail__modify_email", result=modify))

        if registry and thread_id:
            registry.add_thread_non_deal_mapping(thread_id, reason)
            registry.save()

        # Operator feedback dataset (minimal; no raw bodies). Best-effort.
        try:
            actor = payload.created_by
            for ev in reversed(payload.audit_trail or []):
                if getattr(ev, "event", "") == "approved":
                    actor = getattr(ev, "actor", actor) or actor
                    break

            entry = build_feedback_entry(
                decision="reject",
                message_id=message_id,
                thread_id=thread_id or None,
                sender=str(inputs.get("from") or ""),
                subject=str(inputs.get("subject") or ""),
                classification=str(inputs.get("classification") or "") or None,
                confidence=(float(inputs.get("confidence")) if inputs.get("confidence") is not None else None),
                actor=actor,
                action_id=payload.action_id,
                action_type=payload.type,
                deal_id=payload.deal_id,
                extra={"reason": reason},
            )
            append_feedback(entry)
        except Exception:
            pass

        # Optional n8n integration (off by default unless ZAKOPS_N8N_WEBHOOK_URL is set).
        try:
            emit_quarantine_rejected(message_id=message_id, thread_id=thread_id or None, reason=reason)
        except Exception:
            pass

        return ExecutionResult(
            outputs={
                "message_id": message_id,
                "thread_id": thread_id,
                "reason": reason,
                "labels_added": labels_to_add,
                "labels_removed": labels_to_remove,
            },
            artifacts=[],
        )
