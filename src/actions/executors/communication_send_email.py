from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from actions.engine.models import ActionError, ActionPayload, ArtifactMetadata, now_utc_iso
from actions.executors._artifacts import resolve_action_artifact_dir
from tools.gateway import ToolErrorCode, ToolInvocationContext, ToolResult, get_tool_gateway
from zakops_secret_scan import find_secrets_in_text

from .base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _dataroom_root() -> Path:
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()


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

    t = threading.Thread(target=_thread_main, name="send-email-loop", daemon=True)
    t.start()
    t.join()
    if error:
        raise error["exc"]
    return result.get("value")


def _normalize_recipients(raw: Any) -> List[str]:
    candidates: List[str] = []
    if isinstance(raw, list):
        candidates = [str(x or "").strip() for x in raw if str(x or "").strip()]
    else:
        text = str(raw or "").strip()
        if text:
            candidates = [p.strip() for p in re.split(r"[;,]+", text) if p.strip()]

    emails: List[str] = []
    for c in candidates:
        emails.extend(EMAIL_RE.findall(c))

    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: List[str] = []
    for e in emails:
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def _extract_body_from_markdown(content: str) -> str:
    """
    Extract a reasonable email body from common ZakOps draft artifacts.

    Supports:
    - DraftEmailExecutor: "To: ...\\nSubject: ...\\n\\n<body>"
    - RequestDocsExecutor: body between two '---' delimiters
    """
    text = (content or "").strip()
    if not text:
        return ""

    lines = text.splitlines()
    delim = [i for i, ln in enumerate(lines) if ln.strip() == "---"]
    if len(delim) >= 2 and delim[0] < delim[1]:
        body = "\n".join(lines[delim[0] + 1 : delim[1]]).strip()
        if body:
            return body

    # Fallback: first blank line indicates start of body.
    for i, ln in enumerate(lines):
        if i > 0 and not ln.strip():
            body = "\n".join(lines[i + 1 :]).strip()
            if body:
                return body

    return text


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


class SendEmailExecutor(ActionExecutor):
    """
    COMMUNICATION.SEND_EMAIL (irreversible; approval required)

    Uses ToolGateway to send email via a manifested tool (default: gmail__send_email).

    Inputs:
      - to: string|array (must contain at least one email address)
      - subject: string
      - body: string (optional if body_artifact_path provided)
      - body_artifact_path: string (optional path under DATAROOM_ROOT)
    """

    action_type = "COMMUNICATION.SEND_EMAIL"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        inputs = payload.inputs or {}
        to = inputs.get("to")
        subject = str(inputs.get("subject", "")).strip()
        body = str(inputs.get("body", "")).strip()
        body_path = str(inputs.get("body_artifact_path", "")).strip()

        recipients = _normalize_recipients(to)
        if not recipients:
            return False, "Missing/invalid `to` (must include at least one email address)"
        if not subject:
            return False, "Missing required `subject`"
        if not body and not body_path:
            return False, "Missing required `body` (or `body_artifact_path`)"
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}

        tool_name = (os.getenv("ZAKOPS_SEND_EMAIL_TOOL", "gmail__send_email") or "gmail__send_email").strip()

        recipients = _normalize_recipients(inputs.get("to"))
        subject = str(inputs.get("subject", "")).strip()

        body = str(inputs.get("body", "")).strip()
        body_artifact_path = str(inputs.get("body_artifact_path", "")).strip()
        if not body and body_artifact_path:
            path = Path(body_artifact_path).expanduser().resolve()
            try:
                path.relative_to(_dataroom_root())
            except ValueError:
                raise ActionExecutionError(
                    ActionError(
                        code="body_artifact_outside_dataroom",
                        message="body_artifact_path is outside DATAROOM_ROOT",
                        category="validation",
                        retryable=False,
                        details={"path": str(path)},
                    )
                )
            if not path.exists() or not path.is_file():
                raise ActionExecutionError(
                    ActionError(
                        code="body_artifact_missing",
                        message="body_artifact_path does not exist on disk",
                        category="io",
                        retryable=False,
                        details={"path": str(path)},
                    )
                )
            raw = path.read_text(encoding="utf-8", errors="replace")
            body = _extract_body_from_markdown(raw)

        if not recipients or not subject or not body:
            raise ActionExecutionError(
                ActionError(
                    code="validation_failed",
                    message="Validation failed (missing to/subject/body)",
                    category="validation",
                    retryable=False,
                )
            )

        # Secret-scan gate before ToolGateway invocation (irreversible action).
        secrets = find_secrets_in_text(f"To: {', '.join(recipients)}\nSubject: {subject}\n\n{body}\n")
        if secrets:
            raise ActionExecutionError(
                ActionError(
                    code="secret_scan_blocked",
                    message=f"Secret-like patterns detected ({', '.join(secrets)}); refusing to send.",
                    category="policy",
                    retryable=False,
                    details={"tool": tool_name},
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

        result = _run_coro_blocking(
            gateway.invoke(
                tool_name=tool_name,
                args={"to": recipients, "subject": subject, "body": body},
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
                    details={"tool_name": tool_name, "type": str(type(result))},
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
                        details={"tool_name": tool_name},
                    )
                )
            raise ActionExecutionError(_tool_result_to_action_error(tool_name=tool_name, result=result))

        # Artifacts
        out_dir: Path
        if ctx.deal and str((ctx.deal or {}).get("folder_path") or "").strip():
            out_dir = resolve_action_artifact_dir(ctx)
        else:
            out_dir = (_dataroom_root() / "99-ACTIONS" / payload.action_id).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

        sent_md = out_dir / "sent_email.md"
        sent_md.write_text(
            f"To: {', '.join(recipients)}\nSubject: {subject}\n\n{body}\n",
            encoding="utf-8",
        )
        result_path = out_dir / "send_result.json"
        result_path.write_text(
            json.dumps({"tool": tool_name, "output": result.output or {}}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        artifacts = [
            ArtifactMetadata(filename=sent_md.name, mime_type="text/markdown", path=str(sent_md), created_at=now_utc_iso()),
            ArtifactMetadata(filename=result_path.name, mime_type="application/json", path=str(result_path), created_at=now_utc_iso()),
        ]

        outputs = {
            "to": recipients,
            "subject": subject,
            "tool_name": tool_name,
            "tool_output": result.output or {},
            "duration_ms": int(result.duration_ms),
        }
        return ExecutionResult(outputs=outputs, artifacts=artifacts)
