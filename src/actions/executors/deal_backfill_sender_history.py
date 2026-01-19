from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from typing_extensions import Literal

from actions.engine.models import ActionError, ActionPayload, ArtifactMetadata, compute_idempotency_key, now_utc_iso
from actions.executors._artifacts import resolve_action_artifact_dir
from actions.executors.base import ActionExecutionError, ActionExecutor, ExecutionContext, ExecutionResult
from email_ingestion.enrichment.link_extractor import LinkExtractor
from integrations.gmail_thread_fetch import GmailThreadFetchConfig, get_thread_message_ids, load_thread_fetch_config
from tools.gateway import ToolErrorCode, ToolInvocationContext, ToolResult, get_tool_gateway


_ID_LINE_RE = re.compile(r"^ID:\s*(?P<id>.+?)\s*$")
_SUBJECT_LINE_RE = re.compile(r"^Subject:\s*(?P<subject>.*)$")
_FROM_LINE_RE = re.compile(r"^From:\s*(?P<from>.*)$")
_DATE_LINE_RE = re.compile(r"^Date:\s*(?P<date>.*)$")

_THREAD_ID_RE = re.compile(r"^Thread ID:\s*(?P<tid>.*)$")
_TO_RE = re.compile(r"^To:\s*(?P<to>.*)$")

_ATTACHMENT_HEADER_RE = re.compile(r"^Attachments\s*\((?P<count>\d+)\):\s*$")
_ATTACHMENT_LINE_RE = re.compile(
    r"^-\s+(?P<filename>.+?)\s+\((?P<mime>[^,]+),\s*(?P<size_kb>\d+)\s*KB,\s*ID:\s*(?P<id>[^)]+)\)\s*$"
)


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

    t = threading.Thread(target=_thread_main, name="deal-backfill-loop", daemon=True)
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


def _sanitize_filename(name: str) -> str:
    base = (name or "").strip()
    base = base.replace("\\\\", "/").split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9._ -]+", "_", base)
    base = re.sub(r"\s+", " ", base).strip()
    if not base:
        return "attachment"
    if len(base) > 180:
        stem = Path(base).stem[:160]
        suf = Path(base).suffix[:20]
        base = f"{stem}{suf}"
    return base


def _dataroom_root() -> Path:
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()


def _quarantine_root() -> Path:
    return _dataroom_root() / "00-PIPELINE" / "_INBOX_QUARANTINE"


def _backfill_log_path() -> Path:
    override = (os.getenv("EMAIL_BACKFILL_LOG_PATH") or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (_dataroom_root() / ".deal-registry" / "logs" / "email_backfill.jsonl").resolve()


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return


@dataclass(frozen=True)
class EmailSearchHit:
    message_id: str
    subject: str
    sender: str
    date: str


@dataclass(frozen=True)
class EmailAttachment:
    attachment_id: str
    filename: str
    mime_type: str
    size_bytes: int

    @property
    def ext_lower(self) -> str:
        return Path(self.filename).suffix.lower().lstrip(".")


@dataclass(frozen=True)
class EmailMessage:
    message_id: str
    thread_id: str
    subject: str
    sender: str
    to: str
    date: str
    body: str
    attachments: List[EmailAttachment]


def _parse_search_emails_text(text: str) -> List[EmailSearchHit]:
    hits: List[EmailSearchHit] = []
    current: Dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip("\\r")
        if not line.strip():
            continue
        m = _ID_LINE_RE.match(line)
        if m:
            if current.get("id"):
                hits.append(
                    EmailSearchHit(
                        message_id=current.get("id", ""),
                        subject=current.get("subject", ""),
                        sender=current.get("from", ""),
                        date=current.get("date", ""),
                    )
                )
            current = {"id": m.group("id").strip()}
            continue
        m = _SUBJECT_LINE_RE.match(line)
        if m:
            current["subject"] = m.group("subject").strip()
            continue
        m = _FROM_LINE_RE.match(line)
        if m:
            current["from"] = m.group("from").strip()
            continue
        m = _DATE_LINE_RE.match(line)
        if m:
            current["date"] = m.group("date").strip()
            continue

    if current.get("id"):
        hits.append(
            EmailSearchHit(
                message_id=current.get("id", ""),
                subject=current.get("subject", ""),
                sender=current.get("from", ""),
                date=current.get("date", ""),
            )
        )
    return hits


def _parse_read_email_text(message_id: str, text: str) -> EmailMessage:
    thread_id = ""
    subject = ""
    sender = ""
    to = ""
    date = ""

    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\\r")
        if not line.strip():
            i += 1
            break
        m = _THREAD_ID_RE.match(line)
        if m:
            thread_id = m.group("tid").strip()
        m = _SUBJECT_LINE_RE.match(line)
        if m:
            subject = m.group("subject").strip()
        m = _FROM_LINE_RE.match(line)
        if m:
            sender = m.group("from").strip()
        m = _TO_RE.match(line)
        if m:
            to = m.group("to").strip()
        m = _DATE_LINE_RE.match(line)
        if m:
            date = m.group("date").strip()
        i += 1

    body_lines: List[str] = []
    attachments: List[EmailAttachment] = []
    in_attachments = False
    while i < len(lines):
        line = lines[i].rstrip("\\r")
        if _ATTACHMENT_HEADER_RE.match(line.strip()):
            in_attachments = True
            i += 1
            continue
        if in_attachments:
            m = _ATTACHMENT_LINE_RE.match(line.strip())
            if m:
                filename = m.group("filename").strip()
                mime = m.group("mime").strip()
                size_kb = int(m.group("size_kb"))
                aid = m.group("id").strip()
                attachments.append(
                    EmailAttachment(
                        attachment_id=aid,
                        filename=filename,
                        mime_type=mime,
                        size_bytes=size_kb * 1024,
                    )
                )
        else:
            body_lines.append(line)
        i += 1

    body = "\\n".join(body_lines).strip()
    return EmailMessage(
        message_id=message_id,
        thread_id=thread_id,
        subject=subject,
        sender=sender,
        to=to,
        date=date,
        body=body,
        attachments=attachments,
    )


class BackfillEvidence(BaseModel):
    message_index: int = Field(ge=0)
    snippet: str = Field(default="", max_length=600)
    why_it_matters: str = Field(default="", max_length=400)

    model_config = {"extra": "forbid"}


class BackfillLinkDecision(BaseModel):
    belongs_to_deal: Literal["same", "different", "uncertain"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=600)
    evidence: List[BackfillEvidence] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


def _decision_to_backfill_v1_item(
    *,
    message_id: str,
    decision: Literal["BELONGS_TO_APPROVED_DEAL", "NEW_DEAL_CANDIDATE", "IGNORE"],
    confidence: float,
    reason: str,
    evidence: List[BackfillEvidence],
) -> Dict[str, Any]:
    evidence_out: List[Dict[str, str]] = []
    for e in evidence[:10]:
        quote = _sanitize_evidence_quote(e.snippet)
        if not quote:
            continue
        evidence_out.append({"quote": quote, "source": "THREAD", "reason": (e.why_it_matters or "")[:400]})

    return {
        "message_id": str(message_id),
        "decision": decision,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "reason": (reason or "")[:600],
        "evidence": evidence_out,
    }


def _extract_first_json_object(text: str) -> Optional[dict]:
    """
    Best-effort JSON object extractor for LLM responses.

    Supports:
    - Raw JSON
    - Markdown fenced code blocks
    """
    raw = (text or "").strip()
    if not raw:
        return None

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    brace = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
    if brace:
        candidate = brace.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def _assert_local_vllm_base_url(url: str) -> None:
    from urllib.parse import urlparse

    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").strip().lower()
    if host not in {"localhost", "127.0.0.1"}:
        raise ActionExecutionError(
            ActionError(
                code="vllm_base_url_not_local",
                message="Refusing to call non-local LLM base URL for backfill",
                category="validation",
                retryable=False,
                details={"base_url": url},
            )
        )


def _call_local_vllm_json(*, system: str, user: str, max_tokens: int = 900, temperature: float = 0.2) -> Tuple[Optional[dict], Optional[str], Dict[str, Any]]:
    import urllib.error
    import urllib.request
    from urllib.parse import urljoin

    base = (os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1") or "").strip().rstrip("/") + "/"
    _assert_local_vllm_base_url(base)
    endpoint = urljoin(base, "chat/completions")
    model = (os.getenv("VLLM_MODEL") or os.getenv("DEFAULT_MODEL") or "Qwen/Qwen2.5-32B-Instruct-AWQ").strip()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False,
    }

    started = time.monotonic()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, method="POST", data=body, headers={"Content-Type": "application/json"})
    timeout_s = float(os.getenv("VLLM_TIMEOUT_S", "45") or "45")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_raw = ""
        try:
            err_raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_raw = ""
        meta = {"endpoint": endpoint, "model": model, "status": int(getattr(e, "code", 0) or 0), "latency_ms": int((time.monotonic() - started) * 1000)}
        return None, f"vllm_http_error:{meta['status']}:{err_raw[:300]}", meta
    except Exception as e:
        meta = {"endpoint": endpoint, "model": model, "error": type(e).__name__, "latency_ms": int((time.monotonic() - started) * 1000)}
        return None, f"vllm_request_error:{type(e).__name__}", meta

    meta = {"endpoint": endpoint, "model": model, "latency_ms": int((time.monotonic() - started) * 1000)}
    try:
        data = json.loads(raw)
        content = str(data["choices"][0]["message"]["content"])
    except Exception:
        return None, "vllm_bad_response", meta

    parsed = _extract_first_json_object(content)
    if not isinstance(parsed, dict):
        return None, "vllm_no_json_object", meta
    return parsed, None, meta


def _email_to_sender_email(sender: str) -> str:
    m = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", sender or "")
    return m.group(1).lower() if m else ""


_URL_LIKE_RE = re.compile(r"https?://\\S+", re.IGNORECASE)
_TRAILING_URL_PUNCT = ").,;\"')]>“”"


def _safe_url_no_query(url: str) -> str:
    """
    Strip query/fragment to avoid persisting access tokens.
    """
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        return (url or "").split("?", 1)[0].split("#", 1)[0]


def _sanitize_evidence_quote(text: str) -> str:
    """
    Keep evidence quotes short and avoid URL query params.
    """
    raw = (text or "").strip().replace("\\r", " ").replace("\\n", " ").strip()
    if not raw:
        return ""

    def _repl(match: re.Match[str]) -> str:
        return _safe_url_no_query(match.group(0))

    raw = _URL_LIKE_RE.sub(_repl, raw)
    if len(raw) > 240:
        raw = raw[:239] + "…"
    return raw


def _sanitize_urls_in_text(text: str) -> str:
    """
    Remove URL query/fragment from any URLs embedded in text before persisting to disk.
    This avoids leaking access tokens into local artifacts that may later be indexed or logged.
    """
    raw = text or ""
    if not raw:
        return ""

    def _repl(match: re.Match[str]) -> str:
        url = match.group(0)
        suffix = ""
        while url and url[-1] in _TRAILING_URL_PUNCT:
            suffix = url[-1] + suffix
            url = url[:-1]
        return _safe_url_no_query(url) + suffix

    return _URL_LIKE_RE.sub(_repl, raw)


def _safe_link_dicts(*, message_id: str, date: str, links: List[Any]) -> List[Dict[str, Any]]:
    def _safe_url(url: str) -> str:
        """
        Strip query/fragment to avoid persisting access tokens.
        """
        try:
            from urllib.parse import urlsplit, urlunsplit

            parts = urlsplit(url)
            return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        except Exception:
            return (url or "").split("?", 1)[0].split("#", 1)[0]

    safe: List[Dict[str, Any]] = []
    for l in links:
        if not isinstance(l, dict):
            continue
        url = str(l.get("url") or "").strip()
        if not url:
            continue
        url = _safe_url_no_query(url)
        safe.append(
            {
                "url": url,
                "type": str(l.get("type") or "other"),
                "auth_required": bool(l.get("auth_required", True)),
                "vendor_hint": l.get("vendor_hint"),
                "source_email_id": message_id,
                "source_timestamp": date,
            }
        )
    return safe


def _ensure_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sender_history_scans (
            scan_key TEXT PRIMARY KEY,
            deal_id TEXT NOT NULL,
            sender_email TEXT NOT NULL,
            lookback_days INTEGER NOT NULL,
            max_messages INTEGER NOT NULL,
            mode TEXT NOT NULL,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            last_error TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sender_history_processed (
            message_id TEXT PRIMARY KEY,
            thread_id TEXT,
            deal_id TEXT,
            decision TEXT,
            confidence REAL,
            processed_at TEXT NOT NULL,
            scan_key TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def _scan_already_completed(conn: sqlite3.Connection, scan_key: str) -> bool:
    row = conn.execute("SELECT status FROM sender_history_scans WHERE scan_key=?", (scan_key,)).fetchone()
    return bool(row and str(row["status"] or "") == "completed")


def _mark_scan_started(conn: sqlite3.Connection, *, scan_key: str, deal_id: str, sender_email: str, lookback_days: int, max_messages: int, mode: str) -> None:
    now = now_utc_iso()
    conn.execute(
        """
        INSERT INTO sender_history_scans (scan_key, deal_id, sender_email, lookback_days, max_messages, mode, created_at, status, last_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'started', NULL)
        ON CONFLICT(scan_key) DO UPDATE SET
            status='started',
            last_error=NULL
        """,
        (scan_key, deal_id, sender_email, int(lookback_days), int(max_messages), mode, now),
    )
    conn.commit()


def _mark_scan_completed(conn: sqlite3.Connection, *, scan_key: str, status: str, error: str = "") -> None:
    now = now_utc_iso()
    conn.execute(
        "UPDATE sender_history_scans SET completed_at=?, status=?, last_error=? WHERE scan_key=?",
        (now, status, (error or "")[:2000] if status != "completed" else None, scan_key),
    )
    conn.commit()


def _record_processed(conn: sqlite3.Connection, *, scan_key: str, deal_id: str, message_id: str, thread_id: str, decision: str, confidence: float) -> None:
    conn.execute(
        """
        INSERT INTO sender_history_processed (message_id, thread_id, deal_id, decision, confidence, processed_at, scan_key)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(message_id) DO UPDATE SET
            decision=excluded.decision,
            confidence=excluded.confidence,
            processed_at=excluded.processed_at,
            scan_key=excluded.scan_key
        """,
        (message_id, thread_id, deal_id, decision, float(confidence), now_utc_iso(), scan_key),
    )
    conn.commit()


def _already_processed(conn: sqlite3.Connection, message_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM sender_history_processed WHERE message_id=? LIMIT 1", (message_id,)).fetchone()
    return bool(row)


def _triage_state_has_message(message_id: str) -> bool:
    """
    Best-effort idempotency check against the Email 3H triage state DB.
    """
    try:
        path = Path(os.getenv("EMAIL_TRIAGE_STATE_DB", "/home/zaks/DataRoom/.deal-registry/email_triage_state.db")).resolve()
        if not path.exists():
            return False
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT status FROM email_triage_messages WHERE message_id=? LIMIT 1", (message_id,)).fetchone()
        conn.close()
        return bool(row and str(row["status"] or "") in {"processed", "processing"})
    except Exception:
        return False


class DealBackfillSenderHistoryExecutor(ActionExecutor):
    """
    DEAL.BACKFILL_SENDER_HISTORY

    Scans prior emails from the approved sender and routes them:
    - same deal (confident) -> DEAL.APPEND_EMAIL_MATERIALS (auto-approved)
    - different/uncertain -> EMAIL_TRIAGE.REVIEW_EMAIL (approval-gated) when mode=classify_and_quarantine
    """

    action_type = "DEAL.BACKFILL_SENDER_HISTORY"

    def validate(self, payload: ActionPayload) -> tuple[bool, Optional[str]]:
        inputs = payload.inputs or {}
        deal_id = str(inputs.get("deal_id") or payload.deal_id or "").strip()
        sender_email = str(inputs.get("sender_email") or "").strip()
        if not deal_id:
            return False, "Missing required deal_id (inputs.deal_id or payload.deal_id)"
        if "@" not in sender_email:
            return False, "Missing/invalid inputs.sender_email"
        return True, None

    def execute(self, payload: ActionPayload, ctx: ExecutionContext) -> ExecutionResult:
        inputs = payload.inputs or {}

        deal_id = str(inputs.get("deal_id") or payload.deal_id or "").strip()
        approved_message_id = str(inputs.get("approved_message_id") or "").strip()
        sender_email = str(inputs.get("sender_email") or "").strip().lower()
        lookback_days = int(inputs.get("lookback_days") or 365)
        max_messages = int(inputs.get("max_messages") or 50)
        mode = str(inputs.get("mode") or "classify_and_quarantine").strip() or "classify_and_quarantine"
        if mode not in {"same_deal_only", "classify_and_quarantine"}:
            mode = "classify_and_quarantine"

        min_confidence_same = float(inputs.get("min_confidence_same") or 0.9)
        max_thread_messages = int(inputs.get("max_thread_messages") or 25)
        started = time.monotonic()
        log_path = _backfill_log_path()

        registry = getattr(ctx, "registry", None)
        if registry is None:
            raise ActionExecutionError(
                ActionError(
                    code="registry_unavailable",
                    message="DealRegistry unavailable in executor context",
                    category="dependency",
                    retryable=False,
                )
            )

        deal_obj = registry.get_deal(deal_id)
        if not deal_obj or not getattr(deal_obj, "folder_path", ""):
            raise ActionExecutionError(
                ActionError(
                    code="deal_not_found",
                    message="Deal not found or missing folder_path",
                    category="validation",
                    retryable=False,
                    details={"deal_id": deal_id},
                )
            )

        scan_key = compute_idempotency_key("sender_history_scan", deal_id, sender_email, str(lookback_days), str(max_messages), mode)
        backfill_db_path = _dataroom_root() / ".deal-registry" / "email_backfill_state.db"
        conn = _ensure_sqlite(backfill_db_path)

        try:
            if _scan_already_completed(conn, scan_key):
                _append_jsonl(
                    log_path,
                    {
                        "timestamp": now_utc_iso(),
                        "event": "scan_skip_already_completed",
                        "deal_id": deal_id,
                        "sender_email": sender_email,
                        "scan_key": scan_key,
                    },
                )
                return ExecutionResult(
                    outputs={
                        "deal_id": deal_id,
                        "sender_email": sender_email,
                        "scan_key": scan_key,
                        "already_completed": True,
                        "next_actions": [],
                    },
                    artifacts=[],
                )

            _mark_scan_started(conn, scan_key=scan_key, deal_id=deal_id, sender_email=sender_email, lookback_days=lookback_days, max_messages=max_messages, mode=mode)
            _append_jsonl(
                log_path,
                {
                    "timestamp": now_utc_iso(),
                    "event": "scan_start",
                    "deal_id": deal_id,
                    "sender_email": sender_email,
                    "scan_key": scan_key,
                    "lookback_days": lookback_days,
                    "max_messages": max_messages,
                    "mode": mode,
                    "min_confidence_same": min_confidence_same,
                    "max_thread_messages": max_thread_messages,
                },
            )

            gateway = get_tool_gateway()
            invocation_context = ToolInvocationContext(
                action_id=payload.action_id,
                action_status=payload.status,
                deal_id=deal_id,
                user_id=payload.created_by,
                session_id=None,
                approved=True,
                bypass_db_approval=False,
            )

            query = f"in:anywhere -in:trash -in:spam from:{sender_email} newer_than:{int(lookback_days)}d"
            search = _run_coro_blocking(
                gateway.invoke(
                    tool_name="gmail__search_emails",
                    args={"query": query, "maxResults": int(max_messages)},
                    context=invocation_context,
                )
            )
            if not isinstance(search, ToolResult):
                raise ActionExecutionError(
                    ActionError(
                        code="tool_gateway_invalid_response",
                        message="Tool gateway returned invalid response type",
                        category="dependency",
                        retryable=False,
                        details={"tool_name": "gmail__search_emails", "type": str(type(search))},
                    )
                )
            if not search.success:
                if search.error_code == ToolErrorCode.GATEWAY_DISABLED:
                    raise ActionExecutionError(
                        ActionError(
                            code="tool_gateway_disabled",
                            message=search.error_message or "Tool gateway disabled",
                            category="dependency",
                            retryable=False,
                            details={"tool_name": "gmail__search_emails"},
                        )
                    )
                raise ActionExecutionError(_tool_result_to_action_error(tool_name="gmail__search_emails", result=search))

            hits = _parse_search_emails_text(_extract_first_text(search.output))

            cfg: GmailThreadFetchConfig = load_thread_fetch_config()
            thread_fetch_cache: Dict[str, Tuple[List[str], Optional[str]]] = {}

            next_actions: List[Dict[str, Any]] = []
            backfill_decisions_v1: List[Dict[str, Any]] = []
            scanned = skipped = appended = quarantined = 0

            extractor = LinkExtractor()

            for hit in hits[: max(0, int(max_messages))]:
                mid = (hit.message_id or "").strip()
                if not mid:
                    continue
                if approved_message_id and mid == approved_message_id:
                    skipped += 1
                    continue
                if _already_processed(conn, mid):
                    skipped += 1
                    continue
                if _triage_state_has_message(mid):
                    skipped += 1
                    continue
                # If already mapped to any deal, skip (deterministic).
                mapped = registry.get_email_deal_mapping(mid)
                if mapped:
                    skipped += 1
                    continue

                read = _run_coro_blocking(
                    gateway.invoke(
                        tool_name="gmail__read_email",
                        args={"messageId": mid},
                        context=invocation_context,
                    )
                )
                if not isinstance(read, ToolResult):
                    raise ActionExecutionError(
                        ActionError(
                            code="tool_gateway_invalid_response",
                            message="Tool gateway returned invalid response type",
                            category="dependency",
                            retryable=False,
                            details={"tool_name": "gmail__read_email", "type": str(type(read))},
                        )
                    )
                if not read.success:
                    raise ActionExecutionError(_tool_result_to_action_error(tool_name="gmail__read_email", result=read))

                msg = _parse_read_email_text(mid, _extract_first_text(read.output))
                tid = (msg.thread_id or "").strip()
                if not tid:
                    skipped += 1
                    continue

                resolved, resolved_deal_id, non_deal_reason = registry.is_thread_resolved(tid)
                if resolved:
                    if non_deal_reason:
                        _record_processed(conn, scan_key=scan_key, deal_id=deal_id, message_id=mid, thread_id=tid, decision="skip_non_deal_thread", confidence=1.0)
                        _append_jsonl(
                            log_path,
                            {
                                "timestamp": now_utc_iso(),
                                "event": "message_skipped",
                                "deal_id": deal_id,
                                "scan_key": scan_key,
                                "message_id": mid,
                                "thread_id": tid,
                                "reason": "thread_to_non_deal",
                            },
                        )
                        skipped += 1
                        continue
                    if resolved_deal_id and resolved_deal_id != deal_id:
                        _record_processed(conn, scan_key=scan_key, deal_id=deal_id, message_id=mid, thread_id=tid, decision="skip_other_deal_thread", confidence=1.0)
                        _append_jsonl(
                            log_path,
                            {
                                "timestamp": now_utc_iso(),
                                "event": "message_skipped",
                                "deal_id": deal_id,
                                "scan_key": scan_key,
                                "message_id": mid,
                                "thread_id": tid,
                                "reason": "thread_mapped_to_other_deal",
                                "other_deal_id": resolved_deal_id,
                            },
                        )
                        skipped += 1
                        continue
                    # Same deal thread; append with deterministic mapping.
                    decision = BackfillLinkDecision(belongs_to_deal="same", confidence=1.0, reason="thread_to_deal_mapping", evidence=[])
                else:
                    # Unresolved thread: LLM linking decision using full thread.
                    if tid not in thread_fetch_cache:
                        ids, err = get_thread_message_ids(cfg=cfg, thread_id=tid)
                        thread_fetch_cache[tid] = (ids, err)
                    thread_message_ids, thread_err = thread_fetch_cache.get(tid, ([], "thread_fetch_missing"))
                    if thread_err:
                        thread_message_ids = [mid]

                    # Cap thread size (include most recent messages if large).
                    if len(thread_message_ids) > max(1, int(max_thread_messages)):
                        thread_message_ids = thread_message_ids[-max(1, int(max_thread_messages)) :]

                    thread_messages: List[EmailMessage] = []
                    for tm in thread_message_ids:
                        if tm == mid:
                            thread_messages.append(msg)
                            continue
                        r = _run_coro_blocking(
                            gateway.invoke(
                                tool_name="gmail__read_email",
                                args={"messageId": tm},
                                context=invocation_context,
                            )
                        )
                        if not isinstance(r, ToolResult) or not r.success:
                            continue
                        thread_messages.append(_parse_read_email_text(tm, _extract_first_text(r.output)))

                    deal_name = str(getattr(deal_obj, "canonical_name", "") or getattr(deal_obj, "display_name", "") or deal_id)

                    # Build a compact thread input for the model (no tool calls).
                    thread_lines: List[str] = []
                    for idx, m in enumerate(thread_messages):
                        body = (m.body or "").strip()
                        if body:
                            body = _URL_LIKE_RE.sub(lambda match: _safe_url_no_query(match.group(0)), body)
                        if len(body) > 6000:
                            body = body[:6000] + "…"
                        thread_lines.append(
                            "\\n".join(
                                [
                                    f"--- MESSAGE {idx} ---",
                                    f"Message ID: {m.message_id}",
                                    f"From: {m.sender}",
                                    f"To: {m.to}",
                                    f"Date: {m.date}",
                                    f"Subject: {m.subject}",
                                    "Body:",
                                    body,
                                ]
                            )
                        )

                    system = (
                        "You are a deal linking classifier.\n"
                        "You decide whether a candidate email thread belongs to an existing deal.\n"
                        "Return ONLY strict JSON (no markdown) with schema:\n"
                        '{ "belongs_to_deal": "same"|"different"|"uncertain", "confidence": 0-1, "reason": string, "evidence": [{"message_index": int, "snippet": string, "why_it_matters": string}] }\n'
                        "Rules:\n"
                        "- Do not invent facts.\n"
                        "- If evidence is insufficient, choose uncertain.\n"
                        "- confidence must reflect certainty based on the provided thread.\n"
                    )
                    user = (
                        f"Existing deal:\n- deal_id: {deal_id}\n- deal_name: {deal_name}\n\n"
                        f"Approved sender: {sender_email}\n"
                        f"Approved message id: {approved_message_id or 'unknown'}\n\n"
                        "Candidate thread messages (chronological):\n"
                        + "\\n\\n".join(thread_lines)
                    )

                    raw_json, llm_err, llm_meta = _call_local_vllm_json(system=system, user=user, max_tokens=900, temperature=0.2)
                    if llm_err or not isinstance(raw_json, dict):
                        decision = BackfillLinkDecision(belongs_to_deal="uncertain", confidence=0.0, reason=llm_err or "llm_no_json", evidence=[])
                    else:
                        try:
                            decision = BackfillLinkDecision.model_validate(raw_json)
                        except Exception:
                            decision = BackfillLinkDecision(belongs_to_deal="uncertain", confidence=0.0, reason="llm_schema_invalid", evidence=[])
                        decision = decision.model_copy(update={"reason": (decision.reason or "")[:600]})
                        decision_meta = {"llm": llm_meta, "thread_fetch_error": thread_err}
                        _ = decision_meta  # metadata recorded in report artifact below

                scanned += 1

                # Extract links/attachments for action inputs + quarantine preview.
                extracted_links = extractor.extract(
                    body_text=msg.body or "",
                    body_html=None,
                    source_email_id=mid,
                    source_timestamp=msg.date or None,
                )
                links_payload = _safe_link_dicts(
                    message_id=mid,
                    date=msg.date,
                    links=[
                        {
                            "url": l.normalized_url or l.url,
                            "type": (l.link_type.value if getattr(l, "link_type", None) else "other"),
                            "auth_required": bool(getattr(l, "requires_auth", False)),
                            "vendor_hint": getattr(l, "vendor_hint", None),
                        }
                        for l in extracted_links
                    ],
                )

                attachment_filename_by_id: Dict[str, str] = {}
                used_attachment_names: set[str] = set()
                for a in msg.attachments:
                    base = _sanitize_filename(a.filename)
                    candidate = base
                    stem = Path(base).stem
                    suf = Path(base).suffix
                    for i in range(2, 50):
                        if candidate and candidate not in used_attachment_names:
                            break
                        candidate = f"{stem}_{i}{suf}"
                    if not candidate:
                        candidate = f"attachment_{a.attachment_id}"
                    used_attachment_names.add(candidate)
                    attachment_filename_by_id[a.attachment_id] = candidate

                attachments_payload: List[Dict[str, Any]] = []
                for a in msg.attachments:
                    attachments_payload.append(
                        {
                            "attachment_id": a.attachment_id,
                            "filename": attachment_filename_by_id.get(a.attachment_id) or _sanitize_filename(a.filename),
                            "mime_type": a.mime_type,
                            "size_bytes": int(a.size_bytes or 0),
                        }
                    )

                quarantine_dir = (_quarantine_root() / mid).resolve()
                try:
                    quarantine_dir.relative_to(_dataroom_root())
                except ValueError:
                    raise ActionExecutionError(
                        ActionError(
                            code="quarantine_dir_outside_dataroom",
                            message="Computed quarantine_dir outside DATAROOM_ROOT",
                            category="validation",
                            retryable=False,
                            details={"quarantine_dir": str(quarantine_dir)},
                        )
                    )
                quarantine_dir.mkdir(parents=True, exist_ok=True)

                # Materialize base files.
                (quarantine_dir / "email_body.txt").write_text(_sanitize_urls_in_text(msg.body or ""), encoding="utf-8")
                decision_dump = decision.model_dump()
                if isinstance(decision_dump.get("evidence"), list):
                    for ev in decision_dump["evidence"]:
                        if not isinstance(ev, dict):
                            continue
                        ev["snippet"] = _sanitize_evidence_quote(str(ev.get("snippet") or ""))
                (quarantine_dir / "email.json").write_text(
                    json.dumps(
                        {
                            "message_id": msg.message_id,
                            "thread_id": msg.thread_id,
                            "from": msg.sender,
                            "to": msg.to,
                            "date": msg.date,
                            "subject": msg.subject,
                            "sender_email": _email_to_sender_email(msg.sender),
                            "decision": decision_dump,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                # Best-effort attachment download (safe allowlist).
                safe_exts = {e.strip().lower().lstrip(".") for e in (os.getenv("EMAIL_BACKFILL_SAFE_EXTS") or "pdf,doc,docx,xls,xlsx,csv,ppt,pptx,txt,zip").split(",") if e.strip()}
                max_mb = int(os.getenv("EMAIL_BACKFILL_MAX_ATTACHMENT_MB", "25") or "25")
                max_bytes = max(1, max_mb) * 1024 * 1024

                for a in msg.attachments:
                    ext = a.ext_lower
                    if ext and ext not in safe_exts:
                        continue
                    if int(a.size_bytes or 0) > max_bytes:
                        continue
                    if (a.mime_type or "").lower().startswith("image/"):
                        continue
                    filename = attachment_filename_by_id.get(a.attachment_id) or _sanitize_filename(a.filename)
                    if (quarantine_dir / filename).exists():
                        continue
                    dl = _run_coro_blocking(
                        gateway.invoke(
                            tool_name="gmail__download_attachment",
                            args={"messageId": mid, "attachmentId": a.attachment_id, "savePath": str(quarantine_dir), "filename": filename},
                            context=invocation_context,
                        )
                    )
                    if isinstance(dl, ToolResult) and not dl.success:
                        # Do not fail the whole backfill for one attachment; quarantine can still proceed.
                        continue

                # Route outcomes.
                if decision.belongs_to_deal == "same" and float(decision.confidence) >= float(min_confidence_same):
                    backfill_decisions_v1.append(
                        _decision_to_backfill_v1_item(
                            message_id=mid,
                            decision="BELONGS_TO_APPROVED_DEAL",
                            confidence=float(decision.confidence),
                            reason=decision.reason,
                            evidence=list(decision.evidence or []),
                        )
                    )
                    # Persist deterministic mapping (high confidence only).
                    try:
                        registry.add_email_deal_mapping(mid, deal_id)
                        if tid:
                            registry.add_thread_deal_mapping(tid, deal_id)
                        registry.save()
                    except Exception:
                        pass

                    next_actions.append(
                        {
                            "action_type": "DEAL.APPEND_EMAIL_MATERIALS",
                            "capability_id": "deal.append_email_materials.v1",
                            "title": "Append sender-history email to deal correspondence bundle",
                            "inputs": {
                                "deal_id": deal_id,
                                "message_id": mid,
                                "thread_id": tid,
                                "from": msg.sender,
                                "to": msg.to,
                                "date": msg.date,
                                "subject": msg.subject,
                                "links": links_payload,
                                "attachments": attachments_payload,
                                "quarantine_dir": str(quarantine_dir),
                                "backfill_source": "sender_history",
                            },
                            "requires_approval": False,
                            "idempotency_key": f"append_email_materials:{deal_id}:{mid}",
                        }
                    )
                    appended += 1
                    _record_processed(conn, scan_key=scan_key, deal_id=deal_id, message_id=mid, thread_id=tid, decision="append_same_deal", confidence=float(decision.confidence))
                    _append_jsonl(
                        log_path,
                        {
                            "timestamp": now_utc_iso(),
                            "event": "message_routed",
                            "deal_id": deal_id,
                            "scan_key": scan_key,
                            "message_id": mid,
                            "thread_id": tid,
                            "belongs_to_deal": decision.belongs_to_deal,
                            "confidence": float(decision.confidence),
                            "outcome": "append",
                        },
                    )
                    continue

                # different/uncertain
                if mode == "classify_and_quarantine":
                    backfill_decisions_v1.append(
                        _decision_to_backfill_v1_item(
                            message_id=mid,
                            decision="NEW_DEAL_CANDIDATE",
                            confidence=float(decision.confidence),
                            reason=decision.reason,
                            evidence=list(decision.evidence or []),
                        )
                    )
                    summary = f"Sender-history backfill: {decision.belongs_to_deal} ({decision.confidence:.2f}). {decision.reason}"[:500]
                    classification = "DEAL_SIGNAL" if decision.belongs_to_deal == "different" else "UNCERTAIN"
                    next_actions.append(
                        {
                            "action_type": "EMAIL_TRIAGE.REVIEW_EMAIL",
                            "capability_id": "email_triage.review_email.v1",
                            "title": f"Review sender-history email ({classification}): {(msg.subject or '').strip()[:80]}",
                            "summary": summary,
                            "inputs": {
                                "message_id": mid,
                                "thread_id": tid,
                                "from": msg.sender,
                                "to": msg.to,
                                "date": msg.date,
                                "subject": msg.subject,
                                "company": None,
                                "sender_email": _email_to_sender_email(msg.sender),
                                "classification": classification,
                                "confidence": float(decision.confidence),
                                "deal_likelihood_reason": decision.reason,
                                "evidence": [
                                    {
                                        "message_index": int(e.message_index),
                                        "snippet": _sanitize_evidence_quote(e.snippet),
                                        "why_it_matters": e.why_it_matters,
                                    }
                                    for e in (decision.evidence or [])
                                ],
                                "links": links_payload,
                                "attachments": attachments_payload,
                                "quarantine_dir": str(quarantine_dir),
                                "backfill_source": "sender_history",
                                "target_deal_id": deal_id,
                            },
                            "requires_approval": True,
                            # Use the same idempotency key as the triage agent to avoid duplicates.
                            "idempotency_key": f"email_triage:{mid}:review_email",
                        }
                    )
                    quarantined += 1
                    _record_processed(conn, scan_key=scan_key, deal_id=deal_id, message_id=mid, thread_id=tid, decision=f"quarantine_{classification.lower()}", confidence=float(decision.confidence))
                    _append_jsonl(
                        log_path,
                        {
                            "timestamp": now_utc_iso(),
                            "event": "message_routed",
                            "deal_id": deal_id,
                            "scan_key": scan_key,
                            "message_id": mid,
                            "thread_id": tid,
                            "belongs_to_deal": decision.belongs_to_deal,
                            "confidence": float(decision.confidence),
                            "outcome": "quarantine",
                            "classification": classification,
                        },
                    )
                else:
                    backfill_decisions_v1.append(
                        _decision_to_backfill_v1_item(
                            message_id=mid,
                            decision="IGNORE",
                            confidence=float(decision.confidence),
                            reason=f"mode_same_deal_only:{decision.reason}",
                            evidence=list(decision.evidence or []),
                        )
                    )
                    skipped += 1
                    _record_processed(conn, scan_key=scan_key, deal_id=deal_id, message_id=mid, thread_id=tid, decision=f"skip_{decision.belongs_to_deal}", confidence=float(decision.confidence))
                    _append_jsonl(
                        log_path,
                        {
                            "timestamp": now_utc_iso(),
                            "event": "message_skipped",
                            "deal_id": deal_id,
                            "scan_key": scan_key,
                            "message_id": mid,
                            "thread_id": tid,
                            "belongs_to_deal": decision.belongs_to_deal,
                            "confidence": float(decision.confidence),
                            "reason": "mode_same_deal_only",
                        },
                    )

            # Write artifacts (deal-local) for observability.
            out_dir = resolve_action_artifact_dir(ctx)
            report = {
                "deal_id": deal_id,
                "sender_email": sender_email,
                "approved_message_id": approved_message_id,
                "scan_key": scan_key,
                "lookback_days": lookback_days,
                "max_messages": max_messages,
                "mode": mode,
                "min_confidence_same": min_confidence_same,
                "max_thread_messages": max_thread_messages,
                "counts": {"scanned": scanned, "skipped": skipped, "appended": appended, "quarantined": quarantined},
                "created_at": now_utc_iso(),
                "next_actions_count": len(next_actions),
            }
            json_path = out_dir / "sender_history_backfill.json"
            md_path = out_dir / "sender_history_backfill.md"
            json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            md_path.write_text(
                "\\n".join(
                    [
                        "# Sender History Backfill",
                        "",
                        f"- Deal: {deal_id}",
                        f"- Sender: {sender_email}",
                        f"- Lookback: {lookback_days}d",
                        f"- Mode: {mode}",
                        "",
                        "## Counts",
                        f"- scanned: {scanned}",
                        f"- appended: {appended}",
                        f"- quarantined: {quarantined}",
                        f"- skipped: {skipped}",
                        "",
                        "## Next actions",
                        f"- count: {len(next_actions)}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            report_v1 = {
                "schema_version": "zakops.email_backfill.v1",
                "approved_deal_id": deal_id,
                "sender": sender_email,
                "decisions": backfill_decisions_v1,
            }
            v1_path = out_dir / "sender_history_backfill_v1.json"
            v1_path.write_text(json.dumps(report_v1, ensure_ascii=False, indent=2), encoding="utf-8")

            # Optional: write a copy under the deal folder for operator convenience.
            try:
                deal_folder_raw = str(getattr(deal_obj, "folder_path", "") or "").strip()
                deal_folder = Path(deal_folder_raw).expanduser()
                if not deal_folder.is_absolute():
                    deal_folder = (_dataroom_root() / deal_folder).resolve()
                else:
                    deal_folder = deal_folder.resolve()
                deal_folder.relative_to(_dataroom_root())
                corr_dir = deal_folder / "07-Correspondence"
                corr_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                corr_path = corr_dir / f"backfill_sender_history_{ts}.json"
                corr_path.write_text(json.dumps(report_v1, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            artifacts = [
                ArtifactMetadata(filename=json_path.name, mime_type="application/json", path=str(json_path), created_at=now_utc_iso()),
                ArtifactMetadata(filename=md_path.name, mime_type="text/markdown", path=str(md_path), created_at=now_utc_iso()),
                ArtifactMetadata(filename=v1_path.name, mime_type="application/json", path=str(v1_path), created_at=now_utc_iso()),
            ]

            _mark_scan_completed(conn, scan_key=scan_key, status="completed")
            _append_jsonl(
                log_path,
                {
                    "timestamp": now_utc_iso(),
                    "event": "scan_complete",
                    "deal_id": deal_id,
                    "sender_email": sender_email,
                    "scan_key": scan_key,
                    "counts": report["counts"],
                    "next_actions_count": len(next_actions),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )

            return ExecutionResult(
                outputs={
                    "deal_id": deal_id,
                    "sender_email": sender_email,
                    "scan_key": scan_key,
                    "counts": report["counts"],
                    "next_actions": next_actions,
                },
                artifacts=artifacts,
            )
        except ActionExecutionError:
            raise
        except Exception as e:
            _mark_scan_completed(conn, scan_key=scan_key, status="failed", error=str(e))
            _append_jsonl(
                log_path,
                {
                    "timestamp": now_utc_iso(),
                    "event": "scan_failed",
                    "deal_id": deal_id,
                    "sender_email": sender_email,
                    "scan_key": scan_key,
                    "error": f"{type(e).__name__}:{str(e)[:200]}",
                    "duration_ms": int((time.monotonic() - started) * 1000),
                },
            )
            raise ActionExecutionError(
                ActionError(
                    code="backfill_failed",
                    message="Sender history backfill failed",
                    category="unknown",
                    retryable=True,
                    details={"error": type(e).__name__},
                )
            )
        finally:
            try:
                conn.close()
            except Exception:
                pass
