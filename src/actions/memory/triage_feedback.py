from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _dataroom_root() -> Path:
    return Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()


def _feedback_path() -> Path:
    return _dataroom_root() / ".deal-registry" / "triage_feedback.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_feedback_entry(
    *,
    decision: str,
    message_id: str,
    thread_id: Optional[str],
    sender: str,
    subject: str,
    classification: Optional[str],
    confidence: Optional[float],
    actor: str,
    action_id: str,
    action_type: str,
    deal_id: Optional[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sender_l = (sender or "").strip()
    subject_l = (subject or "").strip()
    return {
        "timestamp": _now_iso(),
        "decision": decision,
        "message_id": (message_id or "").strip(),
        "thread_id": (thread_id or "").strip() or None,
        "sender": sender_l[:200],
        "sender_domain": (sender_l.split("@", 1)[1].lower() if "@" in sender_l else None),
        "subject_prefix": subject_l[:120],
        "subject_sha256": _sha256(subject_l),
        "classification": (classification or "").strip() or None,
        "confidence": float(confidence) if confidence is not None else None,
        "actor": (actor or "").strip()[:120] or "unknown",
        "action_id": (action_id or "").strip(),
        "action_type": (action_type or "").strip(),
        "deal_id": (deal_id or "").strip() or None,
        "extra": extra or {},
    }


def append_feedback(entry: Dict[str, Any]) -> None:
    """
    Append a single feedback entry to triage_feedback.jsonl.

    - Best-effort: never raises (errors are swallowed).
    - No secrets: callers should pass only minimal metadata (no raw email bodies).
    """
    try:
        path = _feedback_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"

        # Use an advisory lock when available to avoid interleaving lines.
        try:
            import fcntl  # type: ignore
        except Exception:  # pragma: no cover
            fcntl = None  # type: ignore

        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        fd = os.open(str(path), flags, 0o664)
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                except Exception:
                    pass
            os.write(fd, line.encode("utf-8"))
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except Exception:
                    pass
            os.close(fd)

        # If running as root, align ownership/group with parent dir so future non-root writes succeed.
        try:
            if os.geteuid() == 0:
                parent_stat = path.parent.stat()
                os.chown(str(path), parent_stat.st_uid, parent_stat.st_gid)
                os.chmod(str(path), 0o664)
        except Exception:
            pass
    except Exception:
        return

