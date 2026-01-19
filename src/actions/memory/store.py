from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_state_db_path() -> Path:
    return Path(os.getenv("ZAKOPS_STATE_DB", "/home/zaks/DataRoom/.deal-registry/ingest_state.db"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(raw: Optional[str]) -> Any:
    if not raw:
        return None
    return json.loads(raw)


def _tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    if not cleaned:
        return []
    return [t for t in cleaned.split() if len(t) > 1]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def fingerprint_inputs(inputs: Dict[str, Any]) -> str:
    raw = _json_dumps(inputs or {}).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ActionSummary:
    memory_id: str
    created_at: str
    action_id: str
    action_type: str
    deal_id: Optional[str]
    inputs_fingerprint: str
    plan_spec: Dict[str, Any]
    outcome_status: str
    artifacts: List[Dict[str, Any]]
    user_edits: Dict[str, Any]
    summary_text: str


@dataclass(frozen=True)
class ActionSummaryMatch:
    summary: ActionSummary
    score: float
    reason: str


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS action_memory (
  memory_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  action_id TEXT NOT NULL UNIQUE,
  action_type TEXT NOT NULL,
  deal_id TEXT,
  inputs_fingerprint TEXT NOT NULL,
  plan_spec_json TEXT NOT NULL,
  outcome_status TEXT NOT NULL,
  artifacts_json TEXT NOT NULL,
  user_edits_json TEXT NOT NULL,
  summary_text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_action_memory_created_at ON action_memory(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_action_memory_action_type ON action_memory(action_type);
CREATE INDEX IF NOT EXISTS idx_action_memory_deal_id ON action_memory(deal_id);

-- Optional FTS index (best-effort; not all SQLite builds include FTS5).
"""


def _try_create_fts(conn: sqlite3.Connection) -> None:
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS action_memory_fts
            USING fts5(memory_id, summary_text, content='action_memory', content_rowid='rowid');
            """
        )
    except Exception:
        return


class ActionMemoryStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or _default_state_db_path())
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            _try_create_fts(conn)
            conn.commit()

    def record(self, summary: ActionSummary) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO action_memory (
                  memory_id, created_at, action_id, action_type, deal_id, inputs_fingerprint,
                  plan_spec_json, outcome_status, artifacts_json, user_edits_json, summary_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    summary.memory_id,
                    summary.created_at,
                    summary.action_id,
                    summary.action_type,
                    summary.deal_id,
                    summary.inputs_fingerprint,
                    _json_dumps(summary.plan_spec or {}),
                    summary.outcome_status,
                    _json_dumps(summary.artifacts or []),
                    _json_dumps(summary.user_edits or {}),
                    summary.summary_text,
                ),
            )
            # Best-effort keep FTS in sync (if present).
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO action_memory_fts(memory_id, summary_text) VALUES (?, ?)",
                    (summary.memory_id, summary.summary_text),
                )
            except Exception:
                pass
            conn.commit()

    def list_recent(self, *, limit: int = 50) -> List[ActionSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT memory_id, created_at, action_id, action_type, deal_id, inputs_fingerprint,
                       plan_spec_json, outcome_status, artifacts_json, user_edits_json, summary_text
                FROM action_memory
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()

        out: List[ActionSummary] = []
        for r in rows:
            out.append(
                ActionSummary(
                    memory_id=r["memory_id"],
                    created_at=r["created_at"],
                    action_id=r["action_id"],
                    action_type=r["action_type"],
                    deal_id=r["deal_id"],
                    inputs_fingerprint=r["inputs_fingerprint"],
                    plan_spec=_json_loads(r["plan_spec_json"]) or {},
                    outcome_status=r["outcome_status"],
                    artifacts=_json_loads(r["artifacts_json"]) or [],
                    user_edits=_json_loads(r["user_edits_json"]) or {},
                    summary_text=r["summary_text"],
                )
            )
        return out

    def find_similar(
        self,
        query: str,
        *,
        deal_id: Optional[str] = None,
        top_k: int = 3,
        search_limit: int = 200,
    ) -> List[ActionSummaryMatch]:
        intent = (query or "").strip()
        if not intent:
            return []

        # Prefer FTS when available.
        with self._connect() as conn:
            try:
                params: List[Any] = [intent]
                deal_filter = ""
                if deal_id:
                    deal_filter = " AND m.deal_id = ?"
                    params.append(deal_id)

                rows = conn.execute(
                    f"""
                    SELECT m.memory_id, m.created_at, m.action_id, m.action_type, m.deal_id, m.inputs_fingerprint,
                           m.plan_spec_json, m.outcome_status, m.artifacts_json, m.user_edits_json, m.summary_text
                    FROM action_memory_fts f
                    JOIN action_memory m ON m.memory_id = f.memory_id
                    WHERE action_memory_fts MATCH ?
                    {deal_filter}
                    ORDER BY m.created_at DESC
                    LIMIT ?
                    """,
                    (*params, max(1, int(search_limit))),
                ).fetchall()

                matches: List[ActionSummaryMatch] = []
                for r in rows:
                    summary = ActionSummary(
                        memory_id=r["memory_id"],
                        created_at=r["created_at"],
                        action_id=r["action_id"],
                        action_type=r["action_type"],
                        deal_id=r["deal_id"],
                        inputs_fingerprint=r["inputs_fingerprint"],
                        plan_spec=_json_loads(r["plan_spec_json"]) or {},
                        outcome_status=r["outcome_status"],
                        artifacts=_json_loads(r["artifacts_json"]) or [],
                        user_edits=_json_loads(r["user_edits_json"]) or {},
                        summary_text=r["summary_text"],
                    )
                    matches.append(ActionSummaryMatch(summary=summary, score=1.0, reason="fts"))
                if matches:
                    return matches[: max(1, int(top_k))]
            except Exception:
                pass

        # Fallback: scan recent entries + token similarity.
        intent_tokens = _tokenize(intent)
        pool = self.list_recent(limit=max(10, int(search_limit)))
        scored: List[ActionSummaryMatch] = []
        for s in pool:
            if deal_id and s.deal_id and s.deal_id != deal_id:
                continue
            score = _jaccard(intent_tokens, _tokenize(s.summary_text))
            if score <= 0:
                continue
            scored.append(ActionSummaryMatch(summary=s, score=float(score), reason="jaccard"))
        scored.sort(key=lambda m: m.score, reverse=True)
        return scored[: max(1, int(top_k))]


def build_summary_from_action(
    *,
    action_id: str,
    action_type: str,
    deal_id: Optional[str],
    inputs: Dict[str, Any],
    plan_spec: Dict[str, Any],
    outcome_status: str,
    artifacts: List[Dict[str, Any]],
    user_edits: Optional[Dict[str, Any]] = None,
) -> ActionSummary:
    try:
        from tools.gateway import SecretRedactor

        safe_inputs = SecretRedactor.redact(inputs or {})
    except Exception:
        safe_inputs = inputs or {}
    memory_id = f"MEM-{action_id}"
    fp = fingerprint_inputs(inputs or {})
    summary_text = " ".join(
        [
            f"action_type={action_type}",
            f"deal_id={deal_id or ''}",
            f"outcome={outcome_status}",
            _json_dumps({"inputs": safe_inputs or {}, "artifacts": artifacts or []}),
        ]
    )
    return ActionSummary(
        memory_id=memory_id,
        created_at=_now_iso(),
        action_id=action_id,
        action_type=action_type,
        deal_id=deal_id,
        inputs_fingerprint=fp,
        plan_spec=plan_spec or {},
        outcome_status=outcome_status,
        artifacts=artifacts or [],
        user_edits=user_edits or {},
        summary_text=summary_text,
    )
