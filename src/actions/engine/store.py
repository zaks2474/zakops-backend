from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    ActionError,
    ActionPayload,
    ActionStep,
    ArtifactMetadata,
    AuditEvent,
    RunnerLease,
    StepStatus,
    default_runner_owner_id,
    json_dumps,
    json_loads,
    now_utc,
    now_utc_iso,
)


def _default_state_db_path() -> Path:
    return Path(os.getenv("ZAKOPS_STATE_DB", "/home/zaks/DataRoom/.deal-registry/ingest_state.db"))


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        if v.endswith("Z"):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return datetime.fromisoformat(v)
    except Exception:
        return None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS actions (
  action_id TEXT PRIMARY KEY,
  deal_id TEXT,
  capability_id TEXT,
  type TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  duration_seconds REAL,
  created_by TEXT NOT NULL,
  source TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  requires_human_review INTEGER NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  inputs TEXT NOT NULL,
  outputs TEXT,
  error TEXT,
  retry_count INTEGER NOT NULL DEFAULT 0,
  max_retries INTEGER NOT NULL DEFAULT 3,
  next_attempt_at TEXT,
  runner_lock_owner TEXT,
  runner_lock_expires_at TEXT,
  runner_heartbeat_at TEXT,
  cancelled_by TEXT,
  cancelled_at TEXT,
  cancel_reason TEXT,
  hidden_from_quarantine INTEGER NOT NULL DEFAULT 0,
  quarantine_hidden_at TEXT,
  quarantine_hidden_by TEXT,
  quarantine_hidden_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_actions_deal_id ON actions(deal_id);
CREATE INDEX IF NOT EXISTS idx_actions_status ON actions(status);
CREATE INDEX IF NOT EXISTS idx_actions_type ON actions(type);
CREATE INDEX IF NOT EXISTS idx_actions_created_at ON actions(created_at DESC);

CREATE TABLE IF NOT EXISTS action_audit_events (
  audit_id TEXT PRIMARY KEY,
  action_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  event TEXT NOT NULL,
  actor TEXT NOT NULL,
  details TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_action_id ON action_audit_events(action_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON action_audit_events(timestamp DESC);

CREATE TABLE IF NOT EXISTS action_artifacts (
  artifact_id TEXT PRIMARY KEY,
  action_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  path TEXT NOT NULL,
  size_bytes INTEGER NOT NULL DEFAULT 0,
  sha256 TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_artifacts_action_id ON action_artifacts(action_id);

CREATE TABLE IF NOT EXISTS action_runner_leases (
  runner_name TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  lease_expires_at TEXT NOT NULL,
  heartbeat_at TEXT NOT NULL,
  pid INTEGER NOT NULL,
  host TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS action_steps (
  step_id TEXT PRIMARY KEY,
  action_id TEXT NOT NULL,
  step_index INTEGER NOT NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT,
  ended_at TEXT,
  output_ref TEXT,
  error TEXT,
  requires_approval INTEGER NOT NULL DEFAULT 0,
  approved_by TEXT,
  approved_at TEXT,
  metadata TEXT,
  UNIQUE(action_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_steps_action_id ON action_steps(action_id);
CREATE INDEX IF NOT EXISTS idx_steps_status ON action_steps(status);
"""


class ActionStore:
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
        conn.execute("PRAGMA foreign_keys=OFF;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            self._ensure_quarantine_columns(conn)
            conn.commit()

    def _ensure_quarantine_columns(self, conn: sqlite3.Connection) -> None:
        existing = [row["name"] for row in conn.execute("PRAGMA table_info(actions)").fetchall()]
        columns = [
            ("hidden_from_quarantine", "hidden_from_quarantine INTEGER NOT NULL DEFAULT 0"),
            ("quarantine_hidden_at", "quarantine_hidden_at TEXT"),
            ("quarantine_hidden_by", "quarantine_hidden_by TEXT"),
            ("quarantine_hidden_reason", "quarantine_hidden_reason TEXT"),
        ]
        for name, definition in columns:
            if name not in existing:
                conn.execute(f"ALTER TABLE actions ADD COLUMN {definition}")

    @contextmanager
    def _tx(self) -> sqlite3.Connection:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _row_to_action(self, row: sqlite3.Row) -> ActionPayload:
        error_raw = row["error"]
        error = None
        if error_raw:
            try:
                error = ActionError.model_validate(json_loads(error_raw))
            except Exception:
                error = ActionError(code="unknown_error", message=str(error_raw), category="unknown", retryable=False)

        action = ActionPayload(
            action_id=row["action_id"],
            deal_id=row["deal_id"],
            capability_id=row["capability_id"],
            type=row["type"],
            title=row["title"],
            summary=row["summary"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            duration_seconds=row["duration_seconds"],
            created_by=row["created_by"],
            source=row["source"],
            risk_level=row["risk_level"],
            requires_human_review=bool(row["requires_human_review"]),
            idempotency_key=row["idempotency_key"],
            inputs=json_loads(row["inputs"]) or {},
            outputs=json_loads(row["outputs"]) or {},
            error=error,
            retry_count=int(row["retry_count"] or 0),
            max_retries=int(row["max_retries"] or 3),
            next_attempt_at=row["next_attempt_at"],
            runner_lock_owner=row["runner_lock_owner"],
            runner_lock_expires_at=row["runner_lock_expires_at"],
            runner_heartbeat_at=row["runner_heartbeat_at"],
        )
        action.hidden_from_quarantine = bool(row["hidden_from_quarantine"])
        action.quarantine_hidden_at = row["quarantine_hidden_at"]
        action.quarantine_hidden_by = row["quarantine_hidden_by"]
        action.quarantine_hidden_reason = row["quarantine_hidden_reason"]
        return action

    def _load_audit(self, conn: sqlite3.Connection, action_id: str) -> List[AuditEvent]:
        rows = conn.execute(
            "SELECT audit_id, timestamp, event, actor, details FROM action_audit_events WHERE action_id=? ORDER BY timestamp ASC",
            (action_id,),
        ).fetchall()
        out: List[AuditEvent] = []
        for r in rows:
            details = {}
            raw = r["details"]
            if raw:
                try:
                    details = json_loads(raw) or {}
                except Exception:
                    details = {"raw": raw}
            out.append(
                AuditEvent(
                    audit_id=r["audit_id"],
                    timestamp=r["timestamp"],
                    event=r["event"],
                    actor=r["actor"],
                    details=details,
                )
            )
        return out

    def _load_artifacts(self, conn: sqlite3.Connection, action_id: str) -> List[ArtifactMetadata]:
        rows = conn.execute(
            "SELECT artifact_id, filename, mime_type, path, size_bytes, sha256, created_at FROM action_artifacts WHERE action_id=? ORDER BY created_at ASC",
            (action_id,),
        ).fetchall()
        out: List[ArtifactMetadata] = []
        for r in rows:
            out.append(
                ArtifactMetadata(
                    artifact_id=r["artifact_id"],
                    filename=r["filename"],
                    mime_type=r["mime_type"],
                    path=r["path"],
                    size_bytes=int(r["size_bytes"] or 0),
                    sha256=r["sha256"],
                    created_at=r["created_at"],
                    download_url=None,
                )
            )
        return out

    def _load_steps(self, conn: sqlite3.Connection, action_id: str) -> List[ActionStep]:
        rows = conn.execute(
            """SELECT step_id, action_id, step_index, name, status, started_at, ended_at,
                      output_ref, error, requires_approval, approved_by, approved_at, metadata
               FROM action_steps WHERE action_id=? ORDER BY step_index ASC""",
            (action_id,),
        ).fetchall()
        out: List[ActionStep] = []
        for r in rows:
            error = None
            if r["error"]:
                try:
                    error = ActionError.model_validate(json_loads(r["error"]))
                except Exception:
                    error = ActionError(code="unknown_error", message=str(r["error"]), category="unknown", retryable=False)
            metadata = {}
            if r["metadata"]:
                try:
                    metadata = json_loads(r["metadata"]) or {}
                except Exception:
                    metadata = {}
            out.append(
                ActionStep(
                    step_id=r["step_id"],
                    action_id=r["action_id"],
                    step_index=int(r["step_index"]),
                    name=r["name"],
                    status=r["status"],
                    started_at=r["started_at"],
                    ended_at=r["ended_at"],
                    output_ref=r["output_ref"],
                    error=error,
                    requires_approval=bool(r["requires_approval"]),
                    approved_by=r["approved_by"],
                    approved_at=r["approved_at"],
                    metadata=metadata,
                )
            )
        return out

    def get_artifact(self, *, action_id: str, artifact_id: str) -> Optional[ArtifactMetadata]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT artifact_id, filename, mime_type, path, size_bytes, sha256, created_at FROM action_artifacts WHERE action_id=? AND artifact_id=?",
                (action_id, artifact_id),
            ).fetchone()
            if not row:
                return None
            return ArtifactMetadata(
                artifact_id=row["artifact_id"],
                filename=row["filename"],
                mime_type=row["mime_type"],
                path=row["path"],
                size_bytes=int(row["size_bytes"] or 0),
                sha256=row["sha256"],
                created_at=row["created_at"],
                download_url=None,
            )

    def create_action(self, action: ActionPayload) -> Tuple[ActionPayload, bool]:
        """
        Create action with idempotency.

        Returns: (action, created_new)
        """
        with self._tx() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO actions (
                      action_id, deal_id, capability_id, type, title, summary, status,
                      created_at, updated_at, started_at, completed_at, duration_seconds,
                      created_by, source, risk_level, requires_human_review,
                      idempotency_key, inputs, outputs, error, retry_count, max_retries, next_attempt_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        action.action_id,
                        action.deal_id,
                        action.capability_id,
                        action.type,
                        action.title,
                        action.summary or "",
                        action.status,
                        action.created_at,
                        action.updated_at,
                        action.started_at,
                        action.completed_at,
                        action.duration_seconds,
                        action.created_by,
                        action.source,
                        action.risk_level,
                        1 if action.requires_human_review else 0,
                        action.idempotency_key,
                        json_dumps(action.inputs or {}),
                        json_dumps(action.outputs or {}),
                        json_dumps(action.error.model_dump()) if action.error else None,
                        int(action.retry_count or 0),
                        int(action.max_retries or 3),
                        action.next_attempt_at,
                    ),
                )
                self.add_audit_event(conn, action.action_id, event="created", actor=action.created_by, details={
                    "source": action.source,
                    "type": action.type,
                    "capability_id": action.capability_id,
                })
                created = True
            except sqlite3.IntegrityError:
                # Idempotency hit — return existing action
                existing_row = conn.execute(
                    "SELECT * FROM actions WHERE idempotency_key=?",
                    (action.idempotency_key,),
                ).fetchone()
                if not existing_row:
                    raise
                action = self._row_to_action(existing_row)
                created = False

            # hydrate artifacts/audit
            action.audit_trail = self._load_audit(conn, action.action_id)
            action.artifacts = self._load_artifacts(conn, action.action_id)
            return action, created

    def get_action(self, action_id: str) -> Optional[ActionPayload]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if not row:
                return None
            action = self._row_to_action(row)
            action.audit_trail = self._load_audit(conn, action_id)
            action.artifacts = self._load_artifacts(conn, action_id)
            action.steps = self._load_steps(conn, action_id)
            return action

    def list_actions(
        self,
        *,
        deal_id: Optional[str] = None,
        status: Optional[str] = None,
        action_type: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        exclude_hidden: bool = True,
    ) -> List[ActionPayload]:
        where: List[str] = []
        params: List[Any] = []
        if deal_id:
            where.append("deal_id = ?")
            params.append(deal_id)
        if status:
            where.append("status = ?")
            params.append(status)
        if action_type:
            where.append("type = ?")
            params.append(action_type)
        if created_after:
            where.append("created_at >= ?")
            params.append(created_after)
        if created_before:
            where.append("created_at <= ?")
            params.append(created_before)
        if exclude_hidden:
            where.append("hidden_from_quarantine = 0")

        sql = "SELECT * FROM actions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([max(1, int(limit)), max(0, int(offset))])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            out: List[ActionPayload] = []
            for row in rows:
                action = self._row_to_action(row)
                # Lightweight list response: no audit/artifacts to keep it fast.
                out.append(action)
            return out

    def add_audit_event(
        self,
        conn: sqlite3.Connection,
        action_id: str,
        *,
        event: str,
        actor: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        audit = AuditEvent(event=event, actor=actor, details=details or {})
        conn.execute(
            "INSERT INTO action_audit_events (audit_id, action_id, timestamp, event, actor, details) VALUES (?, ?, ?, ?, ?, ?)",
            (audit.audit_id, action_id, audit.timestamp, audit.event, audit.actor, json_dumps(audit.details or {})),
        )

    def record_artifact(self, conn: sqlite3.Connection, action_id: str, artifact: ArtifactMetadata) -> None:
        conn.execute(
            """
            INSERT INTO action_artifacts (artifact_id, action_id, filename, mime_type, path, size_bytes, sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.artifact_id,
                action_id,
                artifact.filename,
                artifact.mime_type,
                artifact.path,
                int(artifact.size_bytes or 0),
                artifact.sha256,
                artifact.created_at,
            ),
        )

    def add_artifacts(self, *, action_id: str, artifacts: List[ArtifactMetadata]) -> None:
        if not artifacts:
            return
        with self._tx() as conn:
            for artifact in artifacts:
                self.record_artifact(conn, action_id, artifact)

    def update_action_inputs(self, action_id: str, inputs: Dict[str, Any], *, actor: str) -> ActionPayload:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if not row:
                raise KeyError("action_not_found")
            action = self._row_to_action(row)
            if action.status not in {"PENDING_APPROVAL", "READY"}:
                raise ValueError(f"action_inputs_not_editable_in_status:{action.status}")

            updated_at = now_utc_iso()
            conn.execute(
                "UPDATE actions SET inputs=?, updated_at=? WHERE action_id=?",
                (json_dumps(inputs or {}), updated_at, action_id),
            )
            self.add_audit_event(conn, action_id, event="inputs_updated", actor=actor, details={})
            action = self.get_action(action_id)
            if not action:
                raise KeyError("action_not_found")
            return action

    def approve_action(self, action_id: str, *, actor: str) -> ActionPayload:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if not row:
                raise KeyError("action_not_found")
            action = self._row_to_action(row)
            if action.status != "PENDING_APPROVAL":
                raise ValueError(f"invalid_transition:{action.status}->READY")
            updated_at = now_utc_iso()
            conn.execute(
                "UPDATE actions SET status='READY', updated_at=? WHERE action_id=?",
                (updated_at, action_id),
            )
            self.add_audit_event(conn, action_id, event="approved", actor=actor, details={})
        action = self.get_action(action_id)
        if not action:
            raise KeyError("action_not_found")
        return action

    def request_execute(self, action_id: str, *, actor: str) -> ActionPayload:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if not row:
                raise KeyError("action_not_found")
            action = self._row_to_action(row)
            if action.status != "READY":
                raise ValueError(f"invalid_transition:{action.status}->READY(queued)")

            # Queue semantics (v1.2 hardening):
            # - /execute does NOT set status=PROCESSING (runner does when it actually starts work)
            # - this prevents "stuck PROCESSING forever" when runner is down
            updated_at = now_utc_iso()
            conn.execute(
                """
                UPDATE actions
                SET updated_at=?,
                    error=NULL,
                    next_attempt_at=?,
                    runner_lock_owner=NULL,
                    runner_lock_expires_at=NULL,
                    runner_heartbeat_at=NULL
                WHERE action_id=? AND status='READY'
                """,
                (updated_at, updated_at, action_id),
            )
            self.add_audit_event(conn, action_id, event="execution_requested", actor=actor, details={"next_attempt_at": updated_at})
        action = self.get_action(action_id)
        if not action:
            raise KeyError("action_not_found")
        return action

    def cancel_action(self, action_id: str, *, actor: str, reason: str = "") -> ActionPayload:
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if not row:
                raise KeyError("action_not_found")
            action = self._row_to_action(row)
            if action.status not in {"PENDING_APPROVAL", "READY", "PROCESSING"}:
                raise ValueError(f"invalid_cancel_status:{action.status}")
            updated_at = now_utc_iso()
            conn.execute(
                """
                UPDATE actions
                SET status='CANCELLED',
                    updated_at=?,
                    cancelled_by=?,
                    cancelled_at=?,
                    cancel_reason=?,
                    runner_lock_owner=NULL,
                    runner_lock_expires_at=NULL,
                    runner_heartbeat_at=NULL
                WHERE action_id=?
                """,
                (updated_at, actor, updated_at, (reason or "")[:500], action_id),
            )
            self.add_audit_event(conn, action_id, event="cancelled", actor=actor, details={"reason": reason})
        action = self.get_action(action_id)
        if not action:
            raise KeyError("action_not_found")
        return action

    def hide_quarantine_item(self, action_id: str, *, actor: str, reason: Optional[str] = None) -> bool:
        with self._tx() as conn:
            row = conn.execute("SELECT hidden_from_quarantine FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if not row:
                return False
            if bool(row["hidden_from_quarantine"]):
                return False
            now = now_utc_iso()
            conn.execute(
                """
                UPDATE actions
                SET hidden_from_quarantine=1,
                    quarantine_hidden_at=?,
                    quarantine_hidden_by=?,
                    quarantine_hidden_reason=?,
                    updated_at=?
                WHERE action_id=?
                """,
                (now, actor, (reason or "")[:500], now, action_id),
            )
            self.add_audit_event(
                conn,
                action_id,
                event="hidden_from_quarantine",
                actor=actor,
                details={"reason": reason or "deleted_from_quarantine"},
            )
        return True

    # ---------------------------------------------------------------------
    # Runner lease + per-action lock
    # ---------------------------------------------------------------------

    def acquire_runner_lease(
        self,
        *,
        runner_name: str = "kinetic_actions",
        owner_id: Optional[str] = None,
        lease_seconds: int = 30,
        pid: Optional[int] = None,
        host: Optional[str] = None,
    ) -> bool:
        owner_id = owner_id or default_runner_owner_id()
        pid = pid if pid is not None else os.getpid()
        host = host or os.uname().nodename
        now = now_utc()
        expires = now + timedelta(seconds=max(5, int(lease_seconds)))

        now_iso = now.isoformat().replace("+00:00", "Z")
        exp_iso = expires.isoformat().replace("+00:00", "Z")

        with self._tx() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO action_runner_leases (runner_name, owner_id, lease_expires_at, heartbeat_at, pid, host)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (runner_name, owner_id, exp_iso, now_iso, int(pid), str(host)),
                )
                return True
            except sqlite3.IntegrityError:
                # Update only if expired or already owned by us
                cur = conn.execute(
                    """
                    UPDATE action_runner_leases
                    SET owner_id=?, lease_expires_at=?, heartbeat_at=?, pid=?, host=?
                    WHERE runner_name=? AND (lease_expires_at < ? OR owner_id=?)
                    """,
                    (owner_id, exp_iso, now_iso, int(pid), str(host), runner_name, now_iso, owner_id),
                )
                return cur.rowcount == 1

    def heartbeat_runner_lease(
        self,
        *,
        runner_name: str,
        owner_id: str,
        lease_seconds: int = 30,
        pid: Optional[int] = None,
        host: Optional[str] = None,
    ) -> bool:
        pid = pid if pid is not None else os.getpid()
        host = host or os.uname().nodename
        now = now_utc()
        expires = now + timedelta(seconds=max(5, int(lease_seconds)))
        now_iso = now.isoformat().replace("+00:00", "Z")
        exp_iso = expires.isoformat().replace("+00:00", "Z")
        with self._tx() as conn:
            cur = conn.execute(
                """
                UPDATE action_runner_leases
                SET lease_expires_at=?, heartbeat_at=?, pid=?, host=?
                WHERE runner_name=? AND owner_id=?
                """,
                (exp_iso, now_iso, int(pid), str(host), runner_name, owner_id),
            )
            return cur.rowcount == 1

    def release_runner_lease(self, *, runner_name: str, owner_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE action_runner_leases SET lease_expires_at=? WHERE runner_name=? AND owner_id=?",
                (now_utc_iso(), runner_name, owner_id),
            )

    def get_runner_lease(self, *, runner_name: str) -> Optional[RunnerLease]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM action_runner_leases WHERE runner_name=?", (runner_name,)).fetchone()
            if not row:
                return None
            return RunnerLease(
                runner_name=row["runner_name"],
                owner_id=row["owner_id"],
                lease_expires_at=row["lease_expires_at"],
                heartbeat_at=row["heartbeat_at"],
                pid=int(row["pid"]),
                host=row["host"],
            )

    def claim_action_lock(
        self,
        *,
        action_id: str,
        owner_id: str,
        lease_seconds: int = 120,
    ) -> bool:
        now_iso = now_utc_iso()
        exp_iso = (now_utc() + timedelta(seconds=max(10, int(lease_seconds)))).isoformat().replace("+00:00", "Z")
        with self._tx() as conn:
            cur = conn.execute(
                """
                UPDATE actions
                SET runner_lock_owner=?,
                    runner_lock_expires_at=?,
                    runner_heartbeat_at=?
                WHERE action_id=?
                  AND status='PROCESSING'
                  AND (runner_lock_expires_at IS NULL OR runner_lock_expires_at < ? OR runner_lock_owner=?)
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                """,
                (owner_id, exp_iso, now_iso, action_id, now_iso, owner_id, now_iso),
            )
            return cur.rowcount == 1

    def heartbeat_action_lock(self, *, action_id: str, owner_id: str, lease_seconds: int = 120) -> bool:
        now_iso = now_utc_iso()
        exp_iso = (now_utc() + timedelta(seconds=max(10, int(lease_seconds)))).isoformat().replace("+00:00", "Z")
        with self._tx() as conn:
            cur = conn.execute(
                """
                UPDATE actions
                SET runner_lock_expires_at=?,
                    runner_heartbeat_at=?
                WHERE action_id=? AND runner_lock_owner=?
                """,
                (exp_iso, now_iso, action_id, owner_id),
            )
            return cur.rowcount == 1

    def release_action_lock(self, *, action_id: str, owner_id: str) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE actions
                SET runner_lock_owner=NULL,
                    runner_lock_expires_at=NULL,
                    runner_heartbeat_at=NULL
                WHERE action_id=? AND runner_lock_owner=?
                """,
                (action_id, owner_id),
            )

    def get_next_due_processing_action_id(self) -> Optional[str]:
        """
        Return an action_id that is PROCESSING and claimable (lock expired / not held).

        This is used by the runner to find work.
        """
        now_iso = now_utc_iso()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT action_id
                FROM actions
                WHERE status='PROCESSING'
                  AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                  AND (runner_lock_expires_at IS NULL OR runner_lock_expires_at < ?)
                ORDER BY updated_at ASC
                LIMIT 1
                """,
                (now_iso, now_iso),
            ).fetchone()
            return row["action_id"] if row else None

    def get_next_due_action_id(self) -> Optional[str]:
        """
        Return the next action_id that is due for execution.

        Due actions:
        - READY with next_attempt_at <= now (explicitly queued via /execute or retry scheduling)
        - PROCESSING that is claimable (legacy /execute semantics or takeover)
        """
        now_iso = now_utc_iso()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT action_id
                FROM actions
                WHERE (
                    (status='READY' AND next_attempt_at IS NOT NULL AND next_attempt_at <= ?)
                    OR
                    (status='PROCESSING' AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
                )
                  AND (runner_lock_expires_at IS NULL OR runner_lock_expires_at < ?)
                ORDER BY
                  CASE status WHEN 'READY' THEN 0 ELSE 1 END,
                  COALESCE(next_attempt_at, updated_at) ASC
                LIMIT 1
                """,
                (now_iso, now_iso, now_iso),
            ).fetchone()
            return row["action_id"] if row else None

    def begin_processing(
        self,
        *,
        action_id: str,
        owner_id: str,
        lease_seconds: int = 120,
        actor: str = "actions_runner",
    ) -> bool:
        """
        Atomically transition READY → PROCESSING and claim per-action lock.

        For already PROCESSING actions (legacy), this claims the lock and ensures started_at is set.
        """
        now = now_utc()
        now_iso = now.isoformat().replace("+00:00", "Z")
        exp_iso = (now + timedelta(seconds=max(10, int(lease_seconds)))).isoformat().replace("+00:00", "Z")
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if not row:
                return False
            action = self._row_to_action(row)

            # READY: must be explicitly queued.
            if action.status == "READY":
                if action.next_attempt_at and action.next_attempt_at > now_iso:
                    return False
                cur = conn.execute(
                    """
                    UPDATE actions
                    SET status='PROCESSING',
                        updated_at=?,
                        started_at=COALESCE(started_at, ?),
                        error=NULL,
                        next_attempt_at=NULL,
                        runner_lock_owner=?,
                        runner_lock_expires_at=?,
                        runner_heartbeat_at=?
                    WHERE action_id=?
                      AND status='READY'
                      AND (runner_lock_expires_at IS NULL OR runner_lock_expires_at < ? OR runner_lock_owner=?)
                      AND (next_attempt_at IS NOT NULL AND next_attempt_at <= ?)
                    """,
                    (
                        now_iso,
                        now_iso,
                        owner_id,
                        exp_iso,
                        now_iso,
                        action_id,
                        now_iso,
                        owner_id,
                        now_iso,
                    ),
                )
                if cur.rowcount != 1:
                    return False
                self.add_audit_event(conn, action_id, event="started", actor=actor, details={"owner_id": owner_id})
                return True

            # PROCESSING: takeover / resume (legacy).
            if action.status == "PROCESSING":
                cur = conn.execute(
                    """
                    UPDATE actions
                    SET updated_at=?,
                        started_at=COALESCE(started_at, ?),
                        runner_lock_owner=?,
                        runner_lock_expires_at=?,
                        runner_heartbeat_at=?
                    WHERE action_id=?
                      AND status='PROCESSING'
                      AND (runner_lock_expires_at IS NULL OR runner_lock_expires_at < ? OR runner_lock_owner=?)
                      AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                    """,
                    (
                        now_iso,
                        now_iso,
                        owner_id,
                        exp_iso,
                        now_iso,
                        action_id,
                        now_iso,
                        owner_id,
                        now_iso,
                    ),
                )
                return cur.rowcount == 1

            return False

    def list_stuck_processing_action_ids(self, *, older_than_seconds: int = 300, limit: int = 100) -> List[str]:
        """
        Return PROCESSING action_ids that appear stuck (lock held, heartbeat stale).

        This is intended for operator tooling (`make actions-retry-stuck`) and should be used
        conservatively alongside the per-action heartbeat in the runner.
        """
        now = now_utc()
        cutoff = (now - timedelta(seconds=max(1, int(older_than_seconds)))).isoformat().replace("+00:00", "Z")
        now_iso = now.isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT action_id
                FROM actions
                WHERE status='PROCESSING'
                  AND runner_lock_owner IS NOT NULL
                  AND runner_lock_expires_at IS NOT NULL
                  AND runner_lock_expires_at > ?
                  AND runner_heartbeat_at IS NOT NULL
                  AND runner_heartbeat_at < ?
                ORDER BY runner_heartbeat_at ASC
                LIMIT ?
                """,
                (now_iso, cutoff, max(1, int(limit))),
            ).fetchall()
            return [r["action_id"] for r in rows]

    def unstick_action(
        self,
        *,
        action_id: str,
        actor: str,
        reason: str = "unstick",
    ) -> bool:
        """
        Operator action: release per-action lock early and make the action immediately claimable.

        Returns True if an update was applied.
        """
        now_iso = now_utc_iso()
        with self._tx() as conn:
            cur = conn.execute(
                """
                UPDATE actions
                SET runner_lock_owner=NULL,
                    runner_lock_expires_at=NULL,
                    runner_heartbeat_at=NULL,
                    next_attempt_at=?,
                    status='READY'
                WHERE action_id=? AND status='PROCESSING'
                """,
                (now_iso, action_id),
            )
            if cur.rowcount != 1:
                return False
            self.add_audit_event(conn, action_id, event="unstuck", actor=actor, details={"reason": reason})
            return True

    def mark_action_completed(
        self,
        *,
        action_id: str,
        actor: str,
        outputs: Dict[str, Any],
        error: Optional[ActionError] = None,
    ) -> ActionPayload:
        now = now_utc()
        now_iso = now.isoformat().replace("+00:00", "Z")
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if not row:
                raise KeyError("action_not_found")
            action = self._row_to_action(row)

            started = _parse_iso(action.started_at) or now
            duration = max(0.0, (now - started).total_seconds())

            status = "COMPLETED" if error is None else "FAILED"
            conn.execute(
                """
                UPDATE actions
                SET status=?,
                    updated_at=?,
                    completed_at=?,
                    duration_seconds=?,
                    outputs=?,
                    error=?,
                    runner_lock_owner=NULL,
                    runner_lock_expires_at=NULL,
                    runner_heartbeat_at=NULL
                WHERE action_id=?
                """,
                (
                    status,
                    now_iso,
                    now_iso,
                    duration,
                    json_dumps(outputs or {}),
                    json_dumps(error.model_dump()) if error else None,
                    action_id,
                ),
            )
            self.add_audit_event(
                conn,
                action_id,
                event="completed" if error is None else "failed",
                actor=actor,
                details={"status": status, "error_code": error.code if error else None},
            )
        updated = self.get_action(action_id)
        if not updated:
            raise KeyError("action_not_found")
        return updated

    def mark_action_retry(
        self,
        *,
        action_id: str,
        actor: str,
        error: ActionError,
        retry_count: int,
        next_attempt_at: str,
    ) -> ActionPayload:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE actions
                SET status='READY',
                    updated_at=?,
                    error=?,
                    retry_count=?,
                    next_attempt_at=?,
                    runner_lock_owner=NULL,
                    runner_lock_expires_at=NULL,
                    runner_heartbeat_at=NULL
                WHERE action_id=?
                """,
                (now_utc_iso(), json_dumps(error.model_dump()), int(retry_count), next_attempt_at, action_id),
            )
            self.add_audit_event(
                conn,
                action_id,
                event="retry_scheduled",
                actor=actor,
                details={"retry_count": retry_count, "next_attempt_at": next_attempt_at, "error_code": error.code},
            )
        updated = self.get_action(action_id)
        if not updated:
            raise KeyError("action_not_found")
        return updated

    def mark_processing_timeouts(
        self,
        *,
        older_than_seconds: int,
        actor: str = "watchdog",
        limit: int = 100,
    ) -> int:
        """
        Watchdog: mark PROCESSING actions as FAILED if they exceed a TTL.

        This guarantees no action remains PROCESSING forever.
        """
        now = now_utc()
        cutoff = (now - timedelta(seconds=max(1, int(older_than_seconds)))).isoformat().replace("+00:00", "Z")
        now_iso = now.isoformat().replace("+00:00", "Z")
        to_fail: List[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT action_id
                FROM actions
                WHERE status='PROCESSING'
                  AND COALESCE(started_at, updated_at) < ?
                ORDER BY COALESCE(started_at, updated_at) ASC
                LIMIT ?
                """,
                (cutoff, max(1, int(limit))),
            ).fetchall()
            to_fail = [r["action_id"] for r in rows]

        failed = 0
        for action_id in to_fail:
            err = ActionError(
                code="processing_timeout",
                message=f"Action exceeded processing TTL ({int(older_than_seconds)}s)",
                category="unknown",
                retryable=True,
                details={"processing_ttl_seconds": int(older_than_seconds)},
            )
            with self._tx() as conn:
                cur = conn.execute(
                    """
                    UPDATE actions
                    SET status='FAILED',
                        updated_at=?,
                        completed_at=?,
                        duration_seconds=COALESCE(duration_seconds, 0),
                        error=?,
                        runner_lock_owner=NULL,
                        runner_lock_expires_at=NULL,
                        runner_heartbeat_at=NULL
                    WHERE action_id=? AND status='PROCESSING'
                    """,
                    (now_iso, now_iso, json_dumps(err.model_dump()), action_id),
                )
                if cur.rowcount == 1:
                    self.add_audit_event(conn, action_id, event="failed_timeout", actor=actor, details={"cutoff": cutoff})
                    failed += 1

        return failed

    def requeue_failed_action(self, *, action_id: str, actor: str, reason: str = "requeue") -> ActionPayload:
        """
        Admin operation: move FAILED → READY and enqueue immediately.
        """
        now_iso = now_utc_iso()
        with self._tx() as conn:
            row = conn.execute("SELECT * FROM actions WHERE action_id=?", (action_id,)).fetchone()
            if not row:
                raise KeyError("action_not_found")
            action = self._row_to_action(row)
            if action.status != "FAILED":
                raise ValueError(f"invalid_transition:{action.status}->READY(requeue)")

            prev_err = action.error.model_dump() if action.error else None
            conn.execute(
                """
                UPDATE actions
                SET status='READY',
                    updated_at=?,
                    error=NULL,
                    retry_count=0,
                    next_attempt_at=?,
                    runner_lock_owner=NULL,
                    runner_lock_expires_at=NULL,
                    runner_heartbeat_at=NULL
                WHERE action_id=? AND status='FAILED'
                """,
                (now_iso, now_iso, action_id),
            )
            self.add_audit_event(
                conn,
                action_id,
                event="requeued",
                actor=actor,
                details={"reason": reason, "previous_error": prev_err},
            )

        updated = self.get_action(action_id)
        if not updated:
            raise KeyError("action_not_found")
        return updated

    def action_metrics(self, *, window_hours: int = 24) -> Dict[str, Any]:
        now = now_utc()
        cutoff = (now - timedelta(hours=max(1, int(window_hours)))).isoformat().replace("+00:00", "Z")
        now_iso = now.isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM actions").fetchone()["c"]
            pending = conn.execute("SELECT COUNT(*) AS c FROM actions WHERE status='PENDING_APPROVAL'").fetchone()["c"]
            ready = conn.execute("SELECT COUNT(*) AS c FROM actions WHERE status='READY'").fetchone()["c"]
            ready_queued = conn.execute(
                "SELECT COUNT(*) AS c FROM actions WHERE status='READY' AND next_attempt_at IS NOT NULL AND next_attempt_at <= ?",
                (now_iso,),
            ).fetchone()["c"]
            processing = conn.execute("SELECT COUNT(*) AS c FROM actions WHERE status='PROCESSING'").fetchone()["c"]
            completed_24h = conn.execute(
                "SELECT COUNT(*) AS c FROM actions WHERE status='COMPLETED' AND completed_at >= ?",
                (cutoff,),
            ).fetchone()["c"]
            failed_24h = conn.execute(
                "SELECT COUNT(*) AS c FROM actions WHERE status='FAILED' AND updated_at >= ?",
                (cutoff,),
            ).fetchone()["c"]
            avg_duration = conn.execute(
                "SELECT AVG(duration_seconds) AS avg_d FROM actions WHERE status='COMPLETED' AND completed_at >= ?",
                (cutoff,),
            ).fetchone()["avg_d"]

        success_rate = None
        denom = int(completed_24h) + int(failed_24h)
        if denom > 0:
            success_rate = float(completed_24h) / float(denom)

        return {
            "total": int(total),
            "queue": {
                "pending_approval": int(pending),
                "ready": int(ready),
                "ready_queued": int(ready_queued),
                "processing": int(processing),
            },
            "window_hours": int(window_hours),
            "completed": int(completed_24h),
            "failed": int(failed_24h),
            "success_rate": success_rate,
            "avg_duration_seconds": float(avg_duration) if avg_duration is not None else None,
        }

    # =========================================================================
    # Step Management Methods
    # =========================================================================

    def create_step(
        self,
        action_id: str,
        name: str,
        step_index: int,
        requires_approval: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionStep:
        """
        Create a new step for an action.
        """
        from .models import safe_uuid

        step = ActionStep(
            step_id=safe_uuid(),
            action_id=action_id,
            step_index=step_index,
            name=name,
            status="PENDING",
            requires_approval=requires_approval,
            metadata=metadata or {},
        )
        with self._tx() as conn:
            conn.execute(
                """INSERT INTO action_steps
                   (step_id, action_id, step_index, name, status, requires_approval, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    step.step_id,
                    step.action_id,
                    step.step_index,
                    step.name,
                    step.status,
                    1 if step.requires_approval else 0,
                    json_dumps(step.metadata),
                ),
            )
        return step

    def create_steps_for_action(
        self,
        action_id: str,
        step_definitions: List[Dict[str, Any]],
    ) -> List[ActionStep]:
        """
        Bulk create steps for an action.

        step_definitions: [{"name": "gather_context", "requires_approval": False}, ...]
        """
        steps = []
        for idx, defn in enumerate(step_definitions):
            step = self.create_step(
                action_id=action_id,
                name=defn.get("name", f"step_{idx}"),
                step_index=idx,
                requires_approval=defn.get("requires_approval", False),
                metadata=defn.get("metadata"),
            )
            steps.append(step)
        return steps

    def update_step_status(
        self,
        step_id: str,
        status: StepStatus,
        output_ref: Optional[str] = None,
        error: Optional[ActionError] = None,
    ) -> ActionStep:
        """
        Update a step's status, output_ref, and/or error.
        """
        now_iso = now_utc_iso()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM action_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"step_not_found:{step_id}")

            updates = ["status=?"]
            params: List[Any] = [status]

            if status == "IN_PROGRESS" and row["started_at"] is None:
                updates.append("started_at=?")
                params.append(now_iso)

            if status in ("COMPLETED", "FAILED", "SKIPPED"):
                updates.append("ended_at=?")
                params.append(now_iso)

            if output_ref is not None:
                updates.append("output_ref=?")
                params.append(output_ref)

            if error is not None:
                updates.append("error=?")
                params.append(json_dumps(error.model_dump()))

            params.append(step_id)
            conn.execute(
                f"UPDATE action_steps SET {', '.join(updates)} WHERE step_id=?",
                tuple(params),
            )

        # Return the updated step
        with self._connect() as conn:
            updated_row = conn.execute(
                "SELECT * FROM action_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if not updated_row:
                raise KeyError(f"step_not_found:{step_id}")

            err_obj = None
            if updated_row["error"]:
                try:
                    err_obj = ActionError.model_validate(json_loads(updated_row["error"]))
                except Exception:
                    pass

            return ActionStep(
                step_id=updated_row["step_id"],
                action_id=updated_row["action_id"],
                step_index=int(updated_row["step_index"]),
                name=updated_row["name"],
                status=updated_row["status"],
                started_at=updated_row["started_at"],
                ended_at=updated_row["ended_at"],
                output_ref=updated_row["output_ref"],
                error=err_obj,
                requires_approval=bool(updated_row["requires_approval"]),
                approved_by=updated_row["approved_by"],
                approved_at=updated_row["approved_at"],
                metadata=json_loads(updated_row["metadata"]) or {},
            )

    def approve_step(
        self,
        step_id: str,
        actor: str,
    ) -> ActionStep:
        """
        Approve a step that requires approval before execution.
        """
        now_iso = now_utc_iso()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT * FROM action_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"step_not_found:{step_id}")

            if not row["requires_approval"]:
                raise ValueError("step_does_not_require_approval")

            if row["approved_by"] is not None:
                raise ValueError("step_already_approved")

            conn.execute(
                "UPDATE action_steps SET approved_by=?, approved_at=? WHERE step_id=?",
                (actor, now_iso, step_id),
            )

            # Also add audit event to parent action
            self.add_audit_event(
                conn,
                row["action_id"],
                event="step_approved",
                actor=actor,
                details={"step_id": step_id, "step_name": row["name"]},
            )

        return self.get_step(step_id)  # type: ignore

    def get_step(self, step_id: str) -> Optional[ActionStep]:
        """Get a single step by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM action_steps WHERE step_id=?", (step_id,)
            ).fetchone()
            if not row:
                return None

            err_obj = None
            if row["error"]:
                try:
                    err_obj = ActionError.model_validate(json_loads(row["error"]))
                except Exception:
                    pass

            return ActionStep(
                step_id=row["step_id"],
                action_id=row["action_id"],
                step_index=int(row["step_index"]),
                name=row["name"],
                status=row["status"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                output_ref=row["output_ref"],
                error=err_obj,
                requires_approval=bool(row["requires_approval"]),
                approved_by=row["approved_by"],
                approved_at=row["approved_at"],
                metadata=json_loads(row["metadata"]) or {},
            )

    def get_next_pending_step(self, action_id: str) -> Optional[ActionStep]:
        """
        Get the next PENDING step for an action that is ready to execute.
        Returns None if no steps are pending or if next step requires approval but isn't approved.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM action_steps
                   WHERE action_id=? AND status='PENDING'
                   ORDER BY step_index ASC LIMIT 1""",
                (action_id,),
            ).fetchone()
            if not row:
                return None

            # If step requires approval but isn't approved yet, return None
            if row["requires_approval"] and row["approved_by"] is None:
                return None

            err_obj = None
            if row["error"]:
                try:
                    err_obj = ActionError.model_validate(json_loads(row["error"]))
                except Exception:
                    pass

            return ActionStep(
                step_id=row["step_id"],
                action_id=row["action_id"],
                step_index=int(row["step_index"]),
                name=row["name"],
                status=row["status"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                output_ref=row["output_ref"],
                error=err_obj,
                requires_approval=bool(row["requires_approval"]),
                approved_by=row["approved_by"],
                approved_at=row["approved_at"],
                metadata=json_loads(row["metadata"]) or {},
            )

    def get_step_awaiting_approval(self, action_id: str) -> Optional[ActionStep]:
        """
        Get a step that is PENDING and requires_approval but not yet approved.
        Returns None if no such step exists.
        """
        with self._connect() as conn:
            row = conn.execute(
                """SELECT * FROM action_steps
                   WHERE action_id=? AND status='PENDING'
                     AND requires_approval=1 AND approved_by IS NULL
                   ORDER BY step_index ASC LIMIT 1""",
                (action_id,),
            ).fetchone()
            if not row:
                return None

            return ActionStep(
                step_id=row["step_id"],
                action_id=row["action_id"],
                step_index=int(row["step_index"]),
                name=row["name"],
                status=row["status"],
                started_at=row["started_at"],
                ended_at=row["ended_at"],
                output_ref=row["output_ref"],
                error=None,
                requires_approval=bool(row["requires_approval"]),
                approved_by=row["approved_by"],
                approved_at=row["approved_at"],
                metadata=json_loads(row["metadata"]) or {},
            )
