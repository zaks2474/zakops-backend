#!/usr/bin/env python3
"""
Kinetic Action Engine Runner (v1.2)

Responsibilities:
- Acquire a single global runner lease (crash-safe, takeover-capable)
- Claim per-action execution locks
- Execute actions via the ActionExecutor registry
- Persist outputs, artifacts, audit events, retries with backoff

Notes:
- This runner does NOT require LangSmith.
- This runner does NOT execute irreversible comms (email send). Draft-only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import os
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from actions.engine.models import ActionError, ActionPayload, compute_idempotency_key, default_runner_owner_id, now_utc, now_utc_iso
from actions.engine.store import ActionStore
from actions.executors.base import ActionExecutionError, ExecutionContext
from actions.executors.registry import get_executor, load_builtin_executors
from deal_events import DealEventStore
from deal_registry import DealRegistry

try:
    from email_ingestion.run_ledger import run_context, generate_run_id
except Exception:  # pragma: no cover
    run_context = None  # type: ignore
    generate_run_id = None  # type: ignore


logger = logging.getLogger(__name__)

DATAROOM_ROOT = Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()
REGISTRY_PATH = str(DATAROOM_ROOT / ".deal-registry" / "deal_registry.json")
CASE_FILES_DIR = DATAROOM_ROOT / ".deal-registry" / "case_files"


def _load_case_file(deal_id: str) -> Optional[Dict[str, Any]]:
    path = CASE_FILES_DIR / f"{deal_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _next_backoff_seconds(*, retry_count: int, base_seconds: int = 5, max_seconds: int = 300) -> int:
    # retry_count starts at 1 for the first retry
    delay = base_seconds * (2 ** max(0, retry_count - 1))
    return int(min(max_seconds, delay))


def _emit_deal_event(event_store: DealEventStore, *, deal_id: Optional[str], event_type: str, actor: str, data: Dict[str, Any]) -> None:
    if not deal_id:
        return
    try:
        event_store.create_event(deal_id=deal_id, event_type=event_type, actor=actor, data=data)
    except Exception:
        logger.exception("Failed to emit deal event: %s", event_type)


def _enqueue_follow_on_actions(
    *,
    store: ActionStore,
    registry: DealRegistry,
    event_store: DealEventStore,
    parent_action: ActionPayload,
    outputs: Dict[str, Any],
    max_actions: int = 5,
    max_depth: int = 3,
) -> None:
    """
    Enqueue follow-on Kinetic Actions described in `outputs.next_actions[]`.

    Safety:
    - Max chain depth to avoid runaway recursion.
    - Deduplicated by idempotency_key (ActionStore unique constraint).
    - Auto-approve + queue only when requires_approval=false.
    """
    if not isinstance(outputs, dict):
        return

    specs = outputs.get("next_actions")
    if not isinstance(specs, list) or not specs:
        return

    try:
        parent_inputs = parent_action.inputs or {}
    except Exception:
        parent_inputs = {}

    try:
        parent_depth = int(parent_inputs.get("_chain_depth") or 0)
    except Exception:
        parent_depth = 0

    if parent_depth >= max(0, int(max_depth)):
        return

    root_action_id = str(parent_inputs.get("_chain_root_action_id") or parent_action.action_id or "").strip() or parent_action.action_id

    # Capability registry is used for defaults (requires_approval, risk_level) when omitted.
    try:
        from actions.capabilities.registry import get_registry as get_capability_registry

        cap_reg = get_capability_registry()
    except Exception:
        cap_reg = None

    created_count = 0

    for raw in specs[: max(1, int(max_actions))]:
        if not isinstance(raw, dict):
            continue

        action_type = str(raw.get("action_type") or "").strip()
        if not action_type:
            continue

        spec_inputs = raw.get("inputs") if isinstance(raw.get("inputs"), dict) else {}
        child_inputs = dict(spec_inputs or {})
        child_inputs.setdefault("_chain_parent_action_id", parent_action.action_id)
        child_inputs.setdefault("_chain_root_action_id", root_action_id)
        child_inputs.setdefault("_chain_depth", parent_depth + 1)

        deal_id = str(raw.get("deal_id") or child_inputs.get("deal_id") or parent_action.deal_id or "").strip() or None

        capability_id = str(raw.get("capability_id") or "").strip() or None
        title = str(raw.get("title") or action_type).strip()[:200] or action_type
        summary = str(raw.get("summary") or f"Follow-on from {parent_action.action_id}").strip()[:500]

        requires_approval = bool(raw.get("requires_approval")) if "requires_approval" in raw else None
        risk_level = str(raw.get("risk_level") or "").strip().lower() or None

        if cap_reg is not None:
            try:
                manifest = cap_reg.get_capability(capability_id) if capability_id else cap_reg.get_by_action_type(action_type)
            except Exception:
                manifest = None
            if manifest is not None:
                if not capability_id:
                    capability_id = str(manifest.capability_id)
                if requires_approval is None:
                    requires_approval = bool(getattr(manifest, "requires_approval", True))
                if not risk_level:
                    risk_level = str(getattr(manifest, "risk_level", "medium") or "medium").strip().lower()

        if requires_approval is None:
            requires_approval = True
        if not risk_level:
            risk_level = "medium"
        if risk_level not in {"low", "medium", "high"}:
            risk_level = "medium"

        idem = str(raw.get("idempotency_key") or "").strip()
        if not idem:
            idem = compute_idempotency_key(parent_action.action_id, action_type, json.dumps(child_inputs, sort_keys=True))

        child = ActionPayload(
            deal_id=deal_id,
            capability_id=capability_id,
            type=action_type,
            title=title,
            summary=summary,
            status="PENDING_APPROVAL",
            created_by="actions_runner",
            source="system",
            risk_level=risk_level,  # type: ignore[arg-type]
            requires_human_review=bool(requires_approval),
            idempotency_key=idem,
            inputs=child_inputs,
        )

        try:
            created_action, created_new = store.create_action(child)
        except Exception:
            logger.exception("Failed to enqueue follow-on action: %s", action_type)
            continue

        if created_new:
            created_count += 1
            _emit_deal_event(
                event_store,
                deal_id=created_action.deal_id,
                event_type="kinetic_action_created",
                actor="actions_runner",
                data={
                    "action_id": created_action.action_id,
                    "action_type": created_action.type,
                    "created_new": True,
                    "capability_id": created_action.capability_id,
                    "chain_parent_action_id": parent_action.action_id,
                },
            )

        # Auto-approve + queue when the follow-on action is explicitly safe to run without approval.
        if created_new and not bool(requires_approval):
            try:
                store.approve_action(created_action.action_id, actor="actions_runner")
                store.request_execute(created_action.action_id, actor="actions_runner")
            except Exception:
                logger.exception("Failed to auto-approve/queue follow-on action: %s", created_action.action_id)

    if created_count:
        logger.info("Enqueued %d follow-on actions for %s", created_count, parent_action.action_id)


def process_one_action(
    *,
    store: ActionStore,
    registry: DealRegistry,
    event_store: DealEventStore,
    owner_id: str,
    action_id: str,
    action_lock_seconds: int,
) -> None:
    if not store.begin_processing(action_id=action_id, owner_id=owner_id, lease_seconds=action_lock_seconds):
        return

    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        interval = max(5.0, min(30.0, float(action_lock_seconds) / 3.0))
        while not stop_heartbeat.wait(interval):
            try:
                store.heartbeat_action_lock(action_id=action_id, owner_id=owner_id, lease_seconds=action_lock_seconds)
            except Exception:
                logger.exception("Failed to heartbeat action lock: %s", action_id)

    heartbeat_thread = threading.Thread(target=_heartbeat, name=f"action-heartbeat-{action_id}", daemon=True)
    heartbeat_thread.start()

    try:
        action = store.get_action(action_id)
        if not action:
            return
        if action.status != "PROCESSING":
            return

        deal = registry.get_deal(action.deal_id) if action.deal_id else None
        case_file = _load_case_file(action.deal_id) if action.deal_id else None

        _emit_deal_event(
            event_store,
            deal_id=action.deal_id,
            event_type="kinetic_action_started",
            actor="actions_runner",
            data={"action_id": action.action_id, "action_type": action.type, "retry_count": action.retry_count},
        )

        # Run-ledger wrapper (optional, used when available)
        if run_context and generate_run_id:
            run_id = generate_run_id(prefix="ACTION")
            correlation = {"action_id": action.action_id, "deal_id": action.deal_id or ""}
            ctx_mgr = run_context(run_id, component="kinetic_action_runner", correlation=correlation)
        else:
            ctx_mgr = None

        def _execute() -> None:
            # Timing diagnostics
            timings: Dict[str, float] = {}
            exec_start = time.time()

            # Step 1: Get executor
            t0 = time.time()
            executor = get_executor(action.type)
            timings["get_executor_ms"] = (time.time() - t0) * 1000
            if executor is None:
                raise ActionExecutionError(
                    ActionError(
                        code="executor_not_found",
                        message=f"No executor registered for action type: {action.type}",
                        category="dependency",
                        retryable=False,
                        details={"action_type": action.type},
                    )
                )

            # Step 2: Validate
            t0 = time.time()
            ok, err = executor.validate(action)
            timings["validate_ms"] = (time.time() - t0) * 1000
            if not ok:
                raise ActionExecutionError(
                    ActionError(
                        code="validation_failed",
                        message=err or "Validation failed",
                        category="validation",
                        retryable=False,
                    )
                )

            # Step 3: Capability lookup
            t0 = time.time()
            cloud_allowed = False
            try:
                from actions.capabilities.registry import get_registry as get_capability_registry

                cap_reg = get_capability_registry()
                manifest = None
                if action.capability_id:
                    manifest = cap_reg.get_capability(action.capability_id)
                if manifest is None:
                    manifest = cap_reg.get_by_action_type(action.type)
                cloud_allowed = bool(getattr(manifest, "cloud_required", False)) if manifest else False
            except Exception:
                cloud_allowed = False
            timings["capability_lookup_ms"] = (time.time() - t0) * 1000

            # Step 4: Build execution context
            t0 = time.time()
            exec_ctx = ExecutionContext(
                action=action,
                deal=asdict(deal) if deal else None,
                case_file=case_file,
                tool_gateway=None,
                cloud_allowed=cloud_allowed,
                registry=registry,
            )
            if action.type.upper().startswith("TOOL."):
                try:
                    from tools.gateway import get_tool_gateway

                    exec_ctx = ExecutionContext(
                        action=action,
                        deal=asdict(deal) if deal else None,
                        case_file=case_file,
                        tool_gateway=get_tool_gateway(),
                        cloud_allowed=cloud_allowed,
                        registry=registry,
                    )
                except Exception:
                    exec_ctx = ExecutionContext(
                        action=action,
                        deal=asdict(deal) if deal else None,
                        case_file=case_file,
                        tool_gateway=None,
                        cloud_allowed=cloud_allowed,
                        registry=registry,
                    )
            timings["build_context_ms"] = (time.time() - t0) * 1000

            # Step 5: Execute (main work)
            t0 = time.time()
            logger.info(
                "[ACTION %s] Starting execution: type=%s deal=%s",
                action.action_id,
                action.type,
                action.deal_id,
            )
            result = executor.execute(action, exec_ctx)
            timings["execute_ms"] = (time.time() - t0) * 1000
            timings["total_ms"] = (time.time() - exec_start) * 1000

            # Log timing summary
            logger.info(
                "[ACTION %s] Execution complete: total=%.0fms (validate=%.0fms, execute=%.0fms)",
                action.action_id,
                timings["total_ms"],
                timings["validate_ms"],
                timings["execute_ms"],
            )

            # Hydrate file metadata for artifacts.
            artifacts = []
            for art in result.artifacts:
                path = Path(art.path).resolve()
                try:
                    path.relative_to(DATAROOM_ROOT)
                except ValueError:
                    raise ActionExecutionError(
                        ActionError(
                            code="artifact_outside_dataroom",
                            message="Executor produced artifact outside DataRoom root",
                            category="validation",
                            retryable=False,
                            details={"path": str(path)},
                        )
                    )
                if not path.exists() or not path.is_file():
                    raise ActionExecutionError(
                        ActionError(
                            code="artifact_missing_on_disk",
                            message="Executor reported artifact but file is missing on disk",
                            category="io",
                            retryable=False,
                            details={"path": str(path)},
                        )
                    )

                sha256 = art.sha256 or _sha256_file(path)
                size_bytes = art.size_bytes or path.stat().st_size
                artifacts.append(
                    art.model_copy(update={"sha256": sha256, "size_bytes": int(size_bytes), "path": str(path)})
                )

            if artifacts:
                store.add_artifacts(action_id=action.action_id, artifacts=artifacts)

            store.mark_action_completed(action_id=action.action_id, actor="actions_runner", outputs=result.outputs, error=None)

            # Phase 6: chain follow-on actions deterministically (append-only).
            try:
                _enqueue_follow_on_actions(
                    store=store,
                    registry=registry,
                    event_store=event_store,
                    parent_action=action,
                    outputs=result.outputs or {},
                )
            except Exception:
                logger.exception("Follow-on action enqueue failed for %s", action.action_id)

            _emit_deal_event(
                event_store,
                deal_id=action.deal_id,
                event_type="kinetic_action_completed",
                actor="actions_runner",
                data={"action_id": action.action_id, "action_type": action.type},
            )

            # Memory (best-effort): store compact summary for retrieval-based planning.
            try:
                from actions.contracts.plan_spec import PlanSpec
                from actions.memory.store import ActionMemoryStore, build_summary_from_action
                from tools.manifest.registry import get_unified_manifest_registry

                refreshed = store.get_action(action.action_id)
                if refreshed and refreshed.status in {"COMPLETED", "FAILED"}:
                    reg = get_unified_manifest_registry()
                    entry = None
                    try:
                        for e in reg.list_entries():
                            if e.action_type == refreshed.type:
                                entry = e
                                break
                    except Exception:
                        entry = None

                    cap_id = (refreshed.capability_id or "").strip() or (entry.capability_id if entry else "")
                    tool_name = entry.tool_name if entry else refreshed.type
                    safety_class = (entry.safety_class if entry else "reversible") if entry else "reversible"
                    irreversible = bool(entry.irreversible) if entry else False
                    gated = safety_class in {"gated", "irreversible"} or irreversible

                    plan = PlanSpec(
                        status="OK",
                        plan_id=f"PLAN-{refreshed.action_id}",
                        created_by="actions_runner_memory",
                        goal=refreshed.title or refreshed.type,
                        deal_id=refreshed.deal_id,
                        steps=[
                            {
                                "step_id": "step_1",
                                "capability_id": cap_id or refreshed.type,
                                "action_type": refreshed.type,
                                "tool_name": tool_name or refreshed.type,
                                "title": refreshed.title or refreshed.type,
                                "summary": refreshed.summary or "",
                                "inputs": refreshed.inputs or {},
                                "depends_on": [],
                                "expected_artifacts": [],
                                "safety": {
                                    "safety_class": safety_class,
                                    "irreversible": irreversible,
                                    "gated": gated,
                                    "requires_human_approval": True,
                                },
                            }
                        ],
                    )

                    artifacts_payload = [
                        {
                            "artifact_id": a.artifact_id,
                            "filename": a.filename,
                            "mime_type": a.mime_type,
                            "path": a.path,
                            "sha256": a.sha256,
                            "size_bytes": a.size_bytes,
                            "created_at": a.created_at,
                        }
                        for a in (refreshed.artifacts or [])
                    ]

                    edits = {
                        "inputs_updated": any(ev.event == "inputs_updated" for ev in (refreshed.audit_trail or [])),
                    }

                    summary = build_summary_from_action(
                        action_id=refreshed.action_id,
                        action_type=refreshed.type,
                        deal_id=refreshed.deal_id,
                        inputs=refreshed.inputs or {},
                        plan_spec=plan.model_dump(),
                        outcome_status=refreshed.status,
                        artifacts=artifacts_payload,
                        user_edits=edits,
                    )
                    ActionMemoryStore().record(summary)
            except Exception:
                logger.exception("Action memory write failed: %s", action.action_id)

        if ctx_mgr is None:
            _execute()
        else:
            with ctx_mgr as run_ctx:
                _execute()
                refreshed = store.get_action(action.action_id)
                for art in (refreshed.artifacts or []) if refreshed else []:
                    try:
                        run_ctx.add_artifact(art.path)
                    except Exception:
                        pass

    except ActionExecutionError as e:
        # Retry scheduling or fail
        action = store.get_action(action_id)
        if not action:
            return

        err = e.error
        retry_count = int(action.retry_count or 0) + 1
        if err.retryable and retry_count <= int(action.max_retries or 3):
            delay_s = _next_backoff_seconds(retry_count=retry_count)
            next_attempt_at = (now_utc() + timedelta(seconds=delay_s)).isoformat().replace("+00:00", "Z")
            store.mark_action_retry(
                action_id=action.action_id,
                actor="actions_runner",
                error=err,
                retry_count=retry_count,
                next_attempt_at=next_attempt_at,
            )
            _emit_deal_event(
                event_store,
                deal_id=action.deal_id,
                event_type="kinetic_action_retry_scheduled",
                actor="actions_runner",
                data={
                    "action_id": action.action_id,
                    "action_type": action.type,
                    "retry_count": retry_count,
                    "next_attempt_at": next_attempt_at,
                    "error_code": err.code,
                },
            )
        else:
            store.mark_action_completed(action_id=action.action_id, actor="actions_runner", outputs=action.outputs or {}, error=err)
            _emit_deal_event(
                event_store,
                deal_id=action.deal_id,
                event_type="kinetic_action_failed",
                actor="actions_runner",
                data={"action_id": action.action_id, "action_type": action.type, "error_code": err.code},
            )
            try:
                from actions.memory.store import ActionMemoryStore, build_summary_from_action

                refreshed = store.get_action(action.action_id)
                if refreshed and refreshed.status in {"COMPLETED", "FAILED"}:
                    artifacts_payload = [
                        {
                            "artifact_id": a.artifact_id,
                            "filename": a.filename,
                            "mime_type": a.mime_type,
                            "path": a.path,
                            "sha256": a.sha256,
                            "size_bytes": a.size_bytes,
                            "created_at": a.created_at,
                        }
                        for a in (refreshed.artifacts or [])
                    ]
                    summary = build_summary_from_action(
                        action_id=refreshed.action_id,
                        action_type=refreshed.type,
                        deal_id=refreshed.deal_id,
                        inputs=refreshed.inputs or {},
                        plan_spec={},
                        outcome_status=refreshed.status,
                        artifacts=artifacts_payload,
                        user_edits={"inputs_updated": any(ev.event == "inputs_updated" for ev in (refreshed.audit_trail or []))},
                    )
                    ActionMemoryStore().record(summary)
            except Exception:
                logger.exception("Action memory write failed (failure path): %s", action.action_id)
    except Exception as e:
        logger.exception("Unhandled runner exception for action %s", action_id)
        # Best-effort mark failed (retryable) to avoid silent drops.
        action = store.get_action(action_id)
        if action:
            err = ActionError(
                code="runner_exception",
                message=f"{type(e).__name__}: {str(e)}",
                category="unknown",
                retryable=True,
            )
            retry_count = int(action.retry_count or 0) + 1
            if retry_count <= int(action.max_retries or 3):
                delay_s = _next_backoff_seconds(retry_count=retry_count)
                next_attempt_at = (now_utc() + timedelta(seconds=delay_s)).isoformat().replace("+00:00", "Z")
                store.mark_action_retry(
                    action_id=action.action_id,
                    actor="actions_runner",
                    error=err,
                    retry_count=retry_count,
                    next_attempt_at=next_attempt_at,
                )
            else:
                store.mark_action_completed(action_id=action.action_id, actor="actions_runner", outputs=action.outputs or {}, error=err)
    finally:
        stop_heartbeat.set()
        try:
            heartbeat_thread.join(timeout=1.0)
        except Exception:
            pass
        store.release_action_lock(action_id=action_id, owner_id=owner_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kinetic Action Engine Runner (v1.2)")
    parser.add_argument("--runner-name", default="kinetic_actions")
    parser.add_argument("--lease-seconds", type=int, default=30)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--action-lock-seconds", type=int, default=300)
    parser.add_argument("--processing-ttl-seconds", type=int, default=int(os.getenv("ZAKOPS_ACTION_PROCESSING_TTL_SECONDS", "3600")))
    parser.add_argument("--owner-id", default="")
    parser.add_argument("--once", action="store_true", help="Process at most one due action and exit")
    parser.add_argument("--max-actions", type=int, default=0, help="Max actions to process before exit (0=unlimited)")
    parser.add_argument("--action-id", default="", help="Process a specific action_id (debug)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    store = ActionStore()
    event_store = DealEventStore()

    owner_id = (args.owner_id or "").strip() or default_runner_owner_id()
    runner_name = str(args.runner_name)

    load_builtin_executors()

    processed = 0
    last_heartbeat = 0.0

    while True:
        ok = store.acquire_runner_lease(
            runner_name=runner_name,
            owner_id=owner_id,
            lease_seconds=int(args.lease_seconds),
            pid=os.getpid(),
            host=os.uname().nodename,
        )
        if not ok:
            time.sleep(args.poll_seconds)
            if args.once:
                return 2
            continue

        now = time.monotonic()
        if now - last_heartbeat > max(1.0, float(args.lease_seconds) / 2.0):
            store.heartbeat_runner_lease(
                runner_name=runner_name,
                owner_id=owner_id,
                lease_seconds=int(args.lease_seconds),
                pid=os.getpid(),
                host=os.uname().nodename,
            )
            last_heartbeat = now

        if args.action_id:
            action_id = args.action_id.strip()
        else:
            # Watchdog: ensure no action stays PROCESSING forever.
            try:
                store.mark_processing_timeouts(older_than_seconds=int(args.processing_ttl_seconds), actor="actions_runner_watchdog")
            except Exception:
                logger.exception("Processing watchdog failed")

            action_id = store.get_next_due_action_id() or store.get_next_due_processing_action_id()

        if not action_id:
            if args.once:
                return 0
            time.sleep(args.poll_seconds)
            continue

        # IMPORTANT: do not cache DealRegistry across actions.
        # The API process can mutate the registry (e.g., archive/delete deals). Executors call
        # registry.save(), so using a stale in-memory registry here can overwrite those changes.
        registry = DealRegistry(REGISTRY_PATH)

        process_one_action(
            store=store,
            registry=registry,
            event_store=event_store,
            owner_id=owner_id,
            action_id=action_id,
            action_lock_seconds=int(args.action_lock_seconds),
        )
        processed += 1

        if args.once:
            return 0
        if args.max_actions and processed >= args.max_actions:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
