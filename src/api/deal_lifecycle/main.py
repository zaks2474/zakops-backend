#!/usr/bin/env python3
"""
Deal Lifecycle REST API

FastAPI-based REST API for the deal lifecycle system.
Provides endpoints for managing deals, actions, quarantine, and agent invocation.

Security:
- Binds to localhost by default
- Tracks approved_by for stage transitions
- Enforces approval gates via state machine

Usage:
    python3 deal_lifecycle_api.py  # Start API server on port 8090
    python3 deal_lifecycle_api.py --port 8091 --host 0.0.0.0  # Custom binding

Endpoints:
    GET  /api/deals                      - List all deals
    GET  /api/deals/{id}                 - Get deal details
    GET  /api/deals/{id}/events          - Get deal events
    GET  /api/deals/{id}/case-file       - Get deal case file
    POST /api/deals/{id}/transition      - Stage transition
    POST /api/deals/{id}/note            - Add operator note
    GET  /api/actions                    - List all actions
    GET  /api/actions/due                - Get due actions
    POST /api/actions/{id}/execute       - Execute action
    POST /api/actions/{id}/cancel        - Cancel action
    GET  /api/quarantine                 - List quarantine items
    POST /api/quarantine/{id}/resolve    - Resolve quarantine item
    GET  /api/pipeline                   - Pipeline summary
    GET  /api/alerts                     - System alerts
    POST /api/agents/{name}/invoke       - Invoke agent
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from typing import Literal
from datetime import timedelta

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Canonical roots
DATAROOM_ROOT = Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom")).resolve()

# Dashboard path
DASHBOARD_DIR = DATAROOM_ROOT / "_dashboard"

from deal_registry import DealRegistry
from deal_events import DealEventStore
from link_normalizer import LinkNormalizer, process_links as normalize_links, classify_link, LinkCategory
from deferred_actions import DeferredActionQueue
from deal_state_machine import DealStateMachine, DealStage
from lifecycle_event_emitter import QuarantineManager

# Configuration
REGISTRY_PATH = str(DATAROOM_ROOT / ".deal-registry" / "deal_registry.json")
CASE_FILES_DIR = DATAROOM_ROOT / ".deal-registry" / "case_files"

# Create FastAPI app
app = FastAPI(
    title="Deal Lifecycle API",
    description="REST API for ZakOps Deal Lifecycle System",
    version="1.0.0",
)

# Add CORS middleware for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TransitionRequest(BaseModel):
    to_stage: str
    reason: str
    approved_by: str


class NoteRequest(BaseModel):
    content: str
    category: str = "general"


class QuarantineResolveRequest(BaseModel):
    resolution: str  # link_to_deal, create_new_deal, discard
    deal_id: Optional[str] = None
    resolved_by: str = "operator"


class AgentInvokeRequest(BaseModel):
    task: str
    deal_id: str
    params: Optional[Dict[str, Any]] = None


class DealArchiveRequest(BaseModel):
    operator: str = "operator"
    reason: Optional[str] = None


class DealBulkArchiveRequest(BaseModel):
    deal_ids: List[str]
    operator: str = "operator"
    reason: Optional[str] = None


class QuarantineDeleteRequest(BaseModel):
    deleted_by: str = "operator"
    reason: Optional[str] = None


class QuarantineBulkDeleteRequest(BaseModel):
    action_ids: List[str]
    deleted_by: str = "operator"
    reason: Optional[str] = None


class ChatScopeModel(BaseModel):
    type: str = "global"  # global, deal, document
    deal_id: Optional[str] = None
    doc: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    query: str
    scope: ChatScopeModel = ChatScopeModel()
    session_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class ProposalExecuteRequest(BaseModel):
    proposal_id: str
    approved_by: str
    session_id: str
    action: Literal["approve", "reject"] = "approve"
    reject_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Kinetic Action Engine (v1.2) API models
# ---------------------------------------------------------------------------

class KineticActionCreateRequest(BaseModel):
    action_type: str
    title: str
    summary: str = ""
    deal_id: Optional[str] = None
    capability_id: Optional[str] = None
    created_by: str = "operator"
    source: Literal["chat", "ui", "system"] = "ui"
    risk_level: Literal["low", "medium", "high"] = "medium"
    requires_human_review: bool = True
    idempotency_key: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None


class KineticActionApproveRequest(BaseModel):
    approved_by: str = "operator"


class KineticActionExecuteRequest(BaseModel):
    requested_by: str = "operator"


class KineticActionCancelRequest(BaseModel):
    cancelled_by: str = "operator"
    reason: str = "Cancelled via API"


class KineticActionUpdateRequest(BaseModel):
    updated_by: str = "operator"
    inputs: Dict[str, Any]


class KineticActionUnstickRequest(BaseModel):
    unstuck_by: str = "operator"
    reason: str = "operator_unstick"


def _action_payload_to_frontend(action: Any) -> Dict[str, Any]:
    """
    Normalize ActionPayload to the frontend contract used by `zakops-dashboard/src/lib/api.ts`.

    Key fixes:
    - ActionPayload uses `type`; frontend expects `action_type`
    - ActionError.details is a dict; frontend expects `details` as string (optional)
    - Ensure `deal_id` is always a string (never null) - use "GLOBAL" for non-deal actions
    - OMIT null values for optional fields (Zod .optional() != .nullable())
    - Always include artifacts as [] (never null)
    """
    try:
        payload: Dict[str, Any] = action.model_dump()
    except Exception:
        payload = dict(action or {})

    # Required: action_type (frontend schema expects this)
    action_type = payload.get("action_type") or payload.get("type") or ""
    payload["action_type"] = action_type

    # Required: deal_id must be string (never null). Use "GLOBAL" for non-deal actions.
    if not payload.get("deal_id"):
        payload["deal_id"] = "GLOBAL"

    # Required: artifacts must be array (never null)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        payload["artifacts"] = []
    else:
        normalized = []
        for a in artifacts:
            if not isinstance(a, dict):
                continue
            if (not a.get("download_url")) and a.get("artifact_id") and payload.get("action_id"):
                a = dict(a)
                a["download_url"] = f"/api/actions/{payload.get('action_id')}/artifact/{a.get('artifact_id')}"
            normalized.append(a)
        payload["artifacts"] = normalized

    # Optional fields: OMIT if null/None (Zod .optional() expects missing or undefined, not null)
    optional_fields = [
        "capability_id", "summary", "outputs", "error",
        "approved_at", "approved_by", "started_at", "completed_at",
        "duration_seconds", "next_attempt_at", "created_by",
        "progress", "progress_message",
        "runner_lock_owner", "runner_lock_expires_at", "runner_heartbeat_at",
    ]
    for field in optional_fields:
        if field in payload and payload[field] is None:
            del payload[field]

    # error: if present and is a dict, normalize details to string
    err = payload.get("error")
    if isinstance(err, dict):
        details = err.get("details")
        if details is not None and not isinstance(details, str):
            try:
                err["details"] = json.dumps(details, sort_keys=True)
            except Exception:
                err["details"] = str(details)
        payload["error"] = err
    elif err is None and "error" in payload:
        del payload["error"]

    # Ensure required string fields aren't null
    for field in ["action_id", "title", "status", "created_at", "updated_at"]:
        if not payload.get(field):
            payload[field] = payload.get(field) or ""

    # inputs/outputs: ensure dict (never null)
    if not isinstance(payload.get("inputs"), dict):
        payload["inputs"] = {}
    if "outputs" in payload and not isinstance(payload.get("outputs"), dict):
        del payload["outputs"]

    return payload


def _capability_manifest_to_frontend(capability: Any) -> Dict[str, Any]:
    """
    Normalize CapabilityManifest to the frontend contract in `zakops-dashboard/src/lib/api.ts`.

    Frontend expects:
    - `version` as a separate field
    - `output_artifacts[].type` (not `kind`)
    """
    try:
        raw: Dict[str, Any] = capability.model_dump()
    except Exception:
        raw = dict(capability or {})

    capability_id = str(raw.get("capability_id") or "").strip()
    version = str(raw.get("version") or "").strip()
    if not version and capability_id:
        # Best-effort: parse trailing ".v<major>" (e.g., document.generate_loi.v1)
        tail = capability_id.split(".")[-1]
        if tail.startswith("v") and tail[1:].isdigit():
            version = tail
        else:
            version = "v1"

    input_schema = raw.get("input_schema") or {}
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}, "required": []}

    output_artifacts_out: List[Dict[str, Any]] = []
    for art in raw.get("output_artifacts") or []:
        if not isinstance(art, dict):
            continue
        output_artifacts_out.append(
            {
                "type": str(art.get("type") or art.get("kind") or "").strip(),
                "description": str(art.get("description") or "").strip(),
                "mime_type": str(art.get("mime_type") or "").strip(),
            }
        )

    examples_out: List[Dict[str, Any]] = []
    for ex in raw.get("examples") or []:
        if not isinstance(ex, dict):
            continue
        desc = str(ex.get("description") or ex.get("user_intent") or "").strip()
        inputs = ex.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        if desc:
            examples_out.append({"description": desc, "inputs": inputs})

    tags = raw.get("tags") or []
    if not isinstance(tags, list):
        tags = []

    return {
        "capability_id": capability_id,
        "version": version,
        "title": str(raw.get("title") or "").strip(),
        "description": str(raw.get("description") or "").strip(),
        "action_type": str(raw.get("action_type") or "").strip(),
        "input_schema": input_schema,
        "output_artifacts": output_artifacts_out,
        "risk_level": str(raw.get("risk_level") or "medium").strip().lower(),
        "requires_approval": bool(raw.get("requires_approval", True)),
        # Optional fields (frontend can ignore if unused). These help operators understand
        # why an action may be blocked (e.g., cloud-disabled) and align with policy gating.
        "cloud_required": bool(raw.get("cloud_required", False)),
        "llm_allowed": bool(raw.get("llm_allowed", False)),
        "constraints": list(raw.get("constraints") or []),
        "examples": examples_out or [],
        "tags": list(tags),
    }

class KineticActionPlanRequest(BaseModel):
    query: str
    deal_id: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None


class KineticActionRequeueRequest(BaseModel):
    requeued_by: str = "operator"
    reason: str = "operator_requeue"


# Helper functions
def get_registry() -> DealRegistry:
    return DealRegistry(REGISTRY_PATH)


def get_event_store() -> DealEventStore:
    return DealEventStore()


def get_action_queue() -> DeferredActionQueue:
    return DeferredActionQueue()


def get_quarantine() -> QuarantineManager:
    return QuarantineManager()


# Kinetic Actions (v1.2) store singleton
_KINETIC_ACTION_STORE = None


def get_kinetic_action_store():
    global _KINETIC_ACTION_STORE
    if _KINETIC_ACTION_STORE is None:
        from actions.engine.store import ActionStore

        _KINETIC_ACTION_STORE = ActionStore()
    return _KINETIC_ACTION_STORE


def load_case_file(deal_id: str) -> Optional[Dict[str, Any]]:
    path = CASE_FILES_DIR / f"{deal_id}.json"
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def days_ago(iso_timestamp: str) -> int:
    try:
        if iso_timestamp.endswith("Z"):
            iso_timestamp = iso_timestamp[:-1] + "+00:00"
        ts = datetime.fromisoformat(iso_timestamp)
        now = datetime.now(timezone.utc)
        return max(0, (now - ts).days)
    except (ValueError, TypeError):
        return 0


# ===== DASHBOARD =====

@app.get("/", response_class=HTMLResponse)
def dashboard_redirect():
    """Redirect root to dashboard."""
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the dashboard HTML."""
    dashboard_path = DASHBOARD_DIR / "index.html"
    if dashboard_path.exists():
        with open(dashboard_path) as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


# ===== DEAL ENDPOINTS =====

@app.get("/api/deals")
def list_deals(
    stage: Optional[str] = None,
    status: str = "active",
    broker: Optional[str] = None,
    age_gt: Optional[int] = None,
):
    """List all deals with optional filtering."""
    registry = get_registry()
    deals = registry.list_deals(stage=stage, status=status)

    # Filter by broker if specified
    if broker:
        broker_lower = broker.lower()
        deals = [d for d in deals if d.broker and broker_lower in d.broker.name.lower()]

    # Filter by age if specified
    if age_gt:
        deals = [d for d in deals if days_ago(d.updated_at) > age_gt]

    # Load enrichment data for all deals
    enrichment_data = {}
    try:
        registry_path = Path("/home/zaks/DataRoom/.deal-registry/deal_registry.json")
        if registry_path.exists():
            with open(registry_path) as f:
                raw_data = json.load(f)
            enrichment_data = raw_data.get("deals", {})
    except Exception:
        pass

    def get_enrichment_summary(deal_id: str) -> dict:
        """Get enrichment summary for a deal."""
        data = enrichment_data.get(deal_id, {})
        materials = data.get("materials", [])
        return {
            "display_name": data.get("display_name"),
            "materials_count": len(materials),
            "has_cim": any(m.get("link_type") == "cim" for m in materials),
            "last_email_at": data.get("last_email_at"),
        }

    return {
        "count": len(deals),
        "deals": [
            {
                "deal_id": d.deal_id,
                "canonical_name": d.canonical_name,
                "display_name": enrichment_data.get(d.deal_id, {}).get("display_name"),
                "stage": d.stage,
                "status": d.status,
                "broker": d.broker.name if d.broker else None,
                "priority": d.metadata.priority,
                "updated_at": d.updated_at,
                "days_since_update": days_ago(d.updated_at),
                "enrichment": get_enrichment_summary(d.deal_id),
            }
            for d in deals
        ],
    }


@app.get("/api/deals/{deal_id}")
def get_deal(deal_id: str):
    """Get full deal details with case file."""
    registry = get_registry()
    deal = registry.get_deal(deal_id)

    if not deal:
        raise HTTPException(status_code=404, detail=f"Deal not found: {deal_id}")

    event_store = get_event_store()
    action_queue = get_action_queue()

    # Get state machine info
    sm = DealStateMachine(deal.stage, deal_id)

    # Load case file
    case_file = load_case_file(deal_id)

    # Load deal profile if available (for enriched fields from triage)
    deal_profile = None
    if deal.folder_path:
        try:
            profile_path = Path(deal.folder_path) / "deal_profile.json"
            if profile_path.exists():
                deal_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            else:
                # Try to extract enrichment from quarantine triage_summary.json (backfill)
                deal_profile = _extract_enrichment_from_quarantine(deal.folder_path)
        except Exception:
            pass

    # Merge deal profile fields with registry fields (profile takes precedence if non-null)
    company_info = {
        "sector": deal.company_info.sector,
        "location": deal.company_info.location.__dict__ if deal.company_info.location else None,
    }
    if deal_profile:
        profile_company = deal_profile.get("company_info") or {}
        if profile_company.get("sector"):
            company_info["sector"] = profile_company["sector"]
        if profile_company.get("location"):
            company_info["location"] = profile_company["location"]
        if profile_company.get("name"):
            company_info["name"] = profile_company["name"]
        if profile_company.get("website"):
            company_info["website"] = profile_company["website"]

    metadata = {
        "priority": deal.metadata.priority,
        "asking_price": deal.metadata.asking_price,
        "ebitda": deal.metadata.ebitda,
        "nda_status": deal.metadata.nda_status,
        "cim_received": deal.metadata.cim_received,
    }
    if deal_profile:
        profile_financials = deal_profile.get("financials") or {}
        if profile_financials.get("asking_price"):
            metadata["asking_price"] = profile_financials["asking_price"]
        if profile_financials.get("ebitda"):
            metadata["ebitda"] = profile_financials["ebitda"]
        if profile_financials.get("revenue"):
            metadata["revenue"] = profile_financials["revenue"]
        if profile_financials.get("sde"):
            metadata["sde"] = profile_financials["sde"]
        if profile_financials.get("multiple"):
            metadata["multiple"] = profile_financials["multiple"]
        profile_status = deal_profile.get("deal_status") or {}
        if profile_status.get("nda_status"):
            metadata["nda_status"] = profile_status["nda_status"]
        if profile_status.get("cim_received"):
            metadata["cim_received"] = profile_status["cim_received"]

    broker = {
        "name": deal.broker.name if deal.broker else None,
        "email": deal.broker.email if deal.broker else None,
        "company": deal.broker.company if deal.broker else None,
        "phone": deal.broker.phone if deal.broker else None,
    }
    if deal_profile:
        profile_broker = deal_profile.get("broker") or {}
        if profile_broker.get("company") and not broker.get("company"):
            broker["company"] = profile_broker["company"]
        if profile_broker.get("role"):
            broker["role"] = profile_broker["role"]
        if profile_broker.get("name") and not broker.get("name"):
            broker["name"] = profile_broker["name"]

    # Triage summary (rich context from email triage)
    triage_summary = None
    evidence = []
    if deal_profile:
        triage_summary = deal_profile.get("triage_summary")
        evidence = deal_profile.get("evidence") or []

    return {
        "deal_id": deal.deal_id,
        "canonical_name": deal.canonical_name,
        "display_name": deal.display_name,
        "folder_path": deal.folder_path,
        "stage": deal.stage,
        "status": deal.status,
        "broker": broker,
        "company_info": company_info,
        "metadata": metadata,
        "triage_summary": triage_summary,
        "evidence": evidence,
        "state_machine": {
            "current_stage": deal.stage,
            "is_terminal": sm.is_terminal(),
            "allowed_transitions": [s.value for s in sm.get_allowed_transitions()],
            "advisory_context": sm.get_advisory_context(),
        },
        "case_file": case_file,
        "event_count": event_store.count_events(deal_id),
        "pending_actions": len(action_queue.get_actions_for_deal(deal_id, status="pending")),
        # Pipeline outputs summary - addresses "no visible outputs" UX issue
        "pipeline_summary": _get_pipeline_summary(action_queue, deal_id),
        "created_at": deal.created_at,
        "updated_at": deal.updated_at,
    }


def _extract_enrichment_from_quarantine(deal_folder: str) -> Optional[Dict[str, Any]]:
    """
    Extract enrichment data from quarantine triage_summary.json for deals
    that don't have deal_profile.json yet (backfill for existing deals).
    """
    import re

    def _parse_money(text: str) -> Optional[float]:
        """Parse money string like '$899,000' or '$225K' or '$1.5M' to float."""
        if not text:
            return None
        money_re = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)\s*([KkMm])?", re.IGNORECASE)
        m = money_re.search(str(text))
        if not m:
            return None
        num_str = m.group(1).replace(",", "")
        try:
            value = float(num_str)
        except ValueError:
            return None
        suffix = (m.group(2) or "").upper()
        if suffix == "K":
            value *= 1_000
        elif suffix == "M":
            value *= 1_000_000
        return value

    try:
        deal_path = Path(deal_folder)
        corr_dir = deal_path / "07-Correspondence"
        if not corr_dir.exists():
            return None

        # Find first correspondence bundle with quarantine_dir
        for bundle_dir in sorted(corr_dir.iterdir()):
            if not bundle_dir.is_dir():
                continue
            manifest_path = bundle_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                quarantine_dir = manifest.get("quarantine_dir")
                if not quarantine_dir:
                    continue

                triage_path = Path(quarantine_dir) / "triage_summary.json"
                if not triage_path.exists():
                    continue

                triage = json.loads(triage_path.read_text(encoding="utf-8"))

                # Extract target company info
                target_company = triage.get("target_company") or {}
                deal_signals = triage.get("deal_signals") or {}
                valuation = deal_signals.get("valuation_terms") or {}

                # Build deal_profile-like structure
                profile = {
                    "company_info": {
                        "name": target_company.get("name"),
                        "sector": target_company.get("industry"),
                        "location": target_company.get("location"),
                        "website": target_company.get("website"),
                    },
                    "financials": {
                        "asking_price": _parse_money(valuation.get("ask_price")),
                        "ebitda": _parse_money(valuation.get("ebitda")),
                        "revenue": _parse_money(valuation.get("revenue")),
                        "sde": _parse_money(valuation.get("sde")),
                        "multiple": valuation.get("multiple"),
                    },
                    "triage_summary": triage.get("summary"),
                    "evidence": triage.get("evidence"),
                    "source": "quarantine_backfill",
                }

                # Only return if we have meaningful data
                if any(profile["company_info"].values()) or any(profile["financials"].values()):
                    return profile

            except Exception:
                continue

        return None
    except Exception:
        return None


def _get_pipeline_summary(action_store, deal_id: str) -> Dict[str, Any]:
    """Get a summary of pipeline actions for UI display."""
    all_actions = action_store.get_actions_for_deal(deal_id, status=None)
    pipeline_types = []
    completed_count = 0
    total_count = 0

    for action in all_actions:
        if action.type.startswith("DEAL.") or action.type.startswith("RAG."):
            total_count += 1
            if action.status == "COMPLETED":
                completed_count += 1
            # Track which pipeline actions have run
            action_name = action.type.split(".")[-1].replace("_", " ").title()
            pipeline_types.append({
                "type": action.type,
                "name": action_name,
                "status": action.status,
                "completed_at": action.completed_at,
            })

    return {
        "total_actions": total_count,
        "completed_actions": completed_count,
        "all_completed": completed_count == total_count and total_count > 0,
        "actions_summary": pipeline_types[:10],  # Limit to recent 10
    }


@app.post("/api/deals/{deal_id}/archive")
def archive_deal(deal_id: str, request: DealArchiveRequest):
    registry = get_registry()
    success = registry.mark_deal_deleted(deal_id, request.operator, reason=request.reason)
    if not success:
        raise HTTPException(status_code=404, detail=f"Deal not found or already deleted: {deal_id}")
    registry.save()
    deal = registry.get_deal(deal_id)
    return {
        "archived": True,
        "deal_id": deal_id,
        "deleted_at": deal.deleted_at,
        "deleted_by": deal.deleted_by,
        "deleted_reason": deal.deleted_reason,
    }


@app.post("/api/deals/{deal_id}/restore")
def restore_deal(deal_id: str, request: DealArchiveRequest):
    registry = get_registry()
    success = registry.restore_deal(deal_id, request.operator, reason=request.reason)
    if not success:
        raise HTTPException(status_code=404, detail=f"Deal not found or not deleted: {deal_id}")
    registry.save()
    deal = registry.get_deal(deal_id)
    return {
        "restored": True,
        "deal_id": deal_id,
        "stage": deal.stage,
        "status": deal.status,
    }


@app.post("/api/deals/bulk-archive")
def bulk_archive_deals(request: DealBulkArchiveRequest):
    registry = get_registry()
    summary = {"archived": [], "skipped": []}
    for deal_id in request.deal_ids:
        if registry.mark_deal_deleted(deal_id, request.operator, reason=request.reason):
            summary["archived"].append(deal_id)
        else:
            summary["skipped"].append(deal_id)
    if summary["archived"]:
        registry.save()
    return summary


@app.get("/api/deals/{deal_id}/pipeline-outputs")
def get_deal_pipeline_outputs(deal_id: str):
    """
    Get pipeline execution status and outputs for a deal.

    Shows all actions run for the deal, their status, and what outputs they produced.
    This addresses the "actions completed but no visible outputs" UX issue.
    """
    action_store = get_action_queue()

    # Get all actions for this deal
    all_actions = action_store.get_actions_for_deal(deal_id, status=None)

    # Group actions by type for easier UI consumption
    pipeline_actions = []
    for action in all_actions:
        action_type = action.type
        # Filter to pipeline actions (not approval/review actions)
        if action_type.startswith("DEAL.") or action_type.startswith("RAG."):
            outputs = action.outputs or {}
            artifacts = [
                {
                    "filename": a.filename,
                    "path": a.path,
                    "mime_type": a.mime_type,
                }
                for a in action.artifacts
            ] if action.artifacts else []

            pipeline_actions.append({
                "action_id": action.action_id,
                "type": action_type,
                "title": action.title,
                "status": action.status,
                "created_at": action.created_at,
                "completed_at": action.completed_at,
                "duration_seconds": action.duration_seconds,
                "outputs": {
                    "bundle_path": outputs.get("bundle_path"),
                    "deal_path": outputs.get("deal_path"),
                    "artifact_paths": outputs.get("artifact_paths", []),
                    "extracted_entities": outputs.get("extracted_entities"),
                    "placed_files": outputs.get("placed_files", []),
                    "rag_indexed": outputs.get("rag_indexed", False),
                },
                "artifacts": artifacts,
            })

    # Sort by created_at
    pipeline_actions.sort(key=lambda a: a.get("created_at", ""), reverse=True)

    # Summary stats
    completed_count = sum(1 for a in pipeline_actions if a["status"] == "COMPLETED")
    failed_count = sum(1 for a in pipeline_actions if a["status"] == "FAILED")
    pending_count = sum(1 for a in pipeline_actions if a["status"] in ("PENDING_APPROVAL", "PENDING", "QUEUED"))

    # Get classified links summary from the deal folder if available
    classified_links_summary = {"primary_count": 0, "tracking_count": 0, "social_count": 0}
    registry = get_registry()
    deal = registry.get_deal(deal_id)
    if deal and deal.folder_path:
        raw_folder = Path(str(deal.folder_path).strip()).expanduser()
        deal_path = (DATAROOM_ROOT / raw_folder).resolve() if not raw_folder.is_absolute() else raw_folder.resolve()
        corr_dir = deal_path / "07-Correspondence"

        for links_file in ["classified_links.json", "links.json"]:
            links_path = corr_dir / links_file
            if links_path.exists():
                try:
                    links_data = json.loads(links_path.read_text(encoding="utf-8"))
                    if links_file == "classified_links.json":
                        summary = links_data.get("summary", {})
                        classified_links_summary = {
                            "primary_count": summary.get("material_count", 0),
                            "tracking_count": summary.get("tracking_count", 0),
                            "social_count": summary.get("social_count", 0),
                            "unsubscribe_count": summary.get("unsubscribe_count", 0),
                            "total_raw": summary.get("total_raw", 0),
                        }
                    else:
                        # Process links.json through normalizer
                        raw_links = links_data.get("links", []) if isinstance(links_data, dict) else []
                        processed = normalize_links(raw_links)
                        groups = processed.get("groups", {})
                        classified_links_summary = {
                            "primary_count": len(groups.get("deal_material", [])) + len(groups.get("other", [])),
                            "tracking_count": len(groups.get("tracking", [])),
                            "social_count": len(groups.get("social", [])),
                            "unsubscribe_count": len(groups.get("unsubscribe", [])),
                            "total_raw": processed.get("total_count", len(raw_links)),
                            "duplicates_removed": processed.get("duplicates_removed", 0),
                        }
                    break
                except Exception:
                    pass

    return {
        "deal_id": deal_id,
        "pipeline_status": {
            "total_actions": len(pipeline_actions),
            "completed": completed_count,
            "failed": failed_count,
            "pending": pending_count,
            "all_completed": completed_count == len(pipeline_actions) and len(pipeline_actions) > 0,
        },
        "actions": pipeline_actions,
        "classified_links": classified_links_summary,
    }


@app.get("/api/deals/{deal_id}/events")
def get_deal_events(
    deal_id: str,
    limit: int = Query(default=100, le=1000),
    since: Optional[str] = None,
):
    """Get event history for a deal."""
    event_store = get_event_store()
    events = event_store.get_events(deal_id, since=since, limit=limit)

    return {
        "deal_id": deal_id,
        "count": len(events),
        "events": [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "timestamp": e.timestamp,
                "actor": e.actor,
                "data": e.data,
            }
            for e in reversed(events)  # Most recent first
        ],
    }


@app.get("/api/deals/{deal_id}/case-file")
def get_deal_case_file(deal_id: str):
    """Get deal case file."""
    case_file = load_case_file(deal_id)
    if not case_file:
        raise HTTPException(status_code=404, detail=f"No case file for: {deal_id}")
    return case_file


@app.get("/api/deals/{deal_id}/materials")
def get_deal_materials(deal_id: str):
    """
    Filesystem-backed materials view (correspondence bundles + links + attachments).

    This endpoint is designed for the Deal UI "Materials" section and reflects the
    progressive directory growth model:
    - deal folder is created on approval
    - follow-up emails append bundles under 07-Correspondence/
    """
    registry = get_registry()
    deal = registry.get_deal(deal_id)
    if not deal:
        raise HTTPException(status_code=404, detail=f"Deal not found: {deal_id}")
    folder_path = str(deal.folder_path or "").strip()
    if not folder_path:
        raise HTTPException(status_code=404, detail="deal_folder_missing")

    raw = Path(folder_path).expanduser()
    deal_path = (DATAROOM_ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        deal_path.relative_to(DATAROOM_ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail="deal_folder_outside_dataroom")

    corr_dir = (deal_path / "07-Correspondence").resolve()
    if not corr_dir.exists():
        return {"deal_id": deal_id, "deal_path": str(deal_path), "correspondence": [], "aggregate_links": {"links": []}, "pending_auth": []}

    # Use link normalizer for classification and grouping
    def _process_bundle_links(links: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process links with normalizer and return UI-ready structure."""
        try:
            result = normalize_links(links)
            # Build grouped structure with categorized links
            grouped = {}
            for category, category_links in result.get("groups", {}).items():
                if category_links:
                    grouped[category] = category_links

            return {
                "all": result.get("all_unique", []),
                "groups": grouped,
                "stats": {
                    "total_raw": result.get("total_count", len(links)),
                    "unique_count": result.get("unique_count", 0),
                    "duplicates_removed": result.get("duplicates_removed", 0),
                },
            }
        except Exception:
            # Fallback to basic structure on error
            return {"all": links, "groups": {}, "stats": {}}

    bundles: List[Dict[str, Any]] = []

    # New-style bundle directories: 07-Correspondence/<bundle>/manifest.json
    for p in sorted(corr_dir.iterdir()):
        if not p.is_dir():
            continue
        manifest = p / "manifest.json"
        if not manifest.exists() or not manifest.is_file():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        links = data.get("links") if isinstance(data, dict) and isinstance(data.get("links"), list) else []
        links_list = [l for l in links if isinstance(l, dict)]

        attachments_dir = p / "attachments"
        attachments: List[Dict[str, Any]] = []
        if attachments_dir.exists() and attachments_dir.is_dir():
            for f in sorted(attachments_dir.iterdir()):
                if f.is_file():
                    attachments.append({"filename": f.name, "path": str(f), "size_bytes": int(f.stat().st_size)})

        pending_auth = p / "pending_auth_links.json"
        pending_links: List[Dict[str, Any]] = []
        if pending_auth.exists() and pending_auth.is_file():
            try:
                pdata = json.loads(pending_auth.read_text(encoding="utf-8"))
                raw_links = pdata.get("links") if isinstance(pdata, dict) else None
                if isinstance(raw_links, list):
                    pending_links = [l for l in raw_links if isinstance(l, dict)]
            except Exception:
                pending_links = []

        bundles.append(
            {
                "bundle_id": p.name,
                "bundle_path": str(p),
                "format": "bundle_dir",
                "message_id": str(data.get("message_id") or ""),
                "thread_id": str(data.get("thread_id") or ""),
                "from": str(data.get("from") or ""),
                "to": str(data.get("to") or ""),
                "date": str(data.get("date") or ""),
                "subject": str(data.get("subject") or ""),
                "email_md": str((p / "email.md").resolve()) if (p / "email.md").exists() else None,
                "manifest_json": str(manifest),
                "links": _process_bundle_links(links_list),
                "pending_auth_links": pending_links,
                "attachments": attachments,
            }
        )

    # Legacy review-email artifacts: *_manifest.json + *_email.md + *_attachments/
    for manifest in sorted(corr_dir.glob("*_manifest.json")):
        if manifest.name == "links.json":
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        base = manifest.name[: -len("_manifest.json")] if manifest.name.endswith("_manifest.json") else manifest.stem
        email_md = None
        candidates = sorted(corr_dir.glob(f"{base}_*_email.md"))
        if candidates:
            email_md = str(candidates[0])

        att_dir = corr_dir / f"{base}_attachments"
        attachments: List[Dict[str, Any]] = []
        if att_dir.exists() and att_dir.is_dir():
            for f in sorted(att_dir.iterdir()):
                if f.is_file():
                    attachments.append({"filename": f.name, "path": str(f), "size_bytes": int(f.stat().st_size)})

        links = data.get("links") if isinstance(data, dict) and isinstance(data.get("links"), list) else []
        links_list = [l for l in links if isinstance(l, dict)]
        pending_links = [l for l in links_list if bool(l.get("auth_required"))]

        bundles.append(
            {
                "bundle_id": base,
                "bundle_path": str(corr_dir),
                "format": "legacy_review_email",
                "message_id": str(data.get("message_id") or ""),
                "thread_id": str(data.get("thread_id") or ""),
                "from": str(data.get("from") or ""),
                "to": str(data.get("to") or ""),
                "date": str(data.get("date") or ""),
                "subject": str(data.get("subject") or ""),
                "email_md": email_md,
                "manifest_json": str(manifest),
                "links": _process_bundle_links(links_list),
                "pending_auth_links": pending_links,
                "attachments": attachments,
            }
        )

    # Aggregate links: prefer classified_links.json (pre-categorized) over links.json (raw).
    # Always process through link normalizer for deduplication and categorization.
    def _build_classified_links_response(raw_links: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build UI-ready classified links structure."""
        processed = _process_bundle_links(raw_links)
        groups = processed.get("groups", {})
        stats = processed.get("stats", {})

        # Extract links by category for top-level access
        primary_links = (
            groups.get("deal_material", []) +
            groups.get("portal", []) +
            groups.get("other", [])
        )
        tracking_links = groups.get("tracking", [])
        social_links = groups.get("social", [])
        unsubscribe_links = groups.get("unsubscribe", [])
        contact_links = groups.get("contact", []) + groups.get("calendar", [])

        return {
            "primary_links": primary_links,
            "tracking_links": tracking_links,
            "social_links": social_links,
            "unsubscribe_links": unsubscribe_links,
            "contact_links": contact_links,
            "summary": {
                "primary_count": len(primary_links),
                "tracking_count": len(tracking_links),
                "social_count": len(social_links),
                "unsubscribe_count": len(unsubscribe_links),
                "contact_count": len(contact_links),
                "total_raw": stats.get("total_raw", len(raw_links)),
                "unique_count": stats.get("unique_count", 0),
                "duplicates_removed": stats.get("duplicates_removed", 0),
            },
            # Preserve raw for audit/debug (hidden by default)
            "_raw_links": raw_links,
            "_all_groups": groups,
        }

    aggregate_links: Dict[str, Any] = {
        "primary_links": [],
        "tracking_links": [],
        "social_links": [],
        "unsubscribe_links": [],
        "contact_links": [],
        "summary": {},
    }

    # Try classified_links.json first (pre-categorized by executor)
    classified_path = corr_dir / "classified_links.json"
    agg_path = corr_dir / "links.json"

    if classified_path.exists() and classified_path.is_file():
        try:
            classified_data = json.loads(classified_path.read_text(encoding="utf-8")) or {}
            # classified_links.json has pre-categorized structure
            aggregate_links = {
                "primary_links": classified_data.get("links", []),
                "tracking_links": classified_data.get("_tracking_links", []),
                "social_links": classified_data.get("_social_links", []),
                "unsubscribe_links": classified_data.get("_unsubscribe_links", []),
                "contact_links": [],
                "summary": classified_data.get("summary", {}),
                "_raw_links": classified_data.get("_all_links_raw", []),
            }
        except Exception:
            pass  # Fall through to links.json

    # Fall back to links.json and process through normalizer
    if not aggregate_links.get("primary_links") and agg_path.exists() and agg_path.is_file():
        try:
            raw_agg = json.loads(agg_path.read_text(encoding="utf-8")) or {"links": []}
        except Exception:
            raw_agg = {"links": []}
        if not isinstance(raw_agg, dict) or not isinstance(raw_agg.get("links"), list):
            raw_agg = {"links": []}

        # Process aggregate links with normalizer
        aggregate_links = _build_classified_links_response(raw_agg.get("links", []))

    # Filter pending_auth to exclude tracking/unsubscribe/social links
    # These should never require auth and shouldn't clutter the pending-auth UI
    EXCLUDED_CATEGORIES = {LinkCategory.TRACKING, LinkCategory.UNSUBSCRIBE, LinkCategory.SOCIAL}
    # Known public broker listing sites - these don't require auth
    PUBLIC_BROKER_DOMAINS = {
        "quietlight.com",
        "bizbuysell.com",
        "businessforsale.com",
        "loopnet.com",
        "acquire.com",
        "flippa.com",
        "empireflippers.com",
        "microacquire.com",
        "dealstream.com",
    }
    pending_auth_all: List[Dict[str, Any]] = []
    for b in bundles:
        for l in b.get("pending_auth_links") or []:
            if not isinstance(l, dict):
                continue
            url = str(l.get("url") or "").strip()
            if not url:
                continue
            # Check for public broker sites - these are public listings, not auth-required
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                domain = (parsed.netloc or "").lower()
                if domain.startswith("www."):
                    domain = domain[4:]
                if any(domain == d or domain.endswith("." + d) for d in PUBLIC_BROKER_DOMAINS):
                    continue  # Skip public broker listings
            except Exception:
                pass
            # Classify the link to determine if it's a tracking/unsubscribe/social link
            classified = classify_link(url)
            if classified.category in EXCLUDED_CATEGORIES:
                # Skip tracking/unsubscribe/social - they don't need auth
                continue
            # Add classification info to the link for UI display
            pending_auth_all.append({
                **l,
                "bundle_id": b.get("bundle_id"),
                "category": classified.category.value,
                "link_type": classified.link_type,
                "meaning_label": classified.meaning_label,
            })

    # Most recent first (best-effort: sort by date, then bundle_id).
    def _sort_key(item: Dict[str, Any]) -> str:
        return str(item.get("date") or item.get("bundle_id") or "")

    bundles.sort(key=_sort_key, reverse=True)

    return {
        "deal_id": deal_id,
        "deal_path": str(deal_path),
        "correspondence": bundles,
        "aggregate_links": aggregate_links,
        "pending_auth": pending_auth_all,
    }


# =============================================================================
# Enrichment Endpoints
# =============================================================================

def _get_enrichment_data(deal_id: str) -> Optional[dict]:
    """Get enrichment data for a deal from the registry JSON."""
    registry_path = Path("/home/zaks/DataRoom/.deal-registry/deal_registry.json")
    if not registry_path.exists():
        return None

    try:
        with open(registry_path) as f:
            data = json.load(f)

        deal_data = data.get("deals", {}).get(deal_id, {})

        # Extract enrichment fields
        return {
            "display_name": deal_data.get("display_name"),
            "target_company_name": deal_data.get("target_company_name"),
            "materials": deal_data.get("materials", []),
            "broker": deal_data.get("broker"),
            "last_email_at": deal_data.get("last_email_at"),
            "enrichment_confidence": deal_data.get("enrichment_confidence"),
            "enriched_at": deal_data.get("enriched_at"),
            "aliases": deal_data.get("aliases", []),
        }
    except Exception as e:
        return None


@app.get("/api/deals/{deal_id}/enrichment")
def get_deal_enrichment(deal_id: str):
    """Get enrichment data for a deal (materials, resolved names, etc.)."""
    enrichment = _get_enrichment_data(deal_id)

    if not enrichment:
        raise HTTPException(status_code=404, detail=f"No enrichment data for: {deal_id}")

    # Categorize materials by type
    materials_by_type = {}
    for mat in enrichment.get("materials", []):
        link_type = mat.get("link_type", "unknown")
        if link_type not in materials_by_type:
            materials_by_type[link_type] = []
        materials_by_type[link_type].append(mat)

    # Count auth-required links
    auth_required = [m for m in enrichment.get("materials", []) if m.get("requires_auth")]

    return {
        "deal_id": deal_id,
        "display_name": enrichment.get("display_name"),
        "target_company_name": enrichment.get("target_company_name"),
        "broker": enrichment.get("broker"),
        "last_email_at": enrichment.get("last_email_at"),
        "enrichment_confidence": enrichment.get("enrichment_confidence"),
        "enriched_at": enrichment.get("enriched_at"),
        "materials": {
            "total": len(enrichment.get("materials", [])),
            "by_type": materials_by_type,
            "auth_required_count": len(auth_required),
        },
        "aliases": enrichment.get("aliases", []),
    }


@app.get("/api/enrichment/audit")
def get_enrichment_audit():
    """Get enrichment audit report across all deals."""
    registry_path = Path("/home/zaks/DataRoom/.deal-registry/deal_registry.json")

    if not registry_path.exists():
        return {"error": "Registry not found"}

    try:
        with open(registry_path) as f:
            data = json.load(f)

        deals = data.get("deals", {})

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_deals": len(deals),
            "deals_with_materials": 0,
            "deals_with_display_name": 0,
            "deals_with_broker": 0,
            "deals_enriched": 0,
            "total_materials": 0,
            "materials_by_type": {},
            "auth_required_pending": 0,
            "missing_display_name": [],
        }

        for deal_id, deal in deals.items():
            materials = deal.get("materials", [])
            if materials:
                report["deals_with_materials"] += 1
                report["total_materials"] += len(materials)

                for mat in materials:
                    link_type = mat.get("link_type", "unknown")
                    report["materials_by_type"][link_type] = \
                        report["materials_by_type"].get(link_type, 0) + 1

                    if mat.get("requires_auth") and mat.get("status") == "pending":
                        report["auth_required_pending"] += 1

            if deal.get("display_name"):
                report["deals_with_display_name"] += 1
            else:
                report["missing_display_name"].append(deal_id)

            if deal.get("broker"):
                report["deals_with_broker"] += 1

            if deal.get("enriched_at"):
                report["deals_enriched"] += 1

        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/enrichment/pending-links")
def get_pending_auth_links():
    """Get all pending auth-required links that need manual download."""
    link_queue_path = Path("/home/zaks/DataRoom/.deal-registry/link_intake_queue.json")

    if not link_queue_path.exists():
        return {"count": 0, "links": []}

    try:
        with open(link_queue_path) as f:
            queue = json.load(f)

        pending = [l for l in queue if l.get("status") == "pending"]

        # Group by vendor
        by_vendor = {}
        for link in pending:
            vendor = link.get("vendor_hint", "unknown")
            if vendor not in by_vendor:
                by_vendor[vendor] = []
            by_vendor[vendor].append(link)

        return {
            "count": len(pending),
            "by_vendor": {v: len(links) for v, links in by_vendor.items()},
            "links": pending,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MarkLinkFetchedRequest(BaseModel):
    """Request to mark a link as fetched."""
    url: str
    local_path: str
    operator_note: Optional[str] = None


@app.post("/api/enrichment/mark-link-fetched")
def mark_link_fetched(request: MarkLinkFetchedRequest):
    """Mark an auth-required link as manually fetched."""
    link_queue_path = Path("/home/zaks/DataRoom/.deal-registry/link_intake_queue.json")

    if not link_queue_path.exists():
        raise HTTPException(status_code=404, detail="Link queue not found")

    try:
        with open(link_queue_path) as f:
            queue = json.load(f)

        # Find and update the link
        found = False
        for link in queue:
            if link.get("url") == request.url or link.get("normalized_url") == request.url:
                link["status"] = "fetched"
                link["local_path"] = request.local_path
                link["fetched_at"] = datetime.now(timezone.utc).isoformat()
                if request.operator_note:
                    link["operator_note"] = request.operator_note
                found = True
                break

        if not found:
            raise HTTPException(status_code=404, detail="Link not found in queue")

        with open(link_queue_path, 'w') as f:
            json.dump(queue, f, indent=2)

        return {"success": True, "message": "Link marked as fetched"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/deals/{deal_id}/transition")
def transition_deal(deal_id: str, request: TransitionRequest):
    """Transition deal to a new stage."""
    registry = get_registry()
    deal = registry.get_deal(deal_id)

    if not deal:
        raise HTTPException(status_code=404, detail=f"Deal not found: {deal_id}")

    sm = DealStateMachine(deal.stage, deal_id)
    target = DealStage.from_str(request.to_stage)

    # Check if transition is valid
    if not sm.can_transition_to(target):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid transition: {deal.stage} -> {request.to_stage}. Allowed: {[s.value for s in sm.get_allowed_transitions()]}",
        )

    # Check approval requirement
    needs_approval = sm.requires_approval(target)
    if needs_approval and not request.approved_by:
        raise HTTPException(
            status_code=400,
            detail=f"Transition to {request.to_stage} requires approval (approved_by field)",
        )

    # Execute transition
    result = sm.transition(target, approved=bool(request.approved_by))

    if result.success:
        # Update registry
        registry.update_deal(deal_id, stage=request.to_stage)
        registry.save()

        # Emit event
        event_store = get_event_store()
        event_store.create_event(
            deal_id=deal_id,
            event_type="stage_changed",
            actor=request.approved_by or "api",
            data={
                "from_stage": deal.stage,
                "to_stage": request.to_stage,
                "reason": request.reason,
                "approved_by": request.approved_by,
            },
        )

    return {
        "success": result.success,
        "message": result.message,
        "from_stage": result.from_stage.value,
        "to_stage": result.to_stage.value,
        "approval_required": result.approval_required,
    }


@app.post("/api/deals/{deal_id}/note")
def add_deal_note(deal_id: str, request: NoteRequest):
    """Add operator note to deal."""
    registry = get_registry()
    deal = registry.get_deal(deal_id)

    if not deal:
        raise HTTPException(status_code=404, detail=f"Deal not found: {deal_id}")

    event_store = get_event_store()
    event = event_store.create_event(
        deal_id=deal_id,
        event_type="note_added",
        actor="operator",
        data={
            "content": request.content,
            "category": request.category,
        },
    )

    return {
        "success": True,
        "event_id": event.event_id,
    }


# ===== DEFERRED ACTIONS (LEGACY) =====

@app.get("/api/deferred-actions")
def list_deferred_actions(
    deal_id: Optional[str] = None,
    status: Optional[str] = None,
):
    """List all scheduled actions."""
    action_queue = get_action_queue()

    if deal_id:
        actions = action_queue.get_actions_for_deal(deal_id, status=status)
    else:
        data = action_queue._load()
        actions = [
            action_queue.get_action(aid)
            for aid in data.get("actions", {}).keys()
        ]
        actions = [a for a in actions if a and (not status or a.status == status)]

    return {
        "count": len(actions),
        "actions": [
            {
                "action_id": a.action_id,
                "deal_id": a.deal_id,
                "action_type": a.action_type,
                "scheduled_for": a.scheduled_for,
                "status": a.status,
                "priority": a.priority,
                "is_due": a.is_due(),
            }
            for a in actions
        ],
    }


@app.get("/api/deferred-actions/due")
def get_due_deferred_actions():
    """Get actions that are due for execution."""
    action_queue = get_action_queue()
    due = action_queue.get_due_actions()

    return {
        "count": len(due),
        "actions": [
            {
                "action_id": a.action_id,
                "deal_id": a.deal_id,
                "action_type": a.action_type,
                "scheduled_for": a.scheduled_for,
                "priority": a.priority,
                "data": a.data,
            }
            for a in due
        ],
    }


@app.post("/api/deferred-actions/{action_id}/execute")
def execute_deferred_action(action_id: str):
    """Manually trigger action execution."""
    action_queue = get_action_queue()
    action = action_queue.get_action(action_id)

    if not action:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")

    if action.status != "pending":
        raise HTTPException(status_code=400, detail=f"Action not pending: {action.status}")

    # Mark as executed
    action_queue.mark_executed(action_id, {"executed_via": "api"})

    # Emit event
    event_store = get_event_store()
    event_store.create_event(
        deal_id=action.deal_id,
        event_type="action_executed",
        actor="api",
        data={
            "action_id": action_id,
            "action_type": action.action_type,
            "result": "success",
        },
    )

    return {"success": True, "action_id": action_id}


@app.post("/api/deferred-actions/{action_id}/cancel")
def cancel_deferred_action(action_id: str, reason: str = "Cancelled via API"):
    """Cancel a scheduled action."""
    action_queue = get_action_queue()
    action = action_queue.get_action(action_id)

    if not action:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")

    if action.status != "pending":
        raise HTTPException(status_code=400, detail=f"Action not pending: {action.status}")

    action_queue.cancel(action_id, reason)

    return {"success": True, "action_id": action_id}


# ===== KINETIC ACTIONS (v1.2) =====


@app.get("/api/actions")
def list_kinetic_actions(
    deal_id: Optional[str] = None,
    status: Optional[str] = None,
    type_: Optional[str] = Query(default=None, alias="type"),
    action_type: Optional[str] = None,
    created_after: Optional[str] = None,
    created_before: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """
    List Kinetic Actions (v1.2).

    Filters:
    - deal_id
    - status
    - action_type
    - created_after / created_before (ISO8601)
    """
    store = get_kinetic_action_store()
    actions = store.list_actions(
        deal_id=deal_id,
        status=status,
        action_type=action_type or type_,
        created_after=created_after,
        created_before=created_before,
        limit=limit,
        offset=offset,
    )
    return {"count": len(actions), "actions": [_action_payload_to_frontend(a) for a in actions]}


@app.post("/api/actions")
def create_kinetic_action(request: KineticActionCreateRequest):
    from actions.engine.models import ActionPayload, compute_idempotency_key
    from actions.engine.validation import ActionCreationValidationError, validate_action_creation

    store = get_kinetic_action_store()

    try:
        validate_action_creation(action_type=request.action_type, capability_id=request.capability_id)
    except ActionCreationValidationError as e:
        raise HTTPException(status_code=400, detail=e.to_dict())

    inputs = request.inputs or {}
    idem = request.idempotency_key or compute_idempotency_key(
        request.deal_id or "",
        request.action_type,
        request.title,
        json.dumps(inputs, sort_keys=True),
    )

    action = ActionPayload(
        deal_id=request.deal_id,
        capability_id=request.capability_id,
        type=request.action_type,
        title=request.title,
        summary=request.summary or "",
        status="PENDING_APPROVAL",
        created_by=request.created_by,
        source=request.source,
        risk_level=request.risk_level,
        requires_human_review=bool(request.requires_human_review),
        idempotency_key=idem,
        inputs=inputs,
    )

    created_action, created_new = store.create_action(action)

    # Emit deal event when deal_id present
    if created_action.deal_id:
        try:
            event_store = get_event_store()
            event_store.create_event(
                deal_id=created_action.deal_id,
                event_type="kinetic_action_created",
                actor=request.created_by,
                data={
                    "action_id": created_action.action_id,
                    "action_type": created_action.type,
                    "created_new": created_new,
                    "capability_id": created_action.capability_id,
                },
            )
        except Exception:
            pass

    return {
        "success": True,
        "created_new": created_new,
        "action_id": created_action.action_id,
        "action": _action_payload_to_frontend(created_action),
    }


@app.post("/api/actions/{action_id}/approve")
def approve_kinetic_action(action_id: str, request: KineticActionApproveRequest):
    store = get_kinetic_action_store()
    try:
        action = store.approve_action(action_id, actor=request.approved_by)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if action.deal_id:
        try:
            event_store = get_event_store()
            event_store.create_event(
                deal_id=action.deal_id,
                event_type="kinetic_action_approved",
                actor=request.approved_by,
                data={"action_id": action.action_id, "action_type": action.type},
            )
        except Exception:
            pass

    return {"success": True, "action": _action_payload_to_frontend(action)}


@app.post("/api/actions/{action_id}/execute")
def execute_kinetic_action(action_id: str, request: Optional[KineticActionExecuteRequest] = Body(default=None)):
    """
    Request execution of a READY action.

    Idempotent: returns success if action already in terminal state (COMPLETED/FAILED/CANCELLED).
    """
    store = get_kinetic_action_store()
    actor = request.requested_by if request else "operator"
    try:
        action = store.request_execute(action_id, actor=actor)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    except ValueError as e:
        # Graceful idempotent handling: if action already terminal, return success
        current_action = store.get_action(action_id)
        if current_action and current_action.status in ("COMPLETED", "FAILED", "CANCELLED"):
            return {
                "success": True,
                "already_terminal": True,
                "message": f"Action already {current_action.status}",
                "action": _action_payload_to_frontend(current_action),
            }
        raise HTTPException(status_code=409, detail=str(e))

    if action.deal_id:
        try:
            event_store = get_event_store()
            event_store.create_event(
                deal_id=action.deal_id,
                event_type="kinetic_action_execution_requested",
                actor=actor,
                data={"action_id": action.action_id, "action_type": action.type},
            )
        except Exception:
            pass

    return {"success": True, "action": _action_payload_to_frontend(action)}


@app.post("/api/actions/{action_id}/cancel")
def cancel_kinetic_action(action_id: str, request: Optional[KineticActionCancelRequest] = Body(default=None)):
    store = get_kinetic_action_store()
    actor = request.cancelled_by if request else "operator"
    reason = request.reason if request else "Cancelled via API"
    try:
        action = store.cancel_action(action_id, actor=actor, reason=reason)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    if action.deal_id:
        try:
            event_store = get_event_store()
            event_store.create_event(
                deal_id=action.deal_id,
                event_type="kinetic_action_cancelled",
                actor=actor,
                data={"action_id": action.action_id, "action_type": action.type, "reason": reason},
            )
        except Exception:
            pass

    return {"success": True, "action": _action_payload_to_frontend(action)}


@app.post("/api/actions/{action_id}/update")
def update_kinetic_action(action_id: str, request: KineticActionUpdateRequest):
    store = get_kinetic_action_store()
    try:
        action = store.update_action_inputs(action_id, request.inputs, actor=request.updated_by)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"success": True, "action": _action_payload_to_frontend(action)}


@app.post("/api/actions/{action_id}/retry")
def retry_kinetic_action(action_id: str):
    """
    Frontend compatibility endpoint: retries a FAILED action.

    The backend's canonical admin operation is `/api/actions/{id}/requeue`, but the UI calls `/retry`.
    """
    store = get_kinetic_action_store()
    try:
        action = store.requeue_failed_action(action_id=action_id, actor="operator", reason="ui_retry")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "action": _action_payload_to_frontend(action)}


@app.get("/api/actions/{action_id}/artifacts")
def list_kinetic_action_artifacts(action_id: str):
    store = get_kinetic_action_store()
    action = store.get_action(action_id)
    if not action:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")

    artifacts = []
    for art in action.artifacts or []:
        payload = art.model_dump()
        payload["download_url"] = f"/api/actions/{action_id}/artifact/{art.artifact_id}"
        artifacts.append(payload)
    return {"count": len(artifacts), "artifacts": artifacts}


@app.get("/api/actions/{action_id}/artifact/{artifact_id}")
def download_kinetic_action_artifact(action_id: str, artifact_id: str):
    store = get_kinetic_action_store()
    artifact = store.get_artifact(action_id=action_id, artifact_id=artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    path = Path(artifact.path).resolve()
    try:
        path.relative_to(DATAROOM_ROOT)
    except ValueError:
        raise HTTPException(status_code=400, detail="Artifact path is outside DataRoom")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact file missing on disk")

    return FileResponse(
        str(path),
        media_type=artifact.mime_type,
        filename=artifact.filename,
    )


@app.get("/api/actions/capabilities")
def list_action_capabilities():
    from actions.capabilities.registry import get_registry
    from tools.registry import get_tool_registry

    reg = get_registry()
    try:
        reg.index_tools(get_tool_registry())
    except Exception:
        pass
    caps = [_capability_manifest_to_frontend(c) for c in reg.list_capabilities()]
    return {"count": len(caps), "capabilities": caps}


@app.get("/api/actions/capabilities/{capability_id}")
def get_action_capability(capability_id: str):
    from actions.capabilities.registry import get_registry
    from tools.registry import get_tool_registry

    reg = get_registry()
    try:
        reg.index_tools(get_tool_registry())
    except Exception:
        pass
    cap = reg.get_capability(capability_id)
    if not cap:
        raise HTTPException(status_code=404, detail=f"Capability not found: {capability_id}")
    return _capability_manifest_to_frontend(cap)


@app.get("/api/actions/metrics")
def action_metrics(window_hours: int = Query(default=24, ge=1, le=168)):
    store = get_kinetic_action_store()
    metrics = store.action_metrics(window_hours=window_hours)

    # Frontend (zakops-dashboard/src/lib/api.ts) schema expects these fields:
    # - queue_lengths, avg_duration_by_type, success_rate_24h, total_24h, completed_24h, failed_24h, error_breakdown
    try:
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=max(1, int(window_hours)))).isoformat().replace("+00:00", "Z")

        queue_lengths: Dict[str, int] = {}
        avg_duration_by_type: Dict[str, Dict[str, Any]] = {}
        error_counts: Dict[str, int] = {}

        with store._connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute("SELECT status, COUNT(*) AS c FROM actions GROUP BY status").fetchall()
            queue_lengths = {str(r["status"]): int(r["c"]) for r in rows}
            for s in ("PENDING_APPROVAL", "READY", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED"):
                queue_lengths.setdefault(s, 0)

            completed_24h = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM actions WHERE status='COMPLETED' AND completed_at >= ?",
                    (cutoff,),
                ).fetchone()["c"]
            )
            failed_24h = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM actions WHERE status='FAILED' AND updated_at >= ?",
                    (cutoff,),
                ).fetchone()["c"]
            )

            rows = conn.execute(
                "SELECT type, AVG(duration_seconds) AS avg_d, COUNT(*) AS c FROM actions WHERE status='COMPLETED' AND completed_at >= ? GROUP BY type",
                (cutoff,),
            ).fetchall()
            for r in rows:
                avg_duration_by_type[str(r["type"])] = {"avg_seconds": float(r["avg_d"] or 0.0), "count": int(r["c"])}

            rows = conn.execute(
                "SELECT error FROM actions WHERE status='FAILED' AND updated_at >= ? AND error IS NOT NULL",
                (cutoff,),
            ).fetchall()
            for r in rows:
                raw = r["error"]
                key = "unknown"
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else {}
                    if isinstance(parsed, dict):
                        key = str(parsed.get("code") or parsed.get("message") or "unknown")
                    else:
                        key = str(parsed)
                except Exception:
                    key = str(raw)[:120]
                error_counts[key] = error_counts.get(key, 0) + 1

        total_24h = int(completed_24h) + int(failed_24h)
        success_rate_24h = float(completed_24h) / float(total_24h) if total_24h else 0.0

        metrics["queue_lengths"] = queue_lengths
        metrics["avg_duration_by_type"] = avg_duration_by_type
        metrics["success_rate_24h"] = success_rate_24h
        metrics["total_24h"] = total_24h
        metrics["completed_24h"] = completed_24h
        metrics["failed_24h"] = failed_24h
        metrics["error_breakdown"] = [
            {"error": k, "count": int(v)} for k, v in sorted(error_counts.items(), key=lambda kv: kv[1], reverse=True)[:25]
        ]
    except Exception:
        # Never fail the endpoint due to metrics enrichment; keep legacy keys.
        pass

    lease = store.get_runner_lease(runner_name="kinetic_actions")
    metrics["runner_lease"] = lease.model_dump() if lease else None
    stuck_ids = store.list_stuck_processing_action_ids(older_than_seconds=180, limit=20)
    metrics["stuck_processing"] = {"older_than_seconds": 180, "count": len(stuck_ids), "action_ids": stuck_ids}
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    return metrics


@app.post("/api/actions/plan")
def plan_action(request: KineticActionPlanRequest):
    """
    Deterministic-first planner endpoint (v1.2).

    This is intentionally offline and safe by default; LangGraph brain integration can layer on top.
    """
    from actions.planner import ActionPlanner

    planner = ActionPlanner()
    plan = planner.plan(request.query, provided_inputs=request.inputs or {})
    return plan.model_dump()


# Cached dependency health checks for ops endpoints (avoid spawning processes per request).
_DEPENDENCY_HEALTH_CACHE: Dict[str, Any] = {}


def _gmail_mcp_health_cached() -> Dict[str, Any]:
    ttl_s = int(os.getenv("ZAKOPS_GMAIL_MCP_HEALTH_TTL_SECONDS", "30") or "30")
    now = datetime.now(timezone.utc)
    cached = _DEPENDENCY_HEALTH_CACHE.get("gmail_mcp")
    if isinstance(cached, dict):
        checked_at = cached.get("_checked_at")
        if isinstance(checked_at, datetime) and (now - checked_at).total_seconds() < ttl_s:
            res = cached.get("result")
            if isinstance(res, dict):
                return res

    try:
        from tools.mcp_health import check_gmail_mcp_health

        result = check_gmail_mcp_health(timeout_ms=2500)
    except Exception as e:
        result = {"ok": False, "reason": f"exception:{type(e).__name__}", "checked_at": now.isoformat()}

    _DEPENDENCY_HEALTH_CACHE["gmail_mcp"] = {"_checked_at": now, "result": result}
    return result if isinstance(result, dict) else {"ok": False, "reason": "invalid_health_result"}


@app.get("/api/actions/runner-status")
def actions_runner_status():
    """
    Runner observability endpoint (world-class ops).

    Returns:
    - runner lease holder + heartbeat
    - queue counts by status
    - whether the runner appears alive
    """
    store = get_kinetic_action_store()
    lease = store.get_runner_lease(runner_name="kinetic_actions")
    metrics = store.action_metrics(window_hours=24)
    now_iso = datetime.now(timezone.utc).isoformat()

    runner_alive = False
    if lease:
        try:
            from datetime import datetime as _dt

            lease_exp = _dt.fromisoformat(lease.lease_expires_at.replace("Z", "+00:00"))
            runner_alive = lease_exp > _dt.now(timezone.utc)
        except Exception:
            runner_alive = False

    stuck_processing = {"older_than_seconds": 180, "count": 0, "action_ids": []}
    try:
        stuck_ids = store.list_stuck_processing_action_ids(older_than_seconds=180, limit=20)
        stuck_processing = {"older_than_seconds": 180, "count": len(stuck_ids), "action_ids": stuck_ids}
    except Exception:
        pass

    error_breakdown: List[Dict[str, Any]] = []
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat().replace("+00:00", "Z")
        error_counts: Dict[str, int] = {}
        with store._connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute(
                "SELECT error FROM actions WHERE status='FAILED' AND updated_at >= ? AND error IS NOT NULL",
                (cutoff,),
            ).fetchall()
            for r in rows:
                raw = r["error"]
                key = "unknown"
                try:
                    parsed = json.loads(raw) if isinstance(raw, str) else {}
                    if isinstance(parsed, dict):
                        key = str(parsed.get("code") or parsed.get("message") or "unknown")
                    else:
                        key = str(parsed)
                except Exception:
                    key = str(raw)[:120]
                error_counts[key] = error_counts.get(key, 0) + 1
        error_breakdown = [
            {"error": k, "count": int(v)}
            for k, v in sorted(error_counts.items(), key=lambda kv: kv[1], reverse=True)[:25]
        ]
    except Exception:
        error_breakdown = []

    return {
        "timestamp": now_iso,
        "runner_alive": runner_alive,
        "runner_lease": lease.model_dump() if lease else None,
        "queue": metrics.get("queue") or {},
        "stuck_processing": stuck_processing,
        "error_breakdown": error_breakdown,
        "processing_ttl_seconds": int(os.getenv("ZAKOPS_ACTION_PROCESSING_TTL_SECONDS", "3600")),
        "gmail_mcp": _gmail_mcp_health_cached(),
    }


@app.get("/api/gmail/health")
def gmail_health():
    """
    Gmail MCP health endpoint (ops diagnostic).

    Uses the same cached health check as `/api/actions/runner-status` to avoid spawning a node process per request.
    """
    return _gmail_mcp_health_cached()


@app.get("/metrics")
def prometheus_metrics():
    """
    Prometheus text-format metrics (no prometheus_client dependency).

    This endpoint is designed to be cheap to compute and safe to scrape frequently.
    """
    store = get_kinetic_action_store()
    metrics = store.action_metrics(window_hours=24)
    queue = metrics.get("queue") or {}

    lease = store.get_runner_lease(runner_name="kinetic_actions")
    runner_alive = 0
    if lease:
        try:
            lease_exp = datetime.fromisoformat(str(lease.lease_expires_at).replace("Z", "+00:00"))
            runner_alive = 1 if lease_exp > datetime.now(timezone.utc) else 0
        except Exception:
            runner_alive = 0

    quarantine_pending = 0
    try:
        with store._connect() as conn:  # type: ignore[attr-defined]
            row = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM actions
                WHERE type='EMAIL_TRIAGE.REVIEW_EMAIL'
                  AND status='PENDING_APPROVAL'
                  AND hidden_from_quarantine=0
                """
            ).fetchone()
            quarantine_pending = int(row["c"] or 0) if row else 0
    except Exception:
        quarantine_pending = 0

    stuck_processing = 0
    try:
        stuck_processing = len(store.list_stuck_processing_action_ids(older_than_seconds=600, limit=500))
    except Exception:
        stuck_processing = 0

    gmail_ok = 1 if bool((_gmail_mcp_health_cached() or {}).get("ok")) else 0

    lines: List[str] = []
    lines.append("# HELP zakops_actions_total Total actions stored in the action DB.")
    lines.append("# TYPE zakops_actions_total gauge")
    lines.append(f"zakops_actions_total {int(metrics.get('total') or 0)}")
    lines.append("")

    lines.append("# HELP zakops_actions_by_status Count of actions by status (queue view).")
    lines.append("# TYPE zakops_actions_by_status gauge")
    lines.append(f'zakops_actions_by_status{{status="PENDING_APPROVAL"}} {int(queue.get("pending_approval") or 0)}')
    lines.append(f'zakops_actions_by_status{{status="READY"}} {int(queue.get("ready") or 0)}')
    lines.append(f'zakops_actions_by_status{{status="READY_QUEUED"}} {int(queue.get("ready_queued") or 0)}')
    lines.append(f'zakops_actions_by_status{{status="PROCESSING"}} {int(queue.get("processing") or 0)}')
    lines.append("")

    lines.append("# HELP zakops_quarantine_pending Pending quarantine items (EMAIL_TRIAGE.REVIEW_EMAIL).")
    lines.append("# TYPE zakops_quarantine_pending gauge")
    lines.append(f"zakops_quarantine_pending {int(quarantine_pending)}")
    lines.append("")

    lines.append("# HELP zakops_actions_stuck_processing Count of PROCESSING actions with stale heartbeats.")
    lines.append("# TYPE zakops_actions_stuck_processing gauge")
    lines.append(f"zakops_actions_stuck_processing {int(stuck_processing)}")
    lines.append("")

    lines.append("# HELP zakops_runner_alive Whether the kinetic actions runner lease appears alive.")
    lines.append("# TYPE zakops_runner_alive gauge")
    lines.append(f"zakops_runner_alive {int(runner_alive)}")
    lines.append("")

    lines.append("# HELP zakops_gmail_mcp_ok Whether Gmail MCP healthcheck succeeded.")
    lines.append("# TYPE zakops_gmail_mcp_ok gauge")
    lines.append(f"zakops_gmail_mcp_ok {int(gmail_ok)}")
    lines.append("")

    body = "\n".join(lines).strip() + "\n"
    return PlainTextResponse(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/api/diagnostics")
def system_diagnostics():
    """
    Unified diagnostics endpoint for debugging ZakOps services.

    Returns:
    - triage: last run stats, health status
    - runner: alive status, queue sizes
    - vllm: model availability
    """
    from datetime import datetime, timezone, timedelta

    now = datetime.now(timezone.utc)
    result: Dict[str, Any] = {"timestamp": now.isoformat()}

    # 1. Triage stats from persisted file
    triage_stats_path = DATAROOM_ROOT / ".triage_stats.json"
    triage_stats_fallback = DATAROOM_ROOT / ".deal-registry" / "triage_stats.json"
    if (not triage_stats_path.exists()) and triage_stats_fallback.exists():
        triage_stats_path = triage_stats_fallback
    triage: Dict[str, Any] = {"healthy": False, "reason": "no_stats_file"}
    if triage_stats_path.exists():
        try:
            stats = json.loads(triage_stats_path.read_text())
            last_run = datetime.fromisoformat(stats.get("last_run_at", "").replace("Z", "+00:00"))
            age_hours = (now - last_run).total_seconds() / 3600
            triage = {
                "healthy": age_hours < 2 and stats.get("failed", 0) < 5,
                "last_run_at": stats.get("last_run_at"),
                "age_hours": round(age_hours, 2),
                "processed": stats.get("processed", 0),
                "skipped": stats.get("skipped", 0),
                "failed": stats.get("failed", 0),
                "query": stats.get("query"),
                "stats_path": str(triage_stats_path),
            }
            if age_hours >= 2:
                triage["reason"] = "stale_run"
            elif stats.get("failed", 0) >= 5:
                triage["reason"] = "too_many_failures"
        except Exception as e:
            triage = {"healthy": False, "reason": f"parse_error:{str(e)}"}
    result["triage"] = triage

    # 2. Runner status (reuse existing logic)
    store = get_kinetic_action_store()
    lease = store.get_runner_lease(runner_name="kinetic_actions")
    metrics = store.action_metrics(window_hours=24)

    runner_alive = False
    if lease:
        try:
            lease_exp = datetime.fromisoformat(lease.lease_expires_at.replace("Z", "+00:00"))
            runner_alive = lease_exp > now
        except Exception:
            pass

    result["runner"] = {
        "healthy": runner_alive,
        "lease_owner": lease.owner_id if lease else None,
        "heartbeat_at": lease.heartbeat_at if lease else None,
        "queue": metrics.get("queue") or {},
    }

    # 3. vLLM status
    vllm_healthy = False
    vllm_model = None
    try:
        import urllib.request
        vllm_url = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
        req = urllib.request.Request(f"{vllm_url}/models", method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            if data.get("data"):
                vllm_model = data["data"][0].get("id")
                vllm_healthy = True
    except Exception:
        pass
    result["vllm"] = {"healthy": vllm_healthy, "model": vllm_model}

    # 4. Overall health
    result["overall_healthy"] = all([
        triage.get("healthy", False),
        runner_alive,
        vllm_healthy,
    ])

    return result


@app.post("/api/actions/{action_id}/unstick")
def unstick_kinetic_action(action_id: str, request: Optional[KineticActionUnstickRequest] = Body(default=None)):
    """
    Operator admin endpoint: force a stuck PROCESSING action back to READY.

    This is safe and idempotent; it does not execute the action. The runner must claim it again.
    """
    store = get_kinetic_action_store()
    actor = request.unstuck_by if request else "operator"
    reason = request.reason if request else "operator_unstick"

    action = store.get_action(action_id)
    if not action:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    if action.status != "PROCESSING":
        raise HTTPException(status_code=409, detail=f"Action not PROCESSING: {action.status}")

    ok = store.unstick_action(action_id=action_id, actor=actor, reason=reason)
    if not ok:
        raise HTTPException(status_code=409, detail="unstick_failed")

    refreshed = store.get_action(action_id)
    return {"success": True, "action": _action_payload_to_frontend(refreshed or action)}


@app.post("/api/actions/{action_id}/requeue")
def requeue_kinetic_action(action_id: str, request: KineticActionRequeueRequest):
    store = get_kinetic_action_store()
    try:
        action = store.requeue_failed_action(action_id=action_id, actor=request.requeued_by, reason=request.reason)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "action": _action_payload_to_frontend(action)}


@app.get("/api/actions/{action_id}/debug")
def debug_kinetic_action(action_id: str):
    """
    Debug endpoint for a single action: shows lock/lease state, audit tail, and last tool invocation (if any).
    """
    store = get_kinetic_action_store()
    action = store.get_action(action_id)
    if not action:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")

    # Last tool invocation (best-effort)
    last_tool = None
    try:
        import sqlite3

        db_path = os.getenv("ZAKOPS_STATE_DB", "/home/zaks/DataRoom/.deal-registry/ingest_state.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT invocation_id, tool_name, provider, success, error_code, error_message,
                   attempt_number, attempt_started_at, attempt_completed_at, attempt_duration_ms
            FROM tool_invocation_log
            WHERE action_id = ?
            ORDER BY attempt_started_at DESC
            LIMIT 1
            """,
            (action_id,),
        ).fetchone()
        conn.close()
        if row:
            last_tool = dict(row)
    except Exception:
        last_tool = None

    audit_tail = (action.audit_trail or [])[-10:]

    transitions: Dict[str, Optional[str]] = {}
    try:
        for ev in action.audit_trail or []:
            name = (ev.event or "").strip()
            if not name:
                continue
            transitions.setdefault(name, ev.timestamp)
    except Exception:
        transitions = {}

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action.model_dump(),
        "audit_tail": [e.model_dump() for e in audit_tail],
        "last_tool_invocation": last_tool,
        "transition_times": transitions,
    }


def _triage_action_to_quarantine_item(action: Any) -> Dict[str, Any]:
    """Normalize an EMAIL_TRIAGE.REVIEW_EMAIL action to the legacy /api/quarantine item shape."""
    try:
        payload: Dict[str, Any] = action.model_dump()
    except Exception:
        payload = dict(action or {})

    inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
    subject = str(inputs.get("subject") or payload.get("title") or "").strip()
    sender = str(inputs.get("from") or "").strip()
    received_at = str(inputs.get("date") or payload.get("created_at") or "").strip()
    classification = str(inputs.get("classification") or "").strip()
    urgency = str(inputs.get("urgency") or "").strip()

    reason = payload.get("summary") or ""
    if not reason:
        reason = "Deal signal review" if classification else "Review inbound email"
        if urgency:
            reason = f"{reason} ({urgency})"

    return {
        "id": payload.get("action_id"),
        "quarantine_id": payload.get("action_id"),
        "action_id": payload.get("action_id"),
        "deal_id": payload.get("deal_id") or "GLOBAL",
        "status": payload.get("status") or "",
        "email_subject": subject,
        "subject": subject,
        "sender": sender,
        "from": sender,
        "received_at": received_at,
        "timestamp": payload.get("created_at") or received_at,
        "quarantine_reason": reason,
        "reason": reason,
        # Extra fields (safe to ignore by frontend)
        "classification": classification,
        "urgency": urgency,
        "company": inputs.get("company"),
        "links": inputs.get("links") or [],
        "attachments": inputs.get("attachments") or [],
        "quarantine_dir": inputs.get("quarantine_dir"),
        "capability_id": payload.get("capability_id"),
    }


@app.get("/api/actions/quarantine")
def list_quarantine_actions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """
    Canonical quarantine queue (v2): approval-gated triage actions.

    Backed by the kinetic actions store:
      type=EMAIL_TRIAGE.REVIEW_EMAIL AND status=PENDING_APPROVAL
    """
    store = get_kinetic_action_store()
    actions = store.list_actions(
        action_type="EMAIL_TRIAGE.REVIEW_EMAIL",
        status="PENDING_APPROVAL",
        limit=limit,
        offset=offset,
    )
    items = [_triage_action_to_quarantine_item(a) for a in actions]
    return {"count": len(items), "items": items}


@app.get("/api/actions/quarantine/{action_id}")
def get_quarantine_action(action_id: str):
    store = get_kinetic_action_store()
    action = store.get_action(action_id)
    if not action:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    if action.type != "EMAIL_TRIAGE.REVIEW_EMAIL":
        raise HTTPException(status_code=400, detail="not_a_triage_quarantine_action")
    return {
        "action": _action_payload_to_frontend(action),
        "quarantine_item": _triage_action_to_quarantine_item(action),
    }


@app.get("/api/actions/quarantine/{action_id}/preview")
def get_quarantine_action_preview(action_id: str):
    """
    Local-only preview payload for the Quarantine decision UI right-side panel.

    IMPORTANT:
    - Must not require cloud/LLM.
    - Must enforce quarantine_dir stays within DATAROOM_ROOT.
    """
    import re

    store = get_kinetic_action_store()
    action = store.get_action(action_id)
    if not action:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    if action.type != "EMAIL_TRIAGE.REVIEW_EMAIL":
        raise HTTPException(status_code=400, detail="not_a_triage_quarantine_action")

    inputs = action.inputs or {}
    message_id = str(inputs.get("message_id") or "").strip()
    thread_id = str(inputs.get("thread_id") or "").strip()
    email_from = str(inputs.get("from") or "").strip()
    email_to = str(inputs.get("to") or "").strip()
    email_date = str(inputs.get("date") or "").strip()
    subject = str(inputs.get("subject") or action.title or "").strip()
    company = str(inputs.get("company") or "").strip() or None
    classification = str(inputs.get("classification") or "").strip() or None
    urgency = str(inputs.get("urgency") or "").strip() or None

    quarantine_dir_raw = str(inputs.get("quarantine_dir") or "").strip()
    quarantine_dir = (
        Path(quarantine_dir_raw).expanduser().resolve()
        if quarantine_dir_raw
        else (DATAROOM_ROOT / "00-PIPELINE" / "_INBOX_QUARANTINE" / message_id).resolve()
    )
    try:
        quarantine_dir.relative_to(DATAROOM_ROOT)
    except ValueError:
        quarantine_dir = None  # type: ignore[assignment]

    email_body = ""
    if quarantine_dir and (quarantine_dir / "email_body.txt").exists():
        try:
            email_body = (quarantine_dir / "email_body.txt").read_text(encoding="utf-8", errors="replace")
        except Exception:
            email_body = ""
    if email_body and len(email_body) > 20000:
        email_body = email_body[:20000] + "…"

    # Attachments inventory (from inputs + filesystem presence).
    attachments_in = inputs.get("attachments") or []
    att_items: List[Dict[str, Any]] = []
    if isinstance(attachments_in, list):
        for a in attachments_in:
            if not isinstance(a, dict):
                continue
            fname = str(a.get("filename") or "").strip()
            if not fname:
                continue
            att_items.append(
                {
                    "filename": fname,
                    "mime_type": a.get("mime_type"),
                    "size_bytes": a.get("size_bytes"),
                    "downloaded_path": str((quarantine_dir / fname).resolve()) if quarantine_dir and (quarantine_dir / fname).exists() else None,
                }
            )

    # Links - use link normalizer for classification, deduplication, and grouping
    links_in = inputs.get("links") or []
    links_for_processing = []
    if isinstance(links_in, list):
        for item in links_in:
            if isinstance(item, dict) and str(item.get("url") or "").strip():
                links_for_processing.append(item)

    # Process links with normalizer (classify, dedupe, group)
    try:
        normalized_result = normalize_links(links_for_processing)
    except Exception:
        normalized_result = {"groups": {}, "all_unique": [], "unique_count": 0, "total_count": 0}

    # Build legacy-compatible grouped structure while adding new categorization
    grouped: Dict[str, List[Dict[str, Any]]] = {
        k: [] for k in ["deal_material", "tracking", "social", "unsubscribe", "calendar", "portal", "contact", "other"]
    }
    # Also maintain backwards-compatible groups for UI
    legacy_grouped: Dict[str, List[Dict[str, Any]]] = {
        k: [] for k in ["dataroom", "cim", "teaser", "nda", "financials", "calendar", "docs", "other"]
    }

    all_links: List[Dict[str, Any]] = []
    for link in normalized_result.get("all_unique", []):
        entry = {
            "type": link.get("link_type", "other"),
            "category": link.get("category", "other"),
            "url": link.get("canonical_url") or link.get("url", ""),
            "original_url": link.get("url", ""),
            "auth_required": bool(link.get("auth_required", False)),
            "vendor_hint": link.get("vendor_hint"),
            "resolved_url": link.get("resolved_url"),
        }

        # Add to new category groups
        category = link.get("category", "other")
        if category in grouped:
            grouped[category].append(entry)
        else:
            grouped["other"].append(entry)

        # Map to legacy groups for backwards compatibility
        link_type = link.get("link_type", "other")
        if link_type in legacy_grouped:
            legacy_grouped[link_type].append(entry)
        elif link_type in ("dataroom", "cim", "teaser", "nda", "financials"):
            legacy_grouped[link_type].append(entry) if link_type in legacy_grouped else legacy_grouped["other"].append(entry)
        else:
            legacy_grouped["other"].append(entry)

        all_links.append(entry)

    # Deterministic extracted fields (best-effort).
    email_addr = ""
    m = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", email_from)
    if m:
        email_addr = m.group(1).lower()

    company_guess = company
    if not company_guess:
        subj = re.sub(r"^(re:|fw:|fwd:)\\s*", "", subject, flags=re.I).strip()
        if " - " in subj:
            left = subj.split(" - ", 1)[0].strip()
            if 3 <= len(left) <= 80:
                company_guess = left

    asking_price = None
    if email_body:
        # Cheap money heuristic (first mention near "asking" or "price", else first $ amount).
        money = re.findall(r"(\\$\\s?\\d[\\d,]*(?:\\.\\d+)?\\s*(?:mm|m|million|k|thousand|b|bn|billion)?)", email_body, flags=re.I)
        if money:
            asking_price = money[0].strip()

    summary: List[str] = []
    if classification or urgency:
        summary.append(f"Classification: {classification or 'unknown'} ({urgency or 'unknown'} urgency)")
    if company_guess:
        summary.append(f"Company: {company_guess}")
    if all_links:
        auth_required = sum(1 for l in all_links if l.get("auth_required"))
        summary.append(f"Links: {len(all_links)} ({auth_required} auth-required)")
    if att_items:
        summary.append(f"Attachments: {len(att_items)}")
    if asking_price:
        summary.append(f"Asking/price mentioned: {asking_price}")
    if email_body:
        body_lower = email_body.lower()
        keywords = [k for k in ["nda", "cim", "teaser", "dataroom", "financials", "ebitda", "revenue"] if k in body_lower]
        if keywords:
            summary.append(f"Body keywords: {', '.join(keywords[:6])}")
    summary = summary[:6]

    preview = {
        "action_id": action.action_id,
        "status": action.status,
        "created_at": action.created_at,
        "deal_id": action.deal_id or "GLOBAL",
        "message_id": message_id,
        "thread_id": thread_id,
        "from": email_from,
        "to": email_to,
        "received_at": email_date,
        "subject": subject,
        "summary": summary,
        "extracted_fields": {
            "broker_email": email_addr or None,
            "company_guess": company_guess,
            "asking_price": asking_price,
            "industry": None,
            "location": None,
        },
        "attachments": {"count": len(att_items), "items": att_items},
        "links": {
            "groups": grouped,
            "legacy_groups": legacy_grouped,  # Backwards-compatible grouping
            "all": all_links,
            "stats": {
                "total_raw": normalized_result.get("total_count", len(links_for_processing)),
                "unique_count": normalized_result.get("unique_count", len(all_links)),
                "duplicates_removed": normalized_result.get("duplicates_removed", 0),
                "tracking_count": len(grouped.get("tracking", [])),
                "deal_material_count": len(grouped.get("deal_material", [])),
            },
        },
        "email": {
            "body_snippet": (email_body[:1200] + "…") if email_body and len(email_body) > 1200 else email_body,
            "body": email_body,
        },
        "quarantine_dir": str(quarantine_dir) if quarantine_dir else None,
    }
    try:
        reg = get_registry()
        preview["thread_resolution"] = {
            "thread_to_deal": reg.get_thread_deal_mapping(thread_id) if thread_id else None,
            "thread_to_non_deal": reg.get_thread_non_deal_mapping(thread_id) if thread_id else None,
        }
    except Exception:
        preview["thread_resolution"] = {"thread_to_deal": None, "thread_to_non_deal": None}
    return preview


class QuarantineRejectRequest(BaseModel):
    operator: str = Field(..., min_length=1, description="Operator name/initials")
    reason: Optional[str] = Field(default=None, description="Rejection reason")


@app.post("/api/actions/quarantine/{action_id}/reject")
def reject_quarantine_item(action_id: str, request: QuarantineRejectRequest):
    """
    Atomic reject for quarantine items (EMAIL_TRIAGE.REVIEW_EMAIL → non-deal).

    Flow:
    1. Validate original action exists and is PENDING_APPROVAL
    2. Create EMAIL_TRIAGE.REJECT_EMAIL action (auto-approved since labeling is safe)
    3. Execute it synchronously (labels email, records thread mapping)
    4. Cancel the original action
    5. Return success

    This is idempotent: re-rejecting an already-canceled action returns success.
    """
    from actions.engine.models import ActionPayload, compute_idempotency_key

    store = get_kinetic_action_store()

    # 1. Validate original action
    original = store.get_action(action_id)
    if not original:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    if original.type != "EMAIL_TRIAGE.REVIEW_EMAIL":
        raise HTTPException(status_code=400, detail="not_a_triage_quarantine_action")

    # Idempotency: already canceled/rejected means success
    if original.status in ("CANCELLED", "COMPLETED", "FAILED"):
        return {"ok": True, "already_resolved": True, "action_id": action_id}

    if original.status != "PENDING_APPROVAL":
        raise HTTPException(status_code=409, detail=f"invalid_status:{original.status}")

    inputs = original.inputs or {}
    message_id = str(inputs.get("message_id") or "").strip()
    thread_id = str(inputs.get("thread_id") or "").strip()

    if not message_id:
        raise HTTPException(status_code=400, detail="original_action_missing_message_id")

    # 2. Create reject action
    reject_inputs = {
        "message_id": message_id,
        "thread_id": thread_id,
        "reason": request.reason or "operator_rejected",
    }
    idem_key = compute_idempotency_key(
        original.deal_id or "GLOBAL",
        "EMAIL_TRIAGE.REJECT_EMAIL",
        f"reject:{action_id}",
        json.dumps(reject_inputs, sort_keys=True),
    )

    reject_action = ActionPayload(
        deal_id=original.deal_id,
        type="EMAIL_TRIAGE.REJECT_EMAIL",
        title="Reject email (non-deal)",
        summary=f"Rejected: {request.reason}" if request.reason else "Rejected as non-deal",
        status="PENDING_APPROVAL",
        created_by=request.operator,
        source="ui",
        risk_level="low",
        requires_human_review=False,  # Safe labeling only
        idempotency_key=idem_key,
        inputs=reject_inputs,
    )

    created, is_new = store.create_action(reject_action)
    reject_action_id = created.action_id

    # 3. Auto-approve and execute (labeling is safe)
    try:
        store.approve_action(reject_action_id, actor=request.operator)
    except ValueError:
        pass  # Already approved (idempotent re-submit)

    # Queue for execution
    try:
        store.request_execute(reject_action_id, actor=request.operator)
    except ValueError:
        pass  # Already queued

    # Wait for runner to complete (with timeout)
    import time
    max_wait = 15  # seconds
    start = time.monotonic()
    final_status = "READY"
    while time.monotonic() - start < max_wait:
        refreshed = store.get_action(reject_action_id)
        if refreshed and refreshed.status in ("COMPLETED", "FAILED", "CANCELLED"):
            final_status = refreshed.status
            break
        time.sleep(0.5)
    else:
        refreshed = store.get_action(reject_action_id)
        final_status = refreshed.status if refreshed else "UNKNOWN"

    if final_status != "COMPLETED":
        # Reject action didn't complete, but we still proceed with canceling original
        # since we at least tried. The reject action will be retried by runner.
        pass

    # 4. Cancel original action
    try:
        store.cancel_action(action_id, actor=request.operator, reason="Rejected as non-deal")
    except (KeyError, ValueError):
        pass  # Already canceled or invalid state

    return {
        "ok": True,
        "reject_action_id": reject_action_id,
        "reject_status": final_status,
        "original_action_id": action_id,
    }


class QuarantineApproveRequest(BaseModel):
    operator: str = Field(..., min_length=1, description="Operator name/initials")
    link_to_deal_id: Optional[str] = Field(default=None, description="Link to existing deal instead of creating new")


@app.post("/api/actions/quarantine/{action_id}/approve")
def approve_quarantine_item(action_id: str, request: QuarantineApproveRequest):
    """
    Atomic approve for quarantine items (EMAIL_TRIAGE.REVIEW_EMAIL → deal).

    Flow:
    1. Validate original action exists and is PENDING_APPROVAL
    2. Create DEAL.CREATE_FROM_EMAIL or DEAL.APPEND_EMAIL_MATERIALS action
    3. Execute it SYNCHRONOUSLY (not queued) - ensures atomicity
    4. If execution succeeds: cancel original action, return success with deal_id
    5. If execution fails: leave original action pending, return error

    This is idempotent: re-approving already-processed action returns success with deal_id.
    """
    from dataclasses import asdict
    from actions.engine.models import ActionPayload, compute_idempotency_key, now_utc_iso
    from actions.executors.registry import get_executor

    store = get_kinetic_action_store()

    # 1. Validate original action
    original = store.get_action(action_id)
    if not original:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    if original.type != "EMAIL_TRIAGE.REVIEW_EMAIL":
        raise HTTPException(status_code=400, detail="not_a_triage_quarantine_action")

    inputs = original.inputs or {}
    message_id = str(inputs.get("message_id") or "").strip()
    thread_id = str(inputs.get("thread_id") or "").strip()

    # Idempotency: already completed means success - find the deal
    if original.status in ("COMPLETED", "CANCELLED"):
        registry = get_registry()
        deal_id = registry.get_email_deal_mapping(message_id)
        if not deal_id and thread_id:
            deal_id = registry.get_thread_deal_mapping(thread_id)
        return {
            "ok": True,
            "already_resolved": True,
            "action_id": action_id,
            "deal_id": deal_id,
        }

    if original.status != "PENDING_APPROVAL":
        raise HTTPException(status_code=409, detail=f"invalid_status:{original.status}")

    # 2. Check if this should link to existing deal or create new
    registry = get_registry()
    existing_deal_id = request.link_to_deal_id
    if not existing_deal_id and thread_id:
        existing_deal_id = registry.get_thread_deal_mapping(thread_id)

    # 3. Prepare the deal creation/append action
    quarantine_dir = str(inputs.get("quarantine_dir") or "").strip()
    if not quarantine_dir and message_id:
        quarantine_dir = str(DATAROOM_ROOT / "00-PIPELINE" / "_INBOX_QUARANTINE" / message_id)

    if existing_deal_id:
        # Append to existing deal
        action_type = "DEAL.APPEND_EMAIL_MATERIALS"
        title = f"Append approved email to deal {existing_deal_id}"
        existing_deal = registry.get_deal(existing_deal_id)
        if not existing_deal:
            raise HTTPException(status_code=404, detail=f"Deal not found: {existing_deal_id}")
        action_inputs = {
            "deal_id": existing_deal_id,
            "deal_path": existing_deal.folder_path,
            "message_id": message_id,
            "thread_id": thread_id,
            "from": inputs.get("from", ""),
            "to": inputs.get("to", ""),
            "date": inputs.get("date", ""),
            "subject": inputs.get("subject", ""),
            "quarantine_dir": quarantine_dir,
        }
    else:
        # Create new deal
        action_type = "DEAL.CREATE_FROM_EMAIL"
        title = f"Create deal from approved email: {inputs.get('subject', '')[:60]}"
        action_inputs = {
            "gmail_message_id": message_id,
            "gmail_thread_id": thread_id,
            "message_id": message_id,
            "thread_id": thread_id,
            "subject": inputs.get("subject", ""),
            "from_email": inputs.get("from", "").split("<")[-1].rstrip(">").strip() if "<" in inputs.get("from", "") else inputs.get("from", ""),
            "from_header": inputs.get("from", ""),
            "received_at": inputs.get("date", ""),
            "snippet": inputs.get("snippet", ""),
            "quarantine_dir": quarantine_dir,
            "links": inputs.get("links", []),
            "attachments": inputs.get("attachments", []),
        }

    idem_key = compute_idempotency_key(action_type, action_inputs)

    # Check if this action already exists (idempotency)
    existing_action = store.get_action_by_idempotency_key(idem_key)
    if existing_action and existing_action.status == "COMPLETED":
        deal_id = (existing_action.outputs or {}).get("deal_id")
        # Still cancel the original triage action
        try:
            store.cancel_action(action_id, actor=request.operator, reason="Already processed (idempotent)")
        except (KeyError, ValueError):
            pass
        return {
            "ok": True,
            "already_resolved": True,
            "action_id": action_id,
            "deal_id": deal_id,
            "deal_action_id": existing_action.action_id,
        }

    # 4. Create the deal action
    deal_action = store.create_action(
        deal_id=existing_deal_id or "",
        capability_id=f"deal.{action_type.split('.')[-1].lower()}.v1",
        action_type=action_type,
        title=title,
        summary=f"Synchronous execution from quarantine approval by {request.operator}",
        inputs=action_inputs,
        source="quarantine_approve",
        created_by=request.operator,
        risk_level="low",
        requires_human_review=False,
        max_retries=0,  # No retries for synchronous execution
        parent_action_id=action_id,
        idempotency_key=idem_key,
    )

    # Auto-approve and mark as processing
    store.approve_action(deal_action.action_id, actor=request.operator)
    if not store.begin_processing(action_id=deal_action.action_id, owner_id=f"api_sync_{request.operator}", lease_seconds=300):
        raise HTTPException(status_code=500, detail="Failed to acquire action lock for synchronous execution")

    # 5. Execute SYNCHRONOUSLY
    try:
        executor = get_executor(action_type)
        if executor is None:
            store.mark_action_completed(
                action_id=deal_action.action_id,
                actor="api_sync",
                outputs={},
                error={"code": "executor_not_found", "message": f"No executor for {action_type}", "category": "dependency", "retryable": False},
            )
            raise HTTPException(status_code=500, detail=f"No executor registered for {action_type}")

        # Refresh action to get processing state
        deal_action = store.get_action(deal_action.action_id)

        ok, err = executor.validate(deal_action)
        if not ok:
            store.mark_action_completed(
                action_id=deal_action.action_id,
                actor="api_sync",
                outputs={},
                error={"code": "validation_failed", "message": err or "Validation failed", "category": "validation", "retryable": False},
            )
            raise HTTPException(status_code=400, detail=f"Validation failed: {err}")

        # Create execution context
        from actions.executors.base import ExecutionContext

        deal_for_ctx = registry.get_deal(existing_deal_id) if existing_deal_id else None
        exec_ctx = ExecutionContext(
            action=deal_action,
            deal=asdict(deal_for_ctx) if deal_for_ctx else None,
            case_file=None,
            tool_gateway=None,
            cloud_allowed=False,
            registry=registry,
        )

        result = executor.execute(deal_action, exec_ctx)

        # Mark completed
        store.mark_action_completed(
            action_id=deal_action.action_id,
            actor="api_sync",
            outputs=result.outputs or {},
            error=None,
        )

        deal_id = (result.outputs or {}).get("deal_id")

        # 6. Cancel original triage action
        try:
            store.cancel_action(action_id, actor=request.operator, reason=f"Approved: deal {deal_id} created")
        except (KeyError, ValueError):
            pass

        return {
            "ok": True,
            "action_id": action_id,
            "deal_action_id": deal_action.action_id,
            "deal_id": deal_id,
            "deal_path": (result.outputs or {}).get("deal_path"),
            "created_new_deal": action_type == "DEAL.CREATE_FROM_EMAIL",
        }

    except HTTPException:
        raise
    except Exception as e:
        # Mark action as failed but leave original quarantine item pending
        from actions.engine.models import ActionError

        error = ActionError(
            code="sync_execution_failed",
            message=f"{type(e).__name__}: {str(e)}",
            category="unknown",
            retryable=False,
        )
        store.mark_action_completed(
            action_id=deal_action.action_id,
            actor="api_sync",
            outputs={},
            error=error.model_dump() if hasattr(error, "model_dump") else {"code": error.code, "message": error.message},
        )
        raise HTTPException(status_code=500, detail=f"Deal creation failed: {str(e)}. Quarantine item remains pending.")


@app.get("/api/actions/debug/missing-executors")
def debug_actions_missing_executors(limit: int = Query(default=100, ge=1, le=500)):
    """
    Operator tooling: find actions whose `type` has no registered executor.

    This helps diagnose "executor_not_found" failures and prevents silent broken rows.
    """
    store = get_kinetic_action_store()
    from actions.executors.registry import list_executors

    executors = set(list_executors())
    missing_types: List[str] = []
    actions: List[Dict[str, Any]] = []

    try:
        with store._connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute("SELECT DISTINCT type FROM actions WHERE type NOT LIKE 'TOOL.%'").fetchall()
            types = [str(r["type"]) for r in rows]
            missing_types = sorted([t for t in types if t not in executors])

            if missing_types:
                placeholders = ",".join(["?"] * len(missing_types))
                q = (
                    "SELECT action_id, deal_id, capability_id, type, status, created_at, updated_at "
                    f"FROM actions WHERE type IN ({placeholders}) ORDER BY created_at DESC LIMIT ?"
                )
                rows2 = conn.execute(q, (*missing_types, int(limit))).fetchall()
                actions = [dict(r) for r in rows2]
    except Exception:
        # Never fail operator tooling endpoint.
        pass

    return {"count": len(actions), "missing_types": missing_types, "actions": actions}


@app.get("/api/actions/debug/capability-mismatches")
def debug_actions_capability_mismatches(limit: int = Query(default=200, ge=1, le=1000)):
    """
    Operator tooling: find actions whose capability_id is missing/invalid or mismatched vs action.type.
    """
    store = get_kinetic_action_store()
    from actions.capabilities.registry import get_registry as get_capability_registry
    from tools.registry import get_tool_registry

    cap_reg = get_capability_registry()
    try:
        cap_reg.index_tools(get_tool_registry())
    except Exception:
        pass

    mismatches: List[Dict[str, Any]] = []
    try:
        with store._connect() as conn:  # type: ignore[attr-defined]
            rows = conn.execute(
                """
                SELECT action_id, deal_id, capability_id, type, status, created_at, updated_at
                FROM actions
                WHERE capability_id IS NOT NULL AND TRIM(capability_id) != ''
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            for r in rows:
                cap_id = str(r["capability_id"] or "").strip()
                at = str(r["type"] or "").strip()
                manifest = cap_reg.get_capability(cap_id)
                if manifest is None:
                    mismatches.append(
                        {
                            **dict(r),
                            "problem": "capability_not_found",
                        }
                    )
                    continue
                expected = str(getattr(manifest, "action_type", "") or "").strip()
                if expected != at:
                    mismatches.append(
                        {
                            **dict(r),
                            "problem": "capability_action_type_mismatch",
                            "capability_action_type": expected,
                        }
                    )
    except Exception:
        pass

    return {"count": len(mismatches), "mismatches": mismatches}


@app.get("/api/actions/{action_id}")
def get_kinetic_action(action_id: str):
    store = get_kinetic_action_store()
    action = store.get_action(action_id)
    if not action:
        raise HTTPException(status_code=404, detail=f"Action not found: {action_id}")
    return _action_payload_to_frontend(action)


# ===== TOOLING (PHASE 0.5) =====


@app.get("/api/tools")
def list_tools():
    from tools.registry import get_tool_registry, tool_registry_to_json

    reg = get_tool_registry()
    tools = [tool_registry_to_json(t) for t in reg.list_tools()]
    return {"count": len(tools), "tools": tools}


@app.get("/api/tools/health")
async def tools_health():
    from tools.registry import get_tool_registry

    reg = get_tool_registry()
    results = await reg.check_all_health()
    return {
        "healthy": all(s.healthy for s in results.values()),
        "tools": {
            tool_id: {
                "healthy": bool(status.healthy),
                "latency_ms": int(status.latency_ms),
                "error": status.error,
                "last_check": status.last_check.isoformat(),
            }
            for tool_id, status in results.items()
        },
    }


@app.get("/api/tools/{tool_id}")
def get_tool(tool_id: str):
    from tools.registry import get_tool_registry, tool_registry_to_json

    reg = get_tool_registry()
    tool = reg.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_id}")
    return tool_registry_to_json(tool)


# ===== QUARANTINE ENDPOINTS =====
# NOTE: /health must come BEFORE /{quarantine_id} to avoid route collision

@app.get("/api/quarantine/health")
def quarantine_health():
    """Get quarantine health status."""
    items: List[Dict[str, Any]] = []

    # Include action-backed quarantine (canonical).
    try:
        store = get_kinetic_action_store()
        triage = store.list_actions(action_type="EMAIL_TRIAGE.REVIEW_EMAIL", status="PENDING_APPROVAL", limit=200, offset=0, exclude_hidden=True)
        items.extend([_triage_action_to_quarantine_item(a) for a in triage])
    except Exception:
        pass

    # Include legacy filesystem quarantine (best-effort for backwards compatibility).
    try:
        qm = get_quarantine()
        items.extend(list(qm.get_pending() or []))
    except Exception:
        pass

    return {
        "status": "healthy" if len(items) < 10 else "attention_needed",
        "pending_items": len(items),
        "oldest_pending_days": 0,  # TODO: Calculate from timestamps
    }


@app.get("/api/quarantine")
def list_quarantine():
    """List quarantine items needing resolution."""
    items: List[Dict[str, Any]] = []

    # Canonical quarantine: pending triage actions (action-backed queue).
    try:
        store = get_kinetic_action_store()
        triage = store.list_actions(action_type="EMAIL_TRIAGE.REVIEW_EMAIL", status="PENDING_APPROVAL", limit=200, offset=0, exclude_hidden=True)
        items.extend([_triage_action_to_quarantine_item(a) for a in triage])
    except Exception:
        pass

    # Legacy filesystem quarantine (best-effort).
    try:
        qm = get_quarantine()
        items.extend(list(qm.get_pending() or []))
    except Exception:
        pass

    return {
        "count": len(items),
        "items": items,
    }


@app.get("/api/quarantine/{quarantine_id}")
def get_quarantine_item(quarantine_id: str):
    """Get a specific quarantine item."""
    # Prefer action-backed quarantine.
    try:
        store = get_kinetic_action_store()
        action = store.get_action(quarantine_id)
        if action and action.type == "EMAIL_TRIAGE.REVIEW_EMAIL":
            return _triage_action_to_quarantine_item(action)
    except Exception:
        pass

    qm = get_quarantine()
    item = qm.get_item(quarantine_id)

    if not item:
        raise HTTPException(status_code=404, detail=f"Quarantine item not found: {quarantine_id}")

    return item


@app.post("/api/quarantine/{quarantine_id}/delete")
def delete_quarantine_item(quarantine_id: str, request: QuarantineDeleteRequest):
    store = get_kinetic_action_store()
    action = store.get_action(quarantine_id)
    if not action or action.type != "EMAIL_TRIAGE.REVIEW_EMAIL":
        raise HTTPException(status_code=404, detail=f"Quarantine item not found: {quarantine_id}")

    if not store.hide_quarantine_item(quarantine_id, actor=request.deleted_by, reason=request.reason):
        raise HTTPException(status_code=409, detail="Quarantine item already hidden")

    return {"hidden": True, "quarantine_id": quarantine_id, "deleted_by": request.deleted_by}


@app.post("/api/quarantine/bulk-delete")
def bulk_delete_quarantine_items(request: QuarantineBulkDeleteRequest):
    store = get_kinetic_action_store()
    summary = {"hidden": [], "missing": [], "already_hidden": []}
    for action_id in request.action_ids:
        action = store.get_action(action_id)
        if not action or action.type != "EMAIL_TRIAGE.REVIEW_EMAIL":
            summary["missing"].append(action_id)
            continue
        if action.hidden_from_quarantine:
            summary["already_hidden"].append(action_id)
            continue
        if store.hide_quarantine_item(action_id, actor=request.deleted_by, reason=request.reason):
            summary["hidden"].append(action_id)
        else:
            summary["missing"].append(action_id)
    return summary


@app.post("/api/quarantine/{quarantine_id}/resolve")
def resolve_quarantine(quarantine_id: str, request: QuarantineResolveRequest):
    """Resolve a quarantine item.

    For action-backed quarantine (EMAIL_TRIAGE.REVIEW_EMAIL), this now uses
    SYNCHRONOUS deal creation to ensure atomicity - deals appear immediately
    or the quarantine item remains pending with an error.
    """
    # Prefer action-backed quarantine resolution for triage review actions.
    try:
        store = get_kinetic_action_store()
        action = store.get_action(quarantine_id)
        if action and action.type == "EMAIL_TRIAGE.REVIEW_EMAIL":
            if request.resolution == "discard":
                store.cancel_action(quarantine_id, actor=request.resolved_by, reason="quarantine_discard")
                return {"success": True}

            if request.resolution in {"link_to_deal", "create_new_deal"}:
                # Use the atomic approve endpoint for synchronous deal creation
                approve_request = QuarantineApproveRequest(
                    operator=request.resolved_by,
                    link_to_deal_id=request.deal_id if request.resolution == "link_to_deal" else None,
                )
                result = approve_quarantine_item(quarantine_id, approve_request)
                return {
                    "success": result.get("ok", False),
                    "action_id": quarantine_id,
                    "deal_id": result.get("deal_id"),
                    "deal_path": result.get("deal_path"),
                    "created_new_deal": result.get("created_new_deal", False),
                }

            raise HTTPException(status_code=400, detail=f"Invalid resolution: {request.resolution}")
    except HTTPException:
        raise
    except Exception:
        pass

    # Legacy filesystem quarantine fallback.
    qm = get_quarantine()
    item = qm.get_item(quarantine_id)

    if not item:
        raise HTTPException(status_code=404, detail=f"Quarantine item not found: {quarantine_id}")

    if request.resolution == "link_to_deal":
        if not request.deal_id:
            raise HTTPException(status_code=400, detail="deal_id required for link_to_deal")
        qm.resolve(quarantine_id, "link_to_deal", request.deal_id, request.resolved_by)

    elif request.resolution == "create_new_deal":
        registry = get_registry()
        deal_id = registry.generate_deal_id()
        registry.create_deal(
            deal_id=deal_id,
            canonical_name=item.get("email_subject", "")[:100],
            folder_path=str((DATAROOM_ROOT / "00-PIPELINE" / "Inbound" / f"Legacy-Quarantine-{deal_id}").resolve()),
            source="quarantine_resolution",
        )
        registry.save()
        qm.resolve(quarantine_id, "create_new_deal", deal_id, request.resolved_by)
        request.deal_id = deal_id

    elif request.resolution == "discard":
        qm.resolve(quarantine_id, "discard", resolved_by=request.resolved_by)

    else:
        raise HTTPException(status_code=400, detail=f"Invalid resolution: {request.resolution}")

    # Emit event
    event_store = get_event_store()
    event_store.create_event(
        deal_id=request.deal_id or "",
        event_type="quarantine_resolved",
        actor=request.resolved_by,
        data={
            "quarantine_id": quarantine_id,
            "resolution": request.resolution,
        },
    )

    return {"success": True, "resolution": request.resolution, "deal_id": request.deal_id}


# ===== METRICS ENDPOINTS =====

@app.get("/api/metrics/classification")
def classification_metrics():
    """Get classification metrics."""
    qm = get_quarantine()
    pending = qm.get_pending()

    return {
        "decisions_24h": 0,  # TODO: Calculate from events
        "local_24h": 0,
        "cloud_24h": 0,
        "heuristic_24h": 0,
        "quarantine_rate": len(pending) / 100 if pending else 0,
    }


@app.get("/api/checkpoints")
def list_checkpoints():
    """List active checkpoints/operations."""
    checkpoint_dir = Path("/home/zaks/DataRoom/.deal-registry/checkpoints")
    checkpoints = []

    if checkpoint_dir.exists():
        for cp_file in checkpoint_dir.glob("*.json"):
            try:
                with open(cp_file) as f:
                    cp = json.load(f)
                    checkpoints.append({
                        "id": cp.get("checkpoint_id", cp_file.stem),
                        "operation_type": cp.get("operation_type", "unknown"),
                        "status": cp.get("status", "unknown"),
                        "progress": cp.get("progress", 0),
                    })
            except (json.JSONDecodeError, IOError):
                continue

    return checkpoints


# ===== PIPELINE ENDPOINTS =====

@app.get("/api/pipeline")
def get_pipeline():
    """Get pipeline summary by stage."""
    registry = get_registry()
    deals = registry.list_deals(status="active")

    # Group by stage
    stages = {}
    stage_order = [
        "inbound", "screening", "qualified", "loi", "diligence",
        "closing", "integration", "operations", "growth", "exit_planning",
    ]

    for stage in stage_order:
        stages[stage] = {"count": 0, "deals": [], "avg_age": 0}

    for deal in deals:
        stage = deal.stage
        if stage not in stages:
            stages[stage] = {"count": 0, "deals": [], "avg_age": 0}
        stages[stage]["count"] += 1
        stages[stage]["deals"].append({
            "deal_id": deal.deal_id,
            "canonical_name": deal.canonical_name,
            "days_in_stage": days_ago(deal.updated_at),
        })

    # Calculate averages
    for stage, data in stages.items():
        if data["deals"]:
            data["avg_age"] = sum(d["days_in_stage"] for d in data["deals"]) / len(data["deals"])

    return {
        "total_active": len(deals),
        "stages": stages,
    }


@app.get("/api/alerts")
def get_alerts():
    """Get system alerts (stuck deals, due actions, etc.)."""
    registry = get_registry()
    action_queue = get_action_queue()
    qm = get_quarantine()

    alerts = []

    # Check for stuck deals
    deals = registry.list_deals(status="active")
    for deal in deals:
        age = days_ago(deal.updated_at)
        if age > 30:
            sm = DealStateMachine(deal.stage)
            if not sm.is_terminal():
                alerts.append({
                    "type": "stuck_deal",
                    "severity": "warning" if age < 60 else "high",
                    "deal_id": deal.deal_id,
                    "message": f"Deal has not been updated in {age} days",
                })

    # Check for overdue actions
    due_actions = action_queue.get_due_actions()
    if due_actions:
        alerts.append({
            "type": "overdue_actions",
            "severity": "warning",
            "count": len(due_actions),
            "message": f"{len(due_actions)} action(s) overdue",
            "actions": [a.action_id for a in due_actions[:5]],
        })

    # Check quarantine
    quarantine_items = qm.get_pending()
    if quarantine_items:
        alerts.append({
            "type": "pending_quarantine",
            "severity": "info",
            "count": len(quarantine_items),
            "message": f"{len(quarantine_items)} item(s) in quarantine",
        })

    return {
        "alert_count": len(alerts),
        "alerts": alerts,
    }


# ===== AGENT ENDPOINTS =====

@app.post("/api/agents/{agent_name}/invoke")
def invoke_agent(agent_name: str, request: AgentInvokeRequest):
    """Invoke an agent with a task."""
    from deal_case_manager import DealCaseManagerAgent
    from deal_underwriter import UnderwriterAgent
    from deal_diligence_coordinator import DiligenceCoordinatorAgent

    agents = {
        "case_manager": DealCaseManagerAgent,
        "underwriter": UnderwriterAgent,
        "diligence_coordinator": DiligenceCoordinatorAgent,
    }

    if agent_name not in agents:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown agent: {agent_name}. Available: {list(agents.keys())}",
        )

    agent = agents[agent_name]()

    # Map tasks to methods
    task_methods = {
        "case_manager": {
            "summarize": agent.summarize if hasattr(agent, 'summarize') else None,
            "suggest-actions": agent.suggest_actions if hasattr(agent, 'suggest_actions') else None,
            "process-events": agent.process_events if hasattr(agent, 'process_events') else None,
        },
        "underwriter": {
            "analyze": agent.analyze if hasattr(agent, 'analyze') else None,
            "score": agent.score if hasattr(agent, 'score') else None,
        },
        "diligence_coordinator": {
            "initialize": agent.initialize if hasattr(agent, 'initialize') else None,
            "progress": agent.progress if hasattr(agent, 'progress') else None,
            "missing-docs": agent.missing_docs if hasattr(agent, 'missing_docs') else None,
        },
    }

    method = task_methods.get(agent_name, {}).get(request.task)
    if not method:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{request.task}' for agent '{agent_name}'",
        )

    result = method(request.deal_id)

    return {
        "agent": agent_name,
        "task": request.task,
        "deal_id": request.deal_id,
        "result": result.to_dict(),
    }


@app.get("/api/agents/{agent_name}/history")
def get_agent_history(agent_name: str, limit: int = Query(default=20, le=100)):
    """Get recent agent invocations."""
    ledger_path = Path("/home/zaks/logs/run-ledger.jsonl")

    if not ledger_path.exists():
        return {"agent": agent_name, "invocations": []}

    invocations = []
    with open(ledger_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if entry.get("agent") == agent_name:
                    invocations.append(entry)
            except json.JSONDecodeError:
                continue

    # Return most recent first
    invocations = list(reversed(invocations[-limit:]))

    return {
        "agent": agent_name,
        "count": len(invocations),
        "invocations": invocations,
    }


# ===== CHAT ENDPOINTS =====

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """Chat with streaming SSE response."""
    from fastapi.responses import StreamingResponse
    from chat_orchestrator import get_orchestrator

    orchestrator = get_orchestrator()

    scope = {
        "type": request.scope.type,
        "deal_id": request.scope.deal_id,
        "doc": request.scope.doc,
    }

    async def generate():
        async for event in orchestrator.chat_stream(
            query=request.query,
            scope=scope,
            session_id=request.session_id,
            options=request.options,
        ):
            yield event

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/chat/complete")
async def chat_complete(request: ChatRequest):
    """Chat with non-streaming complete response."""
    from chat_orchestrator import get_orchestrator

    orchestrator = get_orchestrator()

    scope = {
        "type": request.scope.type,
        "deal_id": request.scope.deal_id,
        "doc": request.scope.doc,
    }

    response = await orchestrator.chat(
        query=request.query,
        scope=scope,
        session_id=request.session_id,
        options=request.options,
    )

    return {
        "content": response.content,
        "citations": response.citations,
        "proposals": response.proposals,
        "evidence_summary": response.evidence_summary,
        "model_used": response.model_used,
        "latency_ms": response.latency_ms,
        "warnings": response.warnings,
    }


@app.post("/api/chat/execute-proposal")
async def execute_proposal(request: ProposalExecuteRequest):
    """Execute an approved chat proposal."""
    from chat_orchestrator import get_orchestrator

    orchestrator = get_orchestrator()

    result = await orchestrator.execute_proposal(
        proposal_id=request.proposal_id,
        approved_by=request.approved_by,
        session_id=request.session_id,
        action=request.action,
        reject_reason=request.reject_reason,
    )

    if not result.get("success"):
        # Map reason to HTTP status codes
        reason = result.get("reason", "execution_failed")
        status_map = {
            "session_not_found": 404,
            "proposal_not_found": 404,
            "invalid_status_transition": 409,
            "unknown_proposal_type": 400,
            "invalid_proposal_params": 400,
            "cloud_disabled": 400,
            "gemini_unavailable": 503,
            "invalid_worker_output": 502,
            "execution_failed": 500,
        }
        status_code = status_map.get(reason, 400)
        # Return full error details including reason for UI handling
        return JSONResponse(
            status_code=status_code,
            content={
                "error": result.get("error", "Execution failed"),
                "reason": reason,
                "success": False,
                **{k: v for k, v in result.items() if k not in ("success", "error", "reason")},
            },
        )

    return result


@app.get("/api/chat/session/{session_id}")
def get_chat_session(session_id: str):
    """Get chat session history."""
    from chat_orchestrator import get_orchestrator

    orchestrator = get_orchestrator()
    data = orchestrator.get_session(session_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return data


@app.get("/api/chat/llm-health")
async def llm_health():
    """
    Check LLM backend health and return comprehensive provider info.

    Returns multi-provider health status including:
    - vLLM (local)
    - Gemini Flash/Pro (cloud)
    - Budget/rate limits
    - Cache stats
    """
    from chat_orchestrator import get_orchestrator

    try:
        orchestrator = get_orchestrator()
        health = await orchestrator.get_health_status()
        return health
    except Exception as e:
        # Fallback to basic health check if new system fails
        import httpx
        from chat_orchestrator import OPENAI_API_BASE, VLLM_MODEL

        result = {
            "status": "error",
            "endpoint": OPENAI_API_BASE,
            "configured_model": VLLM_MODEL,
            "error": str(e),
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{OPENAI_API_BASE}/models")
                if resp.status_code == 200:
                    result["status"] = "healthy"
        except:
            pass

        return result


# ===== VERSION & HEALTH CHECK =====

# Store server start time at module load
_SERVER_START_TIME = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

@app.get("/api/version")
def get_version():
    """
    Return version handshake info for deployment verification.
    Includes git commit, server PID, start time, and safe config snapshot.
    """
    import subprocess

    # Get git commit from dashboard repo (scripts dir is not a git repo)
    git_commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd="/home/zaks/zakops-dashboard",
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()[:12]  # Short hash
    except Exception:
        pass

    # Safe config snapshot (no secrets)
    return {
        "git_commit": git_commit,
        "server_pid": os.getpid(),
        "server_start_time": _SERVER_START_TIME,
        "config": {
            "openai_api_base": os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
            "allow_cloud_default": os.environ.get("ALLOW_CLOUD_DEFAULT", "false"),
            "gemini_model_pro": os.environ.get("GEMINI_MODEL_PRO", "gemini-2.5-pro"),
            "registry_path": str(REGISTRY_PATH) if 'REGISTRY_PATH' in dir() else "unknown",
            "case_file_dir": str(CASE_FILE_DIR) if 'CASE_FILE_DIR' in dir() else "unknown",
        },
    }


@app.get("/api/debug/config")
def get_debug_config():
    """
    Return full debug configuration for deployment verification.
    Includes all paths, service info, and current working directory.
    """
    import subprocess
    import getpass

    # Get git commits from both repos
    git_commits = {}
    for repo_name, repo_path in [
        ("dashboard", "/home/zaks/zakops-dashboard"),
        ("scripts", "/home/zaks/scripts"),
        ("bookkeeping", "/home/zaks/bookkeeping"),
    ]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                git_commits[repo_name] = result.stdout.strip()[:12]
        except Exception:
            git_commits[repo_name] = "not_a_git_repo"

    # Pipeline root paths
    pipeline_root = DATAROOM_ROOT / "00-PIPELINE"

    return {
        "server": {
            "pid": os.getpid(),
            "start_time": _SERVER_START_TIME,
            "cwd": os.getcwd(),
            "user": getpass.getuser(),
        },
        "paths": {
            "dataroom_root": str(DATAROOM_ROOT),
            "pipeline_root": str(pipeline_root),
            "registry_path": str(REGISTRY_PATH),
            "case_files_dir": str(CASE_FILES_DIR),
            "quarantine_dir": str(pipeline_root / "Quarantine") if pipeline_root.exists() else None,
            "inbound_dir": str(pipeline_root / "Inbound") if pipeline_root.exists() else None,
        },
        "environment": {
            "openai_api_base": os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"),
            "rag_base_url": os.environ.get("RAG_BASE_URL", "http://localhost:8001"),
            "llm_base_url": os.environ.get("LLM_BASE_URL", os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")),
            "allow_cloud_default": os.environ.get("ALLOW_CLOUD_DEFAULT", "false"),
        },
        "git_commits": git_commits,
        "features": {
            "classified_links_enabled": True,
            "deal_profile_enabled": True,
            "idempotent_execute": True,
        },
    }


@app.get("/health")
def health_check():
    """API health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "version": "1.0.0",
    }


def main():
    parser = argparse.ArgumentParser(description="Deal Lifecycle API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8090, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print(f"Starting Deal Lifecycle API on {args.host}:{args.port}")
    uvicorn.run(
        "deal_lifecycle_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
