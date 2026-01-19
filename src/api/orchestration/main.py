#!/usr/bin/env python3
"""
ZakOps Orchestration API
========================

FastAPI-based REST API for the Deal Lifecycle Engine UI.
Provides endpoints for deals, actions, quarantine, and pipeline management.
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Any, Dict
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# Phase 5: API Stabilization - Middleware imports
from ..shared.middleware import register_error_handlers, TraceMiddleware
# Phase 7: Authentication - Middleware and router imports
from ..shared.middleware import AuthMiddleware
from ..shared.routers.auth import router as auth_router
# Phase 8: Health endpoints
from ..shared.routers.health import router as health_router
from ..shared.openapi import setup_openapi
# Phase 10: Agent Integration
from .routers.invoke import router as agent_router
# Phase 13: Production Hardening
from .routers.admin import router as admin_router
from ..shared.security import SecurityMiddleware
# Phase 15: Observability
from ..shared.middleware import TracingMiddleware
# Phase 16: Feature Development
from .routers.workflow import router as workflow_router
from .routers.search import router as search_router
from .routers.timeline import router as timeline_router

# Configuration
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://dealengine:changeme@localhost:5435/zakops"
)
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "9200"))

# Database connection pool
db_pool: Optional[asyncpg.Pool] = None


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class DealBase(BaseModel):
    canonical_name: str
    display_name: Optional[str] = None
    folder_path: Optional[str] = None
    stage: str = "inbound"
    status: str = "active"


class DealCreate(DealBase):
    identifiers: Optional[Dict[str, Any]] = None
    company_info: Optional[Dict[str, Any]] = None
    broker: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class DealUpdate(BaseModel):
    canonical_name: Optional[str] = None
    display_name: Optional[str] = None
    folder_path: Optional[str] = None
    stage: Optional[str] = None
    status: Optional[str] = None
    identifiers: Optional[Dict[str, Any]] = None
    company_info: Optional[Dict[str, Any]] = None
    broker: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class DealResponse(BaseModel):
    deal_id: str
    canonical_name: str
    display_name: Optional[str]
    folder_path: Optional[str]
    stage: str
    status: str
    identifiers: Dict[str, Any]
    company_info: Dict[str, Any]
    broker: Dict[str, Any]
    metadata: Dict[str, Any]
    email_thread_ids: List[str]
    created_at: datetime
    updated_at: datetime
    days_since_update: Optional[float] = None
    action_count: Optional[int] = None
    alias_count: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class ActionResponse(BaseModel):
    action_id: str
    deal_id: Optional[str]
    capability_id: str
    action_type: str
    title: str
    summary: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime
    risk_level: str
    requires_human_review: bool
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    deal_name: Optional[str] = None
    deal_stage: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ActionApprove(BaseModel):
    approved_by: str = "ui_user"
    notes: Optional[str] = None


class ActionReject(BaseModel):
    rejected_by: str = "ui_user"
    reason: str


class QuarantineResponse(BaseModel):
    id: str
    message_id: Optional[str]
    email_subject: Optional[str]
    sender: Optional[str]
    sender_domain: Optional[str]
    received_at: Optional[datetime]
    classification: str
    urgency: str
    confidence: Optional[float]
    company_name: Optional[str]
    broker_name: Optional[str]
    status: str
    created_at: datetime
    sender_name: Optional[str] = None
    sender_company: Optional[str] = None
    is_broker: Optional[bool] = None

    model_config = ConfigDict(from_attributes=True)


class QuarantineProcess(BaseModel):
    action: str = Field(..., pattern="^(approve|reject)$")
    processed_by: str = "ui_user"
    deal_id: Optional[str] = None
    notes: Optional[str] = None


class PipelineSummary(BaseModel):
    stage: str
    count: int
    avg_days_in_stage: Optional[float]


class DealEvent(BaseModel):
    id: int
    deal_id: str
    event_type: str
    source: str
    actor: Optional[str]
    details: Dict[str, Any]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# DATABASE HELPERS
# =============================================================================

async def get_db() -> asyncpg.Pool:
    """Get database connection pool."""
    if db_pool is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db_pool


def record_to_dict(record: asyncpg.Record) -> Dict[str, Any]:
    """Convert asyncpg Record to dict, handling JSON fields."""
    result = dict(record)
    for key, value in result.items():
        if isinstance(value, str):
            # Try to parse JSON strings
            if value.startswith('{') or value.startswith('['):
                try:
                    result[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass
    return result


# =============================================================================
# OBSERVABILITY INITIALIZATION
# =============================================================================

def init_observability():
    """Initialize observability components (tracing, metrics, logging)."""
    from ...core.observability import init_tracing, init_metrics, configure_logging

    # Get OTel config from environment
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    otel_enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"
    console_export = os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_structured = os.getenv("LOG_STRUCTURED", "true").lower() == "true"

    # Configure structured logging first
    configure_logging(
        level=log_level,
        structured=log_structured,
        service_name="zakops-backend"
    )

    # Initialize tracing if enabled
    if otel_enabled or otlp_endpoint:
        init_tracing(
            service_name="zakops-backend",
            service_version=os.getenv("APP_VERSION", "1.0.0"),
            otlp_endpoint=otlp_endpoint,
            console_export=console_export
        )

        init_metrics(
            service_name="zakops-backend",
            otlp_endpoint=otlp_endpoint,
            console_export=console_export
        )

        print("OpenTelemetry observability initialized")


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global db_pool

    # Initialize observability
    init_observability()

    # Startup
    print(f"Connecting to PostgreSQL: {DB_URL.split('@')[1] if '@' in DB_URL else DB_URL}")
    db_pool = await asyncpg.create_pool(
        DB_URL,
        min_size=2,
        max_size=10,
        command_timeout=60
    )
    print("Database pool created")

    yield

    # Shutdown
    if db_pool:
        await db_pool.close()
        print("Database pool closed")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="ZakOps Orchestration API",
    description="REST API for Deal Lifecycle Engine",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase 5: API Stabilization - Error handlers and trace middleware
register_error_handlers(app)
app.add_middleware(TraceMiddleware)

# Phase 15: OpenTelemetry tracing middleware
app.add_middleware(TracingMiddleware)

# Phase 7: Authentication middleware (after trace, before routes)
app.add_middleware(AuthMiddleware)

# Phase 7: Auth router
app.include_router(auth_router)

# Phase 8: Health endpoints
app.include_router(health_router)

# Phase 10: Agent Integration
app.include_router(agent_router)

# Phase 13: Admin/operator endpoints
app.include_router(admin_router)

# Phase 16: Feature Development - Workflow, Search, Timeline
app.include_router(workflow_router)
app.include_router(search_router)
app.include_router(timeline_router)

# Phase 13: Security middleware
app.add_middleware(SecurityMiddleware)

# Setup custom OpenAPI schema
setup_openapi(app)


# =============================================================================
# DEALS ENDPOINTS
# =============================================================================

@app.get("/api/deals", response_model=List[DealResponse])
async def list_deals(
    stage: Optional[str] = Query(None, description="Filter by stage"),
    status: Optional[str] = Query("active", description="Filter by status"),
    search: Optional[str] = Query(None, description="Search in name"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    pool: asyncpg.Pool = Depends(get_db)
):
    """List all deals with optional filtering."""
    conditions = ["deleted = FALSE"]
    params = []
    param_idx = 1

    if stage:
        conditions.append(f"stage = ${param_idx}")
        params.append(stage)
        param_idx += 1

    if status:
        conditions.append(f"status = ${param_idx}")
        params.append(status)
        param_idx += 1

    if search:
        conditions.append(f"(canonical_name ILIKE ${param_idx} OR display_name ILIKE ${param_idx})")
        params.append(f"%{search}%")
        param_idx += 1

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT
            d.*,
            EXTRACT(DAY FROM NOW() - d.updated_at) AS days_since_update,
            (SELECT COUNT(*) FROM zakops.actions a WHERE a.deal_id = d.deal_id) AS action_count,
            (SELECT COUNT(*) FROM zakops.deal_aliases a WHERE a.deal_id = d.deal_id) AS alias_count
        FROM zakops.deals d
        WHERE {where_clause}
        ORDER BY d.updated_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
    """
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [record_to_dict(row) for row in rows]


@app.get("/api/deals/{deal_id}", response_model=DealResponse)
async def get_deal(deal_id: str, pool: asyncpg.Pool = Depends(get_db)):
    """Get a single deal by ID."""
    query = """
        SELECT
            d.*,
            EXTRACT(DAY FROM NOW() - d.updated_at) AS days_since_update,
            (SELECT COUNT(*) FROM zakops.actions a WHERE a.deal_id = d.deal_id) AS action_count,
            (SELECT COUNT(*) FROM zakops.deal_aliases a WHERE a.deal_id = d.deal_id) AS alias_count
        FROM zakops.deals d
        WHERE d.deal_id = $1 AND d.deleted = FALSE
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, deal_id)

    if not row:
        raise HTTPException(status_code=404, detail=f"Deal {deal_id} not found")

    return record_to_dict(row)


@app.post("/api/deals", response_model=DealResponse)
async def create_deal(deal: DealCreate, pool: asyncpg.Pool = Depends(get_db)):
    """Create a new deal."""
    async with pool.acquire() as conn:
        # Generate new deal ID
        deal_id = await conn.fetchval("SELECT zakops.next_deal_id()")

        # Insert deal
        query = """
            INSERT INTO zakops.deals (
                deal_id, canonical_name, display_name, folder_path,
                stage, status, identifiers, company_info, broker, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING *
        """
        row = await conn.fetchrow(
            query,
            deal_id,
            deal.canonical_name,
            deal.display_name,
            deal.folder_path,
            deal.stage,
            deal.status,
            json.dumps(deal.identifiers or {}),
            json.dumps(deal.company_info or {}),
            json.dumps(deal.broker or {}),
            json.dumps(deal.metadata or {})
        )

        # Record event
        await conn.execute(
            """
            SELECT zakops.record_deal_event(
                $1, 'deal_created', 'api', 'ui_user', $2::jsonb
            )
            """,
            deal_id,
            json.dumps({"canonical_name": deal.canonical_name})
        )

    return record_to_dict(row)


@app.patch("/api/deals/{deal_id}", response_model=DealResponse)
async def update_deal(
    deal_id: str,
    update: DealUpdate,
    pool: asyncpg.Pool = Depends(get_db)
):
    """Update an existing deal."""
    # Build dynamic update query
    updates = []
    params = [deal_id]
    param_idx = 2

    update_dict = update.model_dump(exclude_unset=True)

    for field, value in update_dict.items():
        if value is not None:
            if field in ['identifiers', 'company_info', 'broker', 'metadata']:
                updates.append(f"{field} = ${param_idx}::jsonb")
                params.append(json.dumps(value))
            else:
                updates.append(f"{field} = ${param_idx}")
                params.append(value)
            param_idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    query = f"""
        UPDATE zakops.deals
        SET {', '.join(updates)}
        WHERE deal_id = $1 AND deleted = FALSE
        RETURNING *
    """

    async with pool.acquire() as conn:
        # Get previous state
        prev = await conn.fetchrow(
            "SELECT * FROM zakops.deals WHERE deal_id = $1",
            deal_id
        )
        if not prev:
            raise HTTPException(status_code=404, detail=f"Deal {deal_id} not found")

        # Update
        row = await conn.fetchrow(query, *params)

        # Record event
        await conn.execute(
            """
            SELECT zakops.record_deal_event(
                $1, 'deal_updated', 'api', 'ui_user', $2::jsonb, $3::jsonb, $4::jsonb
            )
            """,
            deal_id,
            json.dumps({"fields_updated": list(update_dict.keys())}),
            json.dumps(dict(prev)),
            json.dumps(dict(row))
        )

    return record_to_dict(row)


@app.get("/api/deals/{deal_id}/events", response_model=List[DealEvent])
async def get_deal_events(
    deal_id: str,
    limit: int = Query(50, ge=1, le=200),
    pool: asyncpg.Pool = Depends(get_db)
):
    """Get event history for a deal."""
    query = """
        SELECT id, deal_id, event_type, source, actor, details, created_at
        FROM zakops.deal_events
        WHERE deal_id = $1
        ORDER BY created_at DESC
        LIMIT $2
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, deal_id, limit)

    return [record_to_dict(row) for row in rows]


@app.get("/api/deals/{deal_id}/aliases")
async def get_deal_aliases(deal_id: str, pool: asyncpg.Pool = Depends(get_db)):
    """Get aliases for a deal."""
    query = """
        SELECT id, alias, alias_type, confidence, source, created_at
        FROM zakops.deal_aliases
        WHERE deal_id = $1
        ORDER BY confidence DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, deal_id)

    return [dict(row) for row in rows]


# =============================================================================
# ACTIONS ENDPOINTS
# =============================================================================

@app.get("/api/actions", response_model=List[ActionResponse])
async def list_actions(
    status: Optional[str] = Query(None, description="Filter by status"),
    deal_id: Optional[str] = Query(None, description="Filter by deal"),
    pending_only: bool = Query(False, description="Show only pending actions"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    pool: asyncpg.Pool = Depends(get_db)
):
    """List actions with optional filtering."""
    conditions = []
    params = []
    param_idx = 1

    if pending_only:
        conditions.append("a.status IN ('PENDING_APPROVAL', 'QUEUED')")
    elif status:
        conditions.append(f"a.status = ${param_idx}")
        params.append(status)
        param_idx += 1

    if deal_id:
        conditions.append(f"a.deal_id = ${param_idx}")
        params.append(deal_id)
        param_idx += 1

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    query = f"""
        SELECT
            a.*,
            d.canonical_name AS deal_name,
            d.stage AS deal_stage
        FROM zakops.actions a
        LEFT JOIN zakops.deals d ON a.deal_id = d.deal_id
        WHERE {where_clause}
        ORDER BY
            CASE a.status WHEN 'PENDING_APPROVAL' THEN 0 WHEN 'QUEUED' THEN 1 ELSE 2 END,
            a.created_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
    """
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [record_to_dict(row) for row in rows]


@app.get("/api/actions/{action_id}", response_model=ActionResponse)
async def get_action(action_id: str, pool: asyncpg.Pool = Depends(get_db)):
    """Get a single action by ID."""
    query = """
        SELECT
            a.*,
            d.canonical_name AS deal_name,
            d.stage AS deal_stage
        FROM zakops.actions a
        LEFT JOIN zakops.deals d ON a.deal_id = d.deal_id
        WHERE a.action_id = $1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, action_id)

    if not row:
        raise HTTPException(status_code=404, detail=f"Action {action_id} not found")

    return record_to_dict(row)


@app.post("/api/actions/{action_id}/approve")
async def approve_action(
    action_id: str,
    approval: ActionApprove,
    pool: asyncpg.Pool = Depends(get_db)
):
    """Approve a pending action."""
    async with pool.acquire() as conn:
        # Check action exists and is pending
        action = await conn.fetchrow(
            "SELECT * FROM zakops.actions WHERE action_id = $1",
            action_id
        )
        if not action:
            raise HTTPException(status_code=404, detail=f"Action {action_id} not found")
        if action['status'] not in ('PENDING_APPROVAL', 'READY'):
            raise HTTPException(
                status_code=400,
                detail=f"Action is not pending approval (status: {action['status']})"
            )

        # Update status to QUEUED
        await conn.execute(
            """
            UPDATE zakops.actions
            SET status = 'QUEUED',
                audit_trail = audit_trail || $2::jsonb
            WHERE action_id = $1
            """,
            action_id,
            json.dumps([{
                "action": "approved",
                "by": approval.approved_by,
                "at": datetime.now(timezone.utc).isoformat(),
                "notes": approval.notes
            }])
        )

    return {"status": "approved", "action_id": action_id}


@app.post("/api/actions/{action_id}/reject")
async def reject_action(
    action_id: str,
    rejection: ActionReject,
    pool: asyncpg.Pool = Depends(get_db)
):
    """Reject a pending action."""
    async with pool.acquire() as conn:
        action = await conn.fetchrow(
            "SELECT * FROM zakops.actions WHERE action_id = $1",
            action_id
        )
        if not action:
            raise HTTPException(status_code=404, detail=f"Action {action_id} not found")
        if action['status'] not in ('PENDING_APPROVAL', 'READY'):
            raise HTTPException(
                status_code=400,
                detail=f"Action is not pending approval (status: {action['status']})"
            )

        await conn.execute(
            """
            UPDATE zakops.actions
            SET status = 'REJECTED',
                audit_trail = audit_trail || $2::jsonb
            WHERE action_id = $1
            """,
            action_id,
            json.dumps([{
                "action": "rejected",
                "by": rejection.rejected_by,
                "at": datetime.now(timezone.utc).isoformat(),
                "reason": rejection.reason
            }])
        )

    return {"status": "rejected", "action_id": action_id}


# =============================================================================
# QUARANTINE ENDPOINTS
# =============================================================================

@app.get("/api/quarantine", response_model=List[QuarantineResponse])
async def list_quarantine(
    status: Optional[str] = Query("pending", description="Filter by status"),
    classification: Optional[str] = Query(None, description="Filter by classification"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    pool: asyncpg.Pool = Depends(get_db)
):
    """List quarantine items."""
    conditions = []
    params = []
    param_idx = 1

    if status:
        conditions.append(f"q.status = ${param_idx}")
        params.append(status)
        param_idx += 1

    if classification:
        conditions.append(f"q.classification = ${param_idx}")
        params.append(classification)
        param_idx += 1

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    query = f"""
        SELECT
            q.*,
            sp.name AS sender_name,
            sp.company AS sender_company,
            sp.is_broker
        FROM zakops.quarantine_items q
        LEFT JOIN zakops.sender_profiles sp ON q.sender = sp.email
        WHERE {where_clause}
        ORDER BY
            CASE q.urgency WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END,
            q.created_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
    """
    params.extend([limit, offset])

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [record_to_dict(row) for row in rows]


@app.get("/api/quarantine/{item_id}", response_model=QuarantineResponse)
async def get_quarantine_item(item_id: str, pool: asyncpg.Pool = Depends(get_db)):
    """Get a single quarantine item."""
    query = """
        SELECT
            q.*,
            sp.name AS sender_name,
            sp.company AS sender_company,
            sp.is_broker
        FROM zakops.quarantine_items q
        LEFT JOIN zakops.sender_profiles sp ON q.sender = sp.email
        WHERE q.id = $1
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, item_id)

    if not row:
        raise HTTPException(status_code=404, detail=f"Quarantine item {item_id} not found")

    return record_to_dict(row)


@app.post("/api/quarantine/{item_id}/process")
async def process_quarantine(
    item_id: str,
    process: QuarantineProcess,
    pool: asyncpg.Pool = Depends(get_db)
):
    """Process a quarantine item (approve or reject)."""
    async with pool.acquire() as conn:
        item = await conn.fetchrow(
            "SELECT * FROM zakops.quarantine_items WHERE id = $1",
            item_id
        )
        if not item:
            raise HTTPException(status_code=404, detail=f"Quarantine item {item_id} not found")
        if item['status'] != 'pending':
            raise HTTPException(
                status_code=400,
                detail=f"Item is not pending (status: {item['status']})"
            )

        new_status = 'approved' if process.action == 'approve' else 'rejected'

        await conn.execute(
            """
            UPDATE zakops.quarantine_items
            SET status = $2,
                processed_at = NOW(),
                processed_by = $3,
                processing_result = $4,
                created_deal_id = $5
            WHERE id = $1
            """,
            item_id,
            new_status,
            process.processed_by,
            process.action,
            process.deal_id
        )

    return {
        "status": new_status,
        "item_id": item_id,
        "deal_id": process.deal_id
    }


# =============================================================================
# PIPELINE ENDPOINTS
# =============================================================================

@app.get("/api/pipeline/summary", response_model=List[PipelineSummary])
async def get_pipeline_summary(pool: asyncpg.Pool = Depends(get_db)):
    """Get pipeline summary with deal counts per stage."""
    query = """
        SELECT stage, count, avg_days_in_stage
        FROM zakops.v_pipeline_summary
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    return [dict(row) for row in rows]


@app.get("/api/pipeline/stats")
async def get_pipeline_stats(pool: asyncpg.Pool = Depends(get_db)):
    """Get overall pipeline statistics."""
    async with pool.acquire() as conn:
        # Total deals
        total_deals = await conn.fetchval(
            "SELECT COUNT(*) FROM zakops.deals WHERE deleted = FALSE AND status = 'active'"
        )

        # Pending actions
        pending_actions = await conn.fetchval(
            "SELECT COUNT(*) FROM zakops.actions WHERE status IN ('PENDING_APPROVAL', 'QUEUED')"
        )

        # Quarantine items
        quarantine_pending = await conn.fetchval(
            "SELECT COUNT(*) FROM zakops.quarantine_items WHERE status = 'pending'"
        )

        # Deals by stage
        stage_counts = await conn.fetch(
            """
            SELECT stage, COUNT(*) as count
            FROM zakops.deals
            WHERE deleted = FALSE AND status = 'active'
            GROUP BY stage
            """
        )

        # Recent activity
        recent_events = await conn.fetchval(
            """
            SELECT COUNT(*) FROM zakops.deal_events
            WHERE created_at > NOW() - INTERVAL '24 hours'
            """
        )

    return {
        "total_active_deals": total_deals,
        "pending_actions": pending_actions,
        "quarantine_pending": quarantine_pending,
        "recent_events_24h": recent_events,
        "deals_by_stage": {row['stage']: row['count'] for row in stage_counts}
    }


# =============================================================================
# AGENT THREAD/RUN ENDPOINTS (Phase 2 - Agent Invocation)
# =============================================================================

from starlette.responses import StreamingResponse

# Import agent invocation module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agent_invocation import (
    create_thread,
    get_thread,
    archive_thread,
    create_run,
    get_run,
    list_runs,
    start_run,
    complete_run,
    fail_run,
    emit_stream_token,
    create_tool_call,
    get_tool_call,
    approve_tool_call,
    reject_tool_call,
    complete_tool_call,
    fail_tool_call,
    record_event,
    get_events_since,
    create_sse_response,
    thread_to_dict,
    run_to_dict,
    tool_call_to_dict,
    event_to_dict,
    ThreadStatus,
    RunStatus,
    ToolCallStatus,
    ToolRiskLevel,
    AgentEventType,
)


class ThreadCreate(BaseModel):
    assistant_id: str
    deal_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None


class RunCreate(BaseModel):
    input_message: str
    assistant_id: Optional[str] = None  # Override thread's assistant_id
    metadata: Optional[Dict[str, Any]] = None
    stream: bool = False  # Whether to stream response


class ToolCallApprove(BaseModel):
    approved_by: str = "ui_user"


class ToolCallReject(BaseModel):
    rejected_by: str = "ui_user"
    reason: str


@app.post("/api/threads")
async def api_create_thread(data: ThreadCreate):
    """Create a new agent thread."""
    thread = await create_thread(
        assistant_id=data.assistant_id,
        deal_id=data.deal_id,
        user_id=data.user_id,
        metadata=data.metadata,
        user_context=data.user_context,
    )
    return thread_to_dict(thread)


@app.get("/api/threads/{thread_id}")
async def api_get_thread(thread_id: str):
    """Get a thread by ID."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
    return thread_to_dict(thread)


@app.delete("/api/threads/{thread_id}")
async def api_archive_thread(thread_id: str):
    """Archive a thread."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")
    await archive_thread(thread_id)
    return {"status": "archived", "thread_id": thread_id}


@app.get("/api/threads/{thread_id}/runs")
async def api_list_runs(
    thread_id: str,
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """List runs for a thread."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    status_enum = RunStatus(status) if status else None
    runs = await list_runs(thread_id, limit=limit, status=status_enum)
    return [run_to_dict(r) for r in runs]


@app.post("/api/threads/{thread_id}/runs")
async def api_create_run(thread_id: str, data: RunCreate):
    """Create a new run within a thread."""
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    assistant_id = data.assistant_id or thread.assistant_id

    run = await create_run(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input_message=data.input_message,
        metadata=data.metadata,
    )

    # Record run_created event
    await record_event(
        thread_id=thread_id,
        run_id=run.run_id,
        event_type=AgentEventType.RUN_CREATED,
        event_data={"status": "pending", "input_message": data.input_message[:100]},
    )

    return run_to_dict(run)


@app.post("/api/threads/{thread_id}/runs/stream")
async def api_create_and_stream_run(
    thread_id: str,
    data: RunCreate,
    last_event_id: Optional[str] = Query(None, alias="Last-Event-ID"),
):
    """Create a run and stream events via SSE.

    Supports resume via Last-Event-ID header or query parameter.
    """
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    assistant_id = data.assistant_id or thread.assistant_id

    run = await create_run(
        thread_id=thread_id,
        assistant_id=assistant_id,
        input_message=data.input_message,
        metadata=data.metadata,
    )

    # Record run_created event
    await record_event(
        thread_id=thread_id,
        run_id=run.run_id,
        event_type=AgentEventType.RUN_CREATED,
        event_data={"status": "pending", "input_message": data.input_message[:100]},
    )

    # Return SSE stream
    return create_sse_response(run.run_id, thread_id, last_event_id)


@app.get("/api/threads/{thread_id}/runs/{run_id}")
async def api_get_run(thread_id: str, run_id: str):
    """Get a run by ID."""
    run = await get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return run_to_dict(run)


@app.get("/api/threads/{thread_id}/runs/{run_id}/events")
async def api_get_run_events(
    thread_id: str,
    run_id: str,
    last_event_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    """Get events for a run, optionally starting after a specific event."""
    run = await get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    events = await get_events_since(run_id, last_event_id, limit)
    return [event_to_dict(e) for e in events]


@app.get("/api/threads/{thread_id}/runs/{run_id}/stream")
async def api_stream_run_events(
    thread_id: str,
    run_id: str,
    last_event_id: Optional[str] = Query(None, alias="Last-Event-ID"),
):
    """Stream events for an existing run via SSE.

    Supports resume via Last-Event-ID header or query parameter.
    """
    run = await get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    return create_sse_response(run_id, thread_id, last_event_id)


@app.get("/api/threads/{thread_id}/runs/{run_id}/tool_calls")
async def api_list_tool_calls(
    thread_id: str,
    run_id: str,
    pool: asyncpg.Pool = Depends(get_db),
):
    """List tool calls for a run."""
    run = await get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM zakops.agent_tool_calls
            WHERE run_id = $1
            ORDER BY sequence_number ASC
            """,
            run_id,
        )

    result = []
    for row in rows:
        result.append({
            "tool_call_id": row["tool_call_id"],
            "run_id": row["run_id"],
            "tool_name": row["tool_name"],
            "tool_input": json.loads(row["tool_input"]) if row["tool_input"] else {},
            "tool_output": json.loads(row["tool_output"]) if row["tool_output"] else None,
            "status": row["status"],
            "risk_level": row["risk_level"],
            "requires_approval": row["requires_approval"],
            "approved_by": row["approved_by"],
            "approved_at": row["approved_at"].isoformat() if row["approved_at"] else None,
            "rejection_reason": row["rejection_reason"],
            "sequence_number": row["sequence_number"],
            "created_at": row["created_at"].isoformat(),
        })

    return result


@app.post("/api/threads/{thread_id}/runs/{run_id}/tool_calls/{tool_call_id}/approve")
async def api_approve_tool_call(
    thread_id: str,
    run_id: str,
    tool_call_id: str,
    data: ToolCallApprove,
):
    """Approve a pending tool call."""
    run = await get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    tc = await get_tool_call(tool_call_id)
    if not tc or tc.run_id != run_id:
        raise HTTPException(status_code=404, detail=f"Tool call {tool_call_id} not found")

    if tc.status != ToolCallStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Tool call is not pending (status: {tc.status.value})"
        )

    updated = await approve_tool_call(tool_call_id, data.approved_by)

    # Record approval event
    await record_event(
        thread_id=thread_id,
        run_id=run_id,
        event_type=AgentEventType.TOOL_APPROVAL_GRANTED,
        event_data={
            "tool_call_id": tool_call_id,
            "tool_name": tc.tool_name,
            "approved_by": data.approved_by,
        },
        tool_call_id=tool_call_id,
    )

    return tool_call_to_dict(updated)


@app.post("/api/threads/{thread_id}/runs/{run_id}/tool_calls/{tool_call_id}/reject")
async def api_reject_tool_call(
    thread_id: str,
    run_id: str,
    tool_call_id: str,
    data: ToolCallReject,
):
    """Reject a pending tool call."""
    run = await get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    tc = await get_tool_call(tool_call_id)
    if not tc or tc.run_id != run_id:
        raise HTTPException(status_code=404, detail=f"Tool call {tool_call_id} not found")

    if tc.status != ToolCallStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Tool call is not pending (status: {tc.status.value})"
        )

    updated = await reject_tool_call(tool_call_id, data.rejected_by, data.reason)

    # Record rejection event
    await record_event(
        thread_id=thread_id,
        run_id=run_id,
        event_type=AgentEventType.TOOL_APPROVAL_DENIED,
        event_data={
            "tool_call_id": tool_call_id,
            "tool_name": tc.tool_name,
            "rejected_by": data.rejected_by,
            "reason": data.reason,
        },
        tool_call_id=tool_call_id,
    )

    return tool_call_to_dict(updated)


@app.get("/api/threads/{thread_id}/runs/{run_id}/tool_calls/{tool_call_id}")
async def api_get_tool_call(
    thread_id: str,
    run_id: str,
    tool_call_id: str,
):
    """Get a tool call by ID."""
    run = await get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    tc = await get_tool_call(tool_call_id)
    if not tc or tc.run_id != run_id:
        raise HTTPException(status_code=404, detail=f"Tool call {tool_call_id} not found")

    return tool_call_to_dict(tc)


# View for pending tool approvals
@app.get("/api/pending-tool-approvals")
async def api_pending_tool_approvals(
    limit: int = Query(50, ge=1, le=200),
    pool: asyncpg.Pool = Depends(get_db),
):
    """List all pending tool approvals across all threads."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM zakops.v_pending_tool_approvals
            ORDER BY created_at ASC
            LIMIT $1
            """,
            limit,
        )

    result = []
    for row in rows:
        result.append({
            "tool_call_id": row["tool_call_id"],
            "run_id": row["run_id"],
            "thread_id": row["thread_id"],
            "deal_id": row["deal_id"],
            "deal_name": row["deal_name"],
            "tool_name": row["tool_name"],
            "tool_input": json.loads(row["tool_input"]) if row["tool_input"] else {},
            "risk_level": row["risk_level"],
            "created_at": row["created_at"].isoformat(),
        })

    return result


# =============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle any incoming messages
            data = await websocket.receive_text()
            # Echo back or handle commands
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# SENDER PROFILES
# =============================================================================

@app.get("/api/senders")
async def list_senders(
    is_broker: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    pool: asyncpg.Pool = Depends(get_db)
):
    """List sender profiles."""
    conditions = []
    params = []
    param_idx = 1

    if is_broker is not None:
        conditions.append(f"is_broker = ${param_idx}")
        params.append(is_broker)
        param_idx += 1

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    query = f"""
        SELECT *
        FROM zakops.sender_profiles
        WHERE {where_clause}
        ORDER BY last_email_at DESC NULLS LAST
        LIMIT ${param_idx}
    """
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [dict(row) for row in rows]


@app.get("/api/senders/{email}")
async def get_sender(email: str, pool: asyncpg.Pool = Depends(get_db)):
    """Get a sender profile by email."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM zakops.sender_profiles WHERE email = $1",
            email
        )

    if not row:
        raise HTTPException(status_code=404, detail=f"Sender {email} not found")

    return dict(row)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
