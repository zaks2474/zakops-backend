"""
Admin/Operator API

Endpoints for system administration and operator tooling.

Phase 13: Production Hardening
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Request, Query, HTTPException
from pydantic import BaseModel

from ...shared.middleware.auth import require_auth
from ....core.outbox.dlq import get_dlq_manager, DLQAction
from ....core.database.adapter import get_database

router = APIRouter(prefix="/api/admin", tags=["admin"])


class RetryRequest(BaseModel):
    """Request to retry DLQ entry."""
    reason: Optional[str] = None


class PurgeRequest(BaseModel):
    """Request to purge DLQ entries."""
    days: int = 30
    reason: Optional[str] = None


# DLQ Management Endpoints

@router.get("/dlq")
async def list_dlq_entries(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    correlation_id: Optional[str] = None,
):
    """List Dead Letter Queue entries."""
    operator = require_auth(request)

    manager = await get_dlq_manager()

    entries = await manager.get_entries(
        limit=limit,
        offset=offset,
        correlation_id=correlation_id
    )

    total = await manager.get_count(correlation_id)

    return {
        "entries": [e.to_dict() for e in entries],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/dlq/stats")
async def dlq_stats(request: Request):
    """Get DLQ statistics."""
    operator = require_auth(request)

    manager = await get_dlq_manager()
    return await manager.get_stats()


@router.post("/dlq/{entry_id}/retry")
async def retry_dlq_entry(
    request: Request,
    entry_id: UUID,
    body: RetryRequest,
):
    """Retry a specific DLQ entry."""
    operator = require_auth(request)

    manager = await get_dlq_manager()

    success = await manager.retry_entry(
        entry_id=entry_id,
        operator_id=str(operator.id) if operator else None
    )

    if not success:
        raise HTTPException(status_code=404, detail="DLQ entry not found")

    return {"status": "queued_for_retry", "entry_id": str(entry_id)}


@router.post("/dlq/retry-all")
async def retry_all_dlq(
    request: Request,
    correlation_id: Optional[str] = None,
):
    """Retry all DLQ entries."""
    operator = require_auth(request)

    manager = await get_dlq_manager()

    count = await manager.retry_all(
        correlation_id=correlation_id,
        operator_id=str(operator.id) if operator else None
    )

    return {
        "status": "all_queued_for_retry",
        "correlation_id": correlation_id,
        "count": count
    }


@router.delete("/dlq/{entry_id}")
async def purge_dlq_entry(
    request: Request,
    entry_id: UUID,
):
    """Permanently delete a DLQ entry."""
    operator = require_auth(request)

    manager = await get_dlq_manager()

    success = await manager.purge_entry(
        entry_id=entry_id,
        operator_id=str(operator.id) if operator else None
    )

    if not success:
        raise HTTPException(status_code=404, detail="DLQ entry not found")

    return {"status": "purged", "entry_id": str(entry_id)}


@router.post("/dlq/purge-old")
async def purge_old_dlq(
    request: Request,
    body: PurgeRequest,
):
    """Purge DLQ entries older than specified days."""
    operator = require_auth(request)

    manager = await get_dlq_manager()

    count = await manager.purge_old(
        days=body.days,
        operator_id=str(operator.id) if operator else None
    )

    return {"status": "purged", "older_than_days": body.days, "count": count}


# Outbox Status Endpoints

@router.get("/outbox/stats")
async def outbox_stats(request: Request):
    """Get outbox queue statistics."""
    operator = require_auth(request)

    db = await get_database()

    stats = await db.fetch(
        """
        SELECT status, COUNT(*) as count
        FROM zakops.outbox
        GROUP BY status
        """
    )

    pending_old = await db.fetchrow(
        """
        SELECT COUNT(*) as count
        FROM zakops.outbox
        WHERE status = 'pending'
        AND created_at < NOW() - INTERVAL '1 hour'
        """
    )

    return {
        "by_status": {row["status"]: row["count"] for row in stats},
        "stuck_count": pending_old["count"] if pending_old else 0,
        "healthy": (pending_old["count"] if pending_old else 0) == 0
    }


# SSE Status Endpoints

@router.get("/sse/stats")
async def sse_stats(request: Request):
    """Get SSE connection statistics."""
    operator = require_auth(request)

    from ...shared.sse import get_sse_manager

    manager = get_sse_manager()

    return {
        "total_connections": manager.connection_count,
        "max_connections": manager.config.max_total_connections,
        "max_per_user": manager.config.max_connections_per_user,
        "heartbeat_interval": manager.config.heartbeat_interval,
        "buffered_events": len(manager._event_buffer)
    }


# System Health Endpoints

@router.get("/system/health")
async def system_health(request: Request):
    """Get comprehensive system health status."""
    operator = require_auth(request)

    from ...shared.sse import get_sse_manager
    from ....core.outbox.processor import get_outbox_processor

    db = await get_database()

    # Database health
    try:
        await db.fetchrow("SELECT 1")
        db_healthy = True
    except Exception:
        db_healthy = False

    # Outbox health
    outbox_stats = await db.fetchrow(
        """
        SELECT
            COUNT(*) FILTER (WHERE status = 'pending') as pending,
            COUNT(*) FILTER (WHERE status = 'dead') as dead,
            COUNT(*) FILTER (WHERE status = 'pending' AND created_at < NOW() - INTERVAL '1 hour') as stuck
        FROM zakops.outbox
        """
    )

    # SSE health
    sse_manager = get_sse_manager()
    sse_healthy = sse_manager.connection_count < sse_manager.config.max_total_connections

    # Outbox processor
    processor = await get_outbox_processor()
    processor_running = processor is not None and processor._running

    return {
        "overall": "healthy" if all([db_healthy, sse_healthy, processor_running]) else "degraded",
        "components": {
            "database": {"status": "healthy" if db_healthy else "unhealthy"},
            "sse": {
                "status": "healthy" if sse_healthy else "degraded",
                "connections": sse_manager.connection_count,
                "max": sse_manager.config.max_total_connections
            },
            "outbox": {
                "status": "healthy" if outbox_stats["stuck"] == 0 else "degraded",
                "pending": outbox_stats["pending"],
                "dead": outbox_stats["dead"],
                "stuck": outbox_stats["stuck"]
            },
            "outbox_processor": {
                "status": "running" if processor_running else "stopped"
            }
        }
    }
