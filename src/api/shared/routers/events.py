"""
Event API Endpoints

Provides REST endpoints for querying events.
These endpoints are consumed by the UI for the Agent Visibility Layer.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID
import json
import logging

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["events"])


class EventResponse(BaseModel):
    """Event response model."""
    id: str
    correlation_id: str
    event_type: str
    event_data: dict
    schema_version: int
    source: Optional[str]
    run_id: Optional[str]
    thread_id: Optional[str]
    created_at: str


class EventStatsResponse(BaseModel):
    """Event statistics response."""
    total: int
    by_type: dict
    since: str


def _format_event(event: dict) -> EventResponse:
    """Format database row to response model."""
    # Get id - might be event_id or id
    event_id = event.get("id") or event.get("event_id") or ""

    # Get correlation_id
    correlation_id = event.get("correlation_id") or ""

    # Handle event_data/payload
    event_data = event.get("event_data") or event.get("payload") or {}
    if isinstance(event_data, str):
        try:
            event_data = json.loads(event_data)
        except (json.JSONDecodeError, TypeError):
            event_data = {}

    # Handle created_at
    created_at = event.get("created_at")
    if created_at and hasattr(created_at, 'isoformat'):
        created_at_str = created_at.isoformat()
    elif created_at:
        created_at_str = str(created_at)
    else:
        created_at_str = ""

    return EventResponse(
        id=str(event_id),
        correlation_id=str(correlation_id),
        event_type=event.get("event_type", ""),
        event_data=event_data,
        schema_version=event.get("schema_version", 1),
        source=event.get("source"),
        run_id=str(event["run_id"]) if event.get("run_id") else None,
        thread_id=event.get("thread_id"),
        created_at=created_at_str
    )


@router.get("/recent", response_model=List[EventResponse])
async def get_recent_events(
    limit: int = Query(50, ge=1, le=200),
    event_types: Optional[str] = Query(None, description="Comma-separated event types")
):
    """
    Get recent events.

    Used by: Dashboard Agent Activity Widget
    """
    # Import here to avoid circular imports
    from ....core.events import get_query_service

    service = await get_query_service()

    types_list = event_types.split(",") if event_types else None
    events = await service.get_recent(limit=limit, event_types=types_list)

    return [_format_event(e) for e in events]


@router.get("/by-correlation/{correlation_id}", response_model=List[EventResponse])
async def get_events_by_correlation(
    correlation_id: UUID,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Get events for a specific correlation ID (deal).

    Used by: Deal Workspace Agent Panel
    """
    from ....core.events import get_query_service

    service = await get_query_service()
    events = await service.get_by_correlation_id(
        correlation_id=correlation_id,
        limit=limit,
        offset=offset
    )

    return [_format_event(e) for e in events]


@router.get("/by-run/{run_id}", response_model=List[EventResponse])
async def get_events_by_run(
    run_id: UUID,
    limit: int = Query(100, ge=1, le=500)
):
    """
    Get events for a specific agent run.

    Used by: Agent Activity Page (run detail view)
    """
    from ....core.events import get_query_service

    service = await get_query_service()
    events = await service.get_by_run_id(run_id=run_id, limit=limit)

    return [_format_event(e) for e in events]


@router.get("/stats", response_model=EventStatsResponse)
async def get_event_stats(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back")
):
    """
    Get event statistics.

    Used by: Dashboard stats, Agent Activity page header
    """
    from ....core.events import get_query_service

    service = await get_query_service()
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    stats = await service.get_stats(since=since)

    return EventStatsResponse(**stats)


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: UUID):
    """
    Get a single event by ID.
    """
    from ....core.events import get_query_service

    service = await get_query_service()
    event = await service.get_by_id(event_id)

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    return _format_event(event)
