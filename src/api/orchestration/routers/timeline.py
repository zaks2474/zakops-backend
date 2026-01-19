"""
Activity Timeline API

Provides unified activity view for deals.
"""

from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Query, HTTPException

from ....core.database.adapter import get_database

router = APIRouter(prefix="/api/deals", tags=["activity"])


@router.get("/{deal_id}/timeline")
async def get_deal_timeline(
    deal_id: str,
    limit: int = Query(50, ge=1, le=200),
    before: Optional[datetime] = None,
    activity_type: Optional[str] = Query(None, description="Filter by type: stage_change, action, agent_run, event")
):
    """
    Get unified activity timeline for a deal.

    Includes:
    - Stage changes
    - Actions (created, approved, completed)
    - Agent runs
    - Events
    """
    db = await get_database()

    # Verify deal exists
    deal = await db.fetchrow(
        "SELECT deal_id FROM zakops.deals WHERE deal_id = $1",
        deal_id
    )
    if not deal:
        raise HTTPException(status_code=404, detail=f"Deal {deal_id} not found")

    activities = []

    # Stage changes from deal_events
    if not activity_type or activity_type == "stage_change":
        stages = await db.fetch(
            """
            SELECT event_type, actor, details, created_at
            FROM zakops.deal_events
            WHERE deal_id = $1 AND event_type = 'stage_changed'
            ORDER BY created_at DESC
            LIMIT $2
            """,
            deal_id, limit
        )

        for s in stages:
            details = s.get("details", {})
            if isinstance(details, str):
                import json
                try:
                    details = json.loads(details)
                except:
                    details = {}

            activities.append({
                "type": "stage_change",
                "timestamp": s["created_at"].isoformat() if s.get("created_at") else None,
                "data": {
                    "from_stage": details.get("from_stage"),
                    "to_stage": details.get("to_stage"),
                    "by": s.get("actor"),
                    "reason": details.get("reason")
                }
            })

    # Actions
    if not activity_type or activity_type == "action":
        actions = await db.fetch(
            """
            SELECT action_id, action_type, title, status, created_at, updated_at
            FROM zakops.actions
            WHERE deal_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            deal_id, limit
        )

        for a in actions:
            activities.append({
                "type": "action",
                "timestamp": a["created_at"].isoformat() if a.get("created_at") else None,
                "data": {
                    "action_id": a["action_id"],
                    "action_type": a["action_type"],
                    "title": a.get("title"),
                    "status": a["status"]
                }
            })

    # Agent runs
    if not activity_type or activity_type == "agent_run":
        runs = await db.fetch(
            """
            SELECT run_id, task, status, started_at, duration_ms
            FROM zakops.agent_runs
            WHERE deal_id = $1
            ORDER BY started_at DESC
            LIMIT $2
            """,
            deal_id, limit
        )

        for r in runs:
            activities.append({
                "type": "agent_run",
                "timestamp": r["started_at"].isoformat() if r.get("started_at") else None,
                "data": {
                    "run_id": r["run_id"],
                    "task": r.get("task"),
                    "status": r["status"],
                    "duration_ms": r.get("duration_ms")
                }
            })

    # General events
    if not activity_type or activity_type == "event":
        events = await db.fetch(
            """
            SELECT id, event_type, source, actor, details, created_at
            FROM zakops.deal_events
            WHERE deal_id = $1 AND event_type != 'stage_changed'
            ORDER BY created_at DESC
            LIMIT $2
            """,
            deal_id, limit
        )

        for e in events:
            details = e.get("details", {})
            if isinstance(details, str):
                import json
                try:
                    details = json.loads(details)
                except:
                    details = {}

            activities.append({
                "type": "event",
                "timestamp": e["created_at"].isoformat() if e.get("created_at") else None,
                "data": {
                    "event_id": e["id"],
                    "event_type": e["event_type"],
                    "source": e.get("source"),
                    "actor": e.get("actor"),
                    "details": details
                }
            })

    # Sort by timestamp descending
    activities.sort(
        key=lambda x: x["timestamp"] if x["timestamp"] else "",
        reverse=True
    )

    # Apply before filter if specified
    if before:
        before_iso = before.isoformat()
        activities = [a for a in activities if a["timestamp"] and a["timestamp"] < before_iso]

    return {
        "deal_id": deal_id,
        "activities": activities[:limit],
        "count": len(activities[:limit]),
        "total_available": len(activities)
    }


@router.get("/{deal_id}/activity-summary")
async def get_deal_activity_summary(deal_id: str):
    """
    Get activity summary counts for a deal.
    """
    db = await get_database()

    # Verify deal exists
    deal = await db.fetchrow(
        "SELECT deal_id, canonical_name, stage, status FROM zakops.deals WHERE deal_id = $1",
        deal_id
    )
    if not deal:
        raise HTTPException(status_code=404, detail=f"Deal {deal_id} not found")

    # Count actions by status
    action_counts = await db.fetch(
        """
        SELECT status, COUNT(*) as count
        FROM zakops.actions
        WHERE deal_id = $1
        GROUP BY status
        """,
        deal_id
    )

    # Count events
    event_count = await db.fetchval(
        "SELECT COUNT(*) FROM zakops.deal_events WHERE deal_id = $1",
        deal_id
    )

    # Count agent runs
    run_count = await db.fetchval(
        "SELECT COUNT(*) FROM zakops.agent_runs WHERE deal_id = $1",
        deal_id
    )

    # Recent activity timestamp
    recent_activity = await db.fetchrow(
        """
        SELECT MAX(created_at) as latest
        FROM (
            SELECT created_at FROM zakops.deal_events WHERE deal_id = $1
            UNION ALL
            SELECT created_at FROM zakops.actions WHERE deal_id = $1
            UNION ALL
            SELECT started_at as created_at FROM zakops.agent_runs WHERE deal_id = $1
        ) combined
        """,
        deal_id
    )

    return {
        "deal_id": deal_id,
        "deal_name": deal.get("canonical_name"),
        "stage": deal.get("stage"),
        "status": deal.get("status"),
        "actions": {r["status"]: r["count"] for r in action_counts},
        "events_count": event_count or 0,
        "agent_runs_count": run_count or 0,
        "last_activity": recent_activity["latest"].isoformat() if recent_activity and recent_activity.get("latest") else None
    }
