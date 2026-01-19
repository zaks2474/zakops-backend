"""
Event Query Service

Provides read access to events for UI, debugging, and analytics.
Queries from both zakops.agent_events and zakops.deal_events tables.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID
import json
import logging

from ..database.adapter import get_database, DatabaseAdapter

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


class EventQueryService:
    """
    Query events from both agent_events and deal_events tables.

    Provides a unified view of events across the system:
    - agent_events: Agent execution events (runs, tool calls)
    - deal_events: Deal lifecycle and action events

    Usage:
        service = EventQueryService()
        events = await service.get_by_correlation_id(deal_id)
        events = await service.get_recent(limit=50)
    """

    def __init__(self, db: Optional[DatabaseAdapter] = None):
        self._db = db

    async def _get_db(self) -> DatabaseAdapter:
        if self._db is None:
            self._db = await get_database()
        return self._db

    def _parse_agent_event(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Parse agent_events row into unified event dict."""
        event = dict(row)

        # Handle event_data JSON
        event_data = event.get("event_data") or {}
        if isinstance(event_data, str):
            try:
                event_data = json.loads(event_data)
            except (json.JSONDecodeError, TypeError):
                event_data = {}

        return {
            "id": event.get("event_id") or str(event.get("id", "")),
            "correlation_id": str(event.get("correlation_id") or ""),
            "event_type": event.get("event_type", ""),
            "event_data": event_data,
            "schema_version": 1,
            "source": "agent",
            "run_id": event.get("run_id"),
            "thread_id": event.get("thread_id"),
            "created_at": event.get("created_at"),
        }

    def _parse_deal_event(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Parse deal_events row into unified event dict."""
        event = dict(row)

        # Handle details JSON
        details = event.get("details") or {}
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except (json.JSONDecodeError, TypeError):
                details = {}

        # Extract event_id from details if present
        event_id = details.pop("event_id", None) or str(event.get("id", ""))
        correlation_id = details.pop("correlation_id", None) or event.get("deal_id", "")

        return {
            "id": event_id,
            "correlation_id": str(correlation_id),
            "event_type": event.get("event_type", ""),
            "event_data": details,
            "schema_version": details.get("schema_version", 1),
            "source": event.get("source", "deal"),
            "run_id": None,
            "thread_id": None,
            "created_at": event.get("created_at"),
        }

    async def get_by_id(self, event_id: UUID) -> Optional[Dict[str, Any]]:
        """Get a single event by ID."""
        db = await self._get_db()

        # Try agent_events first
        row = await db.fetchrow(
            "SELECT * FROM zakops.agent_events WHERE event_id = $1",
            str(event_id)
        )
        if row:
            return self._parse_agent_event(row)

        # Try deal_events (check details->event_id)
        row = await db.fetchrow(
            "SELECT * FROM zakops.deal_events WHERE details->>'event_id' = $1",
            str(event_id)
        )
        if row:
            return self._parse_deal_event(row)

        return None

    async def get_by_correlation_id(
        self,
        correlation_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all events for a correlation ID (e.g., deal_id)."""
        db = await self._get_db()

        # Query agent_events
        agent_rows = await db.fetch(
            """
            SELECT * FROM zakops.agent_events
            WHERE correlation_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            str(correlation_id), limit
        )

        # Query deal_events
        deal_rows = await db.fetch(
            """
            SELECT * FROM zakops.deal_events
            WHERE deal_id = $1 OR details->>'correlation_id' = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            str(correlation_id)[:20], limit
        )

        # Combine and sort
        events = [self._parse_agent_event(r) for r in agent_rows]
        events.extend([self._parse_deal_event(r) for r in deal_rows])
        events.sort(key=lambda x: x.get("created_at") or datetime.min, reverse=True)

        return events[offset:offset + limit]

    async def get_by_type(
        self,
        event_type: str,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get events by type from both tables."""
        db = await self._get_db()

        events = []

        # Query agent_events
        if since:
            agent_rows = await db.fetch(
                """
                SELECT * FROM zakops.agent_events
                WHERE event_type = $1 AND created_at >= $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                event_type, since, limit
            )
        else:
            agent_rows = await db.fetch(
                """
                SELECT * FROM zakops.agent_events
                WHERE event_type = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                event_type, limit
            )
        events.extend([self._parse_agent_event(r) for r in agent_rows])

        # Query deal_events
        if since:
            deal_rows = await db.fetch(
                """
                SELECT * FROM zakops.deal_events
                WHERE event_type = $1 AND created_at >= $2
                ORDER BY created_at DESC
                LIMIT $3
                """,
                event_type, since, limit
            )
        else:
            deal_rows = await db.fetch(
                """
                SELECT * FROM zakops.deal_events
                WHERE event_type = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                event_type, limit
            )
        events.extend([self._parse_deal_event(r) for r in deal_rows])

        # Sort and limit
        events.sort(key=lambda x: x.get("created_at") or datetime.min, reverse=True)
        return events[:limit]

    async def get_recent(
        self,
        limit: int = 50,
        event_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get recent events from both tables, optionally filtered by type."""
        db = await self._get_db()

        events = []

        # Query agent_events
        if event_types:
            placeholders = ", ".join(f"${i+1}" for i in range(len(event_types)))
            agent_rows = await db.fetch(
                f"""
                SELECT * FROM zakops.agent_events
                WHERE event_type IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT ${len(event_types) + 1}
                """,
                *event_types, limit
            )
        else:
            agent_rows = await db.fetch(
                """
                SELECT * FROM zakops.agent_events
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit
            )
        events.extend([self._parse_agent_event(r) for r in agent_rows])

        # Query deal_events
        if event_types:
            placeholders = ", ".join(f"${i+1}" for i in range(len(event_types)))
            deal_rows = await db.fetch(
                f"""
                SELECT * FROM zakops.deal_events
                WHERE event_type IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT ${len(event_types) + 1}
                """,
                *event_types, limit
            )
        else:
            deal_rows = await db.fetch(
                """
                SELECT * FROM zakops.deal_events
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit
            )
        events.extend([self._parse_deal_event(r) for r in deal_rows])

        # Sort and limit
        events.sort(key=lambda x: x.get("created_at") or datetime.min, reverse=True)
        return events[:limit]

    async def get_by_run_id(
        self,
        run_id: UUID,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all events for an agent run (agent_events only)."""
        db = await self._get_db()
        rows = await db.fetch(
            """
            SELECT * FROM zakops.agent_events
            WHERE run_id = $1
            ORDER BY created_at ASC
            LIMIT $2
            """,
            str(run_id), limit
        )
        return [self._parse_agent_event(row) for row in rows]

    async def get_stats(
        self,
        since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get event statistics from both tables."""
        db = await self._get_db()

        if since is None:
            since = _utcnow() - timedelta(hours=24)

        # Count by type from agent_events
        agent_counts = await db.fetch(
            """
            SELECT event_type, COUNT(*) as count
            FROM zakops.agent_events
            WHERE created_at >= $1
            GROUP BY event_type
            """,
            since
        )

        # Count by type from deal_events
        deal_counts = await db.fetch(
            """
            SELECT event_type, COUNT(*) as count
            FROM zakops.deal_events
            WHERE created_at >= $1
            GROUP BY event_type
            """,
            since
        )

        # Combine counts
        by_type: Dict[str, int] = {}
        for row in agent_counts:
            by_type[row["event_type"]] = row["count"]
        for row in deal_counts:
            by_type[row["event_type"]] = by_type.get(row["event_type"], 0) + row["count"]

        # Sort by count
        by_type = dict(sorted(by_type.items(), key=lambda x: x[1], reverse=True))

        # Total count
        total = sum(by_type.values())

        return {
            "total": total,
            "by_type": by_type,
            "since": since.isoformat()
        }


# Global query service instance
_query_service: Optional[EventQueryService] = None


async def get_query_service() -> EventQueryService:
    """Get the global event query service instance."""
    global _query_service
    if _query_service is None:
        _query_service = EventQueryService()
    return _query_service
