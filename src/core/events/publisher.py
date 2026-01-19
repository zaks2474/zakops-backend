"""
Event Publisher

Publishes events to the appropriate table based on event type:
- Deal events -> zakops.deal_events
- Agent events -> zakops.agent_events (requires run_id/thread_id)
- Action/Worker events -> zakops.deal_events (with deal context)

Supports:
- Synchronous publishing (immediate)
- Async publishing (via outbox, Phase 3)
- Batch publishing
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
import json
import logging

from ..database.adapter import get_database, DatabaseAdapter
from .models import EventBase, AgentEvent, DealEvent, ActionEvent, WorkerEvent
from .taxonomy import validate_event_type, get_domain

logger = logging.getLogger(__name__)

# Valid event types for agent_events table (from CHECK constraint)
AGENT_EVENT_TYPES = {
    'run_created', 'run_started', 'run_completed', 'run_failed', 'run_cancelled',
    'tool_call_started', 'tool_call_completed', 'tool_call_failed',
    'tool_approval_required', 'tool_approval_granted', 'tool_approval_denied',
    'stream_start', 'stream_token', 'stream_end', 'stream_error', 'custom'
}


class EventPublisher:
    """
    Publishes events to the appropriate database table.

    Routes events based on type:
    - deal.* events -> zakops.deal_events
    - agent.* events -> zakops.agent_events (requires run_id/thread_id)
    - action.*/worker.* events -> zakops.deal_events (general events)

    Usage:
        publisher = EventPublisher()
        await publisher.publish(event)

        # Or batch publish
        await publisher.publish_batch([event1, event2, event3])
    """

    def __init__(self, db: Optional[DatabaseAdapter] = None):
        self._db = db

    async def _get_db(self) -> DatabaseAdapter:
        if self._db is None:
            self._db = await get_database()
        return self._db

    async def publish(self, event: EventBase) -> UUID:
        """
        Publish a single event to the appropriate table.

        Returns the event ID.
        """
        if not validate_event_type(event.event_type):
            logger.warning(f"Unknown event type: {event.event_type} - publishing anyway")

        domain = get_domain(event.event_type)

        # Route to appropriate table based on domain
        if domain == "deal" and hasattr(event, 'deal_id'):
            return await self._publish_to_deal_events(event)
        elif domain == "agent" and hasattr(event, 'run_id') and hasattr(event, 'thread_id'):
            return await self._publish_to_agent_events(event)
        else:
            # Default: publish to deal_events as a general event store
            return await self._publish_to_deal_events(event)

    async def _publish_to_deal_events(self, event: EventBase) -> UUID:
        """Publish event to zakops.deal_events table with full audit metadata."""
        db = await self._get_db()

        # Get deal_id from event
        deal_id = getattr(event, 'deal_id', None)
        if deal_id is None:
            deal_id = str(event.correlation_id)[:20]  # Use correlation as fallback

        # Get actor metadata
        actor_id = getattr(event, 'actor_id', None) or event.source or "system"
        actor_type = getattr(event, 'actor_type', None) or "system"
        idempotency_key = getattr(event, 'idempotency_key', None)

        # Check idempotency if key provided
        if idempotency_key:
            existing = await db.fetchrow(
                """
                SELECT id FROM zakops.deal_events
                WHERE idempotency_key = $1
                  AND created_at > NOW() - INTERVAL '24 hours'
                """,
                idempotency_key
            )
            if existing:
                logger.debug(f"Idempotent event hit: {idempotency_key}")
                return event.id

        # Prepare details with full event data
        details = {
            "event_id": str(event.id),
            "correlation_id": str(event.correlation_id),
            "schema_version": event.schema_version,
            **event.event_data
        }

        await db.execute(
            """
            INSERT INTO zakops.deal_events (
                deal_id, event_type, source, actor, actor_id, actor_type,
                details, idempotency_key, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            str(deal_id)[:20],
            event.event_type,
            event.source or "system",
            actor_id,
            actor_id,
            actor_type,
            json.dumps(details),
            idempotency_key,
            event.created_at
        )

        logger.debug(
            "Published deal event: type=%s deal_id=%s actor=%s",
            event.event_type,
            deal_id,
            actor_id
        )

        return event.id

    async def _publish_to_agent_events(self, event: EventBase) -> UUID:
        """Publish event to zakops.agent_events table."""
        db = await self._get_db()

        run_id = getattr(event, 'run_id', None)
        thread_id = getattr(event, 'thread_id', None)

        if not run_id or not thread_id:
            logger.warning(
                "Agent event requires run_id and thread_id, falling back to deal_events"
            )
            return await self._publish_to_deal_events(event)

        # Map our event type to valid agent_events types
        event_type = self._map_to_agent_event_type(event.event_type)

        # Prepare event_data
        event_data = {
            "original_type": event.event_type,
            "correlation_id": str(event.correlation_id),
            **event.event_data
        }

        await db.execute(
            """
            INSERT INTO zakops.agent_events (
                event_id, thread_id, run_id, event_type, event_data,
                correlation_id, trace_id, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            str(event.id),
            str(thread_id),
            str(run_id),
            event_type,
            json.dumps(event_data),
            event.correlation_id,
            getattr(event, 'trace_id', None),
            event.created_at
        )

        logger.debug(
            "Published agent event: type=%s run_id=%s",
            event_type,
            run_id
        )

        return event.id

    def _map_to_agent_event_type(self, event_type: str) -> str:
        """Map our taxonomy event types to valid agent_events types."""
        mapping = {
            "agent.run_started": "run_started",
            "agent.run_completed": "run_completed",
            "agent.run_failed": "run_failed",
            "agent.tool_called": "tool_call_started",
            "agent.tool_completed": "tool_call_completed",
            "agent.tool_failed": "tool_call_failed",
            "agent.waiting_approval": "tool_approval_required",
        }
        mapped = mapping.get(event_type, "custom")
        if mapped not in AGENT_EVENT_TYPES:
            return "custom"
        return mapped

    async def publish_batch(self, events: List[EventBase]) -> List[UUID]:
        """
        Publish multiple events in a batch.

        Returns list of event IDs.
        """
        ids = []
        for event in events:
            event_id = await self.publish(event)
            ids.append(event_id)
        return ids

    async def publish_deal_event(
        self,
        deal_id: UUID,
        event_type: str,
        event_data: dict,
        source: str = "deal_lifecycle"
    ) -> UUID:
        """Convenience method for deal events."""
        event = DealEvent.create(
            deal_id=deal_id,
            event_type=event_type,
            event_data=event_data,
            source=source
        )
        return await self.publish(event)

    async def publish_action_event(
        self,
        action_id: UUID,
        correlation_id: UUID,
        event_type: str,
        event_data: dict,
        deal_id: Optional[UUID] = None,
        source: str = "action_engine"
    ) -> UUID:
        """Convenience method for action events."""
        event = ActionEvent.create(
            action_id=action_id,
            correlation_id=correlation_id,
            event_type=event_type,
            event_data=event_data,
            deal_id=deal_id,
            source=source
        )
        return await self.publish(event)

    async def publish_agent_event(
        self,
        correlation_id: UUID,
        event_type: str,
        event_data: dict,
        run_id: Optional[UUID] = None,
        thread_id: Optional[str] = None,
        source: str = "agent"
    ) -> UUID:
        """Convenience method for agent events."""
        event = AgentEvent.create(
            correlation_id=correlation_id,
            event_type=event_type,
            event_data=event_data,
            run_id=run_id,
            thread_id=thread_id,
            source=source
        )
        return await self.publish(event)


# Global publisher instance
_publisher: Optional[EventPublisher] = None


async def get_publisher() -> EventPublisher:
    """Get the global event publisher instance."""
    global _publisher
    if _publisher is None:
        _publisher = EventPublisher()
    return _publisher


# Convenience functions
async def publish_event(event: EventBase) -> UUID:
    """Publish a single event."""
    publisher = await get_publisher()
    return await publisher.publish(event)


async def publish_deal_event(
    deal_id: UUID,
    event_type: str,
    event_data: dict,
    source: str = "deal_lifecycle"
) -> UUID:
    """Publish a deal event."""
    publisher = await get_publisher()
    return await publisher.publish_deal_event(
        deal_id, event_type, event_data, source
    )


async def publish_action_event(
    action_id: UUID,
    correlation_id: UUID,
    event_type: str,
    event_data: dict,
    deal_id: Optional[UUID] = None,
    source: str = "action_engine"
) -> UUID:
    """Publish an action event."""
    publisher = await get_publisher()
    return await publisher.publish_action_event(
        action_id, correlation_id, event_type, event_data, deal_id, source
    )


async def publish_agent_event(
    correlation_id: UUID,
    event_type: str,
    event_data: dict,
    run_id: Optional[UUID] = None,
    thread_id: Optional[str] = None,
    source: str = "agent"
) -> UUID:
    """Publish an agent event."""
    publisher = await get_publisher()
    return await publisher.publish_agent_event(
        correlation_id, event_type, event_data, run_id, thread_id, source
    )


async def get_events_after(
    last_sequence: int,
    deal_id: Optional[str] = None,
    limit: int = 100
) -> List[dict]:
    """
    Get events after a given sequence number.

    Used for SSE replay and event sourcing.

    Args:
        last_sequence: Get events with sequence > this value
        deal_id: Optional filter by deal_id
        limit: Maximum events to return

    Returns:
        List of events ordered by sequence_number
    """
    db = await get_database()

    if deal_id:
        events = await db.fetch(
            """
            SELECT id, deal_id, event_type, source, actor, actor_id, actor_type,
                   details, sequence_number, created_at
            FROM zakops.deal_events
            WHERE sequence_number > $1
              AND deal_id = $2
            ORDER BY sequence_number ASC
            LIMIT $3
            """,
            last_sequence, deal_id, limit
        )
    else:
        events = await db.fetch(
            """
            SELECT id, deal_id, event_type, source, actor, actor_id, actor_type,
                   details, sequence_number, created_at
            FROM zakops.deal_events
            WHERE sequence_number > $1
            ORDER BY sequence_number ASC
            LIMIT $2
            """,
            last_sequence, limit
        )

    return [dict(e) for e in events]
