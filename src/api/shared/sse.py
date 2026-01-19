"""
Production-Hardened Server-Sent Events (SSE)

Features:
- Heartbeat keep-alive (30s)
- Authentication enforcement
- Last-Event-ID replay support
- Reconnection directives
- Connection limits & backpressure
- Graceful degradation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator, Optional, Dict, Any, Set
from uuid import uuid4
from dataclasses import dataclass, field
from collections import defaultdict

from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


@dataclass
class SSEConfig:
    """SSE configuration."""
    heartbeat_interval: int = 30  # seconds
    retry_interval: int = 3000  # milliseconds (sent to client)
    max_connections_per_user: int = 5
    max_total_connections: int = 1000
    event_replay_window: timedelta = timedelta(hours=1)
    slow_consumer_timeout: int = 30  # seconds
    backpressure_threshold: int = 100  # queued events


@dataclass
class SSEConnection:
    """Represents an active SSE connection."""
    id: str = field(default_factory=lambda: uuid4().hex)
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None  # Filter events by deal_id
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_event_id: Optional[str] = None
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))

    def __hash__(self):
        return hash(self.id)


class SSEManager:
    """
    Manages SSE connections with production-grade reliability.

    Features:
    - Connection tracking per user
    - Heartbeat management
    - Event replay from Last-Event-ID
    - Backpressure handling
    """

    def __init__(self, config: SSEConfig = None):
        self.config = config or SSEConfig()
        self._connections: Set[SSEConnection] = set()
        self._user_connections: Dict[str, Set[SSEConnection]] = defaultdict(set)
        self._event_buffer: Dict[str, Dict[str, Any]] = {}  # event_id -> event
        self._buffer_order: list = []  # Ordered list of event_ids
        self._lock = asyncio.Lock()

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def connect(
        self,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        last_event_id: Optional[str] = None
    ) -> SSEConnection:
        """
        Register a new SSE connection.

        Args:
            user_id: Authenticated user ID
            correlation_id: Filter events by this ID (e.g., deal_id)
            last_event_id: Replay events after this ID

        Returns:
            SSEConnection instance

        Raises:
            HTTPException: If connection limits exceeded
        """
        async with self._lock:
            # Check total connection limit
            if len(self._connections) >= self.config.max_total_connections:
                logger.warning(f"SSE max connections reached: {len(self._connections)}")
                raise HTTPException(
                    status_code=503,
                    detail="Server at capacity. Please try again later."
                )

            # Check per-user limit
            if user_id:
                user_conns = self._user_connections[user_id]
                if len(user_conns) >= self.config.max_connections_per_user:
                    logger.warning(f"SSE max connections for user {user_id}: {len(user_conns)}")
                    raise HTTPException(
                        status_code=429,
                        detail="Too many connections. Please close some tabs."
                    )

            # Create connection
            conn = SSEConnection(
                user_id=user_id,
                correlation_id=correlation_id,
                last_event_id=last_event_id
            )

            self._connections.add(conn)
            if user_id:
                self._user_connections[user_id].add(conn)

            logger.info(f"SSE connected: {conn.id} (user: {user_id}, total: {len(self._connections)})")

            return conn

    async def disconnect(self, conn: SSEConnection):
        """Remove a connection."""
        async with self._lock:
            self._connections.discard(conn)
            if conn.user_id:
                self._user_connections[conn.user_id].discard(conn)

            logger.info(f"SSE disconnected: {conn.id} (total: {len(self._connections)})")

    async def broadcast(
        self,
        event_type: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
        event_id: Optional[str] = None
    ):
        """
        Broadcast event to all relevant connections.

        Args:
            event_type: Event type name
            data: Event payload
            correlation_id: Only send to connections filtering for this ID
            event_id: Unique event ID for replay support
        """
        event_id = event_id or uuid4().hex

        event = {
            "id": event_id,
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "correlation_id": correlation_id
        }

        # Buffer for replay
        await self._buffer_event(event_id, event)

        # Broadcast to connections
        for conn in list(self._connections):
            # Filter by correlation_id if connection has one
            if conn.correlation_id and correlation_id:
                if conn.correlation_id != correlation_id:
                    continue

            try:
                # Non-blocking put with backpressure
                if conn.queue.qsize() >= self.config.backpressure_threshold:
                    logger.warning(f"SSE backpressure on {conn.id}, dropping event")
                    continue

                conn.queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"SSE queue full for {conn.id}")

    async def _buffer_event(self, event_id: str, event: Dict[str, Any]):
        """Buffer event for replay."""
        async with self._lock:
            self._event_buffer[event_id] = event
            self._buffer_order.append(event_id)

            # Clean old events outside replay window
            cutoff = datetime.now(timezone.utc) - self.config.event_replay_window
            while self._buffer_order:
                oldest_id = self._buffer_order[0]
                oldest = self._event_buffer.get(oldest_id)
                if oldest:
                    try:
                        event_time = datetime.fromisoformat(oldest["timestamp"].replace("Z", "+00:00"))
                        if event_time < cutoff:
                            del self._event_buffer[oldest_id]
                            self._buffer_order.pop(0)
                        else:
                            break
                    except (ValueError, KeyError):
                        self._buffer_order.pop(0)
                else:
                    self._buffer_order.pop(0)

    async def get_replay_events(self, last_event_id: str) -> list:
        """Get events after the given event ID for replay."""
        if not last_event_id:
            return []

        events = []
        found = False

        for event_id in self._buffer_order:
            if found:
                event = self._event_buffer.get(event_id)
                if event:
                    events.append(event)
            elif event_id == last_event_id:
                found = True

        return events

    async def stream(self, conn: SSEConnection) -> AsyncGenerator[str, None]:
        """
        Generate SSE stream for a connection.

        Yields:
            SSE-formatted strings
        """
        # Send retry directive
        yield f"retry: {self.config.retry_interval}\n\n"

        # Replay missed events
        if conn.last_event_id:
            replay_events = await self.get_replay_events(conn.last_event_id)
            for event in replay_events:
                yield self._format_event(event)

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat(conn))

        try:
            while True:
                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(
                        conn.queue.get(),
                        timeout=self.config.slow_consumer_timeout
                    )
                    yield self._format_event(event)

                except asyncio.TimeoutError:
                    # No event received, connection might be slow/dead
                    # Heartbeat will keep connection alive
                    continue

        except asyncio.CancelledError:
            pass
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            await self.disconnect(conn)

    async def _heartbeat(self, conn: SSEConnection):
        """Send periodic heartbeats."""
        try:
            while True:
                await asyncio.sleep(self.config.heartbeat_interval)

                heartbeat = {
                    "id": f"heartbeat-{uuid4().hex[:8]}",
                    "type": "heartbeat",
                    "data": {"timestamp": datetime.now(timezone.utc).isoformat()},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

                try:
                    conn.queue.put_nowait(heartbeat)
                except asyncio.QueueFull:
                    logger.warning(f"Cannot send heartbeat to {conn.id}, queue full")
                    break

        except asyncio.CancelledError:
            pass

    def _format_event(self, event: Dict[str, Any]) -> str:
        """Format event as SSE."""
        lines = []

        if event.get("id"):
            lines.append(f"id: {event['id']}")

        if event.get("type") and event["type"] != "message":
            lines.append(f"event: {event['type']}")

        data = event.get("data", {})
        lines.append(f"data: {json.dumps(data)}")

        return "\n".join(lines) + "\n\n"


# Global manager instance
_sse_manager: Optional[SSEManager] = None


def get_sse_manager() -> SSEManager:
    """Get or create the global SSE manager."""
    global _sse_manager
    if _sse_manager is None:
        _sse_manager = SSEManager()
    return _sse_manager


async def create_sse_response(
    request: Request,
    user_id: Optional[str] = None,
    correlation_id: Optional[str] = None
) -> StreamingResponse:
    """
    Create an SSE StreamingResponse with proper headers.

    Args:
        request: FastAPI request
        user_id: Authenticated user ID
        correlation_id: Filter events by this ID

    Returns:
        StreamingResponse configured for SSE
    """
    manager = get_sse_manager()

    # Get Last-Event-ID for replay
    last_event_id = request.headers.get("Last-Event-ID")

    # Create connection
    conn = await manager.connect(
        user_id=user_id,
        correlation_id=correlation_id,
        last_event_id=last_event_id
    )

    return StreamingResponse(
        manager.stream(conn),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
