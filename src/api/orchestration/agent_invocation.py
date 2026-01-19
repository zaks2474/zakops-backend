"""
ZakOps Agent Invocation Layer
=============================

Thread/Run model for LangSmith Agent Builder integration.
Provides proper SSE streaming with event IDs for resume capability.

API Pattern (per LangSmith Server API):
- POST /threads - Create thread
- GET /threads/{thread_id} - Get thread
- POST /threads/{thread_id}/runs/stream - Create and stream run
- GET /threads/{thread_id}/runs - List runs
- GET /threads/{thread_id}/runs/{run_id} - Get run
- POST /threads/{thread_id}/runs/{run_id}/tool_calls/{tool_call_id}/approve - Approve tool
- POST /threads/{thread_id}/runs/{run_id}/tool_calls/{tool_call_id}/reject - Reject tool
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Optional

import asyncpg
from starlette.responses import StreamingResponse

logger = logging.getLogger("agent_invocation")

# =============================================================================
# Configuration
# =============================================================================

DATABASE_URL = os.getenv(
    "ZAKOPS_DATABASE_URL",
    "postgresql://dealengine:changeme@localhost:5435/zakops"
)

# =============================================================================
# Enums
# =============================================================================

class ThreadStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolCallStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentEventType(str, Enum):
    # Run lifecycle
    RUN_CREATED = "run_created"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"

    # Tool events
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    TOOL_CALL_FAILED = "tool_call_failed"
    TOOL_APPROVAL_REQUIRED = "tool_approval_required"
    TOOL_APPROVAL_GRANTED = "tool_approval_granted"
    TOOL_APPROVAL_DENIED = "tool_approval_denied"

    # Streaming
    STREAM_START = "stream_start"
    STREAM_TOKEN = "stream_token"
    STREAM_END = "stream_end"
    STREAM_ERROR = "stream_error"

    # Custom
    CUSTOM = "custom"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AgentThread:
    thread_id: str
    assistant_id: str
    deal_id: Optional[str] = None
    status: ThreadStatus = ThreadStatus.ACTIVE
    metadata: dict = field(default_factory=dict)
    user_id: Optional[str] = None
    user_context: dict = field(default_factory=dict)
    message_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentRun:
    run_id: str
    thread_id: str
    assistant_id: str
    status: RunStatus = RunStatus.PENDING
    input_message: Optional[str] = None
    output_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    last_event_id: Optional[str] = None
    stream_position: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentToolCall:
    tool_call_id: str
    run_id: str
    tool_name: str
    tool_input: dict = field(default_factory=dict)
    tool_output: Optional[dict] = None
    status: ToolCallStatus = ToolCallStatus.PENDING
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    requires_approval: bool = False
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    error: Optional[str] = None
    retry_count: int = 0
    created_action_id: Optional[str] = None
    sequence_number: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentEvent:
    event_id: str
    thread_id: str
    run_id: str
    event_type: AgentEventType
    event_data: dict = field(default_factory=dict)
    tool_call_id: Optional[str] = None
    sequence_number: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Database Pool
# =============================================================================

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create database connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
        )
    return _pool


async def close_pool():
    """Close database connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# =============================================================================
# Thread Operations
# =============================================================================

async def create_thread(
    assistant_id: str,
    deal_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    user_context: Optional[dict] = None,
) -> AgentThread:
    """Create a new agent thread."""
    pool = await get_pool()

    thread_id = f"thread_{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO zakops.agent_threads (
                thread_id, assistant_id, deal_id, status,
                metadata, user_id, user_context,
                created_at, updated_at, last_active_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8, $8)
            """,
            thread_id,
            assistant_id,
            deal_id,
            ThreadStatus.ACTIVE.value,
            json.dumps(metadata or {}),
            user_id,
            json.dumps(user_context or {}),
            now,
        )

    return AgentThread(
        thread_id=thread_id,
        assistant_id=assistant_id,
        deal_id=deal_id,
        metadata=metadata or {},
        user_id=user_id,
        user_context=user_context or {},
        created_at=now,
        updated_at=now,
        last_active_at=now,
    )


async def get_thread(thread_id: str) -> Optional[AgentThread]:
    """Get a thread by ID."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM zakops.agent_threads WHERE thread_id = $1
            """,
            thread_id,
        )

    if not row:
        return None

    return AgentThread(
        thread_id=row["thread_id"],
        assistant_id=row["assistant_id"],
        deal_id=row["deal_id"],
        status=ThreadStatus(row["status"]),
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        user_id=row["user_id"],
        user_context=json.loads(row["user_context"]) if row["user_context"] else {},
        message_count=row["message_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_active_at=row["last_active_at"],
    )


async def update_thread_activity(thread_id: str):
    """Update thread's last_active_at timestamp."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE zakops.agent_threads
            SET last_active_at = NOW()
            WHERE thread_id = $1
            """,
            thread_id,
        )


async def archive_thread(thread_id: str):
    """Archive a thread."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE zakops.agent_threads
            SET status = $1
            WHERE thread_id = $2
            """,
            ThreadStatus.ARCHIVED.value,
            thread_id,
        )


# =============================================================================
# Run Operations
# =============================================================================

async def create_run(
    thread_id: str,
    assistant_id: str,
    input_message: str,
    metadata: Optional[dict] = None,
) -> AgentRun:
    """Create a new run within a thread."""
    pool = await get_pool()

    run_id = f"run_{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO zakops.agent_runs (
                run_id, thread_id, assistant_id, status,
                input_message, metadata, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $7)
            """,
            run_id,
            thread_id,
            assistant_id,
            RunStatus.PENDING.value,
            input_message,
            json.dumps(metadata or {}),
            now,
        )

        # Update thread activity and message count
        await conn.execute(
            """
            UPDATE zakops.agent_threads
            SET last_active_at = NOW(), message_count = message_count + 1
            WHERE thread_id = $1
            """,
            thread_id,
        )

    return AgentRun(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=assistant_id,
        input_message=input_message,
        metadata=metadata or {},
        created_at=now,
        updated_at=now,
    )


async def get_run(run_id: str) -> Optional[AgentRun]:
    """Get a run by ID."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM zakops.agent_runs WHERE run_id = $1
            """,
            run_id,
        )

    if not row:
        return None

    return AgentRun(
        run_id=row["run_id"],
        thread_id=row["thread_id"],
        assistant_id=row["assistant_id"],
        status=RunStatus(row["status"]),
        input_message=row["input_message"],
        output_message=row["output_message"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        duration_ms=row["duration_ms"],
        error=row["error"],
        error_code=row["error_code"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        last_event_id=row["last_event_id"],
        stream_position=row["stream_position"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def list_runs(
    thread_id: str,
    limit: int = 20,
    status: Optional[RunStatus] = None,
) -> list[AgentRun]:
    """List runs for a thread."""
    pool = await get_pool()

    query = """
        SELECT * FROM zakops.agent_runs
        WHERE thread_id = $1
    """
    params = [thread_id]

    if status:
        query += " AND status = $2"
        params.append(status.value)

    query += f" ORDER BY created_at DESC LIMIT {limit}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [
        AgentRun(
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            assistant_id=row["assistant_id"],
            status=RunStatus(row["status"]),
            input_message=row["input_message"],
            output_message=row["output_message"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            duration_ms=row["duration_ms"],
            error=row["error"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            last_event_id=row["last_event_id"],
            stream_position=row["stream_position"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]


async def update_run_status(
    run_id: str,
    status: RunStatus,
    output_message: Optional[str] = None,
    error: Optional[str] = None,
    error_code: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
):
    """Update run status and related fields."""
    pool = await get_pool()

    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        # Get current run for duration calculation
        row = await conn.fetchrow(
            "SELECT started_at FROM zakops.agent_runs WHERE run_id = $1",
            run_id,
        )

        duration_ms = None
        if row and row["started_at"] and status in (
            RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED
        ):
            duration_ms = int((now - row["started_at"]).total_seconds() * 1000)

        started_at = now if status == RunStatus.RUNNING else None
        completed_at = now if status in (
            RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED
        ) else None

        await conn.execute(
            """
            UPDATE zakops.agent_runs
            SET status = $2,
                output_message = COALESCE($3, output_message),
                error = $4,
                error_code = $5,
                input_tokens = COALESCE($6, input_tokens),
                output_tokens = COALESCE($7, output_tokens),
                total_tokens = COALESCE($6, input_tokens, 0) + COALESCE($7, output_tokens, 0),
                started_at = COALESCE(started_at, $8),
                completed_at = $9,
                duration_ms = COALESCE($10, duration_ms)
            WHERE run_id = $1
            """,
            run_id,
            status.value,
            output_message,
            error,
            error_code,
            input_tokens,
            output_tokens,
            started_at,
            completed_at,
            duration_ms,
        )


# =============================================================================
# Tool Call Operations
# =============================================================================

async def create_tool_call(
    run_id: str,
    tool_name: str,
    tool_input: dict,
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
    requires_approval: bool = False,
) -> AgentToolCall:
    """Create a new tool call within a run."""
    pool = await get_pool()

    tool_call_id = f"tc_{uuid.uuid4().hex}"
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        # Get next sequence number
        row = await conn.fetchrow(
            """
            SELECT COALESCE(MAX(sequence_number), -1) + 1 AS next_seq
            FROM zakops.agent_tool_calls WHERE run_id = $1
            """,
            run_id,
        )
        sequence_number = row["next_seq"]

        # Determine initial status based on approval requirement
        initial_status = (
            ToolCallStatus.PENDING.value
            if requires_approval
            else ToolCallStatus.RUNNING.value
        )

        await conn.execute(
            """
            INSERT INTO zakops.agent_tool_calls (
                tool_call_id, run_id, tool_name, tool_input,
                status, risk_level, requires_approval,
                sequence_number, created_at, updated_at,
                started_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $9, $10)
            """,
            tool_call_id,
            run_id,
            tool_name,
            json.dumps(tool_input),
            initial_status,
            risk_level.value,
            requires_approval,
            sequence_number,
            now,
            None if requires_approval else now,
        )

    return AgentToolCall(
        tool_call_id=tool_call_id,
        run_id=run_id,
        tool_name=tool_name,
        tool_input=tool_input,
        status=ToolCallStatus.PENDING if requires_approval else ToolCallStatus.RUNNING,
        risk_level=risk_level,
        requires_approval=requires_approval,
        sequence_number=sequence_number,
        started_at=None if requires_approval else now,
        created_at=now,
        updated_at=now,
    )


async def get_tool_call(tool_call_id: str) -> Optional[AgentToolCall]:
    """Get a tool call by ID."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT * FROM zakops.agent_tool_calls WHERE tool_call_id = $1
            """,
            tool_call_id,
        )

    if not row:
        return None

    return AgentToolCall(
        tool_call_id=row["tool_call_id"],
        run_id=row["run_id"],
        tool_name=row["tool_name"],
        tool_input=json.loads(row["tool_input"]) if row["tool_input"] else {},
        tool_output=json.loads(row["tool_output"]) if row["tool_output"] else None,
        status=ToolCallStatus(row["status"]),
        risk_level=ToolRiskLevel(row["risk_level"]),
        requires_approval=row["requires_approval"],
        approved_by=row["approved_by"],
        approved_at=row["approved_at"],
        rejection_reason=row["rejection_reason"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        duration_ms=row["duration_ms"],
        error=row["error"],
        retry_count=row["retry_count"],
        created_action_id=row["created_action_id"],
        sequence_number=row["sequence_number"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def approve_tool_call(
    tool_call_id: str,
    approved_by: str,
) -> AgentToolCall:
    """Approve a pending tool call."""
    pool = await get_pool()
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE zakops.agent_tool_calls
            SET status = $2, approved_by = $3, approved_at = $4, started_at = $4
            WHERE tool_call_id = $1 AND status = 'pending'
            """,
            tool_call_id,
            ToolCallStatus.APPROVED.value,
            approved_by,
            now,
        )

    return await get_tool_call(tool_call_id)


async def reject_tool_call(
    tool_call_id: str,
    rejected_by: str,
    reason: str,
) -> AgentToolCall:
    """Reject a pending tool call."""
    pool = await get_pool()
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE zakops.agent_tool_calls
            SET status = $2, approved_by = $3, approved_at = $4, rejection_reason = $5
            WHERE tool_call_id = $1 AND status = 'pending'
            """,
            tool_call_id,
            ToolCallStatus.REJECTED.value,
            rejected_by,
            now,
            reason,
        )

    return await get_tool_call(tool_call_id)


async def complete_tool_call(
    tool_call_id: str,
    tool_output: dict,
    created_action_id: Optional[str] = None,
):
    """Mark a tool call as completed with output."""
    pool = await get_pool()
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        # Get started_at for duration calculation
        row = await conn.fetchrow(
            "SELECT started_at FROM zakops.agent_tool_calls WHERE tool_call_id = $1",
            tool_call_id,
        )

        duration_ms = None
        if row and row["started_at"]:
            duration_ms = int((now - row["started_at"]).total_seconds() * 1000)

        await conn.execute(
            """
            UPDATE zakops.agent_tool_calls
            SET status = $2, tool_output = $3, completed_at = $4,
                duration_ms = $5, created_action_id = $6
            WHERE tool_call_id = $1
            """,
            tool_call_id,
            ToolCallStatus.COMPLETED.value,
            json.dumps(tool_output),
            now,
            duration_ms,
            created_action_id,
        )


async def fail_tool_call(tool_call_id: str, error: str):
    """Mark a tool call as failed."""
    pool = await get_pool()
    now = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        # Get started_at for duration calculation
        row = await conn.fetchrow(
            "SELECT started_at FROM zakops.agent_tool_calls WHERE tool_call_id = $1",
            tool_call_id,
        )

        duration_ms = None
        if row and row["started_at"]:
            duration_ms = int((now - row["started_at"]).total_seconds() * 1000)

        await conn.execute(
            """
            UPDATE zakops.agent_tool_calls
            SET status = $2, error = $3, completed_at = $4, duration_ms = $5
            WHERE tool_call_id = $1
            """,
            tool_call_id,
            ToolCallStatus.FAILED.value,
            error,
            now,
            duration_ms,
        )


# =============================================================================
# Event Operations
# =============================================================================

async def record_event(
    thread_id: str,
    run_id: str,
    event_type: AgentEventType,
    event_data: dict,
    tool_call_id: Optional[str] = None,
) -> str:
    """Record an agent event and return the event_id."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT zakops.record_agent_event($1, $2, $3, $4, $5) AS event_id
            """,
            thread_id,
            run_id,
            event_type.value,
            json.dumps(event_data),
            tool_call_id,
        )

    return row["event_id"]


async def get_events_since(
    run_id: str,
    last_event_id: Optional[str] = None,
    limit: int = 100,
) -> list[AgentEvent]:
    """Get events for a run, optionally starting after a specific event_id."""
    pool = await get_pool()

    query = """
        SELECT * FROM zakops.agent_events
        WHERE run_id = $1
    """
    params = [run_id]

    if last_event_id:
        # Extract sequence number from event_id (format: run_id:000000)
        try:
            last_seq = int(last_event_id.split(":")[-1])
            query += " AND sequence_number > $2"
            params.append(last_seq)
        except (ValueError, IndexError):
            pass

    query += f" ORDER BY sequence_number ASC LIMIT {limit}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    return [
        AgentEvent(
            event_id=row["event_id"],
            thread_id=row["thread_id"],
            run_id=row["run_id"],
            event_type=AgentEventType(row["event_type"]),
            event_data=json.loads(row["event_data"]) if row["event_data"] else {},
            tool_call_id=row["tool_call_id"],
            sequence_number=row["sequence_number"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


# =============================================================================
# SSE Streaming
# =============================================================================

async def stream_events(
    run_id: str,
    thread_id: str,
    last_event_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream events as SSE, supporting resume via Last-Event-ID.

    Event format:
    id: {event_id}
    event: {event_type}
    data: {json_data}
    """
    pool = await get_pool()

    # Parse last_event_id to get starting sequence
    start_seq = 0
    if last_event_id:
        try:
            start_seq = int(last_event_id.split(":")[-1]) + 1
        except (ValueError, IndexError):
            pass

    current_seq = start_seq
    poll_interval = 0.1  # 100ms polling

    while True:
        async with pool.acquire() as conn:
            # Check if run is still active
            run = await conn.fetchrow(
                "SELECT status FROM zakops.agent_runs WHERE run_id = $1",
                run_id,
            )

            if not run:
                yield format_sse_event("error", {"error": "Run not found"})
                break

            run_status = run["status"]

            # Fetch new events
            rows = await conn.fetch(
                """
                SELECT * FROM zakops.agent_events
                WHERE run_id = $1 AND sequence_number >= $2
                ORDER BY sequence_number ASC
                LIMIT 100
                """,
                run_id,
                current_seq,
            )

            for row in rows:
                event_id = row["event_id"]
                event_type = row["event_type"]
                event_data = json.loads(row["event_data"]) if row["event_data"] else {}

                yield format_sse_event(event_type, event_data, event_id)
                current_seq = row["sequence_number"] + 1

            # If run is terminal and no more events, end stream
            if run_status in ("completed", "failed", "cancelled") and not rows:
                break

        await asyncio.sleep(poll_interval)


def format_sse_event(
    event_type: str,
    data: dict,
    event_id: Optional[str] = None,
) -> str:
    """Format an SSE event."""
    lines = []

    if event_id:
        lines.append(f"id: {event_id}")

    lines.append(f"event: {event_type}")
    lines.append(f"data: {json.dumps(data)}")
    lines.append("")
    lines.append("")

    return "\n".join(lines)


def create_sse_response(
    run_id: str,
    thread_id: str,
    last_event_id: Optional[str] = None,
) -> StreamingResponse:
    """Create a StreamingResponse for SSE events."""
    return StreamingResponse(
        stream_events(run_id, thread_id, last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# High-Level Operations
# =============================================================================

async def start_run(run_id: str, thread_id: str) -> str:
    """Start a run and emit the run_started event."""
    await update_run_status(run_id, RunStatus.RUNNING)

    event_id = await record_event(
        thread_id=thread_id,
        run_id=run_id,
        event_type=AgentEventType.RUN_STARTED,
        event_data={"status": "running"},
    )

    return event_id


async def complete_run(
    run_id: str,
    thread_id: str,
    output_message: str,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> str:
    """Complete a run and emit the run_completed event."""
    await update_run_status(
        run_id=run_id,
        status=RunStatus.COMPLETED,
        output_message=output_message,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    event_id = await record_event(
        thread_id=thread_id,
        run_id=run_id,
        event_type=AgentEventType.RUN_COMPLETED,
        event_data={
            "status": "completed",
            "output_message": output_message,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    )

    return event_id


async def fail_run(
    run_id: str,
    thread_id: str,
    error: str,
    error_code: Optional[str] = None,
) -> str:
    """Fail a run and emit the run_failed event."""
    await update_run_status(
        run_id=run_id,
        status=RunStatus.FAILED,
        error=error,
        error_code=error_code,
    )

    event_id = await record_event(
        thread_id=thread_id,
        run_id=run_id,
        event_type=AgentEventType.RUN_FAILED,
        event_data={
            "status": "failed",
            "error": error,
            "error_code": error_code,
        },
    )

    return event_id


async def emit_stream_token(
    run_id: str,
    thread_id: str,
    token: str,
) -> str:
    """Emit a stream token event."""
    return await record_event(
        thread_id=thread_id,
        run_id=run_id,
        event_type=AgentEventType.STREAM_TOKEN,
        event_data={"token": token},
    )


# =============================================================================
# Utility Functions
# =============================================================================

def thread_to_dict(thread: AgentThread) -> dict:
    """Convert AgentThread to dictionary."""
    return {
        "thread_id": thread.thread_id,
        "assistant_id": thread.assistant_id,
        "deal_id": thread.deal_id,
        "status": thread.status.value,
        "metadata": thread.metadata,
        "user_id": thread.user_id,
        "user_context": thread.user_context,
        "message_count": thread.message_count,
        "created_at": thread.created_at.isoformat(),
        "updated_at": thread.updated_at.isoformat(),
        "last_active_at": thread.last_active_at.isoformat(),
    }


def run_to_dict(run: AgentRun) -> dict:
    """Convert AgentRun to dictionary."""
    return {
        "run_id": run.run_id,
        "thread_id": run.thread_id,
        "assistant_id": run.assistant_id,
        "status": run.status.value,
        "input_message": run.input_message,
        "output_message": run.output_message,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "duration_ms": run.duration_ms,
        "error": run.error,
        "error_code": run.error_code,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "total_tokens": run.total_tokens,
        "metadata": run.metadata,
        "last_event_id": run.last_event_id,
        "stream_position": run.stream_position,
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
    }


def tool_call_to_dict(tc: AgentToolCall) -> dict:
    """Convert AgentToolCall to dictionary."""
    return {
        "tool_call_id": tc.tool_call_id,
        "run_id": tc.run_id,
        "tool_name": tc.tool_name,
        "tool_input": tc.tool_input,
        "tool_output": tc.tool_output,
        "status": tc.status.value,
        "risk_level": tc.risk_level.value,
        "requires_approval": tc.requires_approval,
        "approved_by": tc.approved_by,
        "approved_at": tc.approved_at.isoformat() if tc.approved_at else None,
        "rejection_reason": tc.rejection_reason,
        "started_at": tc.started_at.isoformat() if tc.started_at else None,
        "completed_at": tc.completed_at.isoformat() if tc.completed_at else None,
        "duration_ms": tc.duration_ms,
        "error": tc.error,
        "retry_count": tc.retry_count,
        "created_action_id": tc.created_action_id,
        "sequence_number": tc.sequence_number,
        "created_at": tc.created_at.isoformat(),
        "updated_at": tc.updated_at.isoformat(),
    }


def event_to_dict(event: AgentEvent) -> dict:
    """Convert AgentEvent to dictionary."""
    return {
        "event_id": event.event_id,
        "thread_id": event.thread_id,
        "run_id": event.run_id,
        "event_type": event.event_type.value,
        "event_data": event.event_data,
        "tool_call_id": event.tool_call_id,
        "sequence_number": event.sequence_number,
        "created_at": event.created_at.isoformat(),
    }
