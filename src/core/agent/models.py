"""
Agent Models

Data models for agent invocation and execution.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class AgentRunStatus(str, Enum):
    """Status of an agent run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolCall(BaseModel):
    """A tool call made by the agent."""
    id: UUID = Field(default_factory=uuid4)
    tool_name: str
    tool_input: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ToolResult(BaseModel):
    """Result of a tool call."""
    tool_call_id: UUID
    tool_name: str
    output: Any
    error: Optional[str] = None
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentRunRequest(BaseModel):
    """Request to invoke an agent."""
    deal_id: UUID
    task: str
    context: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "deal_id": "123e4567-e89b-12d3-a456-426614174000",
                "task": "Analyze this deal and suggest next steps",
                "context": {"stage": "initial_review"}
            }
        }


class AgentRunResponse(BaseModel):
    """Response from an agent run."""
    run_id: UUID
    deal_id: UUID
    status: AgentRunStatus
    trace_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    tool_calls: List[ToolCall] = []
    actions_created: List[UUID] = []
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "123e4567-e89b-12d3-a456-426614174001",
                "deal_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "trace_id": "trace-abc-123",
                "started_at": "2026-01-19T12:00:00Z",
                "completed_at": "2026-01-19T12:00:05Z",
                "tool_calls": [],
                "actions_created": []
            }
        }


class AgentRun(BaseModel):
    """Database model for agent run."""
    id: UUID = Field(default_factory=uuid4)
    deal_id: UUID
    correlation_id: str  # Same as deal_id for correlation
    trace_id: str
    status: AgentRunStatus = AgentRunStatus.PENDING
    task: str
    context: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tool_calls_count: int = 0
    actions_created_count: int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    def to_response(self, tool_calls: List[ToolCall] = None, actions: List[UUID] = None) -> AgentRunResponse:
        """Convert to API response."""
        return AgentRunResponse(
            run_id=self.id,
            deal_id=self.deal_id,
            status=self.status,
            trace_id=self.trace_id,
            started_at=self.started_at,
            completed_at=self.completed_at,
            tool_calls=tool_calls or [],
            actions_created=actions or [],
            result=self.result,
            error=self.error
        )
