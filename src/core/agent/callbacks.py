"""
Agent Callback Handler

Handles events from agent execution and emits to event system.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID
import logging

from .models import ToolCall, ToolResult, AgentRunStatus

logger = logging.getLogger(__name__)


async def _safe_publish_event(event_data: dict, event_type: str, correlation_id: str):
    """
    Safely publish an event, catching any database errors.

    Event publishing is best-effort - failures should not block agent execution.
    """
    try:
        from ..events import publish_event, AgentEvent

        await publish_event(AgentEvent(
            correlation_id=correlation_id,
            event_type=event_type,
            event_data=event_data
        ))
    except Exception as e:
        # Log but don't fail - event publishing is best-effort
        logger.warning(f"Failed to publish event {event_type}: {e}")


class AgentCallbackHandler:
    """
    Callback handler for agent execution events.

    Emits events to the event system with proper trace_id propagation.
    """

    def __init__(
        self,
        run_id: UUID,
        deal_id: UUID,
        trace_id: str,
        correlation_id: str
    ):
        self.run_id = run_id
        self.deal_id = deal_id
        self.trace_id = trace_id
        self.correlation_id = correlation_id
        self.tool_calls: List[ToolCall] = []
        self.tool_results: List[ToolResult] = []

    async def on_run_start(self, task: str, context: Dict[str, Any] = None):
        """Called when agent run starts."""
        logger.info(f"Agent run started: {self.run_id} (trace: {self.trace_id})")

        await _safe_publish_event(
            event_data={
                "run_id": str(self.run_id),
                "deal_id": str(self.deal_id),
                "trace_id": self.trace_id,
                "task": task,
                "context": context or {},
                "timestamp": datetime.utcnow().isoformat()
            },
            event_type="agent.run_started",
            correlation_id=self.correlation_id
        )

    async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any]) -> ToolCall:
        """Called when a tool is invoked."""
        tool_call = ToolCall(
            tool_name=tool_name,
            tool_input=tool_input
        )
        self.tool_calls.append(tool_call)

        logger.info(f"Tool called: {tool_name} (run: {self.run_id})")

        await _safe_publish_event(
            event_data={
                "run_id": str(self.run_id),
                "deal_id": str(self.deal_id),
                "trace_id": self.trace_id,
                "tool_call_id": str(tool_call.id),
                "tool_name": tool_name,
                "tool_input": tool_input,
                "timestamp": tool_call.timestamp.isoformat()
            },
            event_type="agent.tool_called",
            correlation_id=self.correlation_id
        )

        return tool_call

    async def on_tool_end(
        self,
        tool_call: ToolCall,
        output: Any,
        error: Optional[str] = None,
        duration_ms: int = 0
    ) -> ToolResult:
        """Called when a tool completes."""
        result = ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.tool_name,
            output=output,
            error=error,
            duration_ms=duration_ms
        )
        self.tool_results.append(result)

        logger.info(f"Tool completed: {tool_call.tool_name} ({duration_ms}ms)")

        await _safe_publish_event(
            event_data={
                "run_id": str(self.run_id),
                "deal_id": str(self.deal_id),
                "trace_id": self.trace_id,
                "tool_call_id": str(tool_call.id),
                "tool_name": tool_call.tool_name,
                "success": error is None,
                "error": error,
                "duration_ms": duration_ms,
                "timestamp": result.timestamp.isoformat()
            },
            event_type="agent.tool_completed",
            correlation_id=self.correlation_id
        )

        return result

    async def on_run_end(
        self,
        status: AgentRunStatus,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        actions_created: List[UUID] = None
    ):
        """Called when agent run completes."""
        event_type = (
            "agent.run_completed" if status == AgentRunStatus.COMPLETED
            else "agent.run_failed"
        )

        logger.info(f"Agent run {status.value}: {self.run_id}")

        await _safe_publish_event(
            event_data={
                "run_id": str(self.run_id),
                "deal_id": str(self.deal_id),
                "trace_id": self.trace_id,
                "status": status.value,
                "tool_calls_count": len(self.tool_calls),
                "actions_created": [str(a) for a in (actions_created or [])],
                "result": result,
                "error": error,
                "timestamp": datetime.utcnow().isoformat()
            },
            event_type=event_type,
            correlation_id=self.correlation_id
        )

    async def on_action_created(self, action_id: UUID, action_type: str):
        """Called when an action is created from agent output."""
        logger.info(f"Action created: {action_id} ({action_type})")

        await _safe_publish_event(
            event_data={
                "action_id": str(action_id),
                "deal_id": str(self.deal_id),
                "trace_id": self.trace_id,
                "run_id": str(self.run_id),
                "action_type": action_type,
                "source": "agent",
                "timestamp": datetime.utcnow().isoformat()
            },
            event_type="action.created",
            correlation_id=self.correlation_id
        )
