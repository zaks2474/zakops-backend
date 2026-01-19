"""
Agent Invoker

Main entry point for agent invocation.
Phase 15: Added OpenTelemetry tracing.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
import logging
import time

from ..database.adapter import get_database
from ..hitl import assess_risk, RiskLevel
from ..observability.tracing import create_span, traced, add_correlation_id_to_span
from ..observability.metrics import record_counter, record_histogram
from .models import (
    AgentRun, AgentRunRequest, AgentRunResponse, AgentRunStatus,
    ToolCall, ToolResult
)
from .tools import get_tool_registry, ToolRegistry
from .callbacks import AgentCallbackHandler

logger = logging.getLogger(__name__)


class AgentInvoker:
    """
    Invokes the agent and manages execution lifecycle.

    Responsibilities:
    - Create agent_runs record
    - Execute tools with proper callbacks
    - Create actions from agent output
    - Propagate trace_id through entire flow
    """

    def __init__(self, tool_registry: ToolRegistry = None):
        self.tool_registry = tool_registry or get_tool_registry()

    @traced("agent.invoke")
    async def invoke(
        self,
        request: AgentRunRequest,
        trace_id: Optional[str] = None
    ) -> AgentRunResponse:
        """
        Invoke the agent for a deal.

        Args:
            request: Agent run request with deal_id and task
            trace_id: Optional trace ID (generated if not provided)

        Returns:
            AgentRunResponse with run details and created actions
        """
        # Generate trace_id if not provided
        trace_id = trace_id or request.trace_id or f"trace-{uuid4().hex[:12]}"
        correlation_id = str(request.deal_id)

        # Add correlation context to the current span
        add_correlation_id_to_span(correlation_id)

        # Create agent run record
        run = AgentRun(
            deal_id=request.deal_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
            task=request.task,
            context=request.context,
            status=AgentRunStatus.RUNNING
        )

        # Save run to database
        await self._save_run(run)

        # Create callback handler
        callback = AgentCallbackHandler(
            run_id=run.id,
            deal_id=run.deal_id,
            trace_id=trace_id,
            correlation_id=correlation_id
        )

        # Emit run started event
        await callback.on_run_start(request.task, request.context)

        actions_created: List[UUID] = []
        start_time = time.time()

        try:
            # Execute agent logic (mock for now, replace with LangSmith)
            result = await self._execute_agent(
                run=run,
                callback=callback,
                actions_created=actions_created
            )

            # Update run status
            run.status = AgentRunStatus.COMPLETED
            run.result = result
            run.completed_at = datetime.utcnow()
            run.duration_ms = int((time.time() - start_time) * 1000)
            run.tool_calls_count = len(callback.tool_calls)
            run.actions_created_count = len(actions_created)

            # Emit run completed event
            await callback.on_run_end(
                status=AgentRunStatus.COMPLETED,
                result=result,
                actions_created=actions_created
            )

        except Exception as e:
            logger.error(f"Agent run failed: {e}", exc_info=True)

            run.status = AgentRunStatus.FAILED
            run.error = str(e)
            run.completed_at = datetime.utcnow()
            run.duration_ms = int((time.time() - start_time) * 1000)

            await callback.on_run_end(
                status=AgentRunStatus.FAILED,
                error=str(e)
            )

        # Update run in database
        await self._update_run(run)

        # Record metrics
        record_counter("agent_invocations_total", 1, {
            "status": run.status.value
        })
        record_histogram("agent_run_duration_seconds", run.duration_ms / 1000, {
            "status": run.status.value
        })

        return run.to_response(
            tool_calls=callback.tool_calls,
            actions=actions_created
        )

    async def _execute_agent(
        self,
        run: AgentRun,
        callback: AgentCallbackHandler,
        actions_created: List[UUID]
    ) -> Dict[str, Any]:
        """
        Execute agent logic.

        This is a mock implementation. In production, this would:
        1. Call LangSmith Agent Builder
        2. Process streaming responses
        3. Execute tool calls
        4. Create actions from outputs

        For now, we simulate a simple agent workflow.
        """
        context = run.context or {}

        with create_span("agent.execute", {"deal_id": str(run.deal_id)}) as span:
            # Step 1: Fetch deal info
            with create_span("tool.fetch_deal_info") as tool_span:
                tool_call = await callback.on_tool_start("fetch_deal_info", {"deal_id": str(run.deal_id)})
                start = time.time()

                try:
                    result = await self.tool_registry.execute("fetch_deal_info", {"deal_id": str(run.deal_id)})
                    await callback.on_tool_end(tool_call, result, duration_ms=int((time.time() - start) * 1000))
                    tool_span.set_attribute("tool.success", True)
                except Exception as e:
                    await callback.on_tool_end(tool_call, None, error=str(e), duration_ms=int((time.time() - start) * 1000))
                    tool_span.set_attribute("tool.success", False)
                    tool_span.set_attribute("tool.error", str(e))

            # Step 2: List documents
            with create_span("tool.list_documents") as tool_span:
                tool_call = await callback.on_tool_start("list_documents", {"deal_id": str(run.deal_id)})
                start = time.time()

                try:
                    docs = await self.tool_registry.execute("list_documents", {"deal_id": str(run.deal_id)})
                    await callback.on_tool_end(tool_call, docs, duration_ms=int((time.time() - start) * 1000))
                    tool_span.set_attribute("tool.success", True)
                except Exception as e:
                    await callback.on_tool_end(tool_call, None, error=str(e), duration_ms=int((time.time() - start) * 1000))
                    tool_span.set_attribute("tool.success", False)
                    tool_span.set_attribute("tool.error", str(e))

            # Step 3: Create a task action based on analysis
            with create_span("agent.create_action") as action_span:
                task_action = await self._create_action(
                    deal_id=run.deal_id,
                    trace_id=run.trace_id,
                    action_type="create_task",
                    action_data={
                        "title": f"Review: {run.task}",
                        "description": f"Agent-generated task from run {run.id}",
                        "priority": "medium"
                    },
                    callback=callback
                )

                if task_action:
                    actions_created.append(task_action)
                    action_span.set_attribute("action.created", True)
                    record_counter("actions_created_total", 1, {"type": "create_task"})
                else:
                    action_span.set_attribute("action.created", False)

            span.set_attribute("actions_created", len(actions_created))

        return {
            "summary": f"Analyzed deal and created {len(actions_created)} action(s)",
            "deal_id": str(run.deal_id),
            "task": run.task,
            "actions_created": len(actions_created)
        }

    async def _create_action(
        self,
        deal_id: UUID,
        trace_id: str,
        action_type: str,
        action_data: Dict[str, Any],
        callback: AgentCallbackHandler
    ) -> Optional[UUID]:
        """Create an action from agent output."""
        db = await get_database()

        # Assess risk
        risk = assess_risk(action_type, action_data)

        action_id = uuid4()
        status = "PENDING_APPROVAL" if risk.requires_approval else "QUEUED"

        try:
            # Generate action_id in zakops format
            generated_id = await db.fetchval("SELECT zakops.next_action_id()")

            await db.execute(
                """
                INSERT INTO zakops.actions (
                    action_id, deal_id, capability_id, action_type, title, summary,
                    risk_level, requires_human_review, status, inputs, outputs,
                    created_at, updated_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                generated_id,
                str(deal_id),
                "agent",  # capability_id
                action_type,
                action_data.get("title", "Agent Task"),
                action_data.get("description", ""),
                risk.risk_level.value,
                risk.requires_approval,
                status,
                json.dumps(action_data),
                json.dumps({"trace_id": trace_id, "source": "agent"}),
                datetime.utcnow(),
                datetime.utcnow()
            )

            await callback.on_action_created(action_id, action_type)
            logger.info(f"Created action {generated_id} (type: {action_type}, status: {status})")
            return action_id

        except Exception as e:
            logger.error(f"Failed to create action: {e}")
            return None

    async def _save_run(self, run: AgentRun):
        """Save agent run to database."""
        db = await get_database()

        try:
            await db.execute(
                """
                INSERT INTO zakops.agent_runs (
                    id, deal_id, correlation_id, trace_id, status, task,
                    context, started_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                str(run.id),
                str(run.deal_id),
                run.correlation_id,
                run.trace_id,
                run.status.value,
                run.task,
                json.dumps(run.context) if run.context else None,
                run.started_at
            )
        except Exception as e:
            # Table might not exist yet - log and continue
            logger.warning(f"Could not save agent run to database: {e}")

    async def _update_run(self, run: AgentRun):
        """Update agent run in database."""
        db = await get_database()

        try:
            await db.execute(
                """
                UPDATE zakops.agent_runs
                SET status = $2, result = $3, error = $4, completed_at = $5,
                    duration_ms = $6, tool_calls_count = $7, actions_created_count = $8
                WHERE id = $1
                """,
                str(run.id),
                run.status.value,
                json.dumps(run.result) if run.result else None,
                run.error,
                run.completed_at,
                run.duration_ms,
                run.tool_calls_count,
                run.actions_created_count
            )
        except Exception as e:
            # Table might not exist yet - log and continue
            logger.warning(f"Could not update agent run in database: {e}")


# Convenience function
async def invoke_agent(
    request: AgentRunRequest,
    trace_id: Optional[str] = None
) -> AgentRunResponse:
    """Convenience function to invoke agent."""
    invoker = AgentInvoker()
    return await invoker.invoke(request, trace_id)
