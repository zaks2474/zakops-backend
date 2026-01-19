"""
Agent Invocation API

Endpoints for invoking and managing agent runs.
"""

import json
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Request, HTTPException, Query, Depends
from pydantic import BaseModel

from ....core.agent import (
    invoke_agent, AgentRunRequest, AgentRunResponse,
    get_tool_registry
)
from ....core.database.adapter import get_database
from ...shared.middleware import get_trace_id

router = APIRouter(prefix="/api/agent", tags=["agent"])


class InvokeRequest(BaseModel):
    """Request to invoke agent."""
    deal_id: UUID
    task: str
    context: Optional[dict] = None


class ToolSchema(BaseModel):
    """Tool schema for API response."""
    name: str
    description: str
    parameters: dict
    risk_level: str
    requires_approval: bool


@router.post("/invoke", response_model=AgentRunResponse)
async def invoke_agent_endpoint(request: Request, body: InvokeRequest):
    """
    Invoke the agent for a deal.

    The agent will:
    1. Analyze the deal based on the task
    2. Execute relevant tools
    3. Create actions for human review

    Events are emitted in real-time via SSE.
    """
    # Get trace context from middleware
    trace_id = get_trace_id()

    # Create request
    agent_request = AgentRunRequest(
        deal_id=body.deal_id,
        task=body.task,
        context=body.context,
        trace_id=trace_id
    )

    # Invoke agent
    try:
        response = await invoke_agent(agent_request, trace_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {str(e)}")


@router.get("/runs", response_model=List[dict])
async def list_agent_runs(
    deal_id: Optional[UUID] = Query(None, description="Filter by deal ID"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List agent runs with optional filtering."""
    try:
        db = await get_database()

        if deal_id:
            runs = await db.fetch(
                """
                SELECT id, deal_id, trace_id, status, task, started_at, completed_at,
                       duration_ms, tool_calls_count, actions_created_count, error
                FROM zakops.agent_runs
                WHERE deal_id = $1
                ORDER BY started_at DESC
                LIMIT $2 OFFSET $3
                """,
                str(deal_id), limit, offset
            )
        else:
            runs = await db.fetch(
                """
                SELECT id, deal_id, trace_id, status, task, started_at, completed_at,
                       duration_ms, tool_calls_count, actions_created_count, error
                FROM zakops.agent_runs
                ORDER BY started_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit, offset
            )

        return [dict(r) for r in runs]
    except Exception as e:
        # Table might not exist yet
        return []


@router.get("/runs/{run_id}", response_model=dict)
async def get_agent_run(run_id: str):
    """Get details of a specific agent run."""
    try:
        db = await get_database()

        run = await db.fetchrow(
            "SELECT * FROM zakops.agent_runs WHERE id = $1",
            run_id
        )

        if not run:
            raise HTTPException(status_code=404, detail="Agent run not found")

        result = dict(run)
        # Parse JSON fields
        if result.get("context") and isinstance(result["context"], str):
            result["context"] = json.loads(result["context"])
        if result.get("result") and isinstance(result["result"], str):
            result["result"] = json.loads(result["result"])

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail="Agent run not found")


@router.get("/tools", response_model=List[ToolSchema])
async def list_available_tools():
    """List all available tools for agent execution."""
    registry = get_tool_registry()
    tools = registry.list_tools()

    return [
        ToolSchema(
            name=t.name,
            description=t.description,
            parameters=t.parameters,
            risk_level=t.risk_level,
            requires_approval=t.requires_approval
        )
        for t in tools
    ]
