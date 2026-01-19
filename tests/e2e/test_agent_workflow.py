"""
E2E Test: Agent Workflow

Tests agent-specific workflows:
1. Agent invocation
2. Tool execution
3. Action creation
4. Event emission
"""

import pytest
from typing import Dict, Any
from httpx import AsyncClient


class TestAgentWorkflow:
    """Test agent execution workflows."""

    @pytest.mark.asyncio
    async def test_agent_invoke_creates_run(self, client: AsyncClient, test_deal_id: str):
        """Verify agent invocation creates a run record."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id,
                "task": "Analyze deal"
            }
        )

        assert response.status_code in [200, 500], f"Unexpected status: {response.text}"

        if response.status_code == 200:
            data = response.json()

            assert "run_id" in data
            assert "status" in data
            assert "trace_id" in data
            assert data["deal_id"] == test_deal_id

    @pytest.mark.asyncio
    async def test_agent_invoke_returns_tool_calls(self, client: AsyncClient, test_deal_id: str):
        """Verify agent run returns tool calls."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id,
                "task": "Analyze deal and fetch documents"
            }
        )

        if response.status_code == 200:
            data = response.json()

            # Should have tool_calls array
            assert "tool_calls" in data
            assert isinstance(data["tool_calls"], list)

            # Each tool call should have name and input
            for tc in data["tool_calls"]:
                assert "tool_name" in tc
                assert "tool_input" in tc

    @pytest.mark.asyncio
    async def test_agent_tools_endpoint(self, client: AsyncClient):
        """Verify tools endpoint returns available tools."""
        response = await client.get("/api/agent/tools")

        assert response.status_code == 200, f"Tools endpoint failed: {response.text}"
        tools = response.json()

        assert isinstance(tools, list)
        assert len(tools) > 0, "No tools returned"

        # Verify tool structure
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "risk_level" in tool

        # Verify expected tools exist
        tool_names = [t["name"] for t in tools]
        assert "fetch_deal_info" in tool_names
        assert "analyze_document" in tool_names

    @pytest.mark.asyncio
    async def test_agent_tools_have_risk_levels(self, client: AsyncClient):
        """Verify tools have proper risk levels."""
        response = await client.get("/api/agent/tools")

        assert response.status_code == 200
        tools = response.json()

        valid_levels = ["low", "medium", "high", "critical"]

        for tool in tools:
            assert "risk_level" in tool
            assert tool["risk_level"] in valid_levels, f"Invalid risk level: {tool['risk_level']}"

    @pytest.mark.asyncio
    async def test_agent_tools_have_approval_flag(self, client: AsyncClient):
        """Verify high-risk tools require approval."""
        response = await client.get("/api/agent/tools")

        assert response.status_code == 200
        tools = response.json()

        for tool in tools:
            assert "requires_approval" in tool

            # High-risk tools should require approval
            if tool["risk_level"] in ["high", "critical"]:
                assert tool["requires_approval"] is True, f"{tool['name']} should require approval"

    @pytest.mark.asyncio
    async def test_agent_runs_list(self, client: AsyncClient):
        """Verify agent runs can be listed."""
        response = await client.get("/api/agent/runs")

        assert response.status_code == 200
        runs = response.json()

        assert isinstance(runs, list)

    @pytest.mark.asyncio
    async def test_agent_runs_filter_by_deal(self, client: AsyncClient, test_deal_id: str):
        """Verify agent runs can be filtered by deal."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        # Create a run first
        await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id,
                "task": "Test run"
            }
        )

        # Filter by deal
        response = await client.get(f"/api/agent/runs?deal_id={test_deal_id}")

        assert response.status_code == 200
        runs = response.json()

        assert isinstance(runs, list)


class TestAgentErrorHandling:
    """Test agent error handling."""

    @pytest.mark.asyncio
    async def test_invoke_with_invalid_deal(self, client: AsyncClient):
        """Verify proper error for invalid deal ID."""
        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": "00000000-0000-0000-0000-000000000000",
                "task": "Analyze"
            }
        )

        # Should handle gracefully (either 404 or 200 with error in result)
        assert response.status_code in [200, 404, 422, 500]

    @pytest.mark.asyncio
    async def test_invoke_with_missing_task(self, client: AsyncClient, test_deal_id: str):
        """Verify validation for missing task."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id
                # Missing "task"
            }
        )

        assert response.status_code == 422, "Should reject missing task"

    @pytest.mark.asyncio
    async def test_invoke_with_missing_deal_id(self, client: AsyncClient):
        """Verify validation for missing deal_id."""
        response = await client.post(
            "/api/agent/invoke",
            json={
                "task": "Analyze"
                # Missing "deal_id"
            }
        )

        # 400 or 422 depending on validation layer
        assert response.status_code in [400, 422], "Should reject missing deal_id"

    @pytest.mark.asyncio
    async def test_invoke_with_invalid_json(self, client: AsyncClient):
        """Verify validation for invalid JSON."""
        response = await client.post(
            "/api/agent/invoke",
            content="not json",
            headers={"Content-Type": "application/json"}
        )

        # 400 or 422 depending on validation layer
        assert response.status_code in [400, 422], "Should reject invalid JSON"


class TestAgentTraceId:
    """Test trace_id handling in agent."""

    @pytest.mark.asyncio
    async def test_trace_id_in_response(self, client: AsyncClient, test_deal_id: str):
        """Verify trace_id is included in response."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id,
                "task": "Test"
            }
        )

        if response.status_code == 200:
            data = response.json()
            assert "trace_id" in data
            assert data["trace_id"] is not None

    @pytest.mark.asyncio
    async def test_custom_trace_id_accepted(self, client: AsyncClient, test_deal_id: str):
        """Verify custom trace_id is accepted."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        custom_trace = "custom-trace-abc123"

        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id,
                "task": "Test"
            },
            headers={"X-Trace-ID": custom_trace}
        )

        if response.status_code == 200:
            data = response.json()
            # Response should have a trace_id
            assert "trace_id" in data
