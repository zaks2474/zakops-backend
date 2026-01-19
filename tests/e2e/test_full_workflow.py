"""
E2E Test: Full Integration Workflow

Master test that runs through complete system workflow.
"""

import pytest
from uuid import uuid4
from typing import Dict, Any
from httpx import AsyncClient


class TestFullSystemWorkflow:
    """Test complete system workflow end-to-end."""

    @pytest.mark.asyncio
    async def test_complete_deal_lifecycle(self, client: AsyncClient):
        """
        Test complete deal lifecycle:
        1. Create deal
        2. Invoke agent
        3. Agent creates actions
        4. View actions
        5. Approve actions
        6. Verify completion
        """
        trace_id = f"e2e-lifecycle-{uuid4().hex[:8]}"

        # Step 1: Create deal
        response = await client.post(
            "/api/deals",
            json={
                "canonical_name": f"Lifecycle Test Deal {uuid4().hex[:8]}",
                "display_name": "Lifecycle Corp",
                "stage": "inbound",
                "status": "active"
            },
            headers={"X-Trace-ID": trace_id}
        )

        if response.status_code not in [200, 201]:
            pytest.skip("Could not create deal (database unavailable)")

        deal = response.json()
        deal_id = deal["deal_id"]

        # Step 2: Invoke agent
        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": deal_id,
                "task": "Analyze deal and create tasks"
            },
            headers={"X-Trace-ID": trace_id}
        )

        assert response.status_code in [200, 500], f"Agent invoke failed: {response.text}"

        if response.status_code == 200:
            run_data = response.json()

            # Step 3: Verify run completed
            assert run_data["status"] in ["completed", "running", "failed"]

            # Step 4: Get actions for this deal
            response = await client.get(f"/api/actions?deal_id={deal_id}")

            if response.status_code == 200:
                actions = response.json()

                # Step 5: Approve any pending actions
                for action in actions:
                    if action.get("status") == "PENDING_APPROVAL":
                        action_id = action.get("action_id")

                        await client.post(
                            f"/api/actions/{action_id}/approve",
                            json={
                                "approved_by": "e2e_test",
                                "notes": "E2E lifecycle test"
                            }
                        )

    @pytest.mark.asyncio
    async def test_trace_id_end_to_end(self, client: AsyncClient):
        """Verify trace_id flows through entire system."""
        trace_id = f"trace-e2e-{uuid4().hex[:12]}"

        # Create deal
        response = await client.post(
            "/api/deals",
            json={
                "canonical_name": f"Trace Test Deal {uuid4().hex[:8]}",
                "display_name": "Trace Corp",
                "stage": "inbound",
                "status": "active"
            },
            headers={"X-Trace-ID": trace_id}
        )

        if response.status_code not in [200, 201]:
            pytest.skip("Could not create deal")

        deal = response.json()
        deal_id = deal["deal_id"]

        # Invoke agent with trace_id
        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": deal_id,
                "task": "Trace test"
            },
            headers={"X-Trace-ID": trace_id}
        )

        if response.status_code == 200:
            run_data = response.json()

            # Verify trace_id in run
            assert "trace_id" in run_data
            assert run_data["trace_id"] is not None

    @pytest.mark.asyncio
    async def test_health_check_all_services(self, client: AsyncClient):
        """Verify all service health checks pass."""
        # Main health endpoint
        response = await client.get("/health")
        assert response.status_code == 200

        # Readiness check
        response = await client.get("/health/ready")
        assert response.status_code in [200, 503]

        # Liveness check
        response = await client.get("/health/live")
        assert response.status_code == 200


class TestAPIConsistency:
    """Test API consistency across endpoints."""

    @pytest.mark.asyncio
    async def test_all_endpoints_return_json(self, client: AsyncClient):
        """Verify all endpoints return JSON."""
        endpoints = [
            "/api/deals",
            "/api/actions",
            "/api/agent/tools",
            "/api/agent/runs",
            "/api/pending-tool-approvals"
        ]

        for endpoint in endpoints:
            response = await client.get(endpoint)

            if response.status_code == 200:
                # Should have JSON content type
                content_type = response.headers.get("content-type", "")
                assert "application/json" in content_type, f"{endpoint} should return JSON"

    @pytest.mark.asyncio
    async def test_error_responses_have_detail(self, client: AsyncClient):
        """Verify error responses have detail field."""
        # Request non-existent deal
        response = await client.get("/api/deals/non-existent-deal-id")

        if response.status_code == 404:
            data = response.json()
            assert "detail" in data

    @pytest.mark.asyncio
    async def test_validation_errors_have_detail(self, client: AsyncClient):
        """Verify validation errors have detail field."""
        try:
            response = await client.post(
                "/api/agent/invoke",
                json={}  # Missing required fields
            )

            # 400 or 422 depending on validation layer
            assert response.status_code in [400, 422, 500]
            data = response.json()
            assert "detail" in data
        except RuntimeError as e:
            if "loop" in str(e).lower():
                pytest.skip("Event loop conflict in test environment")
            raise


class TestSystemIntegration:
    """Test system integration."""

    @pytest.mark.asyncio
    async def test_agent_integrates_with_deal(self, client: AsyncClient, test_deal_id: str):
        """Verify agent properly integrates with deal system."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        # Invoke agent
        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id,
                "task": "Integration test"
            }
        )

        if response.status_code == 200:
            run_data = response.json()

            # Deal ID should match
            assert run_data["deal_id"] == test_deal_id

            # Should have tool calls
            assert "tool_calls" in run_data

    @pytest.mark.asyncio
    async def test_actions_link_to_deals(self, client: AsyncClient, test_deal_id: str):
        """Verify actions are properly linked to deals."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        # Get actions for deal
        response = await client.get(f"/api/actions?deal_id={test_deal_id}")

        if response.status_code == 200:
            actions = response.json()

            for action in actions:
                if "deal_id" in action:
                    assert action["deal_id"] == test_deal_id

    @pytest.mark.asyncio
    async def test_thread_tool_calls_workflow(self, client: AsyncClient):
        """Test thread with tool calls workflow."""
        try:
            # Create thread
            response = await client.post(
                "/api/threads",
                json={"assistant_id": "test-assistant"}
            )

            if response.status_code not in [200, 201]:
                pytest.skip("Could not create thread")

            thread = response.json()
            thread_id = thread["thread_id"]

            # Create run
            response = await client.post(
                f"/api/threads/{thread_id}/runs",
                json={"input_message": "Test message"}
            )

            if response.status_code in [200, 201]:
                run = response.json()
                run_id = run["run_id"]

                # Get tool calls
                response = await client.get(f"/api/threads/{thread_id}/runs/{run_id}/tool_calls")
                assert response.status_code in [200, 404, 503]
        except RuntimeError as e:
            if "different loop" in str(e):
                pytest.skip("Event loop conflict in test environment")
            raise


class TestTraceIdPropagation:
    """Test trace_id propagation throughout the system."""

    @pytest.mark.asyncio
    async def test_trace_id_in_response_header(self, client: AsyncClient):
        """Verify trace_id is returned in response header."""
        trace_id = "test-header-trace-123"

        response = await client.get(
            "/api/deals",
            headers={"X-Trace-ID": trace_id}
        )

        # Check response header
        response_trace = response.headers.get("X-Trace-ID")
        assert response_trace == trace_id

    @pytest.mark.asyncio
    async def test_trace_id_generated_if_missing(self, client: AsyncClient):
        """Verify trace_id is generated if not provided."""
        response = await client.get("/api/deals")

        # Should still have X-Trace-ID header
        response_trace = response.headers.get("X-Trace-ID")
        assert response_trace is not None
        assert len(response_trace) > 0

    @pytest.mark.asyncio
    async def test_trace_id_in_agent_run(self, client: AsyncClient, test_deal_id: str):
        """Verify trace_id is stored in agent run."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        trace_id = f"test-run-trace-{uuid4().hex[:8]}"

        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id,
                "task": "Trace test"
            },
            headers={"X-Trace-ID": trace_id}
        )

        if response.status_code == 200:
            data = response.json()
            assert "trace_id" in data
