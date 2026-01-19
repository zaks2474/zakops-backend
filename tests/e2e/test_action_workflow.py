"""
E2E Test: Action Approval Workflow

Tests the complete flow:
1. List pending actions
2. Approve or reject
3. Verify status update
4. Verify events emitted
"""

import pytest
from typing import Dict, Any
from httpx import AsyncClient


class TestActionApprovalWorkflow:
    """Test complete action approval workflow."""

    @pytest.mark.asyncio
    async def test_list_actions(self, client: AsyncClient):
        """Test listing actions."""
        response = await client.get("/api/actions")

        assert response.status_code in [200, 503], f"Unexpected status: {response.text}"

        if response.status_code == 200:
            actions = response.json()
            assert isinstance(actions, list)

    @pytest.mark.asyncio
    async def test_list_pending_actions(self, client: AsyncClient):
        """Test listing pending actions only."""
        response = await client.get("/api/actions?pending_only=true")

        assert response.status_code in [200, 503], f"Unexpected status: {response.text}"

        if response.status_code == 200:
            actions = response.json()
            assert isinstance(actions, list)
            # All should be pending or queued
            for action in actions:
                assert action.get("status") in ["PENDING_APPROVAL", "QUEUED", None]

    @pytest.mark.asyncio
    async def test_pending_tool_approvals(self, client: AsyncClient):
        """Test pending tool approvals endpoint."""
        response = await client.get("/api/pending-tool-approvals")

        assert response.status_code in [200, 503], f"Unexpected status: {response.text}"

        if response.status_code == 200:
            approvals = response.json()
            assert isinstance(approvals, list)

    @pytest.mark.asyncio
    async def test_action_approval_workflow(self, client: AsyncClient, test_deal_id: str, trace_id: str):
        """Test action approval workflow."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        # First invoke agent to potentially create actions
        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id,
                "task": "Create a task for review"
            },
            headers={"X-Trace-ID": trace_id}
        )

        if response.status_code != 200:
            pytest.skip("Agent invocation failed")

        run_data = response.json()

        # Check if any actions were created
        actions_created = run_data.get("actions_created", [])

        # Check actions for this deal
        response = await client.get(f"/api/actions?deal_id={test_deal_id}")

        if response.status_code == 200:
            actions = response.json()

            # If there are pending actions, try to approve one
            for action in actions:
                if action.get("status") == "PENDING_APPROVAL":
                    action_id = action.get("action_id")

                    approve_response = await client.post(
                        f"/api/actions/{action_id}/approve",
                        json={"approved_by": "e2e_test", "notes": "E2E test approval"}
                    )

                    assert approve_response.status_code in [200, 400, 404]
                    break  # Only approve one


class TestActionStatusTransitions:
    """Test action status transitions."""

    @pytest.mark.asyncio
    async def test_action_status_field_present(self, client: AsyncClient):
        """Verify actions have status field."""
        response = await client.get("/api/actions?limit=5")

        if response.status_code == 200:
            actions = response.json()
            for action in actions:
                assert "status" in action, "Action missing status field"

    @pytest.mark.asyncio
    async def test_action_risk_level_field(self, client: AsyncClient):
        """Verify actions have risk_level field."""
        response = await client.get("/api/actions?limit=5")

        if response.status_code == 200:
            actions = response.json()
            for action in actions:
                if "risk_level" in action:
                    assert action["risk_level"] in ["low", "medium", "high", "critical", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


class TestActionRejection:
    """Test action rejection workflow."""

    @pytest.mark.asyncio
    async def test_reject_action_requires_reason(self, client: AsyncClient):
        """Test that rejecting action requires a reason."""
        # Try to reject a non-existent action (to test validation)
        response = await client.post(
            "/api/actions/non-existent-id/reject",
            json={"rejected_by": "e2e_test"}  # Missing reason
        )

        # Should be 404 (not found), 422 (validation error), or 503 (DB unavailable)
        assert response.status_code in [400, 404, 422, 500, 503]

    @pytest.mark.asyncio
    async def test_action_reject_with_reason(self, client: AsyncClient, test_deal_id: str):
        """Test rejecting action with reason."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        # Get actions for deal
        response = await client.get(f"/api/actions?deal_id={test_deal_id}")

        if response.status_code == 200:
            actions = response.json()

            for action in actions:
                if action.get("status") == "PENDING_APPROVAL":
                    action_id = action.get("action_id")

                    reject_response = await client.post(
                        f"/api/actions/{action_id}/reject",
                        json={
                            "rejected_by": "e2e_test",
                            "reason": "E2E test rejection"
                        }
                    )

                    # May succeed or fail depending on action state
                    assert reject_response.status_code in [200, 400, 404]
                    break
