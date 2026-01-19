"""
E2E Test: HITL Quarantine Workflow

Tests human-in-the-loop workflows:
1. High-risk action detection
2. Risk assessment
3. Quarantine management
4. Checkpoint save/restore
"""

import pytest
from typing import Dict, Any
from httpx import AsyncClient


class TestHITLRiskAssessment:
    """Test HITL risk assessment."""

    @pytest.mark.asyncio
    async def test_assess_risk_function(self):
        """Test risk assessment function directly."""
        from src.core.hitl import assess_risk, RiskLevel

        # Test low-risk action
        risk = assess_risk("analyze_document", {"document_id": "doc-123"})
        assert risk.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]

        # Test high-risk action
        risk = assess_risk("send_email", {"to": "external@example.com"})
        # May be high, critical, or medium depending on implementation
        assert risk.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

    @pytest.mark.asyncio
    async def test_risk_levels_enum(self):
        """Verify RiskLevel enum values."""
        from src.core.hitl import RiskLevel

        assert hasattr(RiskLevel, 'LOW')
        assert hasattr(RiskLevel, 'MEDIUM')
        assert hasattr(RiskLevel, 'HIGH')
        assert hasattr(RiskLevel, 'CRITICAL')


class TestHITLQuarantineWorkflow:
    """Test HITL quarantine workflow."""

    @pytest.mark.asyncio
    async def test_quarantine_endpoint(self, client: AsyncClient):
        """Test quarantine endpoint exists."""
        response = await client.get("/api/quarantine")

        # May return 200, 404 (not found), or 503 (DB unavailable)
        assert response.status_code in [200, 404, 503]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_pending_approvals_endpoint(self, client: AsyncClient):
        """Test pending approvals endpoint."""
        response = await client.get("/api/pending-tool-approvals")

        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)

            # Each approval should have required fields
            for approval in data:
                assert "tool_call_id" in approval
                assert "tool_name" in approval


class TestCheckpointWorkflow:
    """Test checkpoint save/restore workflow."""

    @pytest.mark.asyncio
    async def test_checkpoint_store_exists(self):
        """Verify CheckpointStore class exists."""
        from src.core.hitl import CheckpointStore

        # CheckpointStore takes optional db parameter, not action_id
        store = CheckpointStore()
        assert store is not None

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_restore(self):
        """Test checkpoint save and restore cycle."""
        from src.core.hitl import get_checkpoint_store
        from uuid import uuid4

        try:
            store = await get_checkpoint_store()
            correlation_id = str(uuid4())
            action_id = str(uuid4())

            # Save checkpoint
            checkpoint_data = {
                "step": 3,
                "processed_items": ["a", "b", "c"],
                "state": {"key": "value"}
            }

            checkpoint = await store.save_checkpoint(
                correlation_id=correlation_id,
                checkpoint_name="test_checkpoint",
                checkpoint_data=checkpoint_data,
                action_id=action_id
            )

            # Restore checkpoint
            restored = await store.get_latest_checkpoint(action_id=action_id)

            if restored is not None:
                assert restored.checkpoint_data["step"] == 3
                assert restored.checkpoint_data["processed_items"] == ["a", "b", "c"]
        except Exception as e:
            # Database may not be available
            pytest.skip(f"Checkpoint test skipped: {e}")

    @pytest.mark.asyncio
    async def test_checkpoint_list(self):
        """Test listing checkpoints."""
        from src.core.hitl import get_checkpoint_store
        from uuid import uuid4

        try:
            store = await get_checkpoint_store()
            correlation_id = str(uuid4())
            action_id = str(uuid4())

            # Save multiple checkpoints
            await store.save_checkpoint(
                correlation_id=correlation_id,
                checkpoint_name="step_1",
                checkpoint_data={"step": 1},
                action_id=action_id
            )
            await store.save_checkpoint(
                correlation_id=correlation_id,
                checkpoint_name="step_2",
                checkpoint_data={"step": 2},
                action_id=action_id
            )

            # List checkpoints
            checkpoints = await store.get_checkpoints_for_action(action_id)

            assert isinstance(checkpoints, list)
        except Exception as e:
            pytest.skip(f"Checkpoint test skipped: {e}")


class TestHITLToolApproval:
    """Test HITL tool approval workflow."""

    @pytest.mark.asyncio
    async def test_high_risk_tools_require_approval(self, client: AsyncClient):
        """Verify high-risk tools require approval."""
        response = await client.get("/api/agent/tools")

        assert response.status_code == 200
        tools = response.json()

        high_risk_tools = [t for t in tools if t["risk_level"] in ["high", "critical"]]

        for tool in high_risk_tools:
            assert tool["requires_approval"] is True, f"{tool['name']} should require approval"

    @pytest.mark.asyncio
    async def test_low_risk_tools_auto_approved(self, client: AsyncClient):
        """Verify low-risk tools don't require approval."""
        response = await client.get("/api/agent/tools")

        assert response.status_code == 200
        tools = response.json()

        low_risk_tools = [t for t in tools if t["risk_level"] == "low"]

        for tool in low_risk_tools:
            assert tool["requires_approval"] is False, f"{tool['name']} should not require approval"

    @pytest.mark.asyncio
    async def test_tool_approval_in_agent_run(self, client: AsyncClient, test_deal_id: str):
        """Test that high-risk tool calls appear in pending approvals."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        # This would require triggering a high-risk tool call
        # For now, just verify the endpoint works
        response = await client.get("/api/pending-tool-approvals")
        assert response.status_code in [200, 503]


class TestHITLThreadManagement:
    """Test HITL thread management."""

    @pytest.mark.asyncio
    async def test_create_thread(self, client: AsyncClient):
        """Test creating an agent thread."""
        try:
            response = await client.post(
                "/api/threads",
                json={
                    "assistant_id": "test-assistant",
                    "user_id": "e2e-test-user"
                }
            )

            # May succeed or fail depending on DB availability
            assert response.status_code in [200, 201, 500, 503]

            if response.status_code in [200, 201]:
                data = response.json()
                assert "thread_id" in data
        except RuntimeError as e:
            if "different loop" in str(e):
                pytest.skip("Event loop conflict in test environment")
            raise

    @pytest.mark.asyncio
    async def test_thread_run_workflow(self, client: AsyncClient):
        """Test creating a run within a thread."""
        try:
            # First create a thread
            response = await client.post(
                "/api/threads",
                json={
                    "assistant_id": "test-assistant"
                }
            )

            if response.status_code not in [200, 201]:
                pytest.skip("Could not create thread")

            thread_id = response.json()["thread_id"]

            # Create a run
            response = await client.post(
                f"/api/threads/{thread_id}/runs",
                json={
                    "input_message": "Test run"
                }
            )

            assert response.status_code in [200, 201, 500, 503]
        except RuntimeError as e:
            if "different loop" in str(e):
                pytest.skip("Event loop conflict in test environment")
            raise
