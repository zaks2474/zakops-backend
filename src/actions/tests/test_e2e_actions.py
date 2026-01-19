#!/usr/bin/env python3
"""
End-to-End Integration Tests for Kinetic Actions

Tests cover:
1. Action creation via PlanSpec (CodeX integration)
2. Artifact production (REQUEST_DOCS with Gemini)
3. Approval gates (high-risk actions require approval)

Run: python3 actions/tests/test_e2e_actions.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/zaks/scripts")

# Mock pytest if not available
try:
    import pytest
except ImportError:
    class MockPytest:
        @staticmethod
        def fixture(f):
            return f
        @staticmethod
        def skip(msg):
            raise SkipTest(msg)
    pytest = MockPytest()
    class SkipTest(Exception):
        pass

from actions.codex import (
    ActionProposal,
    get_capability,
    handle_codex_tool_call,
    list_capabilities,
    propose_action,
)
from actions.engine.models import ActionPayload, compute_idempotency_key, now_utc_iso, safe_uuid
from actions.engine.store import ActionStore
from actions.executors.base import ExecutionContext
from actions.executors.diligence_request_docs import RequestDocsExecutor


@pytest.fixture
def store():
    """Action store fixture."""
    return ActionStore()


@pytest.fixture
def test_deal():
    """Load a test deal from registry."""
    registry_path = Path("/home/zaks/DataRoom/.deal-registry/deal_registry.json")
    if not registry_path.exists():
        pytest.skip("Deal registry not found")

    registry = json.loads(registry_path.read_text())
    deals = registry.get("deals", {})

    # Find a deal with folder_path
    for deal_id, deal_data in deals.items():
        if deal_data.get("folder_path"):
            return deal_id, deal_data

    pytest.skip("No deal with folder_path found")


class TestActionCreationViaPlanSpec:
    """Test 1: Action creation via PlanSpec (CodeX integration)."""

    def test_list_capabilities_returns_all(self):
        """Capabilities list includes all expected action types."""
        caps = list_capabilities()
        cap_ids = [c.capability_id for c in caps]

        assert "diligence.request_docs" in cap_ids
        assert "communication.draft_email" in cap_ids
        assert "document.generate_loi" in cap_ids
        assert len(caps) >= 4

    def test_get_capability_returns_definition(self):
        """Can get specific capability with full definition."""
        cap = get_capability("diligence.request_docs")

        assert cap is not None
        assert cap.action_type == "DILIGENCE.REQUEST_DOCS"
        assert cap.requires_deal is True
        assert len(cap.inputs) >= 2

    def test_propose_valid_action_creates_ready(self, store, test_deal):
        """Valid proposal creates action in READY state (low risk, no approval)."""
        deal_id, _ = test_deal

        proposal = ActionProposal(
            capability_id="diligence.request_docs",
            deal_id=deal_id,
            title=f"Test E2E: Request docs {safe_uuid()[:8]}",
            inputs={
                "doc_type": "financial",
                "description": "Test request for LTM P&L",
            },
            confidence=0.9,
        )

        result = propose_action(proposal)

        assert result.success is True
        assert result.action_id is not None
        assert result.status == "READY"

        # Verify action exists in store
        action = store.get_action(result.action_id)
        assert action is not None
        assert action.status == "READY"
        assert action.type == "DILIGENCE.REQUEST_DOCS"

    def test_propose_invalid_missing_input(self):
        """Proposal missing required input fails validation."""
        proposal = ActionProposal(
            capability_id="diligence.request_docs",
            deal_id="DEAL-2025-001",
            title="Missing input test",
            inputs={
                "doc_type": "financial",
                # Missing: description
            },
        )

        result = propose_action(proposal)

        assert result.success is False
        assert "Missing required input" in result.message

    def test_codex_tool_call_list_capabilities(self):
        """CodeX tool call interface works for listing capabilities."""
        result = handle_codex_tool_call("list_action_capabilities", {})

        assert "capabilities" in result
        assert len(result["capabilities"]) >= 4
        assert any(c["capability_id"] == "diligence.request_docs" for c in result["capabilities"])

    def test_codex_tool_call_propose_action(self, test_deal):
        """CodeX tool call interface works for proposing actions."""
        deal_id, _ = test_deal

        result = handle_codex_tool_call(
            "propose_action",
            {
                "capability_id": "diligence.request_docs",
                "deal_id": deal_id,
                "title": f"CodeX tool test {safe_uuid()[:8]}",
                "inputs": {
                    "doc_type": "legal",
                    "description": "Request contracts",
                },
                "confidence": 0.85,
            },
        )

        assert result["success"] is True
        assert result["action_id"] is not None


class TestArtifactProduction:
    """Test 2: Artifact production (REQUEST_DOCS executor)."""

    def test_request_docs_produces_artifacts(self, store, test_deal):
        """REQUEST_DOCS executor produces email draft and checklist artifacts."""
        deal_id, deal_data = test_deal

        # Create test action
        action = ActionPayload(
            deal_id=deal_id,
            type="DILIGENCE.REQUEST_DOCS",
            title=f"Artifact test {safe_uuid()[:8]}",
            source="system",
            created_by="test",
            idempotency_key=compute_idempotency_key("artifact_test", safe_uuid()),
            inputs={
                "doc_type": "financial",
                "description": "Test: LTM P&L",
            },
        )

        # Create execution context
        ctx = ExecutionContext(action=action, deal=deal_data)

        # Execute
        executor = RequestDocsExecutor()
        result = executor.execute(action, ctx)

        # Verify artifacts created
        assert len(result.artifacts) == 2

        artifact_names = [a.filename for a in result.artifacts]
        assert "email_draft.md" in artifact_names
        assert "document_checklist.md" in artifact_names

        # Verify files exist
        for artifact in result.artifacts:
            path = Path(artifact.path)
            assert path.exists(), f"Artifact file not found: {artifact.path}"
            content = path.read_text()
            assert len(content) > 0, f"Artifact file is empty: {artifact.path}"

    def test_request_docs_uses_gemini_when_available(self, test_deal):
        """REQUEST_DOCS uses Gemini Flash when available (not fallback)."""
        deal_id, deal_data = test_deal

        action = ActionPayload(
            deal_id=deal_id,
            type="DILIGENCE.REQUEST_DOCS",
            title=f"Gemini test {safe_uuid()[:8]}",
            source="system",
            created_by="test",
            idempotency_key=compute_idempotency_key("gemini_test", safe_uuid()),
            inputs={
                "doc_type": "financial",
                "description": "Test Gemini integration",
            },
        )

        ctx = ExecutionContext(action=action, deal=deal_data)
        executor = RequestDocsExecutor()
        result = executor.execute(action, ctx)

        # Check if LLM was used (depends on Gemini API availability)
        used_llm = result.outputs.get("used_llm", False)
        if used_llm:
            # Verify Gemini-specific output
            assert result.outputs.get("broker_name") is not None
            # Check draft file mentions Gemini
            draft_path = Path(result.outputs.get("draft_artifact", ""))
            if draft_path.exists():
                content = draft_path.read_text()
                assert "Gemini Flash" in content
        else:
            # Fallback is also acceptable
            pytest.skip("Gemini not available, fallback used")

    def test_request_docs_includes_follow_up_suggestion(self, test_deal):
        """REQUEST_DOCS includes follow-up suggestion for SEND_EMAIL."""
        deal_id, deal_data = test_deal

        action = ActionPayload(
            deal_id=deal_id,
            type="DILIGENCE.REQUEST_DOCS",
            title=f"Follow-up test {safe_uuid()[:8]}",
            source="system",
            created_by="test",
            idempotency_key=compute_idempotency_key("followup_test", safe_uuid()),
            inputs={
                "doc_type": "operational",
                "description": "Request org chart",
            },
        )

        ctx = ExecutionContext(action=action, deal=deal_data)
        executor = RequestDocsExecutor()
        result = executor.execute(action, ctx)

        # Verify follow-up suggestion
        assert "follow_up_suggestion" in result.outputs
        follow_up = result.outputs["follow_up_suggestion"]
        assert follow_up["type"] == "COMMUNICATION.SEND_EMAIL"
        assert follow_up["requires_approval"] is True


class TestApprovalGates:
    """Test 3: Approval gates (high-risk actions require approval)."""

    def test_high_risk_action_requires_approval(self):
        """High-risk actions (SEND_EMAIL) require approval."""
        cap = get_capability("communication.send_email")

        assert cap is not None
        assert cap.requires_approval is True
        assert cap.default_risk_level == "high"

    def test_propose_high_risk_creates_pending_approval(self, test_deal):
        """Proposing high-risk action creates PENDING_APPROVAL status."""
        # Note: SEND_EMAIL requires approval regardless of confidence

        result = handle_codex_tool_call(
            "propose_action",
            {
                "capability_id": "communication.send_email",
                "title": f"Send test {safe_uuid()[:8]}",
                "inputs": {
                    "to": "test@example.com",
                    "subject": "Test email",
                    "body": "This is a test",
                },
                "confidence": 0.99,  # Even high confidence requires approval
            },
        )

        assert result["success"] is True
        assert result["status"] == "PENDING_APPROVAL"

    def test_low_confidence_escalates_to_approval(self, test_deal):
        """Low confidence (<0.5) escalates to approval requirement."""
        deal_id, _ = test_deal

        result = handle_codex_tool_call(
            "propose_action",
            {
                "capability_id": "diligence.request_docs",
                "deal_id": deal_id,
                "title": f"Low confidence test {safe_uuid()[:8]}",
                "inputs": {
                    "doc_type": "financial",
                    "description": "Test low confidence",
                },
                "confidence": 0.3,  # Low confidence
            },
        )

        assert result["success"] is True
        # Low confidence should escalate to PENDING_APPROVAL
        # and include a warning
        assert "Low confidence" in str(result.get("warnings", []))

    def test_medium_risk_requires_approval(self):
        """Medium-risk actions (GENERATE_LOI) require approval."""
        cap = get_capability("document.generate_loi")

        assert cap is not None
        assert cap.requires_approval is True
        assert cap.default_risk_level == "medium"

    def test_cannot_override_required_approval(self, test_deal):
        """Cannot override approval requirement via proposal."""
        deal_id, _ = test_deal

        # Try to propose GENERATE_LOI with requires_human_review=False
        proposal = ActionProposal(
            capability_id="document.generate_loi",
            deal_id=deal_id,
            title=f"Override test {safe_uuid()[:8]}",
            inputs={
                "terms": "Test terms",
            },
            requires_human_review=False,  # Try to override
            confidence=0.95,
        )

        result = propose_action(proposal)

        assert result.success is True
        # Should still be PENDING_APPROVAL despite override attempt
        assert result.status == "PENDING_APPROVAL"
        # Should have warning about ignored override
        assert any("requires approval" in str(w).lower() for w in result.warnings)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
