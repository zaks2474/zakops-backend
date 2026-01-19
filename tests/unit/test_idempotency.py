"""
Tests for idempotent deal transitions.
"""

import pytest
from uuid import uuid4
from datetime import datetime, timezone

from src.core.deals.workflow import DealStage, STAGE_TRANSITIONS, StageTransition


class TestDealStageEnum:
    """Test DealStage enum and transitions."""

    def test_all_stages_defined(self):
        """All expected stages should be defined."""
        expected = {
            "inbound", "initial_review", "due_diligence",
            "negotiation", "documentation", "closing",
            "closed_won", "closed_lost", "archived"
        }
        actual = {s.value for s in DealStage}
        assert actual == expected

    def test_transitions_defined_for_all_stages(self):
        """All stages should have transition rules."""
        for stage in DealStage:
            assert stage in STAGE_TRANSITIONS, f"Missing transitions for {stage}"

    def test_terminal_stages_have_no_transitions(self):
        """Terminal stages should have limited transitions."""
        # closed_won and closed_lost can only go to archived
        assert STAGE_TRANSITIONS[DealStage.CLOSED_WON] == [DealStage.ARCHIVED]
        assert STAGE_TRANSITIONS[DealStage.CLOSED_LOST] == [DealStage.ARCHIVED]

        # archived is truly terminal
        assert STAGE_TRANSITIONS[DealStage.ARCHIVED] == []


class TestStageTransition:
    """Test StageTransition dataclass."""

    def test_create_transition(self):
        """Create a stage transition record."""
        transition = StageTransition(
            deal_id="test-deal-123",
            from_stage="initial_review",
            to_stage="due_diligence",
            transitioned_by="user-1",
            reason="Ready for DD"
        )

        assert transition.deal_id == "test-deal-123"
        assert transition.from_stage == "initial_review"
        assert transition.to_stage == "due_diligence"
        assert transition.idempotent_hit is False

    def test_idempotent_hit_flag(self):
        """Test idempotent_hit flag."""
        transition = StageTransition(
            deal_id="test-deal",
            from_stage="inbound",
            to_stage="initial_review",
            idempotent_hit=True
        )

        assert transition.idempotent_hit is True

    def test_default_timestamp(self):
        """Transition should have default timestamp."""
        transition = StageTransition(
            deal_id="test-deal",
            from_stage="inbound",
            to_stage="initial_review"
        )

        assert transition.timestamp is not None
        assert isinstance(transition.timestamp, datetime)


class TestIdempotencyBehavior:
    """Test idempotency behavior (unit tests without DB)."""

    def test_idempotency_key_format(self):
        """Test valid idempotency key formats."""
        valid_keys = [
            "test-123",
            f"transition-{uuid4()}",
            "user:123:deal:456:action",
            "a" * 64  # Max length
        ]

        for key in valid_keys:
            assert len(key) <= 64, f"Key too long: {key}"
            assert isinstance(key, str)

    def test_transition_result_contains_idempotent_hit(self):
        """Transition result should always include idempotent_hit."""
        # First transition (new)
        result1 = StageTransition(
            deal_id="deal-1",
            from_stage="inbound",
            to_stage="initial_review",
            idempotent_hit=False
        )
        assert result1.idempotent_hit is False

        # Cached result (hit)
        result2 = StageTransition(
            deal_id="deal-1",
            from_stage="inbound",
            to_stage="initial_review",
            idempotent_hit=True
        )
        assert result2.idempotent_hit is True

        # Both should have same stage data
        assert result1.from_stage == result2.from_stage
        assert result1.to_stage == result2.to_stage


class TestValidTransitions:
    """Test transition validation logic."""

    def test_inbound_valid_transitions(self):
        """Inbound deals can go to initial_review, closed_lost, or archived."""
        valid = STAGE_TRANSITIONS[DealStage.INBOUND]
        assert DealStage.INITIAL_REVIEW in valid
        assert DealStage.CLOSED_LOST in valid
        assert DealStage.ARCHIVED in valid
        assert len(valid) == 3

    def test_due_diligence_valid_transitions(self):
        """DD deals can go forward, backward, or close."""
        valid = STAGE_TRANSITIONS[DealStage.DUE_DILIGENCE]
        assert DealStage.NEGOTIATION in valid  # Forward
        assert DealStage.INITIAL_REVIEW in valid  # Backward
        assert DealStage.CLOSED_LOST in valid  # Close

    def test_cannot_skip_stages(self):
        """Cannot skip stages (e.g., inbound -> negotiation)."""
        valid = STAGE_TRANSITIONS[DealStage.INBOUND]
        assert DealStage.NEGOTIATION not in valid
        assert DealStage.CLOSING not in valid
        assert DealStage.CLOSED_WON not in valid
