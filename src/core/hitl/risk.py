"""
Risk Assessor

Phase 6: HITL & Checkpoints
Spec Reference: Human-in-the-Loop section

Provides configurable risk assessment for actions:
- Evaluates action type, parameters, and context
- Returns risk level (low, medium, high, critical)
- Determines if human approval is required

Risk levels are CONFIGURABLE via:
- Environment variables
- Configuration files
- Database settings (future)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk levels for actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def requires_approval(self) -> bool:
        """Whether this risk level requires human approval by default."""
        return self in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def __lt__(self, other: "RiskLevel") -> bool:
        order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other: "RiskLevel") -> bool:
        return self == other or self < other


@dataclass
class RiskAssessment:
    """Result of a risk assessment."""
    risk_level: RiskLevel
    requires_approval: bool
    reasons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "requires_approval": self.requires_approval,
            "reasons": self.reasons,
            "recommendations": self.recommendations,
            "metadata": self.metadata,
        }


@dataclass
class RiskRule:
    """A configurable rule for risk assessment."""
    name: str
    description: str
    risk_level: RiskLevel
    requires_approval: bool = True

    # Matching criteria
    action_types: Set[str] = field(default_factory=set)  # Empty = match all
    action_type_patterns: List[str] = field(default_factory=list)  # Regex patterns
    input_patterns: Dict[str, str] = field(default_factory=dict)  # Key: regex pattern for value

    # Conditions
    min_amount: Optional[float] = None  # If inputs have 'amount' field
    external_communication: bool = False  # If action sends external comms

    def matches(self, action_type: str, inputs: Dict[str, Any]) -> bool:
        """Check if this rule matches the given action."""
        action_type_upper = action_type.upper()
        action_types_upper = {t.upper() for t in self.action_types}

        # Check action type (case-insensitive)
        type_matched = False
        if self.action_types:
            if action_type_upper in action_types_upper:
                type_matched = True

        # Try patterns if no direct match
        if not type_matched and self.action_type_patterns:
            for pattern in self.action_type_patterns:
                if re.match(pattern, action_type, re.IGNORECASE):
                    type_matched = True
                    break

        # If we have action_types or patterns defined and neither matched, fail
        if (self.action_types or self.action_type_patterns) and not type_matched:
            return False

        # Check input patterns
        for key, pattern in self.input_patterns.items():
            value = inputs.get(key)
            if value is None:
                continue
            if not re.search(pattern, str(value), re.IGNORECASE):
                return False

        # Check amount threshold
        if self.min_amount is not None:
            amount = inputs.get("amount") or inputs.get("value") or inputs.get("total")
            if amount is not None:
                try:
                    if float(amount) < self.min_amount:
                        return False
                except (ValueError, TypeError):
                    pass

        return True


class RiskAssessor:
    """
    Configurable risk assessor for actions.

    Evaluates actions against a set of rules to determine:
    - Risk level (low, medium, high, critical)
    - Whether human approval is required

    Configuration sources (in order of precedence):
    1. Explicit rules passed to constructor
    2. Environment variables
    3. Default rules

    Usage:
        assessor = RiskAssessor()
        assessment = assessor.assess(action_type, inputs)
        if assessment.requires_approval:
            # Route to approval workflow
    """

    # Default risk rules
    DEFAULT_RULES: List[RiskRule] = [
        # Critical: External communication
        RiskRule(
            name="external_email",
            description="Sending external emails requires approval",
            risk_level=RiskLevel.CRITICAL,
            requires_approval=True,
            action_types={"COMMUNICATION.SEND_EMAIL"},
            action_type_patterns=[r".*SEND.*EMAIL.*", r".*EMAIL.*SEND.*"],
        ),
        # High: Document generation
        RiskRule(
            name="document_generation",
            description="Generating legal/financial documents",
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            action_types={"DOCUMENT.GENERATE_LOI", "DOCUMENT.GENERATE_NDA"},
            action_type_patterns=[r"DOCUMENT\.GENERATE.*", r".*GENERATE.*LOI.*"],
        ),
        # High: Financial operations
        RiskRule(
            name="high_value_operation",
            description="High-value financial operations",
            risk_level=RiskLevel.HIGH,
            requires_approval=True,
            action_type_patterns=[r".*PAYMENT.*", r".*TRANSFER.*", r".*FINANCIAL.*"],
            min_amount=10000.0,
        ),
        # Medium: Data enrichment
        RiskRule(
            name="data_enrichment",
            description="External data enrichment",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            action_types={"DEAL.ENRICH_MATERIALS", "DEAL.PROFILE_ENRICHED"},
            action_type_patterns=[r".*ENRICH.*", r".*PROFILE.*"],
        ),
        # Medium: Analysis operations
        RiskRule(
            name="analysis",
            description="Analysis and valuation operations",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            action_type_patterns=[r"ANALYSIS\..*", r".*VALUATION.*", r".*ANALYZE.*"],
        ),
        # Low: Read-only operations
        RiskRule(
            name="read_only",
            description="Read-only data operations",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            action_type_patterns=[r".*GET.*", r".*LIST.*", r".*READ.*", r".*FETCH.*"],
        ),
    ]

    def __init__(
        self,
        rules: Optional[List[RiskRule]] = None,
        default_risk_level: RiskLevel = RiskLevel.MEDIUM,
        default_requires_approval: bool = True,
    ):
        """
        Initialize the risk assessor.

        Args:
            rules: Custom risk rules (uses defaults if not provided)
            default_risk_level: Default risk level when no rules match
            default_requires_approval: Default approval requirement when no rules match
        """
        self.rules = rules or self._load_rules()
        self.default_risk_level = default_risk_level
        self.default_requires_approval = default_requires_approval

        # Load config from environment
        self._load_env_config()

    def _load_rules(self) -> List[RiskRule]:
        """Load rules from configuration or use defaults."""
        # Future: Load from database or config file
        return list(self.DEFAULT_RULES)

    def _load_env_config(self) -> None:
        """Load configuration from environment variables."""
        # ZAKOPS_HITL_DEFAULT_RISK_LEVEL
        env_level = os.getenv("ZAKOPS_HITL_DEFAULT_RISK_LEVEL", "").lower()
        if env_level in ("low", "medium", "high", "critical"):
            self.default_risk_level = RiskLevel(env_level)

        # ZAKOPS_HITL_DEFAULT_REQUIRES_APPROVAL
        env_approval = os.getenv("ZAKOPS_HITL_DEFAULT_REQUIRES_APPROVAL", "").lower()
        if env_approval in ("true", "1", "yes"):
            self.default_requires_approval = True
        elif env_approval in ("false", "0", "no"):
            self.default_requires_approval = False

        # ZAKOPS_HITL_AUTO_APPROVE_LOW_RISK
        self.auto_approve_low_risk = os.getenv(
            "ZAKOPS_HITL_AUTO_APPROVE_LOW_RISK", "false"
        ).lower() in ("true", "1", "yes")

    def assess(
        self,
        action_type: str,
        inputs: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessment:
        """
        Assess the risk level of an action.

        Args:
            action_type: The action type (e.g., "COMMUNICATION.SEND_EMAIL")
            inputs: Action input parameters
            context: Additional context (deal info, user info, etc.)

        Returns:
            RiskAssessment with risk level and approval requirement
        """
        inputs = inputs or {}
        context = context or {}

        reasons: List[str] = []
        recommendations: List[str] = []
        matched_rules: List[RiskRule] = []

        # Find matching rules
        for rule in self.rules:
            if rule.matches(action_type, inputs):
                matched_rules.append(rule)
                reasons.append(f"Matched rule: {rule.name} - {rule.description}")

        # Determine highest risk level from matched rules
        if matched_rules:
            highest_risk = max(rule.risk_level for rule in matched_rules)
            requires_approval = any(rule.requires_approval for rule in matched_rules)
        else:
            highest_risk = self.default_risk_level
            requires_approval = self.default_requires_approval
            reasons.append(f"No specific rules matched, using default: {highest_risk.value}")

        # Apply auto-approve for low risk if configured
        if self.auto_approve_low_risk and highest_risk == RiskLevel.LOW:
            requires_approval = False
            recommendations.append("Auto-approved due to low risk configuration")

        # Add context-based adjustments
        if context.get("is_test_mode"):
            recommendations.append("Running in test mode - consider auto-approval")
        if context.get("deal_stage") == "closed":
            highest_risk = RiskLevel.CRITICAL
            requires_approval = True
            reasons.append("Deal is closed - all actions require approval")

        # Build metadata
        metadata = {
            "action_type": action_type,
            "matched_rules": [r.name for r in matched_rules],
            "input_keys": list(inputs.keys()),
        }

        return RiskAssessment(
            risk_level=highest_risk,
            requires_approval=requires_approval,
            reasons=reasons,
            recommendations=recommendations,
            metadata=metadata,
        )

    def get_approval_requirements(
        self,
        action_type: str,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get approval requirements for an action.

        Returns a dict with:
        - requires_approval: bool
        - approver_roles: List of roles that can approve
        - escalation_roles: List of roles for escalation
        - timeout_hours: Auto-reject timeout
        """
        assessment = self.assess(action_type, inputs)

        # Default approver roles based on risk level
        approver_roles = ["operator"]
        escalation_roles = ["admin"]
        timeout_hours = 24

        if assessment.risk_level == RiskLevel.CRITICAL:
            approver_roles = ["admin"]
            escalation_roles = ["superadmin"]
            timeout_hours = 4
        elif assessment.risk_level == RiskLevel.HIGH:
            approver_roles = ["operator", "admin"]
            escalation_roles = ["admin"]
            timeout_hours = 8

        return {
            "requires_approval": assessment.requires_approval,
            "risk_level": assessment.risk_level.value,
            "approver_roles": approver_roles,
            "escalation_roles": escalation_roles,
            "timeout_hours": timeout_hours,
        }

    def add_rule(self, rule: RiskRule) -> None:
        """Add a custom rule to the assessor."""
        self.rules.insert(0, rule)  # Insert at beginning for priority

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by name. Returns True if removed."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                self.rules.pop(i)
                return True
        return False


# Global assessor instance
_assessor: Optional[RiskAssessor] = None


def get_risk_assessor() -> RiskAssessor:
    """Get the global risk assessor instance."""
    global _assessor
    if _assessor is None:
        _assessor = RiskAssessor()
    return _assessor


def assess_risk(
    action_type: str,
    inputs: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> RiskAssessment:
    """Convenience function to assess risk using global assessor."""
    return get_risk_assessor().assess(action_type, inputs, context)
