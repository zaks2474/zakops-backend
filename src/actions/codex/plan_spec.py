"""
PlanSpec Interface for CodeX Integration

Allows LLMs (CodeX) to propose actions via structured tool calls.
Includes safety constraints: risk capping, rate limits, required approvals.

Usage:
  1. CodeX queries available capabilities via list_capabilities()
  2. CodeX proposes an action via propose_action(spec)
  3. System validates, applies safety constraints, and creates PENDING_APPROVAL action
  4. Human reviews and approves/rejects
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Safety configuration
MAX_ACTIONS_PER_DEAL_PER_HOUR = int(os.getenv("CODEX_MAX_ACTIONS_PER_DEAL_HOUR", "10"))
MAX_RISK_LEVEL_AUTO = os.getenv("CODEX_MAX_RISK_AUTO", "low")  # Auto-approve only low risk
REQUIRE_APPROVAL_FOR = {"high", "medium"}  # Risk levels requiring human approval


class CapabilityInput(BaseModel):
    """Definition of an input field for a capability."""
    name: str = Field(min_length=1)
    type: Literal["string", "number", "boolean", "array", "object"] = "string"
    description: str = ""
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[str]] = None  # For string enums


class CapabilityDefinition(BaseModel):
    """
    Defines an action capability that CodeX can propose.

    This is the "menu" of what actions are available.
    """
    capability_id: str = Field(min_length=1, description="Unique capability identifier, e.g. 'diligence.request_docs'")
    action_type: str = Field(min_length=1, description="Namespaced action type, e.g. 'DILIGENCE.REQUEST_DOCS'")
    name: str = Field(min_length=1, max_length=100, description="Human-readable name")
    description: str = Field(max_length=500, description="What this action does")
    category: str = Field(default="general", description="Category: diligence, communication, document, analysis")

    # Risk and approval
    default_risk_level: Literal["low", "medium", "high"] = "medium"
    requires_deal: bool = True
    requires_approval: bool = True

    # Inputs schema
    inputs: List[CapabilityInput] = Field(default_factory=list)

    # Safety constraints
    max_per_deal_per_day: int = Field(default=5, ge=1)
    cooldown_seconds: int = Field(default=0, ge=0, description="Min seconds between same action on same deal")

    model_config = {"extra": "forbid"}


class ActionProposal(BaseModel):
    """
    A proposed action from CodeX.

    This is what CodeX sends when it wants to create an action.
    """
    capability_id: str = Field(min_length=1)
    deal_id: Optional[str] = None
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=500)
    inputs: Dict[str, Any] = Field(default_factory=dict)

    # Optional overrides (subject to validation)
    risk_level: Optional[Literal["low", "medium", "high"]] = None
    requires_human_review: Optional[bool] = None

    # Metadata
    reasoning: str = Field(default="", max_length=1000, description="Why CodeX is proposing this action")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="CodeX confidence in this proposal")

    @field_validator("inputs", mode="before")
    @classmethod
    def ensure_dict(cls, v: Any) -> Dict[str, Any]:
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        raise ValueError("inputs must be a dict")

    model_config = {"extra": "forbid"}


class ProposalResult(BaseModel):
    """Result of proposing an action."""
    success: bool
    action_id: Optional[str] = None
    status: Optional[str] = None  # PENDING_APPROVAL, READY, etc.
    message: str = ""
    warnings: List[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


# Built-in capability definitions
_CAPABILITIES: Dict[str, CapabilityDefinition] = {}


def _init_builtin_capabilities() -> None:
    """Register built-in capabilities."""
    global _CAPABILITIES

    if _CAPABILITIES:
        return

    # DILIGENCE.REQUEST_DOCS
    _CAPABILITIES["diligence.request_docs"] = CapabilityDefinition(
        capability_id="diligence.request_docs",
        action_type="DILIGENCE.REQUEST_DOCS",
        name="Request Documents",
        description="Draft an email to the broker requesting due diligence documents. Uses Gemini to generate personalized email based on deal context.",
        category="diligence",
        default_risk_level="low",
        requires_deal=True,
        requires_approval=False,  # Draft only, doesn't send
        inputs=[
            CapabilityInput(name="doc_type", type="string", required=True, description="Type of documents: financial, legal, operational, etc."),
            CapabilityInput(name="description", type="string", required=True, description="Specific documents or information needed"),
        ],
        max_per_deal_per_day=3,
        cooldown_seconds=300,
    )

    # COMMUNICATION.DRAFT_EMAIL
    _CAPABILITIES["communication.draft_email"] = CapabilityDefinition(
        capability_id="communication.draft_email",
        action_type="COMMUNICATION.DRAFT_EMAIL",
        name="Draft Email",
        description="Draft a professional email. Does NOT send - creates draft for review.",
        category="communication",
        default_risk_level="low",
        requires_deal=False,
        requires_approval=False,
        inputs=[
            CapabilityInput(name="recipient_type", type="string", required=True, enum=["broker", "seller", "advisor", "other"]),
            CapabilityInput(name="subject_hint", type="string", required=False, description="Subject line hint"),
            CapabilityInput(name="purpose", type="string", required=True, description="Purpose of the email"),
        ],
        max_per_deal_per_day=10,
        cooldown_seconds=60,
    )

    # COMMUNICATION.SEND_EMAIL
    _CAPABILITIES["communication.send_email"] = CapabilityDefinition(
        capability_id="communication.send_email",
        action_type="COMMUNICATION.SEND_EMAIL",
        name="Send Email",
        description="Send an email. ALWAYS requires human approval before sending.",
        category="communication",
        default_risk_level="high",
        requires_deal=False,
        requires_approval=True,  # ALWAYS requires approval
        inputs=[
            CapabilityInput(name="to", type="string", required=True, description="Recipient email"),
            CapabilityInput(name="subject", type="string", required=True),
            CapabilityInput(name="body", type="string", required=True),
        ],
        max_per_deal_per_day=5,
        cooldown_seconds=600,
    )

    # DOCUMENT.GENERATE_LOI
    _CAPABILITIES["document.generate_loi"] = CapabilityDefinition(
        capability_id="document.generate_loi",
        action_type="DOCUMENT.GENERATE_LOI",
        name="Generate LOI",
        description="Generate a Letter of Intent draft based on deal terms.",
        category="document",
        default_risk_level="medium",
        requires_deal=True,
        requires_approval=True,
        inputs=[
            CapabilityInput(name="purchase_price", type="number", required=False, description="Proposed purchase price"),
            CapabilityInput(name="terms", type="string", required=False, description="Key terms to include"),
            CapabilityInput(name="due_diligence_days", type="number", required=False, default=45),
        ],
        max_per_deal_per_day=2,
        cooldown_seconds=3600,
    )

    # ANALYSIS.BUILD_VALUATION_MODEL
    _CAPABILITIES["analysis.build_valuation_model"] = CapabilityDefinition(
        capability_id="analysis.build_valuation_model",
        action_type="ANALYSIS.BUILD_VALUATION_MODEL",
        name="Build Valuation Model",
        description="Create a valuation model based on deal financials.",
        category="analysis",
        default_risk_level="low",
        requires_deal=True,
        requires_approval=False,
        inputs=[
            CapabilityInput(name="method", type="string", required=False, enum=["multiple", "dcf", "comparable"], default="multiple"),
            CapabilityInput(name="revenue", type="number", required=False),
            CapabilityInput(name="ebitda", type="number", required=False),
        ],
        max_per_deal_per_day=5,
        cooldown_seconds=300,
    )


def list_capabilities(category: Optional[str] = None) -> List[CapabilityDefinition]:
    """
    List available capabilities that CodeX can propose.

    Args:
        category: Optional filter by category

    Returns:
        List of capability definitions
    """
    _init_builtin_capabilities()

    caps = list(_CAPABILITIES.values())
    if category:
        caps = [c for c in caps if c.category == category]

    return sorted(caps, key=lambda c: c.capability_id)


def get_capability(capability_id: str) -> Optional[CapabilityDefinition]:
    """Get a specific capability by ID."""
    _init_builtin_capabilities()
    return _CAPABILITIES.get(capability_id)


def validate_proposal(proposal: ActionProposal) -> tuple[bool, List[str]]:
    """
    Validate a proposed action against capability constraints.

    Returns:
        (is_valid, list_of_errors)
    """
    errors: List[str] = []

    # Get capability
    cap = get_capability(proposal.capability_id)
    if not cap:
        errors.append(f"Unknown capability: {proposal.capability_id}")
        return False, errors

    # Check deal requirement
    if cap.requires_deal and not proposal.deal_id:
        errors.append(f"Capability {proposal.capability_id} requires a deal_id")

    # Validate required inputs
    provided_inputs = set(proposal.inputs.keys())
    for inp in cap.inputs:
        if inp.required and inp.name not in provided_inputs:
            errors.append(f"Missing required input: {inp.name}")

    # Validate input types (basic)
    for inp in cap.inputs:
        if inp.name in proposal.inputs:
            value = proposal.inputs[inp.name]
            if inp.enum and value not in inp.enum:
                errors.append(f"Input {inp.name} must be one of: {inp.enum}")

    return len(errors) == 0, errors


def apply_safety_constraints(
    proposal: ActionProposal,
    cap: CapabilityDefinition,
) -> tuple[str, bool, List[str]]:
    """
    Apply safety constraints to determine final status and approval requirement.

    Returns:
        (status, requires_approval, warnings)
    """
    warnings: List[str] = []

    # Determine risk level
    risk = proposal.risk_level or cap.default_risk_level

    # Override confidence-based risk escalation
    if proposal.confidence < 0.5:
        warnings.append("Low confidence proposal - escalating to approval")
        risk = "high"

    # Determine if approval is required
    requires_approval = cap.requires_approval or risk in REQUIRE_APPROVAL_FOR

    # CodeX cannot override approval requirement
    if proposal.requires_human_review is False and cap.requires_approval:
        warnings.append(f"Capability {cap.capability_id} always requires approval - ignoring override")
        requires_approval = True

    # Determine initial status
    if requires_approval:
        status = "PENDING_APPROVAL"
    else:
        status = "READY"

    return status, requires_approval, warnings


def propose_action(
    proposal: ActionProposal,
    created_by: str = "codex",
) -> ProposalResult:
    """
    Process an action proposal from CodeX.

    Creates the action in PENDING_APPROVAL or READY state based on
    safety constraints.

    Args:
        proposal: The proposed action
        created_by: Actor creating the action

    Returns:
        ProposalResult with action_id if successful
    """
    import sys
    sys.path.insert(0, "/home/zaks/scripts")

    from actions.engine.models import ActionPayload, compute_idempotency_key
    from actions.engine.store import ActionStore

    # Validate proposal
    valid, errors = validate_proposal(proposal)
    if not valid:
        return ProposalResult(
            success=False,
            message=f"Invalid proposal: {'; '.join(errors)}",
        )

    # Get capability
    cap = get_capability(proposal.capability_id)
    if not cap:
        return ProposalResult(
            success=False,
            message=f"Unknown capability: {proposal.capability_id}",
        )

    # Apply safety constraints
    status, requires_approval, warnings = apply_safety_constraints(proposal, cap)

    # Create action
    store = ActionStore()
    try:
        from actions.engine.validation import validate_action_creation

        validate_action_creation(action_type=cap.action_type, capability_id=proposal.capability_id)
    except Exception as e:
        return ProposalResult(
            success=False,
            message=f"Cannot create action (validation failed): {e}",
        )

    action = ActionPayload(
        deal_id=proposal.deal_id,
        capability_id=proposal.capability_id,
        type=cap.action_type,
        title=proposal.title,
        summary=proposal.summary or proposal.reasoning[:500] if proposal.reasoning else "",
        status=status,
        source="chat",  # CodeX proposals come through chat
        created_by=created_by,
        idempotency_key=compute_idempotency_key(
            proposal.capability_id,
            proposal.deal_id or "",
            proposal.title,
        ),
        risk_level=proposal.risk_level or cap.default_risk_level,
        requires_human_review=requires_approval,
        inputs=proposal.inputs,
    )

    try:
        store.create_action(action)
        logger.info(f"Created action {action.action_id} from CodeX proposal")
    except Exception as e:
        return ProposalResult(
            success=False,
            message=f"Failed to create action: {e}",
        )

    return ProposalResult(
        success=True,
        action_id=action.action_id,
        status=status,
        message=f"Action created with status {status}",
        warnings=warnings,
    )


# Tool definition for CodeX
CODEX_TOOL_DEFINITIONS = [
    {
        "name": "list_action_capabilities",
        "description": "List available action capabilities that can be proposed. Returns capability definitions with input schemas.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter: diligence, communication, document, analysis",
                    "enum": ["diligence", "communication", "document", "analysis"],
                },
            },
        },
    },
    {
        "name": "propose_action",
        "description": "Propose an action to be created. Actions may require human approval before execution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "capability_id": {
                    "type": "string",
                    "description": "The capability ID from list_action_capabilities",
                },
                "deal_id": {
                    "type": "string",
                    "description": "The deal ID this action relates to (required for most capabilities)",
                },
                "title": {
                    "type": "string",
                    "description": "Short title for the action (max 200 chars)",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what will be done (max 500 chars)",
                },
                "inputs": {
                    "type": "object",
                    "description": "Input parameters for the action (see capability definition)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why you are proposing this action",
                },
                "confidence": {
                    "type": "number",
                    "description": "Your confidence in this proposal (0.0-1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["capability_id", "title", "inputs"],
        },
    },
]


def handle_codex_tool_call(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle a tool call from CodeX.

    Args:
        tool_name: Name of the tool being called
        tool_input: Input parameters

    Returns:
        Tool result as a dictionary
    """
    if tool_name == "list_action_capabilities":
        category = tool_input.get("category")
        caps = list_capabilities(category=category)
        return {
            "capabilities": [
                {
                    "capability_id": c.capability_id,
                    "name": c.name,
                    "description": c.description,
                    "category": c.category,
                    "requires_approval": c.requires_approval,
                    "inputs": [
                        {
                            "name": i.name,
                            "type": i.type,
                            "required": i.required,
                            "description": i.description,
                            "enum": i.enum,
                        }
                        for i in c.inputs
                    ],
                }
                for c in caps
            ],
        }

    elif tool_name == "propose_action":
        proposal = ActionProposal(
            capability_id=tool_input.get("capability_id", ""),
            deal_id=tool_input.get("deal_id"),
            title=tool_input.get("title", ""),
            summary=tool_input.get("summary", ""),
            inputs=tool_input.get("inputs", {}),
            reasoning=tool_input.get("reasoning", ""),
            confidence=tool_input.get("confidence", 0.8),
        )
        result = propose_action(proposal)
        return {
            "success": result.success,
            "action_id": result.action_id,
            "status": result.status,
            "message": result.message,
            "warnings": result.warnings,
        }

    else:
        return {"error": f"Unknown tool: {tool_name}"}
