from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from actions.capabilities.registry import CapabilityManifest, CapabilityMatch, get_registry


class ActionPlan(BaseModel):
    """
    Deterministic-first action planner output.

    Supports:
    - single capability selection
    - multi-step plan
    - clarifying questions for missing required inputs
    - safe refusal with suggested alternatives
    """

    intent: str
    interpretation: str

    selected_capability_id: Optional[str] = None
    action_type: Optional[str] = None
    action_inputs: Dict[str, Any] = Field(default_factory=dict)
    missing_fields: List[str] = Field(default_factory=list)

    plan_steps: List[Dict[str, Any]] = Field(default_factory=list)

    requires_clarification: bool = False
    clarifying_questions: List[str] = Field(default_factory=list)

    is_refusal: bool = False
    refusal_reason: Optional[str] = None
    suggested_alternatives: List[str] = Field(default_factory=list)

    confidence: float = 0.0
    risk_level: str = "medium"

    model_config = {"extra": "forbid"}


def _required_fields_from_schema(schema: Dict[str, Any]) -> List[str]:
    required: List[str] = []
    if not isinstance(schema, dict):
        return required

    raw = schema.get("required")
    if isinstance(raw, list):
        required.extend([str(x) for x in raw if str(x).strip()])

    # Support legacy/nonstandard per-property required: true/false.
    props = schema.get("properties")
    if isinstance(props, dict):
        for key, spec in props.items():
            if key in required:
                continue
            if isinstance(spec, dict) and spec.get("required") is True:
                required.append(str(key))

    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: List[str] = []
    for k in required:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _extract_common_inputs(query: str, capability: CapabilityManifest) -> Dict[str, Any]:
    """
    Best-effort offline extraction for a few obvious fields.

    This is intentionally conservative: if we can't extract confidently, we leave it missing.
    """
    intent = (query or "").strip()
    at = (capability.action_type or "").upper()
    out: Dict[str, Any] = {}

    # Email address extraction for draft_email-like capabilities.
    if "DRAFT_EMAIL" in at:
        m = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\b", intent)
        if m:
            # Prefer "to" but also populate "recipient" if schema uses it.
            out["to"] = m.group(0)
            out["recipient"] = m.group(0)

    # Extract a multiple like "6.0x"
    if "VALUATION_MODEL" in at or "BUILD_MODEL" in at:
        m = re.search(r"(\d+(?:\.\d+)?)\s*x\b", intent.lower())
        if m:
            try:
                out["multiple"] = float(m.group(1))
            except Exception:
                pass

    return out


def _risk_rank(risk: str) -> int:
    r = (risk or "").lower().strip()
    return {"low": 0, "medium": 1, "high": 2}.get(r, 1)


class ActionPlanner:
    """
    Deterministic-first planner.

    This is intentionally offline; it can be upgraded later by delegating to the LangGraph brain
    for decomposition and input extraction.
    """

    def __init__(self):
        self.registry = get_registry()
        try:
            from tools.registry import get_tool_registry

            self.registry.index_tools(get_tool_registry())
        except Exception:
            pass

    def plan(self, query: str, *, provided_inputs: Optional[Dict[str, Any]] = None) -> ActionPlan:
        intent = (query or "").strip()
        if not intent:
            return ActionPlan(
                intent="",
                interpretation="Empty query",
                is_refusal=True,
                refusal_reason="empty_query",
                suggested_alternatives=[c.capability_id for c in self.registry.list_capabilities()],
                confidence=0.0,
                risk_level="low",
            )

        # Deterministic override: if the user provides an explicit email address and asks to draft an email,
        # prefer the draft_email capability even if "LOI" appears in the query.
        has_email_address = bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", intent))
        if has_email_address and re.search(r"\b(e-?mail|draft)\b", intent.lower()):
            cap = self.registry.get_by_action_type("COMMUNICATION.DRAFT_EMAIL")
            if cap:
                matches = [CapabilityMatch(capability_id=cap.capability_id, action_type=cap.action_type, score=1.0, reason="email_override")]
            else:
                matches = self.registry.match_capability(intent, top_k=5)
        else:
            matches = self.registry.match_capability(intent, top_k=5)
        if not matches:
            return ActionPlan(
                intent=intent,
                interpretation="No matching capability found",
                is_refusal=True,
                refusal_reason="no_matching_capability",
                suggested_alternatives=[c.capability_id for c in self.registry.list_capabilities()][:10],
                confidence=0.0,
                risk_level="low",
            )

        # Heuristic: treat as multi-step when the query explicitly asks for a bundle.
        bundle = bool(re.search(r"\b(package|bundle|with|plus|\+|and)\b", intent.lower()))
        step_matches = [m for m in matches if m.score >= 0.20]

        if bundle and len(step_matches) >= 2:
            steps: List[Dict[str, Any]] = []
            all_questions: List[str] = []
            any_missing = False
            highest_risk = "low"
            for m in step_matches[:3]:
                cap = self.registry.get_capability(m.capability_id)
                if not cap:
                    continue
                extracted = _extract_common_inputs(intent, cap)
                inputs = {**(provided_inputs or {}), **extracted}
                required = _required_fields_from_schema(cap.input_schema or {})
                missing = [k for k in required if k not in inputs or inputs.get(k) in (None, "", [])]
                any_missing = any_missing or bool(missing)
                if _risk_rank(getattr(cap, "risk_level", "medium")) > _risk_rank(highest_risk):
                    highest_risk = str(getattr(cap, "risk_level", "medium"))
                for f in missing:
                    all_questions.append(f"For `{cap.action_type}`: provide `{f}`.")
                steps.append(
                    {
                        "capability_id": cap.capability_id,
                        "action_type": cap.action_type,
                        "action_inputs": inputs,
                        "missing_fields": missing,
                        "confidence": float(m.score),
                        "risk_level": str(getattr(cap, "risk_level", "medium")),
                    }
                )

            return ActionPlan(
                intent=intent,
                interpretation="Multi-step plan from deterministic capability matches",
                plan_steps=steps,
                requires_clarification=any_missing,
                clarifying_questions=all_questions,
                confidence=float(step_matches[0].score),
                risk_level=highest_risk,
            )

        # Single capability
        best = matches[0]
        cap = self.registry.get_capability(best.capability_id)
        if not cap:
            return ActionPlan(
                intent=intent,
                interpretation="Matched capability disappeared",
                is_refusal=True,
                refusal_reason="capability_not_found",
                suggested_alternatives=[c.capability_id for c in self.registry.list_capabilities()][:10],
                confidence=float(best.score),
                risk_level="medium",
            )

        extracted = _extract_common_inputs(intent, cap)
        inputs = {**(provided_inputs or {}), **extracted}
        required = _required_fields_from_schema(cap.input_schema or {})
        missing = [k for k in required if k not in inputs or inputs.get(k) in (None, "", [])]

        questions: List[str] = []
        if missing:
            questions = [f"Please provide `{f}`." for f in missing]

        return ActionPlan(
            intent=intent,
            interpretation="Single capability match from deterministic registry matcher",
            selected_capability_id=cap.capability_id,
            action_type=cap.action_type,
            action_inputs=inputs,
            missing_fields=missing,
            requires_clarification=bool(missing),
            clarifying_questions=questions,
            confidence=float(best.score),
            risk_level=str(getattr(cap, "risk_level", "medium")),
        )
