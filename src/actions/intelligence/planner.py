from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Protocol

from actions.contracts.plan_spec import ArtifactTypeSpec, MissingCapabilitySpec, PlanSpec, PlanStep, StepSafety
from actions.intelligence.validator import PlanValidator
from actions.memory.store import ActionMemoryStore
from tools.manifest.registry import ManifestEntry, UnifiedManifestRegistry, get_unified_manifest_registry


class LLMPlannerClient(Protocol):
    def plan_json(self, *, prompt: str, timeout_s: int) -> str: ...


class VLLMPlannerClient:
    def __init__(self):
        self._enabled = True

    def plan_json(self, *, prompt: str, timeout_s: int = 60) -> str:
        from chat_llm_provider import VLLMProvider

        async def _run() -> str:
            provider = VLLMProvider()
            resp = await provider.generate(
                [
                    {"role": "system", "content": "You are a tool planner. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            return resp.content

        # Avoid importing asyncio globally in hot path.
        import asyncio

        return asyncio.run(_run())


def _extract_json(text: str) -> str:
    """
    Best-effort extraction of a JSON object from an LLM response.
    """
    raw = (text or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        return raw
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return raw
    return m.group(0)


def _default_plan_id() -> str:
    return f"PLAN-{uuid.uuid4().hex[:12]}"


def _step_from_entry(
    *,
    step_id: str,
    entry: ManifestEntry,
    title: str,
    summary: str,
    inputs: Dict[str, Any],
) -> PlanStep:
    safety_class = (entry.safety_class or "reversible").strip().lower()
    irreversible = bool(entry.irreversible)
    gated = safety_class in {"gated", "irreversible"} or irreversible

    expected_artifacts: List[ArtifactTypeSpec] = []
    for raw in entry.output_artifacts or []:
        if not isinstance(raw, dict):
            continue
        try:
            expected_artifacts.append(ArtifactTypeSpec.model_validate(raw))
        except Exception:
            continue

    return PlanStep(
        step_id=step_id,
        capability_id=entry.capability_id,
        action_type=entry.action_type,
        tool_name=entry.tool_name or entry.action_type,
        title=title,
        summary=summary,
        inputs=inputs or {},
        depends_on=[],
        expected_artifacts=expected_artifacts,
        safety=StepSafety(
            safety_class=safety_class, irreversible=irreversible, gated=gated, requires_human_approval=bool(entry.requires_approval)
        ),
    )


class ToolRAGPlanner:
    def __init__(
        self,
        *,
        registry: Optional[UnifiedManifestRegistry] = None,
        validator: Optional[PlanValidator] = None,
        memory: Optional[ActionMemoryStore] = None,
        llm: Optional[LLMPlannerClient] = None,
    ):
        self.registry = registry or get_unified_manifest_registry()
        self.validator = validator or PlanValidator(registry=self.registry)
        self.memory = memory or ActionMemoryStore()
        self.llm = llm  # optional; may be None for deterministic-only mode

    def _find_by_action_type(self, action_type: str) -> Optional[ManifestEntry]:
        at = (action_type or "").strip()
        for entry in self.registry.list_entries():
            if entry.action_type == at:
                return entry
        return None

    def _needs_tool(self, *, goal: str, reason: str) -> PlanSpec:
        missing = MissingCapabilitySpec(
            capability_id="custom.capability.v1",
            title="Missing capability",
            description=f"Capability missing for goal: {goal}",
            action_type="CUSTOM.MISSING_CAPABILITY",
            tool_name=None,
            input_schema={"type": "object", "additionalProperties": True, "properties": {}, "required": []},
            output_artifacts=[],
            risk_level="medium",
            safety_class="gated",
            irreversible=False,
            requires_approval=True,
            examples=[{"user_intent": goal, "expected": "Describe what the tool should do."}],
            constraints=["Add this to the manifest before execution."],
        )
        return PlanSpec(status="NEEDS_TOOL", plan_id=_default_plan_id(), goal=goal, steps=[], missing_capability=missing, debug={"reason": reason})

    def plan(
        self,
        goal: str,
        *,
        deal_id: Optional[str] = None,
        provided_inputs: Optional[Dict[str, Any]] = None,
    ) -> PlanSpec:
        intent = (goal or "").strip()
        if not intent:
            return PlanSpec(status="BLOCKED", plan_id=_default_plan_id(), goal="", blocked_reason="empty_goal", steps=[])

        # 1) Memory: reuse prior successful plan if highly similar.
        try:
            mem_matches = self.memory.find_similar(intent, deal_id=deal_id, top_k=1)
        except Exception:
            mem_matches = []

        if mem_matches:
            candidate = mem_matches[0].summary.plan_spec or {}
            try:
                plan = PlanSpec.model_validate(candidate)
                plan = plan.model_copy(update={"plan_id": _default_plan_id(), "goal": intent, "deal_id": deal_id})
                vr = self.validator.validate(plan)
                if vr.status == "ok":
                    plan.debug["memory_reuse"] = {"memory_id": mem_matches[0].summary.memory_id, "reason": mem_matches[0].reason}
                    return plan
            except Exception:
                pass

        # 2) Deterministic-first heuristics for critical actions.
        has_email = bool(re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", intent))
        wants_send = bool(re.search(r"\b(send|deliver|dispatch)\b", intent.lower()))

        if re.search(r"\bloi\b|\bletter of intent\b", intent.lower()):
            entry = self._find_by_action_type("DOCUMENT.GENERATE_LOI")
            if not entry:
                return self._needs_tool(goal=intent, reason="missing_document_generate_loi")
            step = _step_from_entry(
                step_id="step_1",
                entry=entry,
                title="Generate LOI",
                summary="Generate LOI artifact(s) for operator review.",
                inputs=provided_inputs or {},
            )
            plan = PlanSpec(status="OK", plan_id=_default_plan_id(), goal=intent, deal_id=deal_id, steps=[step])
            vr = self.validator.validate(plan)
            if vr.status == "ok":
                return plan
            return PlanSpec(status="BLOCKED", plan_id=plan.plan_id, goal=intent, deal_id=deal_id, steps=plan.steps, blocked_reason=vr.reason, debug=vr.details or {})

        if has_email and wants_send:
            draft = self._find_by_action_type("COMMUNICATION.DRAFT_EMAIL")
            send = self._find_by_action_type("TOOL.gmail__send_email")
            if not draft:
                return self._needs_tool(goal=intent, reason="missing_communication_draft_email")
            if not send:
                return self._needs_tool(goal=intent, reason="missing_gmail_send_email_tool")

            # Conservative: generate a draft first, then gate the irreversible send step.
            to_addr = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", intent).group(0)  # type: ignore[union-attr]
            draft_inputs = {**(provided_inputs or {}), "to": to_addr}
            draft_inputs.setdefault("subject", "TODO: Subject")
            # Keep context non-empty for validation; operator/LLM can refine before execution.
            draft_inputs.setdefault("context", intent)
            step1 = _step_from_entry(
                step_id="step_1",
                entry=draft,
                title="Draft email",
                summary="Create a draft email artifact for review (draft-only).",
                inputs=draft_inputs,
            )
            step2 = _step_from_entry(
                step_id="step_2",
                entry=send,
                title="Send email (GATED)",
                summary="Send the email via Gmail MCP (requires explicit operator approval).",
                # Use template refs so the executor can safely resolve from step_1 outputs/artifacts.
                inputs={
                    "to": [to_addr],
                    "subject": "{{step_1.outputs.subject}}",
                    "body": "{{step_1.outputs.body}}",
                },
            ).model_copy(update={"depends_on": ["step_1"], "safety": StepSafety(safety_class="irreversible", irreversible=True, gated=True, requires_human_approval=True)})

            plan = PlanSpec(status="OK", plan_id=_default_plan_id(), goal=intent, deal_id=deal_id, steps=[step1, step2])
            vr = self.validator.validate(plan)
            if vr.status == "ok":
                return plan
            return PlanSpec(status="BLOCKED", plan_id=plan.plan_id, goal=intent, deal_id=deal_id, steps=plan.steps, blocked_reason=vr.reason, debug=vr.details or {})

        # 3) Tool-RAG: retrieve relevant capabilities and ask LLM to draft a PlanSpec JSON.
        matches = self.registry.match(intent, top_k=8)
        if not matches:
            return self._needs_tool(goal=intent, reason="no_manifest_match")

        candidates: List[ManifestEntry] = []
        for m in matches:
            entry = self.registry.get_entry(m.capability_id)
            if entry:
                candidates.append(entry)

        # Deterministic-only mode.
        if os.getenv("ZAKOPS_PLANNER_USE_LLM", "true").strip().lower() in {"0", "false", "no"}:
            # Pick top candidate and create a single-step plan.
            entry = candidates[0]
            step = _step_from_entry(
                step_id="step_1",
                entry=entry,
                title=entry.title,
                summary=entry.description[:200],
                inputs=provided_inputs or {},
            )
            plan = PlanSpec(status="OK", plan_id=_default_plan_id(), goal=intent, deal_id=deal_id, steps=[step], debug={"mode": "deterministic_only"})
            vr = self.validator.validate(plan)
            if vr.status == "ok":
                return plan
            return PlanSpec(status="BLOCKED", plan_id=plan.plan_id, goal=intent, deal_id=deal_id, steps=plan.steps, blocked_reason=vr.reason, debug=vr.details or {})

        llm = self.llm
        if llm is None:
            llm = VLLMPlannerClient()

        memory_hints = []
        try:
            memory_hints = [asdict(m.summary) for m in self.memory.find_similar(intent, deal_id=deal_id, top_k=3)]
        except Exception:
            memory_hints = []

        prompt = {
            "goal": intent,
            "deal_id": deal_id,
            "instructions": [
                "Return ONLY a JSON object that conforms to PlanSpec (no markdown).",
                "Use ONLY the provided capabilities list (capability_id/action_type).",
                "For irreversible capabilities (irreversible=true), set step.safety.gated=true and do NOT auto-send communications.",
                "If you cannot complete the goal with available capabilities, return PlanSpec with status=NEEDS_TOOL and missing_capability populated.",
                "No LangSmith tracing.",
            ],
            "capabilities": [asdict(e) for e in candidates],
            "memory_hints": memory_hints,
            "provided_inputs": provided_inputs or {},
            "plan_spec_schema_hint": {
                "status": "OK|NEEDS_TOOL|BLOCKED",
                "plan_id": "string",
                "goal": "string",
                "deal_id": "string?",
                "steps": [
                    {
                        "step_id": "step_1",
                        "capability_id": "document.generate_loi.v1",
                        "action_type": "DOCUMENT.GENERATE_LOI",
                        "tool_name": "DOCUMENT.GENERATE_LOI",
                        "title": "Generate LOI",
                        "summary": "string",
                        "inputs": {},
                        "depends_on": [],
                        "expected_artifacts": [],
                        "safety": {"safety_class": "reversible|gated|irreversible", "irreversible": False, "gated": False, "requires_human_approval": True},
                    }
                ],
            },
        }

        raw = llm.plan_json(prompt=json.dumps(prompt, ensure_ascii=False), timeout_s=int(os.getenv("ZAKOPS_PLANNER_LLM_TIMEOUT_SECONDS", "60")))
        raw_json = _extract_json(raw)
        try:
            plan = PlanSpec.model_validate_json(raw_json)
        except Exception as e:
            return PlanSpec(status="BLOCKED", plan_id=_default_plan_id(), goal=intent, deal_id=deal_id, steps=[], blocked_reason="llm_invalid_planspec", debug={"error": str(e), "raw": raw[:2000]})

        # Always run deterministic validation.
        vr = self.validator.validate(plan)
        if vr.status == "ok":
            plan.debug["validated"] = True
            return plan
        if vr.status == "needs_tool":
            plan = plan.model_copy(update={"status": "NEEDS_TOOL"})
            plan.debug["validator"] = vr.details or {}
            return plan
        plan = plan.model_copy(update={"status": "BLOCKED", "blocked_reason": vr.reason or "validation_failed"})
        plan.debug["validator"] = vr.details or {}
        return plan
