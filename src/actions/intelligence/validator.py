from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from actions.contracts.plan_spec import PlanSpec
from tools.manifest.registry import ManifestEntry, UnifiedManifestRegistry, get_unified_manifest_registry


def _required_fields_from_schema(schema: Dict[str, Any]) -> List[str]:
    required: List[str] = []
    if not isinstance(schema, dict):
        return required

    raw = schema.get("required")
    if isinstance(raw, list):
        required.extend([str(x) for x in raw if str(x).strip()])

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


ValidationStatus = str  # ok|blocked|needs_tool


@dataclass(frozen=True)
class ValidationResult:
    status: ValidationStatus
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class PlanValidator:
    def __init__(self, *, registry: Optional[UnifiedManifestRegistry] = None):
        self.registry = registry or get_unified_manifest_registry()

    def validate(self, plan: PlanSpec) -> ValidationResult:
        mode = (os.getenv("ZAKOPS_PLANNER_MODE", "safe") or "safe").strip().lower()

        if plan.status == "NEEDS_TOOL":
            if not plan.missing_capability:
                return ValidationResult(status="blocked", reason="needs_tool_missing_capability")
            return ValidationResult(status="needs_tool", reason="needs_tool")

        if plan.status == "BLOCKED":
            return ValidationResult(status="blocked", reason=plan.blocked_reason or "blocked")

        if plan.status != "OK":
            return ValidationResult(status="blocked", reason=f"invalid_plan_status:{plan.status}")

        if not plan.steps:
            return ValidationResult(status="blocked", reason="no_steps")

        errors: List[Dict[str, Any]] = []
        for step in plan.steps:
            entry = self.registry.get_entry(step.capability_id)
            if not entry:
                errors.append({"step_id": step.step_id, "error": "capability_not_found", "capability_id": step.capability_id})
                continue

            if entry.action_type and step.action_type != entry.action_type:
                errors.append(
                    {
                        "step_id": step.step_id,
                        "error": "action_type_mismatch",
                        "expected": entry.action_type,
                        "actual": step.action_type,
                    }
                )

            required = _required_fields_from_schema(entry.input_schema or {})
            missing = [k for k in required if k not in (step.inputs or {}) or step.inputs.get(k) in (None, "", [])]
            if missing:
                errors.append({"step_id": step.step_id, "error": "missing_inputs", "missing": missing})

            # Safety: irreversible requires explicit gate.
            irreversible = bool(entry.irreversible)
            if irreversible and not bool(step.safety.irreversible):
                errors.append({"step_id": step.step_id, "error": "irreversible_flag_missing"})
            if irreversible and not bool(step.safety.gated):
                errors.append({"step_id": step.step_id, "error": "irreversible_step_not_gated"})

            # "Gated" safety class means the step is not auto-executable.
            safety_class = (entry.safety_class or "reversible").strip().lower()
            if safety_class in {"gated", "irreversible"} and not bool(step.safety.gated):
                errors.append({"step_id": step.step_id, "error": "safety_class_requires_gating", "safety_class": safety_class})

            # Mode-based blocks.
            if mode in {"no_irreversible", "offline"} and irreversible:
                errors.append({"step_id": step.step_id, "error": "mode_blocks_irreversible", "mode": mode})

        if errors:
            # If the only errors are missing capabilities, return needs_tool.
            only_missing_caps = all(e.get("error") == "capability_not_found" for e in errors)
            if only_missing_caps:
                return ValidationResult(status="needs_tool", reason="capability_not_found", details={"errors": errors})
            return ValidationResult(status="blocked", reason="validation_failed", details={"errors": errors})

        return ValidationResult(status="ok", reason="ok")

