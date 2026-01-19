from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal


PlanStatus = Literal["OK", "NEEDS_TOOL", "BLOCKED"]
SafetyClass = Literal["reversible", "gated", "irreversible"]


class ArtifactTypeSpec(BaseModel):
    kind: str = Field(min_length=1, description="Logical artifact type, e.g. md, docx, pdf, xlsx, pptx")
    extension: Optional[str] = Field(default=None, description="File extension including dot, e.g. .docx")
    mime_type: Optional[str] = Field(default=None)
    required: bool = True
    description: Optional[str] = None

    model_config = {"extra": "forbid"}


class StepSafety(BaseModel):
    safety_class: SafetyClass = "reversible"
    irreversible: bool = False

    # If gated=true, the executor MUST NOT auto-execute this step without an explicit operator approval.
    gated: bool = False

    # Human approval is the default for ZakOps kinetic actions; this field makes it explicit for step-level UX.
    requires_human_approval: bool = True

    model_config = {"extra": "forbid"}


class PlanStep(BaseModel):
    step_id: str = Field(min_length=1, description="Stable within-plan identifier, e.g. step_1")

    # Capability registry identifier (single source of truth in manifest).
    capability_id: str = Field(min_length=1, description="Versioned capability id, e.g. document.generate_loi.v1")

    # Execution dispatch:
    # - For Kinetic Action executors: action_type like DOCUMENT.GENERATE_LOI
    # - For ToolGateway-backed tools: action_type like TOOL.gmail__send_email
    action_type: str = Field(min_length=1, description="Namespaced action type to execute")

    # Human-readable.
    tool_name: str = Field(min_length=1, description="Tool/capability name for UI/debug (may equal action_type)")
    title: str = Field(min_length=1)
    summary: str = Field(default="", description="1-2 line summary")

    inputs: Dict[str, Any] = Field(default_factory=dict)

    depends_on: List[str] = Field(default_factory=list)
    expected_artifacts: List[ArtifactTypeSpec] = Field(default_factory=list)

    safety: StepSafety = Field(default_factory=StepSafety)

    model_config = {"extra": "forbid"}


class MissingCapabilitySpec(BaseModel):
    """
    Returned when the planner cannot satisfy the request with the current manifest.
    This is intended to be copy/pastable into the manifest as a starting point.
    """

    capability_id: str = Field(min_length=1, description="Suggested stable id, versioned")
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    action_type: str = Field(min_length=1)

    tool_name: Optional[str] = Field(default=None, description="Underlying tool id/name if applicable")

    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_artifacts: List[ArtifactTypeSpec] = Field(default_factory=list)

    risk_level: Literal["low", "medium", "high"] = "medium"
    safety_class: SafetyClass = "reversible"
    irreversible: bool = False
    requires_approval: bool = True

    examples: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class PlanSpec(BaseModel):
    """
    Stable planner→executor interface.

    - Planner produces PlanSpec only (no execution).
    - Executor consumes PlanSpec and performs gated/approved execution.
    """

    status: PlanStatus = "OK"

    plan_id: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = Field(default="tool_rag_planner")

    goal: str = Field(min_length=1)
    deal_id: Optional[str] = None

    steps: List[PlanStep] = Field(default_factory=list)

    # Global plan safety constraints (executor MUST enforce).
    safety_constraints: List[str] = Field(
        default_factory=lambda: ["no_langsmith_tracing", "no_silent_drops"],
    )

    # Error / gating payloads (only set when status != OK)
    blocked_reason: Optional[str] = None
    missing_capability: Optional[MissingCapabilitySpec] = None

    # Debug info for ops (not user-facing).
    debug: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


PlannerOutput = Union[PlanSpec]


def write_plan_spec_schema(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = PlanSpec.model_json_schema()
    out_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")


def default_schema_path() -> Path:
    env = os.getenv("ZAKOPS_PLAN_SPEC_SCHEMA_PATH", "").strip()
    if env:
        return Path(env)
    return Path(__file__).with_suffix(".schema.json")


if __name__ == "__main__":
    write_plan_spec_schema(default_schema_path())
    print(default_schema_path())

