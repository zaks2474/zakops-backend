from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field
from typing_extensions import Literal


RiskLevel = Literal["low", "medium", "high"]


class CapabilityExample(BaseModel):
    # Support both v1.2 example shapes:
    # - {user_intent, expected}
    # - {description, inputs, expected_output}
    user_intent: Optional[str] = None
    expected: Optional[str] = Field(default=None, description="What this capability should do for the user.")
    description: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    expected_output: Optional[str] = None

    model_config = {"extra": "allow"}


class OutputArtifactSpec(BaseModel):
    kind: str = Field(min_length=1, alias="type", description="Logical artifact type, e.g. docx, pdf, xlsx, pptx, md")
    extension: Optional[str] = Field(default=None, description="File extension including dot, e.g. .docx")
    mime_type: str = Field(min_length=1)
    required: bool = True
    description: Optional[str] = None

    model_config = {"extra": "allow", "populate_by_name": True}


class CapabilityManifest(BaseModel):
    capability_id: str = Field(
        min_length=1,
        description="Stable, versioned capability id, e.g. document.generate_loi.v1",
    )
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    action_type: str = Field(min_length=1, description="Namespaced action type, e.g. DOCUMENT.GENERATE_LOI")

    input_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema for inputs")
    output_artifacts: List[OutputArtifactSpec] = Field(default_factory=list)

    risk_level: RiskLevel = "medium"
    requires_approval: bool = Field(default=True, alias="required_approval")

    deterministic_steps: List[str] = Field(default_factory=list)
    llm_allowed: bool = False
    cloud_required: bool = Field(default=False, description="If true, this capability may call cloud LLMs/tools, but only after explicit approval.")

    examples: List[CapabilityExample] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    model_config = {"extra": "allow", "populate_by_name": True}


@dataclass(frozen=True)
class CapabilityMatch:
    capability_id: str
    action_type: str
    score: float
    reason: str


def _default_capabilities_dir() -> Path:
    env_dir = os.getenv("ZAKOPS_CAPABILITIES_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return Path("/home/zaks/scripts/actions/capabilities")


def _tokenize(text: str) -> List[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()
    if not cleaned:
        return []
    return [t for t in cleaned.split() if len(t) > 1]


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


class CapabilityRegistry:
    def __init__(self, capabilities_dir: Optional[Path] = None):
        self.capabilities_dir = capabilities_dir or _default_capabilities_dir()
        self._by_id: Dict[str, CapabilityManifest] = {}
        self._by_action_type: Dict[str, CapabilityManifest] = {}

    def load(self) -> None:
        if not self.capabilities_dir.exists():
            raise FileNotFoundError(f"capabilities_dir_not_found: {self.capabilities_dir}")

        manifests: List[Tuple[Path, CapabilityManifest]] = []
        for path in sorted(self.capabilities_dir.glob("*.y*ml")):
            raw = path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) or {}
            if not isinstance(data, dict):
                raise ValueError(f"invalid_capability_manifest: {path}")
            manifest = CapabilityManifest.model_validate(data)
            manifests.append((path, manifest))

        by_id: Dict[str, CapabilityManifest] = {}
        by_action: Dict[str, CapabilityManifest] = {}
        for path, manifest in manifests:
            if manifest.capability_id in by_id:
                raise ValueError(f"duplicate_capability_id: {manifest.capability_id} ({path})")
            if manifest.action_type in by_action:
                raise ValueError(f"duplicate_action_type_in_capabilities: {manifest.action_type} ({path})")
            by_id[manifest.capability_id] = manifest
            by_action[manifest.action_type] = manifest

        self._by_id = by_id
        self._by_action_type = by_action

    def list_capabilities(self) -> List[CapabilityManifest]:
        return [self._by_id[k] for k in sorted(self._by_id.keys())]

    def get_capability(self, capability_id: str) -> Optional[CapabilityManifest]:
        return self._by_id.get((capability_id or "").strip())

    def get_by_action_type(self, action_type: str) -> Optional[CapabilityManifest]:
        return self._by_action_type.get((action_type or "").strip())

    def match_capability(self, user_intent: str, *, top_k: int = 3) -> List[CapabilityMatch]:
        """
        Deterministic-first matcher (fast, offline).

        This intentionally favors safety and predictability over cleverness:
        - Keyword/synonym boosts for known hero capabilities.
        - Token overlap between the user intent and manifest fields/examples.
        """
        intent = (user_intent or "").strip()
        if not intent:
            return []

        synonyms = {
            "loi": ["loi", "letter", "intent"],
            "pitch_deck": ["deck", "pitch", "presentation", "slides"],
            "valuation_model": ["valuation", "model", "xlsx", "spreadsheet", "multiple", "ebitda"],
            "draft_email": ["email", "draft", "broker", "outreach", "follow", "follow-up"],
            "request_docs": ["docs", "documents", "diligence", "checklist", "request", "materials", "cim"],
        }

        intent_tokens = _tokenize(intent)
        has_email_address = bool(re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}", intent.lower()))
        results: List[CapabilityMatch] = []

        for cap in self._by_id.values():
            example_texts: List[str] = []
            for ex in cap.examples or []:
                parts = [
                    (ex.user_intent or "").strip(),
                    (ex.description or "").strip(),
                    (ex.expected or "").strip(),
                    (ex.expected_output or "").strip(),
                ]
                example_texts.append(" ".join([p for p in parts if p]))

            haystack = " ".join(
                [
                    cap.action_type,
                    cap.title,
                    cap.description,
                    " ".join(example_texts),
                    " ".join(cap.constraints),
                ]
            )
            cap_tokens = _tokenize(haystack)
            overlap = _jaccard(intent_tokens, cap_tokens)

            boost = 0.0
            at = cap.action_type.upper()
            if "GENERATE_LOI" in at:
                boost += 0.25 if any(t in intent_tokens for t in synonyms["loi"]) else 0.0
            if "GENERATE_PITCH_DECK" in at:
                boost += 0.25 if any(t in intent_tokens for t in synonyms["pitch_deck"]) else 0.0
            if "BUILD_VALUATION_MODEL" in at:
                boost += 0.25 if any(t in intent_tokens for t in synonyms["valuation_model"]) else 0.0
            if "DRAFT_EMAIL" in at:
                boost += 0.25 if any(t in intent_tokens for t in synonyms["draft_email"]) else 0.0
                boost += 0.25 if has_email_address else 0.0
            if "REQUEST_DOCS" in at:
                boost += 0.25 if any(t in intent_tokens for t in synonyms["request_docs"]) else 0.0

            score = min(1.0, overlap + boost)
            if score <= 0:
                continue

            results.append(
                CapabilityMatch(
                    capability_id=cap.capability_id,
                    action_type=cap.action_type,
                    score=score,
                    reason=f"token_overlap={overlap:.3f} boost={boost:.2f}",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: max(1, int(top_k))]

    def index_tools(self, tool_registry: Any) -> int:
        """
        Index ToolRegistry manifests as capabilities for discovery and schema-driven UI.

        Tool capabilities use:
        - capability_id: TOOL.<tool_id>.v<major>
        - action_type:  TOOL.<tool_id>
        """
        if tool_registry is None:
            return 0

        added = 0
        for tool in tool_registry.list_tools():
            tool_id = str(getattr(tool, "tool_id", "") or "").strip()
            if not tool_id:
                continue

            major = "1"
            version = str(getattr(tool, "version", "") or "").strip()
            if version:
                major = version.split(".")[0] or "1"

            cap_id = f"TOOL.{tool_id}.v{major}"
            action_type = f"TOOL.{tool_id}"

            if cap_id in self._by_id or action_type in self._by_action_type:
                continue

            risk = str(getattr(tool, "risk_level", "medium") or "medium").strip().lower()
            if risk not in {"low", "medium", "high"}:
                risk = "medium"

            tags = list(getattr(tool, "tags", []) or [])
            provider = str(getattr(tool, "provider", "") or "").strip()
            tags = tags + ["tool"] + ([provider] if provider else [])

            manifest = CapabilityManifest(
                capability_id=cap_id,
                title=str(getattr(tool, "title", tool_id) or tool_id),
                description=str(getattr(tool, "description", "") or ""),
                action_type=action_type,
                input_schema=getattr(tool, "input_schema", {}) or {},
                output_artifacts=[],
                risk_level=risk,  # type: ignore[arg-type]
                requires_approval=bool(getattr(tool, "requires_approval", True)),
                deterministic_steps=[f"Invoke ToolGateway tool_id={tool_id}"],
                llm_allowed=False,
                examples=[],
                constraints=list(getattr(tool, "constraints", []) or []),
                tags=tags,
                is_tool=True,
                tool_id=tool_id,
                provider=provider,
            )

            self._by_id[cap_id] = manifest
            self._by_action_type[action_type] = manifest
            added += 1

        return added

    def to_jsonable(self, capability: CapabilityManifest) -> Dict[str, Any]:
        data = capability.model_dump()
        # Ensure `input_schema` is JSON-safe (YAML may contain non-JSON types).
        data["input_schema"] = json.loads(json.dumps(data.get("input_schema") or {}))
        return data


_REGISTRY: Optional[CapabilityRegistry] = None


def get_registry() -> CapabilityRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        reg = CapabilityRegistry()
        reg.load()
        _REGISTRY = reg
    return _REGISTRY
