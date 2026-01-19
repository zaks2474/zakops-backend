#!/usr/bin/env python3
"""
Chat Orchestrator

Orchestrates chat interactions with the deal lifecycle system.
Uses Evidence Builder to gather context, routes to appropriate LLM, streams responses.

Features:
- Evidence-grounded responses with citations
- SSE streaming with progress events
- Tool proposals with approval gates
- Secret scanning before any cloud sends
- Session management with conversation history
- Performance Mode v1: Timing, caching, deterministic routing
- Hybrid LLM: Provider abstraction with fallback chain
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from chat_evidence_builder import EvidenceBuilder, EvidenceBundle, scan_for_secrets, redact_secrets
from chat_timing import (
    TimingTrace, create_timing, ProgressStep, ProviderName,
    async_timing_context, timing_to_done_event
)
from chat_cache import get_cache, EvidenceCache
from chat_llm_router import ChatLLMRouter, get_router, RoutingDecision, estimate_complexity
from chat_llm_provider import get_provider, get_all_health
from chat_budget import get_budget_manager

# SQLite session persistence
try:
    from email_ingestion.chat_persistence import get_chat_session_store, ChatSessionStore
    PERSISTENCE_ENABLED = os.getenv("CHAT_PERSISTENCE_ENABLED", "true").lower() == "true"
except ImportError:
    PERSISTENCE_ENABLED = False
    ChatSessionStore = None

# Configuration - use same settings as rest of ZakOps
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", f"{OPENAI_API_BASE}/chat/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", os.getenv("DEFAULT_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ"))
VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "120"))

# Allow cloud fallback (disabled by default for air-gapped mode)
ALLOW_CLOUD = os.getenv("ALLOW_CLOUD_DEFAULT", "false").lower() == "true"

# Internal LangGraph “brain” (zakops-api :8080) integration
ZAKOPS_BRAIN_URL = os.getenv("ZAKOPS_BRAIN_URL", "http://localhost:8080").rstrip("/")
ZAKOPS_BRAIN_TIMEOUT_S = float(os.getenv("ZAKOPS_BRAIN_TIMEOUT_S", os.getenv("ZAKOPS_BRAIN_TIMEOUT", "30")))

# Feature flags
CACHE_ENABLED = os.getenv("CHAT_CACHE_ENABLED", "true").lower() == "true"
DETERMINISTIC_EXTENDED = os.getenv("CHAT_DETERMINISTIC_EXTENDED", "true").lower() == "true"

# System prompt template
SYSTEM_PROMPT = """You are ZakOps Assistant, an AI helping manage deal lifecycle operations.

## Your Role
- Answer questions about deals using the provided evidence
- Suggest actions when appropriate (but NEVER auto-execute)
- Ground all responses in the evidence provided
- Be concise and actionable

## Evidence Available
The following evidence has been gathered for this query:

{evidence_context}

## Citation Format
When referencing evidence, use citation markers like [cite-1], [cite-2], etc.
Always ground your responses in the provided evidence.

## Proposals
If you suggest an action that would modify the system, format it as a PROPOSAL:

```proposal
{"type":"add_note","deal_id":"DEAL-2025-008","params":{"content":"...","category":"chat_note"},"reason":"Why this note matters"}
```

IMPORTANT: Proposals require user approval. Never claim an action has been taken.

STRICT FORMAT RULES:
- The content inside the ```proposal``` block MUST be STRICT JSON (not YAML).
- Do NOT invent placeholder deal IDs like "DEAL-2025-XXX". If you don't know the deal_id, ask the user to select a deal (do not emit a proposal).
- Valid proposal `type` values: add_note, create_task, create_action, draft_email, request_docs, stage_transition.
- `params` MUST be a JSON object. If you need complex inputs, put them under `params.inputs` as JSON (not a string).

Proposal examples:

```proposal
{"type":"stage_transition","deal_id":"DEAL-2025-008","params":{"to_stage":"qualified"},"reason":"NDA signed and CIM received; ready to advance."}
```

```proposal
{"type":"create_task","deal_id":"DEAL-2025-008","params":{"description":"Follow up with broker","due_days":2,"priority":"normal"},"reason":"No response in 48h."}
```

```proposal
{"type":"create_action","deal_id":"DEAL-2025-008","params":{"action_type":"DILIGENCE.REQUEST_DOCS","capability_id":"diligence.request_docs.v1","title":"Request CIM/financials","inputs":{"doc_type":"cim","description":"Please share CIM + LTM financials + customer concentration."}},"reason":"Need baseline materials to underwrite."}
```

```proposal
{"type":"draft_email","deal_id":"DEAL-2025-008","params":{"recipient":"broker@example.com","subject":"Request for CIM","context":"Politely request CIM and LTM financials; do not send automatically."},"reason":"Prepare a draft email for operator review."}
```

## Guidelines
- Be concise - operators are busy
- Cite your sources using [cite-N] format
- If unsure, say so rather than guessing
- Never fabricate information not in the evidence
"""


def _render_system_prompt(*, evidence_context: str) -> str:
    # Avoid `str.format()` because the prompt includes literal braces in examples (e.g. `{ ... }`).
    return SYSTEM_PROMPT.replace("{evidence_context}", evidence_context or "")

CANONICAL_PROPOSAL_TYPES = {
    "add_note",
    "create_task",
    "create_action",
    "draft_email",
    "request_docs",
    "stage_transition",
}

# Normalization mapping (LLM/UI/back-compat)
PROPOSAL_TYPE_ALIASES = {
    "schedule_action": "create_task",
    "schedule-task": "create_task",
    "schedule action": "create_task",
    "create-action": "create_action",
    "create action": "create_action",
}

def _brain_mode() -> str:
    return (os.getenv("ZAKOPS_BRAIN_MODE", "off") or "off").strip().lower()


def _brain_enabled() -> bool:
    return _brain_mode() in {"1", "true", "yes", "on", "auto", "force"}


def _strip_wrapping_quotes(value: str) -> str:
    v = (value or "").strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1].strip()
    return v


def canonicalize_proposal_type(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None

    # Normalize common separators and whitespace.
    text_norm = text.replace("-", "_").replace(" ", "_")
    text_norm = PROPOSAL_TYPE_ALIASES.get(text_norm, PROPOSAL_TYPE_ALIASES.get(text, text_norm))

    # Handle cases like: "stage_transition | add_note" or "schedule_action / create_task"
    for part in re.split(r"[|/,\n]+", text_norm):
        cand = part.strip().replace("-", "_").replace(" ", "_")
        cand = PROPOSAL_TYPE_ALIASES.get(cand, cand)
        if cand in CANONICAL_PROPOSAL_TYPES:
            return cand

    return text_norm if text_norm in CANONICAL_PROPOSAL_TYPES else None


def _looks_like_placeholder_deal_id(value: Any) -> bool:
    v = str(value or "").strip().upper()
    if not v:
        return False
    # Common placeholder used in prompts/examples.
    if v.endswith("-XXX") or v.endswith("_XXX"):
        return True
    if "DEAL-YYYY" in v or "DEAL-XXXX" in v:
        return True
    return False


@dataclass
class ChatMessage:
    """A chat message."""
    role: str  # user, assistant, system
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    citations: List[str] = field(default_factory=list)
    proposals: List[Dict] = field(default_factory=list)


@dataclass
class ChatSession:
    """A chat session with history."""
    session_id: str
    scope: Dict[str, Any]
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_message(self, role: str, content: str, **kwargs) -> ChatMessage:
        msg = ChatMessage(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self.last_activity = datetime.now(timezone.utc).isoformat()
        return msg

    def get_history_for_llm(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get message history in OpenAI format."""
        history = []
        for msg in self.messages[-max_messages:]:
            if msg.role in ("user", "assistant"):
                history.append({"role": msg.role, "content": msg.content})
        return history


@dataclass
class ChatResponse:
    """A chat response."""
    content: str
    citations: List[Dict] = field(default_factory=list)
    proposals: List[Dict] = field(default_factory=list)
    evidence_summary: Optional[Dict] = None
    model_used: str = ""
    latency_ms: int = 0
    warnings: List[str] = field(default_factory=list)


class ChatOrchestrator:
    """Orchestrates chat interactions with hybrid LLM routing."""

    def __init__(self, allow_cloud: Optional[bool] = None):
        self.evidence_builder = EvidenceBuilder()
        self.sessions: Dict[str, ChatSession] = {}
        self.cache = get_cache() if CACHE_ENABLED else None
        self.router = get_router(allow_cloud=allow_cloud if allow_cloud is not None else ALLOW_CLOUD)
        self.budget = get_budget_manager()
        # SQLite-backed session persistence (survives restarts)
        self.session_store = get_chat_session_store() if PERSISTENCE_ENABLED else None

    def _is_send_email_intent(self, query: str) -> bool:
        q = (query or "").strip().lower()
        if not q:
            return False
        # Keep this conservative: only trigger when the user explicitly says "send".
        return bool(re.search(r"\bsend\b", q))

    def _extract_body_from_markdown(self, content: str) -> str:
        text = (content or "").strip()
        if not text:
            return ""
        lines = text.splitlines()
        delim = [i for i, ln in enumerate(lines) if ln.strip() == "---"]
        if len(delim) >= 2 and delim[0] < delim[1]:
            body = "\n".join(lines[delim[0] + 1 : delim[1]]).strip()
            if body:
                return body
        for i, ln in enumerate(lines):
            if i > 0 and not ln.strip():
                body = "\n".join(lines[i + 1 :]).strip()
                if body:
                    return body
        return text

    def _parse_email_draft_artifact(self, path: str) -> Optional[Dict[str, str]]:
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists() or not p.is_file():
                return None
            raw = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        to = ""
        subject = ""
        m_to = re.search(r"^\\s*To:\\s*(.+?)\\s*$", raw, re.M)
        if not m_to:
            m_to = re.search(r"^\\s*\\*\\*To:\\*\\*\\s*(.+?)\\s*$", raw, re.M)
        if m_to:
            to = m_to.group(1).strip()

        m_sub = re.search(r"^\\s*Subject:\\s*(.+?)\\s*$", raw, re.M)
        if not m_sub:
            m_sub = re.search(r"^\\s*\\*\\*Subject:\\*\\*\\s*(.+?)\\s*$", raw, re.M)
        if m_sub:
            subject = m_sub.group(1).strip()

        body = self._extract_body_from_markdown(raw)
        if not body:
            return None

        out: Dict[str, str] = {"body": body, "body_artifact_path": str(p)}
        if to:
            out["to"] = to
        if subject:
            out["subject"] = subject
        return out

    def _find_latest_email_draft_for_send(self, *, deal_id: str, session: ChatSession) -> Optional[Dict[str, Any]]:
        """
        Find the most recent email draft (artifact preferred) to use for a SEND_EMAIL action.

        Priority:
        1) Latest executed create_action proposal that produced a kinetic action_id (draft types)
        2) Latest completed action in ActionStore for this deal (draft types)
        3) Latest executed legacy draft_email proposal result (subject/body in-memory)
        """
        # 1) Session: create_action results that created an action_id for draft types.
        try:
            for msg in reversed(session.messages or []):
                for p in reversed(getattr(msg, "proposals", []) or []):
                    if canonicalize_proposal_type(p.get("type")) != "create_action":
                        continue
                    if str(p.get("status") or "").strip().lower() != "executed":
                        continue
                    result = p.get("result") or {}
                    action_id = (result.get("action_id") or "").strip()
                    action_type = (result.get("action_type") or "").strip().upper()
                    if not action_id:
                        continue
                    if action_type not in {"COMMUNICATION.DRAFT_EMAIL", "DILIGENCE.REQUEST_DOCS"}:
                        continue
                    from actions.engine.store import ActionStore

                    store = ActionStore()
                    action = store.get_action(action_id)
                    if not action or action.status != "COMPLETED":
                        continue
                    for art in action.artifacts or []:
                        if not str(getattr(art, "filename", "")).lower().endswith(".md"):
                            continue
                        parsed = self._parse_email_draft_artifact(str(getattr(art, "path", "")))
                        if parsed:
                            return parsed
        except Exception:
            pass

        # 2) ActionStore scan by deal_id
        try:
            from actions.engine.store import ActionStore

            store = ActionStore()
            candidates = store.list_actions(deal_id=deal_id, status="COMPLETED", limit=25)
            for a in candidates:
                if (a.type or "").upper() not in {"COMMUNICATION.DRAFT_EMAIL", "DILIGENCE.REQUEST_DOCS"}:
                    continue
                full = store.get_action(a.action_id)
                if not full:
                    continue
                for art in full.artifacts or []:
                    name = str(getattr(art, "filename", "") or "").lower()
                    if not name.endswith(".md"):
                        continue
                    if "draft" not in name:
                        continue
                    parsed = self._parse_email_draft_artifact(str(getattr(art, "path", "")))
                    if parsed:
                        return parsed
        except Exception:
            pass

        # 3) Legacy draft_email proposal result.
        try:
            for msg in reversed(session.messages or []):
                for p in reversed(getattr(msg, "proposals", []) or []):
                    if canonicalize_proposal_type(p.get("type")) != "draft_email":
                        continue
                    if str(p.get("status") or "").strip().lower() != "executed":
                        continue
                    params = p.get("params") or {}
                    result = p.get("result") or {}
                    recipient = str(params.get("recipient", params.get("to", "")) or "").strip()
                    subject = str(result.get("subject", params.get("subject", "")) or "").strip()
                    body = str(result.get("body", "") or "").strip()
                    if recipient and subject and body:
                        return {"to": recipient, "subject": subject, "body": body}
        except Exception:
            pass

        return None

    async def _try_send_email_flow(self, *, query: str, scope: Dict[str, Any], session: ChatSession) -> Optional[ChatResponse]:
        """
        Deterministic bridge: if the user asks to "send" and we can locate a prior draft,
        propose a SEND_EMAIL action instead of re-drafting.
        """
        if not self._is_send_email_intent(query):
            return None
        if (scope or {}).get("type") != "deal":
            return None
        deal_id = (scope or {}).get("deal_id")
        if not deal_id:
            return None

        draft = self._find_latest_email_draft_for_send(deal_id=deal_id, session=session)
        if not draft:
            return None

        inputs: Dict[str, Any] = {}
        if draft.get("body_artifact_path"):
            inputs["body_artifact_path"] = draft["body_artifact_path"]
        else:
            inputs["body"] = draft.get("body", "")
        if draft.get("to"):
            inputs["to"] = draft["to"]
        if draft.get("subject"):
            inputs["subject"] = draft["subject"]

        proposal = {
            "type": "create_action",
            "deal_id": deal_id,
            "params": {
                "capability_id": "communication.send_email.v1",
                "action_type": "COMMUNICATION.SEND_EMAIL",
                "title": "Send drafted email",
                "summary": "Send the latest drafted email via ToolGateway (requires approval).",
                "risk_level": "high",
                "inputs": inputs,
            },
            "reason": "You asked to send the email; proposing a SEND_EMAIL action referencing the latest draft.",
        }

        return ChatResponse(
            content="I found the latest email draft for this deal. I can send it as an approval-gated action.",
            proposals=[proposal],
            citations=[],
            warnings=[],
            model_used="deterministic-send-email",
        )

    def get_or_create_session(
        self,
        session_id: Optional[str],
        scope: Dict[str, Any]
    ) -> ChatSession:
        """Get existing session or create new one (with SQLite persistence)."""
        # Check in-memory cache first
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            # Update scope if changed
            session.scope = scope
            return session

        # Try loading from SQLite if not in memory
        if session_id and self.session_store:
            persisted = self.session_store.load_session(session_id)
            if persisted:
                # Reconstruct ChatSession from persisted data
                session = ChatSession(
                    session_id=persisted.session_id,
                    scope={
                        "type": persisted.scope_type,
                        "deal_id": persisted.scope_deal_id,
                        "doc_url": persisted.scope_doc_url,
                    },
                    created_at=persisted.created_at,
                    last_activity=persisted.last_activity,
                )
                # Restore messages
                for msg in persisted.messages:
                    session.messages.append(ChatMessage(
                        role=msg.role,
                        content=msg.content,
                        timestamp=msg.timestamp,
                        citations=msg.citations,
                        proposals=msg.proposals,
                    ))
                # Cache in memory
                self.sessions[session_id] = session
                return session

        # Create new session
        new_id = session_id or str(uuid.uuid4())[:8]
        session = ChatSession(session_id=new_id, scope=scope)
        self.sessions[new_id] = session

        # Persist new session to SQLite
        if self.session_store:
            self.session_store.save_session(new_id, scope)

        return session

    def _persist_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: Optional[List] = None,
        proposals: Optional[List] = None,
        timings: Optional[Dict] = None,
        provider_used: Optional[str] = None,
        cache_hit: bool = False
    ) -> None:
        """Persist a message to SQLite."""
        if self.session_store:
            self.session_store.add_message(
                session_id=session_id,
                role=role,
                content=content,
                citations=citations,
                proposals=proposals,
                timings=timings,
                provider_used=provider_used,
                cache_hit=cache_hit,
            )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a session by ID (for API access).

        Returns session data as a dict, or None if not found.
        Checks in-memory cache first, then SQLite.
        """
        # Check in-memory first
        if session_id in self.sessions:
            session = self.sessions[session_id]
            return {
                "session_id": session.session_id,
                "scope": session.scope,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "citations": msg.citations,
                        "proposals": msg.proposals,
                    }
                    for msg in session.messages
                ],
            }

        # Try loading from SQLite
        if self.session_store:
            persisted = self.session_store.load_session(session_id)
            if persisted:
                return {
                    "session_id": persisted.session_id,
                    "scope": {
                        "type": persisted.scope_type,
                        "deal_id": persisted.scope_deal_id,
                        "doc_url": persisted.scope_doc_url,
                    },
                    "created_at": persisted.created_at,
                    "last_activity": persisted.last_activity,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp,
                            "citations": msg.citations,
                            "proposals": msg.proposals,
                            "timings": msg.timings,
                            "provider_used": msg.provider_used,
                            "cache_hit": msg.cache_hit,
                        }
                        for msg in persisted.messages
                    ],
                }

        return None

    def get_recent_sessions(
        self,
        limit: int = 50,
        scope_type: Optional[str] = None,
        deal_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recently active sessions for the dashboard."""
        if self.session_store:
            return self.session_store.get_recent_sessions(
                limit=limit,
                scope_type=scope_type,
                deal_id=deal_id
            )
        return []

    async def _try_simple_query(self, query: str, scope: Dict[str, Any]) -> Optional[ChatResponse]:
        """
        Try to answer simple queries without LLM for faster response.

        Expanded deterministic patterns:
        - Deal counts (total, by stage, by broker)
        - Deal stage/status lookup
        - Deals in specific stage
        - Deals by broker
        - Deals stuck > N days
        - Actions due
        - What changed today
        """
        from collections import Counter
        from datetime import datetime, timedelta

        query_lower = query.lower().strip()
        API_BASE = "http://localhost:8090"

        # Helper to create deterministic response
        def make_response(content: str, sources: list = None, extra: dict = None) -> ChatResponse:
            evidence = {"sources_queried": sources or ["api"], "method": "deterministic"}
            if extra:
                evidence.update(extra)
            return ChatResponse(
                content=content,
                model_used="direct-api",
                latency_ms=0,
                evidence_summary=evidence
            )

        # Helper to fetch deals
        async def fetch_deals() -> tuple:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{API_BASE}/api/deals")
                if resp.status_code == 200:
                    data = resp.json()
                    deals = data.get("deals", data) if isinstance(data, dict) else data
                    total = data.get("count", len(deals)) if isinstance(data, dict) else len(deals)
                    return deals, total
            return [], 0

        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 1: Deal counts (how many, total, count, list)
        # ═══════════════════════════════════════════════════════════════════════
        deal_count_patterns = ["how many deals", "count deals", "total deals", "number of deals",
                               "deals total", "deals count", "deal count", "list deals"]
        if any(p in query_lower for p in deal_count_patterns):
            try:
                deals, total = await fetch_deals()
                if deals:
                    # Count by stage if asked
                    if "stage" in query_lower or "by stage" in query_lower:
                        stages = Counter(d.get("stage", "unknown") for d in deals)
                        stage_list = ", ".join(f"{s}: {c}" for s, c in sorted(stages.items()))
                        return make_response(f"There are **{total} deals** total. By stage: {stage_list}")
                    # Count by broker if asked
                    if "broker" in query_lower or "by broker" in query_lower:
                        brokers = Counter(d.get("broker", "Unknown") or "Unknown" for d in deals)
                        broker_list = ", ".join(f"{b}: {c}" for b, c in brokers.most_common(10))
                        return make_response(f"There are **{total} deals** total. Top brokers: {broker_list}")
                    return make_response(f"There are **{total} deals** in the system.")
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 2: Deal stage lookup (what stage is DEAL-X in)
        # ═══════════════════════════════════════════════════════════════════════
        deal_match = re.search(r'deal[- ]?(\d{4}[- ]?\d{3})', query_lower)
        if deal_match and any(p in query_lower for p in ["stage", "status", "what is", "show me", "info"]):
            deal_id = f"DEAL-{deal_match.group(1).replace(' ', '-').replace('--', '-')}"
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"{API_BASE}/api/deals/{deal_id}")
                    if resp.status_code == 200:
                        deal = resp.json()
                        broker = deal.get("broker") or "Unknown"
                        return make_response(
                            f"**{deal_id}** ({deal.get('canonical_name', 'Unknown')})\n"
                            f"- Stage: **{deal.get('stage', 'unknown')}**\n"
                            f"- Status: **{deal.get('status', 'unknown')}**\n"
                            f"- Broker: {broker}\n"
                            f"- Priority: {deal.get('priority', 'N/A')}",
                            extra={"deal_id": deal_id}
                        )
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 3: Deals in specific stage (deals in screening, show me inbound deals)
        # ═══════════════════════════════════════════════════════════════════════
        stage_match = re.search(r'deals?\s+in\s+(\w+)(?:\s+stage)?', query_lower)
        if not stage_match:
            stage_match = re.search(r'show\s+(?:me\s+)?(\w+)\s+deals?', query_lower)
        if stage_match:
            target_stage = stage_match.group(1).lower()
            # Valid stages
            valid_stages = ["inbound", "screening", "qualified", "diligence", "loi", "closing", "closed", "dead"]
            if target_stage in valid_stages:
                try:
                    deals, _ = await fetch_deals()
                    filtered = [d for d in deals if d.get("stage", "").lower() == target_stage]
                    if filtered:
                        deal_lines = []
                        for d in filtered[:15]:  # Limit to 15
                            broker = d.get("broker") or "No broker"
                            deal_lines.append(f"- **{d['deal_id']}**: {d.get('canonical_name', 'Unknown')} ({broker})")
                        more_text = f"\n\n...and {len(filtered) - 15} more" if len(filtered) > 15 else ""
                        return make_response(
                            f"**{len(filtered)} deals in {target_stage} stage:**\n\n" + "\n".join(deal_lines) + more_text,
                            extra={"stage_filter": target_stage, "count": len(filtered)}
                        )
                    else:
                        return make_response(f"No deals found in **{target_stage}** stage.")
                except Exception:
                    pass

        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 4: Deals by broker (deals from Eric, Eric's deals)
        # ═══════════════════════════════════════════════════════════════════════
        broker_match = re.search(r'deals?\s+(?:from|by|with)\s+(\w+)', query_lower)
        if not broker_match:
            broker_match = re.search(r"(\w+)'s\s+deals?", query_lower)
        if broker_match:
            target_broker = broker_match.group(1).lower()
            try:
                deals, _ = await fetch_deals()
                filtered = [d for d in deals if target_broker in (d.get("broker") or "").lower()]
                if filtered:
                    deal_lines = []
                    for d in filtered[:10]:
                        deal_lines.append(f"- **{d['deal_id']}**: {d.get('canonical_name', 'Unknown')} ({d.get('stage', 'unknown')})")
                    more_text = f"\n\n...and {len(filtered) - 10} more" if len(filtered) > 10 else ""
                    return make_response(
                        f"**{len(filtered)} deals from broker matching '{target_broker}':**\n\n" + "\n".join(deal_lines) + more_text,
                        extra={"broker_filter": target_broker, "count": len(filtered)}
                    )
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 5: Deals stuck > N days (stale deals, deals without updates)
        # ═══════════════════════════════════════════════════════════════════════
        stuck_match = re.search(r'deals?\s+stuck.*?(\d+)\s*days?', query_lower)
        if not stuck_match and any(p in query_lower for p in ["stale deals", "deals without updates", "old deals"]):
            stuck_match = type('obj', (object,), {'group': lambda self, x: '7'})()  # Default 7 days
        if stuck_match:
            try:
                days_threshold = int(stuck_match.group(1))
                deals, _ = await fetch_deals()
                now = datetime.now()
                stale = []
                for d in deals:
                    updated = d.get("updated_at")
                    if updated:
                        try:
                            updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00")).replace(tzinfo=None)
                            days_old = (now - updated_dt).days
                            if days_old >= days_threshold:
                                stale.append((d, days_old))
                        except:
                            pass
                stale.sort(key=lambda x: x[1], reverse=True)
                if stale:
                    deal_lines = []
                    for d, days in stale[:10]:
                        deal_lines.append(f"- **{d['deal_id']}**: {d.get('canonical_name', 'Unknown')} - {days} days since update ({d.get('stage', '?')})")
                    more_text = f"\n\n...and {len(stale) - 10} more" if len(stale) > 10 else ""
                    return make_response(
                        f"**{len(stale)} deals stuck for {days_threshold}+ days:**\n\n" + "\n".join(deal_lines) + more_text,
                        extra={"days_threshold": days_threshold, "count": len(stale)}
                    )
                else:
                    return make_response(f"No deals have been stuck for {days_threshold}+ days.")
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 6: Actions due (actions due, pending actions, what's due)
        # ═══════════════════════════════════════════════════════════════════════
        actions_patterns = ["actions due", "pending actions", "what's due", "whats due",
                           "upcoming actions", "due actions", "actions this week"]
        if any(p in query_lower for p in actions_patterns):
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(f"{API_BASE}/api/deferred-actions")
                    if resp.status_code == 200:
                        data = resp.json()
                        actions = data if isinstance(data, list) else data.get("actions", [])
                        # Filter pending actions
                        pending = [a for a in actions if a.get("status") == "pending"]
                        if pending:
                            # Sort by due date
                            now = datetime.now()
                            categorized = {"overdue": [], "today": [], "this_week": [], "later": []}
                            for a in pending:
                                due = a.get("due_date")
                                if due:
                                    try:
                                        due_dt = datetime.fromisoformat(due.replace("Z", "+00:00")).replace(tzinfo=None)
                                        delta = (due_dt - now).days
                                        if delta < 0:
                                            categorized["overdue"].append(a)
                                        elif delta == 0:
                                            categorized["today"].append(a)
                                        elif delta <= 7:
                                            categorized["this_week"].append(a)
                                        else:
                                            categorized["later"].append(a)
                                    except:
                                        categorized["later"].append(a)

                            lines = []
                            if categorized["overdue"]:
                                lines.append(f"**Overdue ({len(categorized['overdue'])}):**")
                                for a in categorized["overdue"][:5]:
                                    lines.append(f"- {a.get('deal_id')}: {a.get('action_type')} - {a.get('description', '')[:50]}")
                            if categorized["today"]:
                                lines.append(f"\n**Due Today ({len(categorized['today'])}):**")
                                for a in categorized["today"][:5]:
                                    lines.append(f"- {a.get('deal_id')}: {a.get('action_type')} - {a.get('description', '')[:50]}")
                            if categorized["this_week"]:
                                lines.append(f"\n**Due This Week ({len(categorized['this_week'])}):**")
                                for a in categorized["this_week"][:5]:
                                    lines.append(f"- {a.get('deal_id')}: {a.get('action_type')} - {a.get('description', '')[:50]}")

                            return make_response(
                                f"**{len(pending)} pending actions:**\n\n" + "\n".join(lines),
                                sources=["actions"],
                                extra={"pending_count": len(pending), "overdue_count": len(categorized["overdue"])}
                            )
                        else:
                            return make_response("No pending actions found.", sources=["actions"])
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 7: What changed today (today's changes, recent activity)
        # ═══════════════════════════════════════════════════════════════════════
        changes_patterns = ["what changed today", "today's changes", "changes today",
                           "recent activity", "what happened today", "today's activity"]
        if any(p in query_lower for p in changes_patterns):
            try:
                deals, _ = await fetch_deals()
                now = datetime.now()
                today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

                changed_today = []
                for d in deals:
                    updated = d.get("updated_at")
                    if updated:
                        try:
                            updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00")).replace(tzinfo=None)
                            if updated_dt >= today_start:
                                changed_today.append((d, updated_dt))
                        except:
                            pass

                changed_today.sort(key=lambda x: x[1], reverse=True)
                if changed_today:
                    lines = []
                    for d, updated_dt in changed_today[:10]:
                        time_str = updated_dt.strftime("%H:%M")
                        lines.append(f"- [{time_str}] **{d['deal_id']}**: {d.get('canonical_name', 'Unknown')} ({d.get('stage', '?')})")
                    more_text = f"\n\n...and {len(changed_today) - 10} more" if len(changed_today) > 10 else ""
                    return make_response(
                        f"**{len(changed_today)} deals updated today:**\n\n" + "\n".join(lines) + more_text,
                        extra={"changed_count": len(changed_today), "date": now.strftime("%Y-%m-%d")}
                    )
                else:
                    return make_response(f"No deals have been updated today ({now.strftime('%Y-%m-%d')}).")
            except Exception:
                pass

        # ═══════════════════════════════════════════════════════════════════════
        # PATTERN 8: Deal summary (what's this deal about, tell me about this deal)
        # Works when in deal scope or when deal_id is mentioned
        # ═══════════════════════════════════════════════════════════════════════
        deal_summary_patterns = [
            # Patterns with "this"
            "what's this deal about", "whats this deal about",
            "what is this deal about", "tell me about this deal",
            "what's this deal", "whats this deal",
            "summarize this deal", "overview of this deal",
            "what do we know about this deal",
            # Patterns without "this" (user's actual phrasing)
            "what's deal about", "whats deal about",
            "what is deal about", "what is the deal about",
            "tell me about deal", "tell me about the deal",
            "summarize deal", "summarize the deal",
            "deal summary", "deal overview",
            "about the deal", "about deal"
        ]

        # Check for deal ID from scope or query
        scope_deal_id = scope.get("deal_id") if scope.get("type") == "deal" else None
        query_deal_match = re.search(r'deal[- ]?(\d{4}[- ]?\d{3})', query_lower)
        target_deal_id = None

        if query_deal_match:
            target_deal_id = f"DEAL-{query_deal_match.group(1).replace(' ', '-').replace('--', '-')}"
        elif scope_deal_id and any(p in query_lower for p in deal_summary_patterns):
            target_deal_id = scope_deal_id
        elif scope_deal_id and any(p in query_lower for p in ["about this deal", "this deal about", "deal about", "about deal"]):
            target_deal_id = scope_deal_id

        if target_deal_id and any(p in query_lower for p in deal_summary_patterns + ["about", "what is", "tell me", "summarize"]):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    # Fetch deal details
                    deal_resp = await client.get(f"{API_BASE}/api/deals/{target_deal_id}")
                    if deal_resp.status_code != 200:
                        return None  # Fall back to LLM

                    deal = deal_resp.json()

                    # Fetch recent events (last 5)
                    events_resp = await client.get(f"{API_BASE}/api/deals/{target_deal_id}/events?limit=5")
                    events = []
                    if events_resp.status_code == 200:
                        events_data = events_resp.json()
                        events = events_data if isinstance(events_data, list) else events_data.get("events", [])

                    # Fetch case file summary
                    case_resp = await client.get(f"{API_BASE}/api/deals/{target_deal_id}/case-file")
                    case_file = {}
                    if case_resp.status_code == 200:
                        case_file = case_resp.json()

                    # Fetch pending actions
                    actions_resp = await client.get(f"{API_BASE}/api/deferred-actions?deal_id={target_deal_id}")
                    pending_actions = []
                    if actions_resp.status_code == 200:
                        actions_data = actions_resp.json()
                        actions_list = actions_data if isinstance(actions_data, list) else actions_data.get("actions", [])
                        pending_actions = [a for a in actions_list if a.get("status") == "pending"]

                    # Build comprehensive summary
                    lines = []
                    lines.append(f"## {target_deal_id}: {deal.get('canonical_name', 'Unknown')}\n")

                    # Basic info
                    lines.append(f"**Stage:** {deal.get('stage', 'unknown')} | **Status:** {deal.get('status', 'unknown')}")
                    lines.append(f"**Broker:** {deal.get('broker') or 'Unknown'} | **Priority:** {deal.get('priority', 'N/A')}")

                    # Financial info if available
                    if deal.get("revenue") or deal.get("ebitda"):
                        revenue = deal.get("revenue", "TBD")
                        ebitda = deal.get("ebitda", "TBD")
                        lines.append(f"**Revenue:** {revenue} | **EBITDA:** {ebitda}")

                    # Case file notes
                    if case_file.get("notes"):
                        notes = case_file.get("notes", [])[-3:]  # Last 3 notes
                        if notes:
                            lines.append("\n### Recent Notes")
                            for note in notes:
                                note_text = note.get("content", "")[:100]
                                lines.append(f"- {note_text}...")

                    # Recent events
                    if events:
                        lines.append("\n### Recent Activity")
                        for e in events[:5]:
                            event_type = e.get("event_type", "unknown")
                            event_date = e.get("timestamp", "")[:10]
                            lines.append(f"- [{event_date}] {event_type}")

                    # Pending actions
                    if pending_actions:
                        lines.append(f"\n### Pending Actions ({len(pending_actions)})")
                        for a in pending_actions[:3]:
                            lines.append(f"- {a.get('action_type', 'action')}: {a.get('description', '')[:50]}")

                    return make_response(
                        "\n".join(lines),
                        sources=["deal", "events", "case-file", "actions"],
                        extra={"deal_id": target_deal_id, "method": "deterministic-summary"}
                    )
            except Exception:
                pass  # Fall back to LLM

        return None  # Not a deterministic query, use LLM

    async def chat(
        self,
        query: str,
        scope: Dict[str, Any],
        session_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """Process a chat query and return complete response."""
        options = options or {}
        start_time = time.time()

        # Get or create session
        session = self.get_or_create_session(session_id, scope)
        session.add_message("user", query)
        self._persist_message(session.session_id, "user", query)

        # Deterministic bridge: "send it" should create a SEND_EMAIL action proposal (not re-draft).
        send_flow = await self._try_send_email_flow(query=query, scope=scope, session=session)
        if send_flow:
            send_flow.latency_ms = int((time.time() - start_time) * 1000)
            session.add_message(
                "assistant",
                send_flow.content,
                citations=[c["id"] for c in send_flow.citations] if send_flow.citations else [],
                proposals=send_flow.proposals or [],
            )
            self._persist_message(
                session.session_id,
                "assistant",
                send_flow.content,
                citations=[c["id"] for c in send_flow.citations] if send_flow.citations else [],
                proposals=send_flow.proposals or [],
                provider_used=send_flow.model_used or "deterministic",
            )
            return send_flow

        # Try simple query first for faster response
        simple_response = await self._try_simple_query(query, scope)
        if simple_response:
            simple_response.latency_ms = int((time.time() - start_time) * 1000)
            session.add_message("assistant", simple_response.content)
            self._persist_message(
                session.session_id, "assistant", simple_response.content,
                provider_used="deterministic"
            )
            return simple_response

        # Build evidence
        bundle = await self.evidence_builder.build(query, scope, options)
        evidence_context = bundle.get_context_for_llm()

        # Build prompt (legacy local path) + session history
        system_prompt = _render_system_prompt(evidence_context=evidence_context)
        history = session.get_history_for_llm(max_messages=6)
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
        ]

        # Call LangGraph brain first (optional), fallback to legacy local routing.
        response_text = ""
        model_used = ""
        warnings: List[str] = []
        brain_proposals: List[Dict[str, Any]] = []

        if _brain_enabled():
            brain_text, brain_props, brain_model, brain_warnings = await self._call_brain_complete(
                query=query,
                scope=scope,
                session_id=session.session_id,
                history=history,
                evidence_context=evidence_context,
                options=options,
            )
            warnings.extend(brain_warnings)
            if brain_text:
                response_text = brain_text
                brain_proposals = brain_props
                model_used = brain_model
            elif _brain_mode() == "force":
                response_text = "ZakOps brain is unavailable (ZAKOPS_BRAIN_MODE=force)."
                model_used = "zakops-api"

        if not response_text:
            response_text, model_used, local_warnings = await self._call_llm(messages, stream=False)
            warnings.extend(local_warnings)

        # Parse response for citations and proposals
        citations = self._extract_citations(response_text, bundle)
        proposals_raw = brain_proposals or self._extract_proposals(response_text)
        proposals = self._normalize_proposals_for_scope(proposals_raw, scope)

        # Add response to session
        session.add_message(
            "assistant",
            response_text,
            citations=[c["id"] for c in citations],
            proposals=proposals
        )
        self._persist_message(
            session.session_id, "assistant", response_text,
            citations=[c["id"] for c in citations],
            proposals=proposals,
            provider_used=model_used,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            content=response_text,
            citations=citations,
            proposals=proposals,
            evidence_summary=bundle.get_evidence_summary(),
            model_used=model_used,
            latency_ms=latency_ms,
            warnings=bundle.warnings + warnings,
        )

    async def _call_brain_complete(
        self,
        *,
        query: str,
        scope: Dict[str, Any],
        session_id: str,
        history: List[Dict[str, str]],
        evidence_context: str,
        options: Dict[str, Any],
    ) -> Tuple[Optional[str], List[Dict[str, Any]], str, List[str]]:
        """
        Call the internal LangGraph engine (zakops-api :8080) in non-streaming mode.

        Returns: (final_text | None, proposals, model_used, warnings)
        """
        warnings: List[str] = []
        if not _brain_enabled():
            return None, [], "", warnings

        url = f"{ZAKOPS_BRAIN_URL}/api/deal-chat"
        payload = {
            "query": query,
            "scope": {
                "type": (scope or {}).get("type", "global"),
                "deal_id": (scope or {}).get("deal_id"),
                "doc": (scope or {}).get("doc"),
            },
            "session_id": session_id,
            "options": {
                **(options or {}),
                # Provide context to the brain; it remains stateless.
                "history": history,
                "evidence_context": evidence_context,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=ZAKOPS_BRAIN_TIMEOUT_S) as client:
                resp = await client.post(url, json=payload)
        except Exception as exc:
            warnings.append(f"brain_unavailable: {str(exc)}")
            return None, [], "", warnings

        if resp.status_code != 200:
            warnings.append(f"brain_http_{resp.status_code}")
            return None, [], "", warnings

        try:
            data = resp.json()
        except Exception:
            warnings.append("brain_invalid_json")
            return None, [], "", warnings

        final_text = str(data.get("final_text") or "").strip()
        proposals = data.get("proposals") if isinstance(data.get("proposals"), list) else []
        debug = data.get("debug") if isinstance(data.get("debug"), dict) else {}

        provider = str(debug.get("provider") or "zakops-api").strip()
        model = str(debug.get("model") or "").strip()
        model_used = f"zakops-api:{provider}{('/' + model) if model else ''}"

        if not final_text:
            warnings.append("brain_empty_final_text")
            return None, [], model_used, warnings

        return final_text, list(proposals), model_used, warnings

    def _chunk_text(self, text: str, *, chunk_size: int = 80) -> List[str]:
        """Split text into chunks for SSE token events (proxying non-streaming backends)."""
        if not text:
            return []
        size = max(10, int(chunk_size))
        return [text[i : i + size] for i in range(0, len(text), size)]

    async def chat_stream(
        self,
        query: str,
        scope: Dict[str, Any],
        session_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Process a chat query and stream response via SSE.

        Implements Performance Mode v1:
        - Timing traces for all phases
        - Granular progress events for UX
        - Evidence caching
        - Hybrid LLM routing with fallback
        """
        options = options or {}
        timing = create_timing()

        # Get or create session
        session = self.get_or_create_session(session_id, scope)
        session.add_message("user", query)
        self._persist_message(session.session_id, "user", query)

        # ─────────────────────────────────────────────────────────────────────
        # Phase 1: Routing Analysis
        # ─────────────────────────────────────────────────────────────────────
        yield self._sse_event("progress", {
            "step": "routing",
            "substep": "analyzing",
            "message": "Analyzing query...",
            "phase": 1,
            "total_phases": 4
        })

        # Try deterministic routing first (fastest path)
        if DETERMINISTIC_EXTENDED:
            timing.start_phase()
            simple_response = await self._try_simple_query(query, scope)
            timing.end_phase("deterministic_ms")

            if simple_response:
                timing.provider_used = ProviderName.DETERMINISTIC
                timing.deterministic_ms = timing.evidence_breakdown.get("deterministic_ms", 0)
                timing.end()

                session.add_message("assistant", simple_response.content)
                self._persist_message(
                    session.session_id, "assistant", simple_response.content,
                    provider_used="deterministic",
                    timings=timing.to_dict(),
                )

                yield self._sse_event("progress", {
                    "step": "complete",
                    "substep": "deterministic",
                    "message": "Fast response (deterministic)",
                    "phase": 4,
                    "total_phases": 4
                })
                yield self._sse_event("token", {"token": simple_response.content})
                yield self._sse_event("done", {
                    "citations": [],
                    "proposals": [],
                    "model_used": ProviderName.DETERMINISTIC,
                    "latency_ms": timing.total_ms,
                    "session_id": session.session_id,
                    "warnings": [],
                    "timings": timing.to_dict(),
                    "evidence_summary": simple_response.evidence_summary,
                    "final_text": simple_response.content,  # Include final text for frontend
                })
                return

        # ─────────────────────────────────────────────────────────────────────
        # Phase 2: Evidence Gathering (with cache)
        # ─────────────────────────────────────────────────────────────────────
        scope_type = scope.get("type", "global")
        deal_id = scope.get("deal_id")
        cache_key = self.cache.cache_key(query, scope) if self.cache else None

        # Try cache first
        bundle = None
        if self.cache and cache_key:
            yield self._sse_event("progress", {
                "step": "evidence",
                "substep": "cache_check",
                "message": "Checking evidence cache...",
                "phase": 2,
                "total_phases": 4
            })
            cached_bundle, cache_hit = await self.cache.get(cache_key, scope_type)
            if cache_hit:
                bundle = cached_bundle
                timing.cache_hit = True
                timing.cache_source = scope_type
                yield self._sse_event("progress", {
                    "step": "evidence",
                    "substep": "cache_hit",
                    "message": "Using cached evidence",
                    "phase": 2,
                    "total_phases": 4
                })

        if bundle is None:
            # Emit granular progress events for evidence gathering
            yield self._sse_event("progress", {
                "step": "evidence",
                "substep": "rag_start",
                "message": "Querying RAG (documents)...",
                "phase": 2,
                "total_phases": 4
            })

            timing.start_phase()
            async with async_timing_context(timing, "evidence_ms"):
                bundle = await self.evidence_builder.build(query, scope, options)

            # Emit progress based on what was gathered
            if scope_type == "deal" and deal_id:
                yield self._sse_event("progress", {
                    "step": "evidence",
                    "substep": "rag_done",
                    "message": f"Found {bundle.rag_results_count} docs",
                    "phase": 2,
                    "total_phases": 4
                })
                yield self._sse_event("progress", {
                    "step": "evidence",
                    "substep": "events_done",
                    "message": f"Loaded {bundle.events_count} events",
                    "phase": 2,
                    "total_phases": 4
                })
                if bundle.case_file_loaded:
                    yield self._sse_event("progress", {
                        "step": "evidence",
                        "substep": "casefile_done",
                        "message": "Case file loaded",
                        "phase": 2,
                        "total_phases": 4
                    })
            else:
                yield self._sse_event("progress", {
                    "step": "evidence",
                    "substep": "rag_done",
                    "message": f"Found {bundle.rag_results_count} documents",
                    "phase": 2,
                    "total_phases": 4
                })

            timing.evidence_ms = timing.evidence_breakdown.get("evidence_ms", 0)

            # Cache the bundle
            if self.cache and cache_key:
                await self.cache.set(cache_key, bundle, scope_type, deal_id=deal_id)

        evidence_context = bundle.get_context_for_llm()

        # Emit evidence summary event
        yield self._sse_event("evidence", {
            **bundle.get_evidence_summary(),
            "cache_hit": timing.cache_hit,
        })

        # ─────────────────────────────────────────────────────────────────────
        # Phase 3: Provider Routing
        # ─────────────────────────────────────────────────────────────────────
        # Build prompt
        system_prompt = _render_system_prompt(evidence_context=evidence_context)
        history = session.get_history_for_llm(max_messages=6)
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
        ]

        # Optional: route generation through zakops-api (LangGraph) and proxy as SSE.
        if _brain_enabled():
            brain_text, brain_props, brain_model, brain_warnings = await self._call_brain_complete(
                query=query,
                scope=scope,
                session_id=session.session_id,
                history=history,
                evidence_context=evidence_context,
                options=options,
            )

            # Auto mode: fall back to legacy streaming if brain failed.
            if not brain_text and _brain_mode() != "force":
                bundle.warnings.extend(brain_warnings)
            else:
                yield self._sse_event("progress", {
                    "step": "routing",
                    "substep": "provider_select",
                    "message": "Selected: zakops-api",
                    "phase": 3,
                    "total_phases": 4
                })
                yield self._sse_event("progress", {
                    "step": "llm",
                    "substep": "generating",
                    "message": "Generating via zakops-api...",
                    "phase": 4,
                    "total_phases": 4
                })

                full_response = ""
                stream_warnings: List[str] = []
                timing.start_phase()

                if brain_text:
                    full_response = brain_text
                    timing.provider_used = brain_model or "zakops-api"
                    for chunk in self._chunk_text(full_response):
                        yield self._sse_event("token", {"token": chunk})
                    stream_warnings.extend(brain_warnings)
                else:
                    full_response = "ZakOps brain is unavailable (ZAKOPS_BRAIN_MODE=force)."
                    timing.provider_used = "zakops-api(unavailable)"
                    timing.degraded = True
                    timing.degraded_reason = "brain_unavailable"
                    yield self._sse_event("token", {"token": full_response})
                    stream_warnings.extend(brain_warnings)

                timing.end_phase("llm_ms")
                timing.llm_ms = timing.evidence_breakdown.get("llm_ms", 0)

                citations = self._extract_citations(full_response, bundle)
                proposals_raw = brain_props or self._extract_proposals(full_response)
                proposals = self._normalize_proposals_for_scope(proposals_raw, scope)

                session.add_message(
                    "assistant",
                    full_response,
                    citations=[c["id"] for c in citations],
                    proposals=proposals
                )
                self._persist_message(
                    session.session_id, "assistant", full_response,
                    citations=[c["id"] for c in citations],
                    proposals=proposals,
                    timings=timing.to_dict(),
                    provider_used=timing.provider_used,
                    cache_hit=timing.cache_hit,
                )

                timing.end()
                all_warnings = bundle.warnings + stream_warnings

                yield self._sse_event("progress", {
                    "step": "complete",
                    "substep": "done",
                    "message": "Complete",
                    "phase": 4,
                    "total_phases": 4
                })
                yield self._sse_event("done", {
                    "citations": citations,
                    "proposals": proposals,
                    "model_used": timing.provider_used,
                    "latency_ms": timing.total_ms,
                    "session_id": session.session_id,
                    "warnings": all_warnings,
                    "timings": timing.to_dict(),
                    "evidence_summary": bundle.get_evidence_summary(),
                    "final_text": full_response,
                })
                return

        # Get routing decision
        allow_cloud_override = options.get("allow_cloud")
        route = self.router.decide_route(
            query,
            evidence_size=bundle.total_evidence_size,
            is_deterministic=False,
            allow_cloud_override=allow_cloud_override
        )

        timing.provider_reason = route.reason

        # Cloud safety gate (only for cloud providers)
        if route.decision in (RoutingDecision.GEMINI_FLASH, RoutingDecision.GEMINI_PRO):
            combined_text = " ".join(m["content"] for m in messages)
            has_secrets, matches = scan_for_secrets(combined_text)

            if has_secrets:
                # Fall back to local vLLM
                route = self.router.decide_route(query, bundle.total_evidence_size, allow_cloud_override=False)
                timing.provider_reason = f"Cloud blocked (secrets detected), using {route.decision.value}"
                bundle.warnings.append("SAFETY: Secret patterns detected, switched to local model")

        # ─────────────────────────────────────────────────────────────────────
        # Phase 3: Provider Routing
        # ─────────────────────────────────────────────────────────────────────
        yield self._sse_event("progress", {
            "step": "routing",
            "substep": "provider_select",
            "message": f"Selected: {route.decision.value}",
            "phase": 3,
            "total_phases": 4
        })

        # ─────────────────────────────────────────────────────────────────────
        # Phase 4: LLM Generation (streaming)
        # ─────────────────────────────────────────────────────────────────────
        yield self._sse_event("progress", {
            "step": "llm",
            "substep": "generating",
            "message": f"Generating via {route.decision.value}...",
            "phase": 4,
            "total_phases": 4
        })

        full_response = ""
        stream_warnings = []
        timing.start_phase()

        try:
            async for chunk, provider_name in self.router.stream_with_fallback(
                messages, route,
                temperature=options.get("temperature", 0.7),
                max_tokens=options.get("max_tokens", 2048)
            ):
                full_response += chunk
                yield self._sse_event("token", {"token": chunk})
                timing.provider_used = provider_name

        except Exception as e:
            stream_warnings.append(str(e))
            yield self._sse_event("error", {"message": str(e)})
            full_response = f"Error generating response: {str(e)}"
            timing.degraded = True
            timing.degraded_reason = str(e)

        timing.end_phase("llm_ms")
        timing.llm_ms = timing.evidence_breakdown.get("llm_ms", 0)

        # Check if fallback was used
        if "(fallback)" in timing.provider_used:
            timing.provider_fallback = True

        # Check if degraded (all providers failed, graceful message returned)
        if timing.provider_used == "degraded":
            timing.degraded = True
            timing.degraded_reason = "All LLM providers temporarily unavailable"
            stream_warnings.append("AI service temporarily unavailable")

        # ─────────────────────────────────────────────────────────────────────
        # Phase 5: Post-processing
        # ─────────────────────────────────────────────────────────────────────
        citations = self._extract_citations(full_response, bundle)
        proposals = self._normalize_proposals_for_scope(
            self._extract_proposals(full_response),
            scope,
        )

        session.add_message(
            "assistant",
            full_response,
            citations=[c["id"] for c in citations],
            proposals=proposals
        )
        self._persist_message(
            session.session_id, "assistant", full_response,
            citations=[c["id"] for c in citations],
            proposals=proposals,
            timings=timing.to_dict(),
            provider_used=timing.provider_used,
            cache_hit=timing.cache_hit,
        )

        timing.end()

        # Combine all warnings
        all_warnings = bundle.warnings + stream_warnings

        # Emit final done event with full timing data
        yield self._sse_event("progress", {
            "step": "complete",
            "substep": "done",
            "message": "Complete",
            "phase": 4,
            "total_phases": 4
        })
        yield self._sse_event("done", {
            "citations": citations,
            "proposals": proposals,
            "model_used": timing.provider_used,
            "latency_ms": timing.total_ms,
            "session_id": session.session_id,
            "warnings": all_warnings,
            "timings": timing.to_dict(),
            "evidence_summary": bundle.get_evidence_summary(),
            "final_text": full_response,  # Include final text for frontend
        })

    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Tuple[str, str, List[str]]:
        """Call LLM and return (response, model_used, warnings)."""
        warnings = []

        # Try local vLLM first
        try:
            async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
                payload = {
                    "model": VLLM_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "stream": False,
                }

                resp = await client.post(VLLM_ENDPOINT, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return content, VLLM_MODEL, warnings
                else:
                    warnings.append(f"vLLM returned status {resp.status_code}")

        except httpx.TimeoutException:
            warnings.append("vLLM timeout - response may be incomplete")
        except Exception as e:
            warnings.append(f"vLLM unavailable: {str(e)}")

        # Fallback to cloud if allowed
        if ALLOW_CLOUD and GEMINI_API_KEY:
            # Security gate: scan for secrets before cloud send
            combined_text = " ".join(m["content"] for m in messages)
            has_secrets, matches = scan_for_secrets(combined_text)

            if has_secrets:
                warnings.append(f"BLOCKED: Secret patterns detected, cloud call prevented")
                return "I cannot process this request as it may contain sensitive data.", "none", warnings

            try:
                # Call Gemini (placeholder - implement actual API call)
                warnings.append("Cloud fallback not implemented")
                return "Local model unavailable and cloud fallback not configured.", "none", warnings
            except Exception as e:
                warnings.append(f"Cloud fallback failed: {str(e)}")

        # Final fallback
        return "I'm sorry, the AI service is currently unavailable. Please try again later.", "none", warnings

    async def _stream_llm(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream LLM response."""
        try:
            async with httpx.AsyncClient(timeout=VLLM_TIMEOUT) as client:
                payload = {
                    "model": VLLM_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                    "stream": True,
                }

                async with client.stream("POST", VLLM_ENDPOINT, json=payload) as resp:
                    if resp.status_code != 200:
                        yield {"type": "error", "message": f"vLLM returned status {resp.status_code}"}
                        return

                    yield {"type": "model", "model": VLLM_MODEL}

                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield {"type": "token", "token": delta["content"]}
                            except json.JSONDecodeError:
                                continue

        except httpx.TimeoutException:
            yield {"type": "error", "message": "vLLM timeout during streaming"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}

    def _extract_citations(
        self,
        text: str,
        bundle: EvidenceBundle
    ) -> List[Dict]:
        """Extract citation references from response text."""
        # Find all [cite-N] patterns
        pattern = r'\[cite-(\d+)\]'
        matches = set(re.findall(pattern, text))

        citations = []
        for match in matches:
            cite_id = f"cite-{match}"
            # Find matching citation in bundle
            for c in bundle.citations:
                if c.id == cite_id:
                    citations.append(c.to_dict())
                    break

        return citations

    def _normalize_proposals_for_scope(self, proposals: List[Dict], scope: Dict[str, Any]) -> List[Dict]:
        """Normalize proposals for consistent execution and UI rendering."""
        scope_type = (scope or {}).get("type") or "global"
        scope_deal_id = (scope or {}).get("deal_id") if scope_type == "deal" else None

        normalized: List[Dict[str, Any]] = []
        for p in proposals or []:
            if not isinstance(p, dict):
                continue

            proposal = dict(p)
            proposal_type = canonicalize_proposal_type(proposal.get("type"))
            if not proposal_type:
                continue
            proposal["type"] = proposal_type

            proposal.setdefault("proposal_id", str(uuid.uuid4())[:8])
            proposal.setdefault("status", "pending_approval")
            proposal["params"] = dict(proposal.get("params") or {})

            # Strip common placeholders so we don't create proposals that can never execute.
            if _looks_like_placeholder_deal_id(proposal.get("deal_id")):
                proposal.pop("deal_id", None)

            # If we're in deal scope and the model omitted deal_id, fill it.
            if not proposal.get("deal_id") and scope_deal_id:
                proposal["deal_id"] = scope_deal_id

            # If still missing deal_id and we're not in deal scope, drop the proposal.
            # All current proposal types require a deal context for safe execution.
            if not proposal.get("deal_id") and scope_type != "deal":
                continue

            normalized.append(proposal)

        return normalized

    def _extract_proposals(self, text: str) -> List[Dict]:
        """Extract proposal blocks from response text."""
        proposals = []

        # Find ```proposal ... ``` blocks
        pattern = r'```proposal\s*([\s\S]*?)```'
        matches = re.findall(pattern, text)

        for match in matches:
            try:
                raw_block = match.strip()

                proposal: Dict[str, Any] = {}
                params: Dict[str, Any] = {}

                # Prefer strict JSON if the block looks like JSON.
                if raw_block.startswith("{") and raw_block.endswith("}"):
                    parsed = json.loads(raw_block)
                    if isinstance(parsed, dict):
                        proposal = dict(parsed)
                        params = dict(proposal.get("params") or {})
                else:
                    # Parse a small YAML-like subset (top-level keys + indented params).
                    in_params = False
                    multiline_key: Optional[str] = None
                    multiline_quote: Optional[str] = None
                    multiline_buf: List[str] = []
                    for raw_line in raw_block.splitlines():
                        if multiline_key:
                            # End multiline if we see the closing triple quote marker.
                            stripped_line = raw_line.strip()
                            if multiline_quote and stripped_line == multiline_quote:
                                params[multiline_key] = "\n".join(multiline_buf).rstrip()
                                multiline_key = None
                                multiline_quote = None
                                multiline_buf = []
                                continue
                            multiline_buf.append(raw_line.rstrip("\n"))
                            continue

                        if not raw_line.strip():
                            continue
                        if raw_line.lstrip().startswith("#"):
                            continue

                        stripped = raw_line.lstrip()
                        indent = len(raw_line) - len(stripped)

                        if stripped.startswith("params") and stripped.rstrip().endswith(":"):
                            in_params = True
                            continue

                        # Nested params (2+ spaces indent after params:)
                        if in_params and indent >= 2:
                            if ":" not in stripped:
                                continue
                            k, v = stripped.split(":", 1)
                            key = k.strip()
                            value_raw = v.strip()
                            # Handle simple triple-quoted multiline values (best-effort).
                            if value_raw in {'"""', "'''"}:
                                multiline_key = key
                                multiline_quote = value_raw
                                multiline_buf = []
                                continue
                            if value_raw.startswith('"""') and value_raw.endswith('"""') and len(value_raw) >= 6:
                                params[key] = value_raw[3:-3]
                                continue
                            value = _strip_wrapping_quotes(value_raw)
                            # Parse inline JSON objects/arrays when present (e.g., inputs).
                            if (value.startswith("{") and value.endswith("}")) or (value.startswith("[") and value.endswith("]")):
                                try:
                                    parsed_inline = json.loads(value)
                                    params[key] = parsed_inline
                                    continue
                                except Exception:
                                    pass
                            params[key] = value
                            continue

                        # Top-level key:value
                        in_params = False
                        if ":" not in stripped:
                            continue
                        k, v = stripped.split(":", 1)
                        key = k.strip()
                        value = _strip_wrapping_quotes(v.strip())
                        if key == "params":
                            in_params = True
                            continue
                        proposal[key] = value

                # Normalize and validate type
                proposal_type = canonicalize_proposal_type(proposal.get("type"))
                if not proposal_type:
                    continue
                proposal["type"] = proposal_type

                # Ensure params dict exists
                if params:
                    proposal["params"] = params
                else:
                    proposal["params"] = dict(proposal.get("params") or {})

                # Back-compat: move known params from top-level into params
                for k in (
                    "to_stage",
                    "content",
                    "category",
                    "due_days",
                    "action_type",
                    "title",
                    "summary",
                    "risk_level",
                    "capability_id",
                    "inputs",
                    "description",
                    "priority",
                    "recipient",
                    "to",
                    "subject",
                    "context",
                    "doc_type",
                ):
                    if k in proposal and k not in proposal["params"]:
                        proposal["params"][k] = proposal[k]
                        proposal.pop(k, None)

                # Fill required fields
                proposal["proposal_id"] = proposal.get("proposal_id") or str(uuid.uuid4())[:8]
                proposal["status"] = proposal.get("status") or "pending_approval"

                # Per-action cloud gate: Mark proposals that require cloud (Gemini Pro)
                # so the UI can display appropriate warnings/consent messaging.
                if proposal["type"] == "draft_email":
                    proposal["cloud_required"] = True
                    proposal["cloud_provider"] = "gemini-pro"

                proposals.append(proposal)

            except Exception:
                continue

        return proposals

    def _sse_event(self, event_type: str, data: Any) -> str:
        """Format SSE event."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    async def execute_proposal(
        self,
        proposal_id: str,
        approved_by: str,
        session_id: str,
        action: str = "approve",
        reject_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute (approve) or reject a proposal.

        Supports proposal types:
        - stage_transition: Move deal to new stage
        - add_note: Add note to deal case file
        - create_task (or schedule_action alias): Create deferred action
        - create_action: Create a Kinetic Action record (PENDING_APPROVAL)
        - draft_email: Generate and store email draft (MUST use Gemini Pro)
        - request_docs: Request documents from broker

        Error responses include reason codes for UI handling:
        - session_not_found (404)
        - proposal_not_found (404)
        - invalid_status_transition (409)
        - unknown_proposal_type (400)
        - invalid_proposal_params (400)
        - execution_failed (500)
        """
        # Find the session - check in-memory first, then SQLite
        session = self.sessions.get(session_id)
        if not session:
            # Try loading from SQLite persistence
            if self.session_store:
                persisted = self.session_store.load_session(session_id)
                if persisted:
                    # Reconstruct session with messages
                    session = ChatSession(
                        session_id=persisted.session_id,
                        scope={
                            "type": persisted.scope_type,
                            "deal_id": persisted.scope_deal_id,
                        },
                    )
                    for msg in persisted.messages:
                        session.messages.append(ChatMessage(
                            role=msg.role,
                            content=msg.content,
                            timestamp=msg.timestamp,
                            citations=msg.citations,
                            proposals=msg.proposals,
                        ))
                    self.sessions[session_id] = session

        if not session:
            return {
                "success": False,
                "error": "Session not found",
                "reason": "session_not_found",
                "session_id": session_id,
            }

        # Find the proposal in message history
        proposal = None
        proposal_message = None
        for msg in reversed(session.messages):
            for p in (msg.proposals or []):
                if p.get("proposal_id") == proposal_id:
                    proposal = p
                    proposal_message = msg
                    break
            if proposal:
                break

        if not proposal:
            return {
                "success": False,
                "error": f"Proposal not found: {proposal_id}",
                "reason": "proposal_not_found",
                "session_id": session_id,
                "proposal_id": proposal_id,
            }

        action_norm = (action or "approve").strip().lower()
        if action_norm not in {"approve", "reject"}:
            action_norm = "approve"

        current_status = proposal.get("status", "unknown")

        if action_norm == "reject":
            if current_status not in {"pending_approval", "failed"}:
                return {
                    "success": False,
                    "error": f"Cannot reject proposal with status '{current_status}'.",
                    "reason": "invalid_status_transition",
                    "current_status": current_status,
                    "proposal_id": proposal_id,
                }

            proposal["status"] = "rejected"
            proposal["rejected_by"] = approved_by
            proposal["rejected_at"] = datetime.now(timezone.utc).isoformat()
            if reject_reason:
                proposal["reject_reason"] = str(reject_reason)[:500]

            self._persist_proposal_update(session_id, proposal_id, proposal)
            return {
                "success": True,
                "result": {"status": "rejected"},
                "proposal": proposal,
                "proposal_type": proposal.get("type"),
            }

        # Approve/execute path (allow retry from failed)
        if current_status not in {"pending_approval", "failed"}:
            return {
                "success": False,
                "error": f"Cannot execute proposal with status '{current_status}'. Expected 'pending_approval' (or 'failed' for retry).",
                "reason": "invalid_status_transition",
                "current_status": current_status,
                "proposal_id": proposal_id,
            }

        # Execute based on type
        proposal_type = canonicalize_proposal_type(proposal.get("type")) or proposal.get("type")
        proposal["type"] = proposal_type
        deal_id = (proposal.get("deal_id") or "").strip() or None
        params = dict(proposal.get("params") or {})

        # Back-compat: if old proposals stored params at top-level, recover here.
        for k in (
            "to_stage",
            "content",
            "category",
            "due_days",
            "action_type",
            "title",
            "summary",
            "risk_level",
            "capability_id",
            "inputs",
            "description",
            "priority",
            "recipient",
            "to",
            "subject",
            "context",
            "doc_type",
        ):
            if k not in params and k in proposal:
                params[k] = proposal.get(k)
        proposal["params"] = params

        def _param(key: str, default: Any = None) -> Any:
            return params.get(key, default)

        try:
            if proposal_type == "stage_transition":
                if not deal_id:
                    return {
                        "success": False,
                        "error": "Missing deal_id for stage_transition proposal",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }
                to_stage = _param("to_stage")
                if not to_stage:
                    return {
                        "success": False,
                        "error": "Missing params.to_stage for stage_transition proposal",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }

                result = self._execute_stage_transition(
                    deal_id=deal_id,
                    to_stage=str(to_stage),
                    reason=str(proposal.get("reason") or "Approved via chat"),
                    approved_by=approved_by,
                )

                if result.get("success"):
                    proposal["status"] = "executed"
                    proposal["result"] = result
                    self._persist_proposal_update(session_id, proposal_id, proposal)
                    await self.invalidate_cache_for_deal(deal_id)
                    return {"success": True, "result": result, "proposal": proposal, "proposal_type": proposal_type}

                proposal["status"] = "failed"
                proposal["error"] = result.get("error") or result.get("message") or "Execution failed"
                self._persist_proposal_update(session_id, proposal_id, proposal)
                return {"success": False, "error": proposal["error"], "reason": "execution_failed", "proposal": proposal, "proposal_type": proposal_type}

            elif proposal_type == "add_note":
                if not deal_id:
                    return {
                        "success": False,
                        "error": "Missing deal_id for add_note proposal",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }
                content = _param("content")
                if not isinstance(content, str) or not content.strip():
                    return {
                        "success": False,
                        "error": "Missing params.content for add_note proposal",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }

                result = self._execute_add_note(
                    deal_id=deal_id,
                    content=content.strip(),
                    category=str(_param("category", "chat_note")),
                )
                if result.get("success"):
                    proposal["status"] = "executed"
                    proposal["result"] = result
                    self._persist_proposal_update(session_id, proposal_id, proposal)
                    await self.invalidate_cache_for_deal(deal_id)
                    return {"success": True, "result": result, "proposal": proposal, "proposal_type": proposal_type}

                proposal["status"] = "failed"
                proposal["error"] = result.get("error") or "Execution failed"
                self._persist_proposal_update(session_id, proposal_id, proposal)
                return {"success": False, "error": proposal["error"], "reason": "execution_failed", "proposal": proposal, "proposal_type": proposal_type}

            elif proposal_type == "create_task":
                if not deal_id:
                    return {
                        "success": False,
                        "error": "Missing deal_id for create_task proposal",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }
                # Create deferred action using DeferredActionQueue
                from deferred_actions import DeferredActionQueue

                queue = DeferredActionQueue()

                # Parse due date from params or default to 1 day from now
                due_days = int(_param("due_days", 1) or 1)
                action_type = str(_param("action_type", "follow_up"))
                description = str(_param("description", proposal.get("reason", "Chat-initiated task")))
                priority = str(_param("priority", "normal"))

                action_id = queue.schedule_relative(
                    deal_id=deal_id,
                    action_type=action_type,
                    days_from_now=due_days,
                    priority=priority,
                    data={
                        "description": description,
                        "source": "chat_proposal",
                        "proposal_id": proposal_id,
                        "approved_by": approved_by,
                    },
                    metadata={
                        "session_id": session_id,
                        "created_via": "chat_assistant",
                    },
                )

                proposal["status"] = "executed"
                proposal["result"] = {"action_id": action_id, "scheduled_days": due_days}
                self._persist_proposal_update(session_id, proposal_id, proposal)
                await self.invalidate_cache_for_deal(deal_id)
                return {
                    "success": True,
                    "result": {"action_id": action_id, "scheduled_days": due_days},
                    "proposal": proposal,
                    "proposal_type": proposal_type,
                }

            elif proposal_type == "create_action":
                if not deal_id:
                    return {
                        "success": False,
                        "error": "Missing deal_id for create_action proposal",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }

                action_type = str(_param("action_type", "") or "").strip()
                title = str(_param("title", "") or action_type or "Create Action").strip()
                summary = str(_param("summary", proposal.get("reason") or "") or "").strip()
                capability_id = str(_param("capability_id", "") or "").strip() or None
                risk_level = str(_param("risk_level", "medium") or "medium").strip()
                inputs = _param("inputs", None) or _param("action_inputs", None) or {}
                if not isinstance(inputs, dict):
                    return {
                        "success": False,
                        "error": "params.inputs must be an object/dict for create_action",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }

                if not action_type:
                    return {
                        "success": False,
                        "error": "Missing params.action_type for create_action proposal",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }

                try:
                    from actions.engine.validation import ActionCreationValidationError, validate_action_creation

                    validate_action_creation(action_type=action_type, capability_id=capability_id)
                except ActionCreationValidationError as e:
                    return {
                        "success": False,
                        "error": e.message,
                        "reason": e.code,
                        "details": e.details or {},
                        "proposal_type": proposal_type,
                        "action_type": action_type,
                        "capability_id": capability_id,
                    }

                from actions.engine.models import ActionPayload, compute_idempotency_key
                from actions.engine.store import ActionStore

                store = ActionStore()
                idempotency_key = compute_idempotency_key("create_action", proposal_id)
                auto_run = bool(_param("auto_run", True))
                action_payload = ActionPayload(
                    deal_id=deal_id,
                    capability_id=capability_id,
                    type=action_type,
                    title=title,
                    summary=summary,
                    status="PENDING_APPROVAL",
                    created_by=approved_by,
                    source="chat",
                    risk_level=risk_level,
                    requires_human_review=True,
                    idempotency_key=idempotency_key,
                    inputs=inputs,
                )

                created_action, created_new = store.create_action(action_payload)

                # The human approved this proposal; treat that as approval for the action record.
                # This avoids "double approval" friction while preserving the action audit trail.
                try:
                    if created_action.status == "PENDING_APPROVAL":
                        created_action = store.approve_action(created_action.action_id, actor=approved_by)
                except Exception:
                    pass

                # Optionally enqueue for runner execution (runner will transition to PROCESSING).
                try:
                    if auto_run and created_action.status == "READY":
                        created_action = store.request_execute(created_action.action_id, actor=approved_by)
                except Exception:
                    pass

                # Emit deal event (best-effort; no HTTP self-calls)
                try:
                    from deal_events import DealEventStore

                    DealEventStore().create_event(
                        deal_id=deal_id,
                        event_type="kinetic_action_created_via_chat",
                        actor=approved_by,
                        data={
                            "proposal_id": proposal_id,
                            "action_id": created_action.action_id,
                            "action_type": created_action.type,
                            "created_new": created_new,
                            "capability_id": created_action.capability_id,
                            "action_status": created_action.status,
                            "auto_run": auto_run,
                        },
                    )
                except Exception:
                    pass

                proposal["status"] = "executed"
                proposal["result"] = {
                    "action_id": created_action.action_id,
                    "created_new": created_new,
                    "action_status": created_action.status,
                    "action_type": created_action.type,
                }
                self._persist_proposal_update(session_id, proposal_id, proposal)
                await self.invalidate_cache_for_deal(deal_id)

                return {
                    "success": True,
                    "result": {
                        "action_id": created_action.action_id,
                        "created_new": created_new,
                        "action_status": created_action.status,
                        "action_type": created_action.type,
                    },
                    "proposal": proposal,
                    "proposal_type": proposal_type,
                }

            elif proposal_type == "draft_email":
                # Generate email draft using Gemini Pro (for broker communications)
                # Store the draft for user review before sending
                #
                # Per-action cloud gate: User approval of draft_email proposal implies
                # consent to use cloud (Gemini Pro) for this specific operation.
                if not deal_id:
                    return {
                        "success": False,
                        "error": "Missing deal_id for draft_email proposal",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }

                recipient = str(_param("recipient", _param("to", "")) or "").strip()
                subject = str(_param("subject", "") or "").strip()
                context = str(_param("context", proposal.get("reason", "")) or "").strip()

                if not recipient:
                    return {
                        "success": False,
                        "error": "Missing params.recipient (or params.to) for draft_email proposal",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }

                # Generate email content via Gemini Pro
                # allow_cloud_override=True: User explicitly approved this proposal,
                # which implies consent to use cloud for broker email drafting.
                email_content = await self._generate_broker_email(
                    deal_id=deal_id,
                    recipient=recipient,
                    subject=subject,
                    context=context,
                    allow_cloud_override=True,  # Per-action cloud gate: approval = consent
                )

                if not email_content.get("success", True):
                    proposal["status"] = "failed"
                    proposal["error"] = email_content.get("error") or "Email drafting failed"
                    self._persist_proposal_update(session_id, proposal_id, proposal)
                    return {
                        "success": False,
                        "error": proposal["error"],
                        "reason": email_content.get("reason", "execution_failed"),
                        "proposal": proposal,
                        "proposal_type": proposal_type,
                    }

                proposal["status"] = "executed"
                proposal["result"] = email_content
                self._persist_proposal_update(session_id, proposal_id, proposal)
                await self.invalidate_cache_for_deal(deal_id)
                return {
                    "success": True,
                    "result": {
                        "email_draft": email_content,
                        "recipient": recipient,
                        "subject": email_content.get("subject") or subject,
                        "provider": email_content.get("provider"),
                        "model": email_content.get("model"),
                        "forced_reason": email_content.get("forced_reason"),
                    },
                    "proposal": proposal,
                    "proposal_type": proposal_type,
                }

            elif proposal_type == "request_docs":
                # Record document request as a note + create follow-up action
                if not deal_id:
                    return {
                        "success": False,
                        "error": "Missing deal_id for request_docs proposal",
                        "reason": "invalid_proposal_params",
                        "proposal_type": proposal_type,
                    }

                doc_type = str(_param("doc_type", "general"))
                description = str(_param("description", proposal.get("reason", "Document request")))

                result = self._execute_request_docs(
                    deal_id=deal_id,
                    doc_type=doc_type,
                    description=description,
                    approved_by=approved_by,
                    proposal_id=proposal_id,
                    session_id=session_id,
                )

                # Bridge: also create a Kinetic Action record so the request is visible/executable in Actions UI.
                try:
                    from actions.engine.validation import validate_action_creation
                    from actions.engine.models import ActionPayload, compute_idempotency_key
                    from actions.engine.store import ActionStore

                    store = ActionStore()
                    idem = compute_idempotency_key("request_docs", proposal_id)
                    validate_action_creation(action_type="DILIGENCE.REQUEST_DOCS", capability_id="diligence.request_docs.v1")
                    action_payload = ActionPayload(
                        deal_id=deal_id,
                        capability_id="diligence.request_docs.v1",
                        type="DILIGENCE.REQUEST_DOCS",
                        title=f"Request docs ({doc_type})",
                        summary=description,
                        status="PENDING_APPROVAL",
                        created_by=approved_by,
                        source="chat",
                        risk_level="medium",
                        requires_human_review=True,
                        idempotency_key=idem,
                        inputs={"doc_type": doc_type, "description": description},
                    )
                    action, created_new = store.create_action(action_payload)
                    if action.status == "PENDING_APPROVAL":
                        try:
                            action = store.approve_action(action.action_id, actor=approved_by)
                        except Exception:
                            pass
                    if action.status == "READY":
                        try:
                            action = store.request_execute(action.action_id, actor=approved_by)
                        except Exception:
                            pass

                    result["kinetic_action"] = {
                        "action_id": action.action_id,
                        "created_new": created_new,
                        "action_status": action.status,
                        "action_type": action.type,
                    }
                except Exception:
                    pass
                if result.get("success"):
                    proposal["status"] = "executed"
                    proposal["result"] = result
                    self._persist_proposal_update(session_id, proposal_id, proposal)
                    await self.invalidate_cache_for_deal(deal_id)
                    return {"success": True, "result": result, "proposal": proposal, "proposal_type": proposal_type}

                proposal["status"] = "failed"
                proposal["error"] = result.get("error") or "Execution failed"
                self._persist_proposal_update(session_id, proposal_id, proposal)
                return {"success": False, "error": proposal["error"], "reason": "execution_failed", "proposal": proposal, "proposal_type": proposal_type}

            else:
                return {
                    "success": False,
                    "error": f"Unknown proposal type: {proposal_type}",
                    "reason": "unknown_proposal_type",
                    "proposal_type": proposal_type,
                    "supported_types": sorted(CANONICAL_PROPOSAL_TYPES),
                }

        except Exception as e:
            proposal["status"] = "failed"
            proposal["error"] = str(e)
            self._persist_proposal_update(session_id, proposal_id, proposal)
            return {
                "success": False,
                "error": str(e),
                "reason": "execution_failed",
                "proposal_type": proposal_type,
            }

    def _persist_proposal_update(self, session_id: str, proposal_id: str, proposal: Dict[str, Any]) -> None:
        """Persist proposal status/result updates (best-effort)."""
        if not self.session_store:
            return
        try:
            # Update proposal inside the persisted message JSON so refresh/restart retains state.
            self.session_store.update_proposal(
                session_id=session_id,
                proposal_id=proposal_id,
                proposal=proposal,
            )
        except Exception:
            # Never crash execution due to persistence failures.
            return

    def _execute_stage_transition(self, *, deal_id: str, to_stage: str, reason: str, approved_by: str) -> Dict[str, Any]:
        """Execute a stage transition using the local control plane (no HTTP self-calls)."""
        from deal_events import DealEventStore
        from deal_registry import DealRegistry
        from deal_state_machine import DealStage, DealStateMachine

        registry_path = "/home/zaks/DataRoom/.deal-registry/deal_registry.json"
        registry = DealRegistry(registry_path)
        deal = registry.get_deal(deal_id)
        if not deal:
            return {"success": False, "error": f"Deal not found: {deal_id}"}

        sm = DealStateMachine(deal.stage, deal_id)
        target = DealStage.from_str(to_stage)

        if not sm.can_transition_to(target):
            allowed = [s.value for s in sm.get_allowed_transitions()]
            return {"success": False, "error": f"Invalid transition: {deal.stage} -> {to_stage}. Allowed: {allowed}"}

        needs_approval = sm.requires_approval(target)
        if needs_approval and not approved_by:
            return {"success": False, "error": f"Transition to {to_stage} requires approval (approved_by)"}

        result = sm.transition(target, approved=bool(approved_by))
        if result.success:
            registry.update_deal(deal_id, stage=to_stage)
            registry.save()

            event_store = DealEventStore()
            event_store.create_event(
                deal_id=deal_id,
                event_type="stage_changed",
                actor=approved_by or "chat",
                data={
                    "from_stage": deal.stage,
                    "to_stage": to_stage,
                    "reason": reason,
                    "approved_by": approved_by,
                },
            )

        return {
            "success": result.success,
            "message": result.message,
            "from_stage": result.from_stage.value,
            "to_stage": result.to_stage.value,
            "approval_required": result.approval_required,
        }

    def _execute_add_note(self, *, deal_id: str, content: str, category: str) -> Dict[str, Any]:
        """Add a note event to a deal (local control plane)."""
        from deal_events import DealEventStore
        from deal_registry import DealRegistry

        registry_path = "/home/zaks/DataRoom/.deal-registry/deal_registry.json"
        registry = DealRegistry(registry_path)
        deal = registry.get_deal(deal_id)
        if not deal:
            return {"success": False, "error": f"Deal not found: {deal_id}"}

        event_store = DealEventStore()
        event = event_store.create_event(
            deal_id=deal_id,
            event_type="note_added",
            actor="operator",
            data={"content": content, "category": category},
        )
        return {"success": True, "event_id": getattr(event, "event_id", None)}

    def _execute_request_docs(
        self,
        *,
        deal_id: str,
        doc_type: str,
        description: str,
        approved_by: str,
        proposal_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """Record a document request as a note + a follow-up task."""
        from deferred_actions import DeferredActionQueue

        note_content = f"Document requested: {doc_type}\n{description}".strip()
        note_result = self._execute_add_note(deal_id=deal_id, content=note_content, category="document_request")
        if not note_result.get("success"):
            return {"success": False, "error": note_result.get("error") or "Failed to record document request note"}

        queue = DeferredActionQueue()
        action_id = queue.schedule_relative(
            deal_id=deal_id,
            action_type="doc_follow_up",
            days_from_now=3,
            priority="normal",
            data={
                "doc_type": doc_type,
                "description": description,
                "source": "chat_proposal",
                "proposal_id": proposal_id,
                "approved_by": approved_by,
                "session_id": session_id,
            },
        )

        return {
            "success": True,
            "action_id": action_id,
            "doc_type": doc_type,
            "note_event_id": note_result.get("event_id"),
        }

    async def _generate_broker_email(
        self,
        deal_id: str,
        recipient: str,
        subject: str,
        context: str,
        *,
        allow_cloud_override: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate broker email content.

        Priority:
        1. If cloud allowed (globally or per-action override): Use Gemini Pro
        2. If Gemini unavailable OR cloud disabled: Fall back to local vLLM
        3. If both fail: Return structured error

        Per-action cloud gate: When a user explicitly approves a draft_email proposal,
        `allow_cloud_override=True` enables cloud for this single operation only.
        """
        forced_reason = "email_draft"
        cloud_allowed = ALLOW_CLOUD or allow_cloud_override

        prompt = f"""You are a professional M&A operator. Draft a concise broker email.

Return ONLY strict JSON with keys:
- subject: string
- body: string

Constraints:
- Keep it concise (max 2 short paragraphs).
- Professional tone, direct and respectful.
- Include a clear call-to-action.
- Do NOT include any markdown, code fences, or extra keys.

Inputs:
deal_id: {deal_id}
recipient: {recipient}
subject_hint: {subject}
context: {context}
"""

        messages = [
            {"role": "system", "content": "You draft broker communications for acquisitions. Output strict JSON only."},
            {"role": "user", "content": prompt},
        ]

        provider_used = None
        model_used = None

        # Try Gemini Pro if cloud is allowed
        if cloud_allowed:
            try:
                gemini_pro = get_provider("gemini-pro")
                health = await gemini_pro.health_check()
                if health.healthy:
                    response = await gemini_pro.generate(messages, temperature=0.4, max_tokens=2000)
                    provider_used = response.provider
                    model_used = response.model

                    # Parse JSON from response
                    raw_content = response.content.strip()
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_content)
                    if json_match:
                        raw_content = json_match.group(1).strip()

                    try:
                        data = json.loads(raw_content)
                        draft_subject = str(data.get("subject") or subject or f"Re: {deal_id}").strip()
                        body = str(data.get("body") or "").strip()
                        if body:
                            return {
                                "success": True,
                                "subject": draft_subject,
                                "body": body,
                                "provider": provider_used,
                                "model": model_used,
                                "forced_reason": forced_reason,
                            }
                    except json.JSONDecodeError:
                        pass  # Fall through to local fallback
            except Exception as e:
                # Log but continue to fallback
                pass

        # Fallback to local vLLM
        try:
            vllm_provider = get_provider("vllm")
            health = await vllm_provider.health_check()
            if health.healthy:
                response = await vllm_provider.generate(messages, temperature=0.4, max_tokens=1500)
                provider_used = "local_fallback"
                model_used = response.model

                raw_content = response.content.strip()
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw_content)
                if json_match:
                    raw_content = json_match.group(1).strip()

                try:
                    data = json.loads(raw_content)
                    draft_subject = str(data.get("subject") or subject or f"Re: {deal_id}").strip()
                    body = str(data.get("body") or "").strip()
                    if body:
                        return {
                            "success": True,
                            "subject": draft_subject,
                            "body": body,
                            "provider": provider_used,
                            "model": model_used,
                            "forced_reason": forced_reason,
                            "fallback_used": True,
                        }
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            pass

        # Both providers failed - return error
        return {
            "success": False,
            "error": "Could not generate email draft. Both Gemini Pro and local vLLM failed or are unavailable.",
            "reason": "all_providers_failed",
            "cloud_allowed": cloud_allowed,
            "forced_reason": forced_reason,
        }


    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for all providers."""
        provider_health = await get_all_health()

        # Get budget status
        budget_status = self.budget.get_status()

        # Get cache stats
        cache_stats = self.cache.stats() if self.cache else {"enabled": False}

        # Determine healthy providers
        healthy_providers = [
            name for name, status in provider_health.items()
            if status.healthy
        ]

        # Build diagnostics and recommendations
        diagnostics = []
        recommendations = []

        for name, status in provider_health.items():
            if not status.healthy:
                diagnostics.append({
                    "provider": name,
                    "issue": status.error,
                    "endpoint": status.endpoint,
                })
                if name == "vllm" and status.error:
                    if "not found" in status.error.lower():
                        recommendations.append(
                            f"vLLM: Model mismatch. Check VLLM_MODEL env var. Configured: {status.model}"
                        )
                    elif "connection" in status.error.lower():
                        recommendations.append(
                            f"vLLM: Connection refused at {status.endpoint}. Check if vLLM container is running."
                        )
                if name in ("gemini-flash", "gemini-pro") and "api key" in (status.error or "").lower():
                    recommendations.append(
                        f"{name}: API key not configured. Set GEMINI_API_KEY env var or create ~/.gemini_api file."
                    )

        # Primary provider determination
        primary_provider = None
        if "vllm" in healthy_providers:
            primary_provider = "vllm"
        elif "gemini-flash" in healthy_providers and ALLOW_CLOUD:
            primary_provider = "gemini-flash"
        elif "gemini-pro" in healthy_providers and ALLOW_CLOUD:
            primary_provider = "gemini-pro"

        return {
            "status": "healthy" if healthy_providers else "degraded",
            "primary_provider": primary_provider,
            "providers": {name: status.to_dict() for name, status in provider_health.items()},
            "budget": budget_status,
            "cache": cache_stats,
            "healthy_providers": healthy_providers,
            "config": {
                "allow_cloud": ALLOW_CLOUD,
                "cache_enabled": CACHE_ENABLED,
                "deterministic_extended": DETERMINISTIC_EXTENDED,
                "openai_api_base": OPENAI_API_BASE,
                "vllm_model": VLLM_MODEL,
            },
            "diagnostics": diagnostics if diagnostics else None,
            "recommendations": recommendations if recommendations else None,
        }

    async def invalidate_cache_for_deal(self, deal_id: str):
        """Invalidate cache entries for a deal (call after mutations)."""
        if self.cache:
            await self.cache.invalidate_deal(deal_id)


# Singleton instance
_orchestrator: Optional[ChatOrchestrator] = None


def get_orchestrator(allow_cloud: Optional[bool] = None) -> ChatOrchestrator:
    """Get or create the chat orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ChatOrchestrator(allow_cloud=allow_cloud)
    return _orchestrator
