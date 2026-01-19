"""
ContextPack Builder Module

Centralized context gathering for Kinetic Actions.
Builds a structured context bundle from multiple sources:
- Deal registry (deal record, broker info)
- Event store (recent events)
- Case files (summary, key facts)
- RAG (relevant documents)
- Extracted materials (links from enrichment)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Configuration
RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8052")
DATAROOM_ROOT = Path(os.getenv("DATAROOM_ROOT", "/home/zaks/DataRoom"))
DEAL_REGISTRY_PATH = DATAROOM_ROOT / ".deal-registry" / "deal_registry.json"
EVENTS_DIR = DATAROOM_ROOT / ".deal-registry" / "events"


@dataclass
class BrokerContext:
    """Broker information for the deal."""
    name: Optional[str] = None
    email: Optional[str] = None
    firm: Optional[str] = None
    phone: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class EventContext:
    """A single event from the deal history."""
    event_id: str
    event_type: str
    timestamp: str
    actor: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGChunk:
    """A chunk of evidence from RAG."""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialLink:
    """An extracted material link from enrichment."""
    url: str
    title: Optional[str] = None
    link_type: str = "unknown"  # cim, nda, teaser, etc.
    extracted_at: Optional[str] = None


@dataclass
class ContextPack:
    """
    Complete context bundle for action execution.

    Contains everything an action executor needs to make informed decisions
    and generate personalized content.
    """
    # Core deal info
    deal_id: str
    deal_record: Dict[str, Any] = field(default_factory=dict)
    canonical_name: Optional[str] = None
    display_name: Optional[str] = None

    # Deal state
    stage: Optional[str] = None
    status: Optional[str] = None

    # Broker info
    broker: Optional[BrokerContext] = None

    # Financial highlights
    asking_price: Optional[float] = None
    revenue: Optional[float] = None
    ebitda: Optional[float] = None

    # Evidence
    recent_events: List[EventContext] = field(default_factory=list)
    case_summary: Optional[str] = None
    key_facts: List[str] = field(default_factory=list)

    # RAG evidence (query-specific)
    rag_evidence: List[RAGChunk] = field(default_factory=list)

    # Materials
    extracted_links: List[MaterialLink] = field(default_factory=list)

    # Memory hook (CodeX future)
    prior_actions_summary: Optional[str] = None

    # Metadata
    built_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    sources_queried: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "deal_id": self.deal_id,
            "canonical_name": self.canonical_name,
            "display_name": self.display_name,
            "stage": self.stage,
            "status": self.status,
            "broker": {
                "name": self.broker.name if self.broker else None,
                "email": self.broker.email if self.broker else None,
                "firm": self.broker.firm if self.broker else None,
                "phone": self.broker.phone if self.broker else None,
            } if self.broker else None,
            "asking_price": self.asking_price,
            "revenue": self.revenue,
            "ebitda": self.ebitda,
            "recent_events": [
                {
                    "event_type": e.event_type,
                    "timestamp": e.timestamp,
                    "actor": e.actor,
                    "summary": e.summary,
                }
                for e in self.recent_events
            ],
            "case_summary": self.case_summary,
            "key_facts": self.key_facts,
            "rag_evidence": [
                {"content": r.content[:500], "source": r.source, "score": r.score}
                for r in self.rag_evidence
            ],
            "extracted_links": [
                {"url": m.url, "title": m.title, "type": m.link_type}
                for m in self.extracted_links
            ],
            "prior_actions_summary": self.prior_actions_summary,
            "built_at": self.built_at,
            "sources_queried": self.sources_queried,
        }

    def to_prompt_context(self) -> str:
        """Format as context string for LLM prompts."""
        parts = []

        # Deal header
        parts.append(f"# Deal: {self.display_name or self.canonical_name or self.deal_id}")
        parts.append(f"Stage: {self.stage or 'Unknown'} | Status: {self.status or 'Unknown'}")

        # Financials
        if any([self.asking_price, self.revenue, self.ebitda]):
            fin_parts = []
            if self.asking_price:
                fin_parts.append(f"Asking: ${self.asking_price:,.0f}")
            if self.revenue:
                fin_parts.append(f"Revenue: ${self.revenue:,.0f}")
            if self.ebitda:
                fin_parts.append(f"EBITDA: ${self.ebitda:,.0f}")
            parts.append(f"Financials: {' | '.join(fin_parts)}")

        # Broker
        if self.broker and self.broker.name:
            broker_str = f"Broker: {self.broker.name}"
            if self.broker.firm:
                broker_str += f" ({self.broker.firm})"
            if self.broker.email:
                broker_str += f" - {self.broker.email}"
            parts.append(broker_str)

        # Case summary
        if self.case_summary:
            parts.append(f"\n## Summary\n{self.case_summary}")

        # Key facts
        if self.key_facts:
            parts.append("\n## Key Facts")
            for fact in self.key_facts[:10]:
                parts.append(f"- {fact}")

        # Recent events
        if self.recent_events:
            parts.append("\n## Recent Activity")
            for event in self.recent_events[:5]:
                parts.append(f"- [{event.timestamp[:10]}] {event.summary}")

        # RAG evidence
        if self.rag_evidence:
            parts.append("\n## Relevant Evidence")
            for chunk in self.rag_evidence[:3]:
                parts.append(f"Source: {chunk.source}")
                parts.append(f"{chunk.content[:500]}...")

        return "\n".join(parts)


def _load_deal_registry() -> Dict[str, Any]:
    """Load the deal registry JSON."""
    if not DEAL_REGISTRY_PATH.exists():
        logger.warning(f"Deal registry not found: {DEAL_REGISTRY_PATH}")
        return {}
    try:
        with open(DEAL_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load deal registry: {e}")
        return {}


def _get_deal_from_registry(deal_id: str) -> Optional[Dict[str, Any]]:
    """Get a deal from the registry by ID."""
    registry = _load_deal_registry()
    deals = registry.get("deals", {})
    return deals.get(deal_id)


def _load_deal_events(deal_id: str, max_events: int = 20) -> List[EventContext]:
    """Load events for a deal from the events directory."""
    events_file = EVENTS_DIR / f"{deal_id}.jsonl"
    if not events_file.exists():
        return []

    events: List[EventContext] = []
    try:
        with open(events_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    events.append(
                        EventContext(
                            event_id=e.get("event_id", ""),
                            event_type=e.get("event_type", "unknown"),
                            timestamp=e.get("timestamp", ""),
                            actor=e.get("actor", "system"),
                            summary=e.get("summary", e.get("event_type", "")),
                            details=e.get("details", {}),
                        )
                    )
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Failed to load events for {deal_id}: {e}")

    # Sort by timestamp descending, take most recent
    events.sort(key=lambda x: x.timestamp, reverse=True)
    return events[:max_events]


def _load_case_file(deal_id: str, deal_path: Optional[Path] = None) -> tuple[Optional[str], List[str]]:
    """Load case summary and key facts from case file."""
    # Try to find case file in deal folder
    if deal_path and deal_path.exists():
        case_file = deal_path / "CASE-FILE.md"
        if case_file.exists():
            try:
                content = case_file.read_text(encoding="utf-8")
                # Extract summary section
                summary = None
                key_facts = []

                # Simple parsing - look for ## Summary section
                lines = content.split("\n")
                in_summary = False
                in_facts = False
                summary_lines = []

                for line in lines:
                    if line.startswith("## Summary") or line.startswith("## Executive Summary"):
                        in_summary = True
                        in_facts = False
                        continue
                    elif line.startswith("## Key Facts") or line.startswith("## Facts"):
                        in_summary = False
                        in_facts = True
                        continue
                    elif line.startswith("## "):
                        in_summary = False
                        in_facts = False
                        continue

                    if in_summary and line.strip():
                        summary_lines.append(line.strip())
                    elif in_facts and line.strip().startswith("- "):
                        key_facts.append(line.strip()[2:])

                if summary_lines:
                    summary = " ".join(summary_lines[:10])  # First 10 lines

                return summary, key_facts[:20]
            except Exception as e:
                logger.error(f"Failed to load case file: {e}")

    return None, []


def _query_rag(query: str, deal_id: Optional[str] = None, top_k: int = 5) -> List[RAGChunk]:
    """Query RAG for relevant evidence."""
    try:
        params = {"query": query, "top_k": top_k}
        if deal_id:
            params["filter"] = json.dumps({"deal_id": deal_id})

        response = httpx.post(
            f"{RAG_API_URL}/rag/search",
            json=params,
            timeout=10.0,
        )
        if response.status_code != 200:
            logger.warning(f"RAG query failed: {response.status_code}")
            return []

        data = response.json()
        chunks = []
        for item in data.get("results", []):
            chunks.append(
                RAGChunk(
                    content=item.get("content", ""),
                    source=item.get("source", item.get("url", "unknown")),
                    score=float(item.get("score", 0.0)),
                    metadata=item.get("metadata", {}),
                )
            )
        return chunks
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        return []


def _extract_material_links(deal_record: Dict[str, Any]) -> List[MaterialLink]:
    """Extract material links from deal record enrichment data."""
    links = []

    # Check materials field from enrichment
    materials = deal_record.get("materials", [])
    for mat in materials:
        if isinstance(mat, dict):
            links.append(
                MaterialLink(
                    url=mat.get("url", ""),
                    title=mat.get("title", mat.get("filename")),
                    link_type=mat.get("type", "unknown"),
                    extracted_at=mat.get("extracted_at"),
                )
            )
        elif isinstance(mat, str):
            links.append(MaterialLink(url=mat))

    return links


def _find_deal_path(deal_id: str, canonical_name: Optional[str] = None) -> Optional[Path]:
    """Find the DataRoom path for a deal."""
    pipeline_root = DATAROOM_ROOT / "00-PIPELINE"

    # Search in each stage folder
    for stage_dir in pipeline_root.iterdir():
        if not stage_dir.is_dir():
            continue

        # Try by canonical name first
        if canonical_name:
            deal_folder = stage_dir / canonical_name
            if deal_folder.exists():
                return deal_folder

        # Try to find by deal_id in folder name
        for folder in stage_dir.iterdir():
            if folder.is_dir() and deal_id in folder.name:
                return folder

    return None


def build_context_pack(
    deal_id: str,
    *,
    action_type: Optional[str] = None,
    rag_query: Optional[str] = None,
    include_rag: bool = True,
    include_events: bool = True,
    max_events: int = 10,
    max_rag_chunks: int = 5,
) -> ContextPack:
    """
    Build a complete context pack for an action.

    Args:
        deal_id: The deal ID to build context for
        action_type: Optional action type (used to customize RAG query)
        rag_query: Optional custom RAG query (overrides action_type-based query)
        include_rag: Whether to include RAG evidence
        include_events: Whether to include event history
        max_events: Maximum number of events to include
        max_rag_chunks: Maximum number of RAG chunks to include

    Returns:
        ContextPack with all gathered context
    """
    pack = ContextPack(deal_id=deal_id)

    # 1. Load deal from registry
    deal_record = _get_deal_from_registry(deal_id)
    if deal_record:
        pack.sources_queried.append("deal_registry")
        pack.deal_record = deal_record
        pack.canonical_name = deal_record.get("canonical_name")
        pack.display_name = deal_record.get("display_name", pack.canonical_name)
        pack.stage = deal_record.get("stage")
        pack.status = deal_record.get("status")

        # Financials
        pack.asking_price = deal_record.get("asking_price")
        pack.revenue = deal_record.get("revenue")
        pack.ebitda = deal_record.get("ebitda")

        # Broker info
        broker_info = deal_record.get("broker_info") or deal_record.get("broker")
        if broker_info and isinstance(broker_info, dict):
            pack.broker = BrokerContext(
                name=broker_info.get("name"),
                email=broker_info.get("email"),
                firm=broker_info.get("firm"),
                phone=broker_info.get("phone"),
                domain=broker_info.get("domain"),
            )

        # Material links from enrichment
        pack.extracted_links = _extract_material_links(deal_record)
        if pack.extracted_links:
            pack.sources_queried.append("enrichment_materials")
    else:
        pack.errors.append(f"Deal not found in registry: {deal_id}")

    # 2. Load events
    if include_events:
        events = _load_deal_events(deal_id, max_events=max_events)
        if events:
            pack.sources_queried.append("events")
            pack.recent_events = events

    # 3. Load case file
    deal_path = _find_deal_path(deal_id, pack.canonical_name)
    if deal_path:
        summary, facts = _load_case_file(deal_id, deal_path)
        if summary:
            pack.sources_queried.append("case_file")
            pack.case_summary = summary
            pack.key_facts = facts

    # 4. Query RAG
    if include_rag:
        query = rag_query
        if not query and action_type:
            # Generate query based on action type
            if "REQUEST_DOCS" in action_type.upper():
                query = f"financial documents due diligence {pack.display_name or deal_id}"
            elif "DRAFT_EMAIL" in action_type.upper():
                query = f"broker communication {pack.display_name or deal_id}"
            elif "LOI" in action_type.upper():
                query = f"letter of intent terms {pack.display_name or deal_id}"
            else:
                query = f"deal overview {pack.display_name or deal_id}"

        if query:
            chunks = _query_rag(query, deal_id=deal_id, top_k=max_rag_chunks)
            if chunks:
                pack.sources_queried.append("rag")
                pack.rag_evidence = chunks

    return pack


# Convenience function for quick context in prompts
def get_deal_context_for_prompt(deal_id: str, action_type: Optional[str] = None) -> str:
    """Get formatted context string for LLM prompts."""
    pack = build_context_pack(deal_id, action_type=action_type)
    return pack.to_prompt_context()
