#!/usr/bin/env python3
"""
Chat Evidence Builder

Builds evidence bundles for chat responses by gathering data from:
- RAG (DataRoom documents via localhost:8052)
- Events (deal event history)
- Case files (structured deal projections)
- Registry (deal status and metadata)
- Deferred actions (pending scheduled actions)

Usage:
    from chat_evidence_builder import EvidenceBuilder

    builder = EvidenceBuilder()
    bundle = await builder.build(
        query="What's the status of this deal?",
        scope={"type": "deal", "deal_id": "DEAL-2025-001"},
        options={"rag_k": 8, "max_citations": 12}
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx

# Configuration
RAG_ENDPOINT = os.getenv("RAG_ENDPOINT", "http://localhost:8052/rag/query")
RAG_SOURCE = os.getenv("RAG_SOURCE", "dataroom.local")
RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT", "5"))

# Retrieval caps (Performance Mode v1)
RETRIEVAL_TOP_K = int(os.getenv("CHAT_RETRIEVAL_TOP_K", "6"))
RETRIEVAL_MAX_SNIPPET_CHARS = int(os.getenv("CHAT_RETRIEVAL_SNIPPET_MAX", "600"))
RETRIEVAL_DEDUPE_THRESHOLD = float(os.getenv("CHAT_RETRIEVAL_DEDUPE_THRESHOLD", "0.85"))

# Evidence limits
MAX_EVIDENCE_SIZE = 40_000  # chars
MAX_PER_SOURCE = {
    "rag": 8_000,
    "events": 12_000,
    "case_file": 10_000,
    "registry": 5_000,
    "actions": 5_000,
}


@dataclass
class Citation:
    """A citation reference for grounding responses."""
    id: str
    source: str  # rag, event, case_file, registry, action
    url: Optional[str] = None
    chunk: Optional[int] = None
    similarity: Optional[float] = None
    snippet: str = ""
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    timestamp: Optional[str] = None
    field: Optional[str] = None
    value: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"id": self.id, "source": self.source, "snippet": self.snippet}
        if self.url:
            d["url"] = self.url
        if self.chunk is not None:
            d["chunk"] = self.chunk
        if self.similarity is not None:
            d["similarity"] = self.similarity
        if self.event_id:
            d["event_id"] = self.event_id
        if self.event_type:
            d["event_type"] = self.event_type
        if self.timestamp:
            d["timestamp"] = self.timestamp
        if self.field:
            d["field"] = self.field
        if self.value is not None:
            d["value"] = self.value
        return d


@dataclass
class EvidenceBundle:
    """Collection of evidence for a chat response."""
    citations: List[Citation] = field(default_factory=list)
    rag_results: List[Dict] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    case_file: Optional[Dict] = None
    registry: Optional[Dict] = None
    actions: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Summary fields
    sources_queried: List[str] = field(default_factory=list)
    rag_query: str = ""
    rag_results_count: int = 0
    events_window: str = "last_30_days"
    events_count: int = 0
    case_file_loaded: bool = False
    registry_loaded: bool = False
    actions_count: int = 0
    total_evidence_size: int = 0

    def get_evidence_summary(self) -> Dict[str, Any]:
        return {
            "sources_queried": self.sources_queried,
            "rag": {
                "query": self.rag_query,
                "results_found": self.rag_results_count,
                "top_similarity": max((c.similarity or 0 for c in self.citations if c.source == "rag"), default=0),
            },
            "events": {
                "window": self.events_window,
                "count": self.events_count,
                "types": list(set(e.get("event_type", "") for e in self.events)),
            },
            "case_file": {
                "loaded": self.case_file_loaded,
                "sections_used": list(self.case_file.keys()) if self.case_file else [],
            },
            "registry": {
                "loaded": self.registry_loaded,
                "stage": self.registry.get("stage") if self.registry else None,
            },
            "actions": {
                "count": self.actions_count,
            },
            "total_evidence_size": self.total_evidence_size,
        }

    def get_context_for_llm(self) -> str:
        """Build context string for LLM prompt."""
        parts = []

        if self.registry:
            parts.append(f"## Deal Status\n{json.dumps(self.registry, indent=2)}")

        if self.case_file:
            # Truncate case file if needed
            cf_str = json.dumps(self.case_file, indent=2)
            if len(cf_str) > MAX_PER_SOURCE["case_file"]:
                cf_str = cf_str[:MAX_PER_SOURCE["case_file"]] + "\n... [truncated]"
            parts.append(f"## Case File\n{cf_str}")

        if self.events:
            events_str = "\n".join([
                f"- [{e.get('timestamp', '')}] {e.get('event_type', 'unknown')}: {e.get('summary', e.get('data', ''))}"
                for e in self.events[:20]  # Limit events
            ])
            parts.append(f"## Recent Events\n{events_str}")

        if self.rag_results:
            rag_parts = []
            for i, r in enumerate(self.rag_results[:RETRIEVAL_TOP_K]):
                content = r.get("content", r.get("text", ""))[:RETRIEVAL_MAX_SNIPPET_CHARS]
                url = r.get("url", "unknown")
                sim = r.get("similarity", 0)
                rag_parts.append(f"[Doc {i+1}] (similarity: {sim:.2f}) {url}\n{content}")
            parts.append(f"## Document Excerpts (RAG)\n" + "\n\n".join(rag_parts))

        if self.actions:
            actions_str = "\n".join([
                f"- {a.get('action_type', 'unknown')} due {a.get('scheduled_for', 'unknown')} [{a.get('status', '')}]"
                for a in self.actions[:10]
            ])
            parts.append(f"## Pending Actions\n{actions_str}")

        context = "\n\n".join(parts)

        # Enforce total limit
        if len(context) > MAX_EVIDENCE_SIZE:
            context = context[:MAX_EVIDENCE_SIZE] + "\n\n... [evidence truncated]"

        self.total_evidence_size = len(context)
        return context


def _simple_text_hash(text: str) -> str:
    """Create a simple hash for text deduplication."""
    # Normalize: lowercase, remove extra whitespace
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized[:200].encode()).hexdigest()


def _dedupe_rag_chunks(results: List[Dict], threshold: float = RETRIEVAL_DEDUPE_THRESHOLD) -> List[Dict]:
    """
    Remove near-duplicate RAG chunks based on content similarity.

    Uses a simple approach: hash the first 200 chars of normalized content.
    Chunks with the same hash are considered duplicates; keep highest similarity.

    Args:
        results: List of RAG result dicts with 'content'/'text' and 'similarity'/'score'
        threshold: Similarity threshold (unused in hash approach, kept for future)

    Returns:
        Deduplicated list, preserving order, keeping highest-similarity duplicates
    """
    if not results:
        return results

    seen_hashes: Dict[str, Dict] = {}  # hash -> best result

    for result in results:
        content = result.get("content", result.get("text", ""))
        similarity = result.get("similarity", result.get("score", 0))

        content_hash = _simple_text_hash(content)

        if content_hash not in seen_hashes:
            seen_hashes[content_hash] = result
        else:
            # Keep the one with higher similarity
            existing_sim = seen_hashes[content_hash].get("similarity",
                          seen_hashes[content_hash].get("score", 0))
            if similarity > existing_sim:
                seen_hashes[content_hash] = result

    # Return in original order (by position in results)
    deduped = []
    seen_in_output = set()
    for result in results:
        content = result.get("content", result.get("text", ""))
        content_hash = _simple_text_hash(content)
        if content_hash not in seen_in_output:
            deduped.append(seen_hashes[content_hash])
            seen_in_output.add(content_hash)

    return deduped


def _truncate_snippet(content: str, max_chars: int = RETRIEVAL_MAX_SNIPPET_CHARS) -> str:
    """Truncate content to max chars, preferring word boundaries."""
    if len(content) <= max_chars:
        return content

    # Find last space before limit
    truncated = content[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:  # Keep at least 80% of content
        truncated = truncated[:last_space]

    return truncated + "..."


class EvidenceBuilder:
    """Builds evidence bundles for chat responses."""

    def __init__(self, registry_path: str = "/home/zaks/DataRoom/.deal-registry/deal_registry.json"):
        self.registry_path = Path(registry_path)
        self.case_files_dir = Path("/home/zaks/DataRoom/.deal-registry/case_files")
        self.events_dir = Path("/home/zaks/DataRoom/.deal-registry/events")
        self.actions_path = Path("/home/zaks/DataRoom/.deal-registry/deferred_actions.json")

    async def build(
        self,
        query: str,
        scope: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> EvidenceBundle:
        """Build evidence bundle for a chat query."""
        options = options or {}
        bundle = EvidenceBundle()
        bundle.rag_query = query

        scope_type = scope.get("type", "global")
        deal_id = scope.get("deal_id")

        # Determine what to query based on scope
        if scope_type == "global":
            # Global scope: RAG only, no deal-specific data
            bundle.sources_queried = ["rag"]
            await self._fetch_rag(bundle, query, options)

        elif scope_type == "deal" and deal_id:
            # Deal scope: All sources
            bundle.sources_queried = ["rag", "events", "case_file", "registry", "actions"]

            # Parallel fetch
            await asyncio.gather(
                self._fetch_rag(bundle, query, options, deal_id=deal_id),
                self._fetch_events(bundle, deal_id),
                self._fetch_case_file(bundle, deal_id),
                self._fetch_registry(bundle, deal_id),
                self._fetch_actions(bundle, deal_id),
                return_exceptions=True
            )

        elif scope_type == "document":
            # Document scope: RAG focused on that document
            bundle.sources_queried = ["rag"]
            doc_url = scope.get("doc", {}).get("url")
            await self._fetch_rag(bundle, query, options, doc_url=doc_url)

        # Build citations from evidence
        self._build_citations(bundle)

        return bundle

    async def _fetch_rag(
        self,
        bundle: EvidenceBundle,
        query: str,
        options: Dict[str, Any],
        deal_id: Optional[str] = None,
        doc_url: Optional[str] = None
    ):
        """Fetch from RAG endpoint with retrieval caps."""
        try:
            # Use configured top_k, allow override from options
            k = options.get("rag_k", RETRIEVAL_TOP_K)

            payload = {
                "query": query,
                "match_count": k,
                "source": RAG_SOURCE,
            }

            # Add deal filter if available
            if deal_id:
                payload["filter_metadata"] = {"deal": deal_id}

            async with httpx.AsyncClient(timeout=RAG_TIMEOUT) as client:
                resp = await client.post(RAG_ENDPOINT, json=payload)

                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", data.get("matches", []))

                    # Apply retrieval caps (Performance Mode v1)
                    # 1. Deduplicate similar chunks
                    results = _dedupe_rag_chunks(results)

                    # 2. Truncate content to max snippet length
                    for result in results:
                        content_key = "content" if "content" in result else "text"
                        if content_key in result:
                            result[content_key] = _truncate_snippet(
                                result[content_key],
                                RETRIEVAL_MAX_SNIPPET_CHARS
                            )

                    # 3. Enforce top_k limit after dedupe
                    results = results[:k]

                    bundle.rag_results = results
                    bundle.rag_results_count = len(results)
                else:
                    bundle.warnings.append(f"RAG returned status {resp.status_code}")

        except httpx.TimeoutException:
            bundle.warnings.append("RAG service timeout - using limited evidence")
        except Exception as e:
            bundle.warnings.append(f"RAG service unavailable: {str(e)}")

    async def _fetch_events(self, bundle: EvidenceBundle, deal_id: str):
        """Fetch deal events."""
        try:
            events_file = self.events_dir / f"{deal_id}.jsonl"
            if events_file.exists():
                events = []
                with open(events_file, "r") as f:
                    for line in f:
                        if line.strip():
                            try:
                                events.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue

                # Get last 30 days
                cutoff = datetime.now(timezone.utc) - timedelta(days=30)
                filtered = []
                for e in events:
                    ts_str = e.get("timestamp", "")
                    try:
                        if ts_str.endswith("Z"):
                            ts_str = ts_str[:-1] + "+00:00"
                        ts = datetime.fromisoformat(ts_str)
                        if ts >= cutoff:
                            filtered.append(e)
                    except (ValueError, TypeError):
                        filtered.append(e)  # Include if can't parse date

                # Most recent first, limit to 20
                bundle.events = list(reversed(filtered[-20:]))
                bundle.events_count = len(bundle.events)
        except Exception as e:
            bundle.warnings.append(f"Events fetch failed: {str(e)}")

    async def _fetch_case_file(self, bundle: EvidenceBundle, deal_id: str):
        """Fetch deal case file."""
        try:
            cf_path = self.case_files_dir / f"{deal_id}.json"
            if cf_path.exists():
                with open(cf_path, "r") as f:
                    bundle.case_file = json.load(f)
                    bundle.case_file_loaded = True
        except Exception as e:
            bundle.warnings.append(f"Case file fetch failed: {str(e)}")

    async def _fetch_registry(self, bundle: EvidenceBundle, deal_id: str):
        """Fetch deal from registry."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, "r") as f:
                    registry = json.load(f)
                    deal = registry.get("deals", {}).get(deal_id)
                    if deal:
                        bundle.registry = {
                            "deal_id": deal_id,
                            "stage": deal.get("stage"),
                            "status": deal.get("status"),
                            "canonical_name": deal.get("canonical_name"),
                            "broker": deal.get("broker", {}).get("name") if deal.get("broker") else None,
                            "updated_at": deal.get("updated_at"),
                            "metadata": deal.get("metadata", {}),
                        }
                        bundle.registry_loaded = True
        except Exception as e:
            bundle.warnings.append(f"Registry fetch failed: {str(e)}")

    async def _fetch_actions(self, bundle: EvidenceBundle, deal_id: str):
        """Fetch deferred actions for deal."""
        try:
            if self.actions_path.exists():
                with open(self.actions_path, "r") as f:
                    data = json.load(f)
                    actions = []
                    for aid, action in data.get("actions", {}).items():
                        if action.get("deal_id") == deal_id:
                            actions.append({
                                "action_id": aid,
                                "action_type": action.get("action_type"),
                                "scheduled_for": action.get("scheduled_for"),
                                "status": action.get("status"),
                                "priority": action.get("priority"),
                            })
                    bundle.actions = actions
                    bundle.actions_count = len(actions)
        except Exception as e:
            bundle.warnings.append(f"Actions fetch failed: {str(e)}")

    def _build_citations(self, bundle: EvidenceBundle):
        """Build citation objects from evidence."""
        cite_num = 1

        # RAG citations (use retrieval cap)
        for i, result in enumerate(bundle.rag_results[:RETRIEVAL_TOP_K]):
            content = result.get("content", result.get("text", ""))
            cite = Citation(
                id=f"cite-{cite_num}",
                source="rag",
                url=result.get("url", result.get("source_url", "")),
                chunk=result.get("chunk_number", i),
                similarity=result.get("similarity", result.get("score", 0)),
                snippet=_truncate_snippet(content, 200)  # Citation snippets shorter
            )
            bundle.citations.append(cite)
            cite_num += 1

        # Event citations
        for event in bundle.events[:5]:
            cite = Citation(
                id=f"cite-{cite_num}",
                source="event",
                event_id=event.get("event_id"),
                event_type=event.get("event_type"),
                timestamp=event.get("timestamp"),
                snippet=event.get("summary", str(event.get("data", {}))[:200])
            )
            bundle.citations.append(cite)
            cite_num += 1

        # Case file citation
        if bundle.case_file:
            cite = Citation(
                id=f"cite-{cite_num}",
                source="case_file",
                field="full_case_file",
                value=bundle.case_file.get("status", {}),
                snippet=f"Case file for {bundle.registry.get('deal_id', 'unknown deal')}"
            )
            bundle.citations.append(cite)
            cite_num += 1

        # Registry citation
        if bundle.registry:
            cite = Citation(
                id=f"cite-{cite_num}",
                source="registry",
                field="deal_status",
                value=bundle.registry.get("stage"),
                snippet=f"Deal {bundle.registry.get('deal_id')} is in {bundle.registry.get('stage')} stage"
            )
            bundle.citations.append(cite)
            cite_num += 1


# Secret scanning (safety gate)
SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{40,}",                           # OpenAI keys
    r"AIza[A-Za-z0-9_-]{35}",                         # Google API keys
    r"-----BEGIN (RSA |)PRIVATE KEY-----",            # PEM keys
    r"(password|passwd|pwd)[\s:=]+\S+",               # Password fields
    r"Bearer\s+[A-Za-z0-9\-._~+/]+=*",                # Bearer tokens
]

def scan_for_secrets(text: str) -> Tuple[bool, List[str]]:
    """Scan text for secret patterns. Returns (blocked, matches)."""
    matches = []
    for pattern in SECRET_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append(match.group()[:20] + "...")  # Truncate match
    return (len(matches) > 0, matches)


def redact_secrets(text: str) -> str:
    """Redact detected secrets from text."""
    for pattern in SECRET_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
    return text
