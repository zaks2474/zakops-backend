#!/usr/bin/env python3
"""
Chat Timing Infrastructure

Provides timing/tracing dataclasses and context managers for measuring
chat request performance across all phases.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TimingTrace:
    """Detailed timing breakdown for a chat request."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Total request time
    total_ms: int = 0

    # Evidence gathering breakdown
    evidence_ms: int = 0
    evidence_breakdown: Dict[str, int] = field(default_factory=dict)
    # Keys: rag_ms, events_ms, casefile_ms, registry_ms, actions_ms

    # LLM generation time
    llm_ms: int = 0

    # Deterministic routing time (if pattern matched)
    deterministic_ms: int = 0

    # Cache status
    cache_hit: bool = False
    cache_source: Optional[str] = None  # "global" | "deal" | "doc"

    # Provider info
    provider_used: str = ""  # deterministic | gemini-flash | gemini-pro | vllm | fallback
    provider_fallback: bool = False  # True if primary failed and fell back
    provider_reason: str = ""  # Why this provider was chosen

    # Status flags
    degraded: bool = False  # True if operating in degraded mode
    degraded_reason: Optional[str] = None

    # Internal timestamps for calculation
    _start_time: float = field(default=0.0, repr=False)
    _phase_start: float = field(default=0.0, repr=False)

    def start(self):
        """Mark the start of timing."""
        self._start_time = time.time()
        self._phase_start = self._start_time

    def end(self):
        """Calculate total_ms at end of request."""
        if self._start_time > 0:
            self.total_ms = int((time.time() - self._start_time) * 1000)

    def start_phase(self):
        """Mark start of a phase for breakdown timing."""
        self._phase_start = time.time()

    def end_phase(self, phase_name: str) -> int:
        """
        End a phase and record its duration.

        Args:
            phase_name: Key for evidence_breakdown (e.g., "rag_ms", "events_ms")

        Returns:
            Duration in milliseconds
        """
        duration_ms = int((time.time() - self._phase_start) * 1000)
        self.evidence_breakdown[phase_name] = duration_ms
        self._phase_start = time.time()
        return duration_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization, excluding internal fields."""
        d = asdict(self)
        # Remove internal fields
        d.pop("_start_time", None)
        d.pop("_phase_start", None)
        return d


class TimingContext:
    """Context manager for timing a specific phase."""

    def __init__(self, timing: TimingTrace, phase_name: str):
        self.timing = timing
        self.phase_name = phase_name
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        duration_ms = int((time.time() - self._start) * 1000)
        self.timing.evidence_breakdown[self.phase_name] = duration_ms


@asynccontextmanager
async def async_timing_context(timing: TimingTrace, phase_name: str):
    """Async context manager for timing a phase."""
    start = time.time()
    try:
        yield
    finally:
        duration_ms = int((time.time() - start) * 1000)
        timing.evidence_breakdown[phase_name] = duration_ms


def create_timing(request_id: Optional[str] = None) -> TimingTrace:
    """Create a new timing trace and start the clock."""
    timing = TimingTrace(request_id=request_id or str(uuid.uuid4())[:8])
    timing.start()
    return timing


# Convenience functions for SSE events
def timing_to_progress_event(step: str, message: str) -> Dict[str, Any]:
    """Format a progress event for SSE."""
    return {
        "step": step,
        "message": message,
        "timestamp_ms": int(time.time() * 1000),
    }


def timing_to_done_event(timing: TimingTrace, **extra) -> Dict[str, Any]:
    """Format timing for the 'done' SSE event."""
    result = timing.to_dict()
    result.update(extra)
    return result


# Progress step constants
class ProgressStep:
    """Standard progress step names."""
    ROUTING = "routing"
    EVIDENCE = "evidence"
    RAG = "rag"
    EVENTS = "events"
    CASEFILE = "casefile"
    LLM = "llm"
    COMPLETE = "complete"
    ERROR = "error"


# Provider name constants
class ProviderName:
    """Standard provider names."""
    DETERMINISTIC = "deterministic"
    DIRECT_API = "direct-api"  # Legacy name for deterministic
    VLLM = "vllm"
    GEMINI_FLASH = "gemini-flash"
    GEMINI_PRO = "gemini-pro"
    FALLBACK = "fallback"


if __name__ == "__main__":
    # Quick test
    import asyncio

    async def test_timing():
        timing = create_timing()

        # Simulate phases
        with TimingContext(timing, "rag_ms"):
            await asyncio.sleep(0.1)

        async with async_timing_context(timing, "events_ms"):
            await asyncio.sleep(0.05)

        timing.provider_used = ProviderName.VLLM
        timing.end()

        print("Timing trace:")
        print(timing.to_dict())

    asyncio.run(test_timing())
