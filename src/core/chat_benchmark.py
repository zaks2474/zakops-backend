#!/usr/bin/env python3
"""
Chat Performance Benchmark

Measures chat system performance across different query types.
Generates p50/p95 statistics and markdown reports.

Usage:
    python3 chat_benchmark.py run      # Run benchmark suite
    python3 chat_benchmark.py report   # Generate markdown report from last run
    python3 chat_benchmark.py quick    # Run quick sanity check (5 prompts)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Configuration
API_BASE = os.getenv("BENCHMARK_API_BASE", "http://localhost:8090")
RESULTS_DIR = Path(os.getenv("BENCHMARK_RESULTS_DIR", "/tmp/chat_benchmarks"))
DEFAULT_ITERATIONS = int(os.getenv("BENCHMARK_ITERATIONS", "3"))


@dataclass
class BenchmarkPrompt:
    """A benchmark prompt with metadata."""
    id: str
    category: str  # deterministic, rag, reasoning, deal_context
    query: str
    scope_type: str = "global"
    deal_id: Optional[str] = None
    expected_provider: Optional[str] = None  # deterministic, vllm, gemini-flash, etc.


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    prompt_id: str
    category: str
    iteration: int
    latency_ms: int
    provider_used: str
    cache_hit: bool = False
    success: bool = True
    error: Optional[str] = None
    timing_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class CategoryStats:
    """Statistics for a benchmark category."""
    category: str
    total_runs: int
    success_rate: float
    p50_ms: int
    p95_ms: int
    mean_ms: int
    min_ms: int
    max_ms: int
    cache_hit_rate: float
    provider_distribution: Dict[str, int] = field(default_factory=dict)


# Benchmark prompts
BENCHMARK_PROMPTS = [
    # Deterministic queries (should be fast)
    BenchmarkPrompt("det-1", "deterministic", "How many deals are in the system?", expected_provider="deterministic"),
    BenchmarkPrompt("det-2", "deterministic", "What deals are in screening stage?", expected_provider="deterministic"),
    BenchmarkPrompt("det-3", "deterministic", "Show me deals from Eric", expected_provider="deterministic"),
    BenchmarkPrompt("det-4", "deterministic", "What actions are due?", expected_provider="deterministic"),
    BenchmarkPrompt("det-5", "deterministic", "What changed today?", expected_provider="deterministic"),

    # RAG queries (document search)
    BenchmarkPrompt("rag-1", "rag", "What financial documents do we have?"),
    BenchmarkPrompt("rag-2", "rag", "Show me information about revenue projections"),
    BenchmarkPrompt("rag-3", "rag", "What do the LOI terms say?"),

    # Deal context queries
    BenchmarkPrompt("deal-1", "deal_context", "What's the status of this deal?",
                   scope_type="deal", deal_id="DEAL-2025-001"),
    BenchmarkPrompt("deal-2", "deal_context", "What are the key concerns for this acquisition?",
                   scope_type="deal", deal_id="DEAL-2025-001"),
    BenchmarkPrompt("deal-3", "deal_context", "What happened recently on this deal?",
                   scope_type="deal", deal_id="DEAL-2025-001"),

    # Reasoning queries (complex)
    BenchmarkPrompt("reason-1", "reasoning", "Analyze the risk factors for our current pipeline"),
    BenchmarkPrompt("reason-2", "reasoning", "What deals should we prioritize and why?"),
    BenchmarkPrompt("reason-3", "reasoning", "Summarize the last 30 days of activity across all deals"),

    # Edge cases
    BenchmarkPrompt("edge-1", "edge", ""),  # Empty query
    BenchmarkPrompt("edge-2", "edge", "a" * 500),  # Long query
    BenchmarkPrompt("edge-3", "edge", "What's 2+2?"),  # Off-topic
]


async def run_single_benchmark(
    prompt: BenchmarkPrompt,
    iteration: int,
    client: httpx.AsyncClient
) -> BenchmarkResult:
    """Run a single benchmark iteration."""
    scope = {"type": prompt.scope_type}
    if prompt.deal_id:
        scope["deal_id"] = prompt.deal_id

    payload = {
        "query": prompt.query,
        "scope": scope,
        "options": {"skip_streaming": True},
    }

    start_time = time.time()

    try:
        resp = await client.post(
            f"{API_BASE}/api/chat/complete",
            json=payload,
            timeout=120
        )

        latency_ms = int((time.time() - start_time) * 1000)

        if resp.status_code != 200:
            return BenchmarkResult(
                prompt_id=prompt.id,
                category=prompt.category,
                iteration=iteration,
                latency_ms=latency_ms,
                provider_used="error",
                success=False,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        timings = data.get("timings", {})

        return BenchmarkResult(
            prompt_id=prompt.id,
            category=prompt.category,
            iteration=iteration,
            latency_ms=latency_ms,
            provider_used=data.get("model_used", "unknown"),
            cache_hit=timings.get("cache_hit", False),
            success=True,
            timing_breakdown={
                "evidence_ms": timings.get("evidence_ms", 0),
                "llm_ms": timings.get("llm_ms", 0),
                "deterministic_ms": timings.get("deterministic_ms", 0),
            },
        )

    except httpx.TimeoutException:
        latency_ms = int((time.time() - start_time) * 1000)
        return BenchmarkResult(
            prompt_id=prompt.id,
            category=prompt.category,
            iteration=iteration,
            latency_ms=latency_ms,
            provider_used="timeout",
            success=False,
            error="Request timeout",
        )
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return BenchmarkResult(
            prompt_id=prompt.id,
            category=prompt.category,
            iteration=iteration,
            latency_ms=latency_ms,
            provider_used="error",
            success=False,
            error=str(e),
        )


def calculate_percentile(values: List[int], percentile: float) -> int:
    """Calculate percentile from a list of values."""
    if not values:
        return 0
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    return sorted_values[min(index, len(sorted_values) - 1)]


def calculate_stats(results: List[BenchmarkResult], category: str) -> CategoryStats:
    """Calculate statistics for a category."""
    category_results = [r for r in results if r.category == category]

    if not category_results:
        return CategoryStats(
            category=category,
            total_runs=0,
            success_rate=0,
            p50_ms=0,
            p95_ms=0,
            mean_ms=0,
            min_ms=0,
            max_ms=0,
            cache_hit_rate=0,
        )

    successful = [r for r in category_results if r.success]
    latencies = [r.latency_ms for r in successful]

    # Provider distribution
    providers: Dict[str, int] = {}
    for r in successful:
        providers[r.provider_used] = providers.get(r.provider_used, 0) + 1

    cache_hits = sum(1 for r in successful if r.cache_hit)

    return CategoryStats(
        category=category,
        total_runs=len(category_results),
        success_rate=len(successful) / len(category_results) if category_results else 0,
        p50_ms=calculate_percentile(latencies, 50) if latencies else 0,
        p95_ms=calculate_percentile(latencies, 95) if latencies else 0,
        mean_ms=int(statistics.mean(latencies)) if latencies else 0,
        min_ms=min(latencies) if latencies else 0,
        max_ms=max(latencies) if latencies else 0,
        cache_hit_rate=cache_hits / len(successful) if successful else 0,
        provider_distribution=providers,
    )


async def run_benchmark(
    prompts: List[BenchmarkPrompt],
    iterations: int = DEFAULT_ITERATIONS,
    progress_callback=None
) -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    results: List[BenchmarkResult] = []
    total = len(prompts) * iterations

    async with httpx.AsyncClient() as client:
        for i, prompt in enumerate(prompts):
            for iteration in range(iterations):
                current = i * iterations + iteration + 1
                if progress_callback:
                    progress_callback(current, total, prompt.id, iteration)

                result = await run_single_benchmark(prompt, iteration, client)
                results.append(result)

                # Small delay between requests
                await asyncio.sleep(0.2)

    return results


def generate_markdown_report(
    results: List[BenchmarkResult],
    run_timestamp: str
) -> str:
    """Generate markdown report from results."""
    categories = ["deterministic", "rag", "deal_context", "reasoning", "edge"]
    stats = {cat: calculate_stats(results, cat) for cat in categories}

    lines = [
        "# Chat Performance Benchmark Report",
        f"\nRun: {run_timestamp}",
        f"\nTotal prompts: {len(BENCHMARK_PROMPTS)}",
        f"\nIterations per prompt: {len(results) // len(BENCHMARK_PROMPTS) if BENCHMARK_PROMPTS else 0}",
        "",
        "## Summary by Category",
        "",
        "| Category | p50 (ms) | p95 (ms) | Mean (ms) | Success Rate | Cache Hit Rate |",
        "|----------|----------|----------|-----------|--------------|----------------|",
    ]

    for cat in categories:
        s = stats[cat]
        if s.total_runs > 0:
            lines.append(
                f"| {cat} | {s.p50_ms} | {s.p95_ms} | {s.mean_ms} | "
                f"{s.success_rate:.1%} | {s.cache_hit_rate:.1%} |"
            )

    lines.extend([
        "",
        "## Performance Targets",
        "",
        "| Metric | Target | Actual | Status |",
        "|--------|--------|--------|--------|",
    ])

    det_stats = stats.get("deterministic", CategoryStats("", 0, 0, 0, 0, 0, 0, 0, 0))
    llm_latencies = []
    for r in results:
        if r.success and r.category in ["rag", "reasoning", "deal_context"]:
            llm_latencies.append(r.latency_ms)

    llm_p50 = calculate_percentile(llm_latencies, 50) if llm_latencies else 0
    llm_p95 = calculate_percentile(llm_latencies, 95) if llm_latencies else 0

    # Targets from plan
    targets = [
        ("Deterministic p50", 500, det_stats.p50_ms),
        ("Deterministic p95", 1500, det_stats.p95_ms),
        ("LLM query p50", 6000, llm_p50),
        ("LLM query p95", 12000, llm_p95),
    ]

    for name, target, actual in targets:
        status = "PASS" if actual <= target else "FAIL"
        lines.append(f"| {name} | <{target}ms | {actual}ms | {status} |")

    lines.extend([
        "",
        "## Provider Distribution",
        "",
    ])

    all_providers: Dict[str, int] = {}
    for r in results:
        if r.success:
            all_providers[r.provider_used] = all_providers.get(r.provider_used, 0) + 1

    for provider, count in sorted(all_providers.items(), key=lambda x: -x[1]):
        pct = count / len([r for r in results if r.success]) * 100
        lines.append(f"- {provider}: {count} ({pct:.1f}%)")

    lines.extend([
        "",
        "## Detailed Results",
        "",
    ])

    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        if cat_results:
            lines.append(f"### {cat.title()}")
            lines.append("")
            lines.append("| Prompt | Iter | Latency (ms) | Provider | Cache | Status |")
            lines.append("|--------|------|--------------|----------|-------|--------|")
            for r in cat_results:
                status = "OK" if r.success else f"ERR: {r.error[:20] if r.error else 'unknown'}"
                cache = "HIT" if r.cache_hit else "-"
                lines.append(f"| {r.prompt_id} | {r.iteration} | {r.latency_ms} | {r.provider_used} | {cache} | {status} |")
            lines.append("")

    return "\n".join(lines)


def save_results(results: List[BenchmarkResult], timestamp: str):
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"benchmark_{timestamp}.json"

    with open(path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "results": [asdict(r) for r in results],
        }, f, indent=2)

    return path


def load_latest_results() -> tuple[str, List[BenchmarkResult]]:
    """Load most recent results."""
    if not RESULTS_DIR.exists():
        raise FileNotFoundError("No benchmark results found")

    files = sorted(RESULTS_DIR.glob("benchmark_*.json"), reverse=True)
    if not files:
        raise FileNotFoundError("No benchmark results found")

    with open(files[0], "r") as f:
        data = json.load(f)

    results = [BenchmarkResult(**r) for r in data["results"]]
    return data["timestamp"], results


async def main():
    parser = argparse.ArgumentParser(description="Chat Performance Benchmark")
    parser.add_argument("command", choices=["run", "report", "quick"],
                       help="Command to execute")
    parser.add_argument("--iterations", "-i", type=int, default=DEFAULT_ITERATIONS,
                       help="Number of iterations per prompt")

    args = parser.parse_args()

    if args.command == "run":
        print(f"Running benchmark with {len(BENCHMARK_PROMPTS)} prompts, {args.iterations} iterations each...")

        def progress(current, total, prompt_id, iteration):
            print(f"  [{current}/{total}] {prompt_id} (iter {iteration})")

        results = await run_benchmark(
            BENCHMARK_PROMPTS,
            iterations=args.iterations,
            progress_callback=progress
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = save_results(results, timestamp)
        print(f"\nResults saved to: {path}")

        report = generate_markdown_report(results, timestamp)
        report_path = RESULTS_DIR / f"report_{timestamp}.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Report saved to: {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for cat in ["deterministic", "rag", "deal_context", "reasoning"]:
            stats = calculate_stats(results, cat)
            if stats.total_runs > 0:
                print(f"{cat:15} p50={stats.p50_ms:5}ms  p95={stats.p95_ms:5}ms  success={stats.success_rate:.0%}")

    elif args.command == "quick":
        print("Running quick sanity check (5 prompts, 1 iteration)...")
        quick_prompts = BENCHMARK_PROMPTS[:5]

        results = await run_benchmark(quick_prompts, iterations=1)

        print("\nResults:")
        for r in results:
            status = "OK" if r.success else f"FAIL: {r.error}"
            print(f"  {r.prompt_id}: {r.latency_ms}ms ({r.provider_used}) - {status}")

    elif args.command == "report":
        try:
            timestamp, results = load_latest_results()
            report = generate_markdown_report(results, timestamp)
            print(report)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
