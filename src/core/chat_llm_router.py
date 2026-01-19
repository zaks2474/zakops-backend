#!/usr/bin/env python3
"""
Chat LLM Router

Routing policy and fallback chain for hybrid LLM providers.

Routing hierarchy:
1. Deterministic (if pattern matched) - fastest, no LLM needed
2. Gemini Flash (simple queries, cloud allowed) - fast, cheap
3. Gemini Pro (complex queries, cloud allowed) - better reasoning
4. vLLM local (default) - always available fallback

Features:
- Query complexity estimation
- Provider fallback chain
- Cloud safety gate integration
- Automatic retry with fallback
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from chat_llm_provider import (
    ChatLLMProvider,
    GeminiFlashProvider,
    GeminiProProvider,
    ProviderResponse,
    VLLMProvider,
    get_provider,
)
from chat_budget import get_budget_manager

# Configuration
CLOUD_ENABLED = os.getenv("ALLOW_CLOUD_DEFAULT", "false").lower() == "true"
COMPLEXITY_THRESHOLD_PRO = int(os.getenv("CHAT_COMPLEXITY_PRO_THRESHOLD", "5000"))


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"      # Single fact lookup, short answer
    MODERATE = "moderate"  # Multi-step, but straightforward
    COMPLEX = "complex"    # Reasoning, analysis, long context


class RoutingDecision(Enum):
    """Routing decision outcomes."""
    DETERMINISTIC = "deterministic"
    GEMINI_FLASH = "gemini-flash"
    GEMINI_PRO = "gemini-pro"
    VLLM_LOCAL = "vllm"
    FALLBACK = "fallback"


@dataclass
class RouteResult:
    """Result of routing decision."""
    decision: RoutingDecision
    reason: str
    provider: Optional[ChatLLMProvider] = None
    fallback_chain: List[str] = None

    def __post_init__(self):
        if self.fallback_chain is None:
            self.fallback_chain = []


# Simple heuristics for complexity estimation
SIMPLE_PATTERNS = [
    r"^(what|which|who|how many|list|show|count)\s",
    r"(status|stage|state)\s*(of|is|for)",
    r"^(is|are|does|do|can)\s",
]

COMPLEX_PATTERNS = [
    r"(why|explain|analyze|compare|recommend|suggest|evaluate)",
    r"(strategy|plan|approach|should we)",
    r"(summary|summarize|overview|report)",
    r"(risk|opportunity|concern|issue)",
]


def estimate_complexity(
    query: str,
    evidence_size: int = 0,
    is_deterministic: bool = False
) -> QueryComplexity:
    """
    Estimate query complexity based on patterns and evidence size.

    Args:
        query: User query text
        evidence_size: Size of evidence context in chars
        is_deterministic: True if already matched deterministic pattern

    Returns:
        QueryComplexity enum
    """
    if is_deterministic:
        return QueryComplexity.SIMPLE

    query_lower = query.lower().strip()

    # Check simple patterns
    for pattern in SIMPLE_PATTERNS:
        if re.search(pattern, query_lower):
            # Simple query, but large context might make it moderate
            if evidence_size > 10000:
                return QueryComplexity.MODERATE
            return QueryComplexity.SIMPLE

    # Check complex patterns
    for pattern in COMPLEX_PATTERNS:
        if re.search(pattern, query_lower):
            return QueryComplexity.COMPLEX

    # Default based on evidence size
    if evidence_size > COMPLEXITY_THRESHOLD_PRO:
        return QueryComplexity.COMPLEX
    elif evidence_size > 3000:
        return QueryComplexity.MODERATE
    else:
        return QueryComplexity.SIMPLE


class ChatLLMRouter:
    """
    Routes chat requests to appropriate LLM provider.

    Implements fallback chain for resilience.
    """

    def __init__(self, allow_cloud: Optional[bool] = None):
        """
        Initialize router.

        Args:
            allow_cloud: Override default cloud permission
        """
        self._allow_cloud = allow_cloud if allow_cloud is not None else CLOUD_ENABLED
        self._budget = get_budget_manager()

    def decide_route(
        self,
        query: str,
        evidence_size: int = 0,
        is_deterministic: bool = False,
        allow_cloud_override: Optional[bool] = None
    ) -> RouteResult:
        """
        Decide which provider to use for a query.

        Args:
            query: User query
            evidence_size: Evidence context size in chars
            is_deterministic: True if deterministic pattern matched
            allow_cloud_override: Override cloud permission for this request

        Returns:
            RouteResult with decision and provider
        """
        # Deterministic always wins
        if is_deterministic:
            return RouteResult(
                decision=RoutingDecision.DETERMINISTIC,
                reason="Deterministic pattern matched",
                fallback_chain=[],
            )

        # Estimate complexity
        complexity = estimate_complexity(query, evidence_size, is_deterministic)

        # Check cloud permission
        allow_cloud = allow_cloud_override if allow_cloud_override is not None else self._allow_cloud

        if allow_cloud:
            # Check budget
            can_use, budget_reason = self._budget.can_use_cloud()

            if can_use:
                if complexity == QueryComplexity.COMPLEX:
                    # Complex queries → Gemini Pro (with vLLM fallback)
                    return RouteResult(
                        decision=RoutingDecision.GEMINI_PRO,
                        reason=f"Complex query ({evidence_size} chars evidence)",
                        provider=get_provider("gemini-pro"),
                        fallback_chain=["gemini-flash", "vllm"],
                    )
                else:
                    # Simple/moderate → Gemini Flash (with vLLM fallback)
                    return RouteResult(
                        decision=RoutingDecision.GEMINI_FLASH,
                        reason=f"{complexity.value} query, cloud allowed",
                        provider=get_provider("gemini-flash"),
                        fallback_chain=["vllm"],
                    )
            else:
                # Budget exhausted → fall back to local
                return RouteResult(
                    decision=RoutingDecision.VLLM_LOCAL,
                    reason=f"Cloud unavailable: {budget_reason}",
                    provider=get_provider("vllm"),
                    fallback_chain=[],
                )
        else:
            # Cloud disabled → local only
            return RouteResult(
                decision=RoutingDecision.VLLM_LOCAL,
                reason="Cloud disabled, using local vLLM",
                provider=get_provider("vllm"),
                fallback_chain=[],
            )

    async def invoke(
        self,
        messages: List[Dict[str, str]],
        route: RouteResult,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ProviderResponse:
        """
        Invoke the provider from routing decision (no fallback).

        Args:
            messages: Chat messages
            route: Routing result from decide_route()
            temperature: Generation temperature
            max_tokens: Max output tokens

        Returns:
            ProviderResponse from the provider
        """
        if route.decision == RoutingDecision.DETERMINISTIC:
            raise ValueError("Deterministic routes should be handled before invoke()")

        response = await route.provider.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Record usage for cloud providers
        if route.decision in (RoutingDecision.GEMINI_FLASH, RoutingDecision.GEMINI_PRO):
            if response.usage:
                self._budget.record_usage(
                    route.provider.name,
                    route.provider.model,
                    response.usage.get("prompt_tokens", 0),
                    response.usage.get("completion_tokens", 0),
                )

        return response

    async def invoke_with_fallback(
        self,
        messages: List[Dict[str, str]],
        route: RouteResult,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Tuple[ProviderResponse, bool]:
        """
        Invoke with automatic fallback on failure.

        Args:
            messages: Chat messages
            route: Routing result from decide_route()
            temperature: Generation temperature
            max_tokens: Max output tokens

        Returns:
            (ProviderResponse, used_fallback) tuple
        """
        if route.decision == RoutingDecision.DETERMINISTIC:
            raise ValueError("Deterministic routes should be handled before invoke()")

        # Build provider chain
        providers = [route.provider]
        for fallback_name in route.fallback_chain:
            providers.append(get_provider(fallback_name))

        last_error = None
        tried_providers = []
        for i, provider in enumerate(providers):
            tried_providers.append(provider.name)
            try:
                response = await provider.generate(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Record usage for cloud providers
                if provider.name in ("gemini-flash", "gemini-pro"):
                    if response.usage:
                        self._budget.record_usage(
                            provider.name,
                            provider.model,
                            response.usage.get("prompt_tokens", 0),
                            response.usage.get("completion_tokens", 0),
                        )

                used_fallback = i > 0
                if used_fallback:
                    response.provider = f"{response.provider} (fallback)"

                return response, used_fallback

            except Exception as e:
                last_error = e
                continue

        # All providers failed - return graceful degradation response
        # Log the technical error but show user-friendly message
        import logging
        logging.error(f"LLM providers exhausted. Tried: {tried_providers}. Last error: {last_error}")

        return ProviderResponse(
            content="I'm sorry, I'm currently unable to process your request. The AI service is temporarily unavailable. Please try again in a few moments, or try a simpler question.",
            provider="degraded",
            model="none",
            finish_reason="error",
            usage=None,
            latency_ms=0,
        ), True

    async def stream_with_fallback(
        self,
        messages: List[Dict[str, str]],
        route: RouteResult,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[Tuple[str, str]]:
        """
        Stream with automatic fallback on failure.

        Yields:
            (chunk, provider_name) tuples
        """
        if route.decision == RoutingDecision.DETERMINISTIC:
            raise ValueError("Deterministic routes should be handled before stream()")

        providers = [route.provider]
        for fallback_name in route.fallback_chain:
            providers.append(get_provider(fallback_name))

        last_error = None
        tried_providers = []
        for provider in providers:
            tried_providers.append(provider.name)
            try:
                async for chunk in provider.stream(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield chunk, provider.name
                return  # Success
            except Exception as e:
                last_error = e
                continue

        # All providers failed - yield graceful degradation message
        import logging
        logging.error(f"LLM streaming providers exhausted. Tried: {tried_providers}. Last error: {last_error}")

        yield "I'm sorry, I'm currently unable to process your request. The AI service is temporarily unavailable. Please try again in a few moments.", "degraded"


# Singleton
_router_instance: Optional[ChatLLMRouter] = None


def get_router(allow_cloud: Optional[bool] = None) -> ChatLLMRouter:
    """Get or create the singleton router."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ChatLLMRouter(allow_cloud=allow_cloud)
    return _router_instance


if __name__ == "__main__":
    import asyncio

    async def test_router():
        router = get_router()

        # Test routing decisions
        test_queries = [
            ("How many deals are active?", 500, False),
            ("What's the status of DEAL-2025-001?", 2000, False),
            ("Analyze the risk factors for this acquisition", 8000, False),
            ("List all pending actions", 100, True),  # Deterministic
        ]

        print("Routing decisions:")
        for query, evidence_size, is_det in test_queries:
            route = router.decide_route(query, evidence_size, is_det)
            print(f"  Query: {query[:40]}...")
            print(f"    → {route.decision.value} ({route.reason})")
            print(f"    Fallback: {route.fallback_chain}")
            print()

        # Test complexity estimation
        print("Complexity estimation:")
        queries = [
            "What stage is this deal in?",
            "Summarize the last 30 days of activity",
            "Why did we reject the previous offer?",
        ]
        for q in queries:
            c = estimate_complexity(q, 3000)
            print(f"  {q[:40]}... → {c.value}")

    asyncio.run(test_router())
