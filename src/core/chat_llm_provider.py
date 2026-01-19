#!/usr/bin/env python3
"""
Chat LLM Provider Abstraction

Provides a unified interface for multiple LLM backends:
- VLLMProvider: Local OpenAI-compatible (Qwen via vLLM)
- GeminiFlashProvider: Google Gemini 1.5 Flash (fast, cheap)
- GeminiProProvider: Google Gemini 1.5 Pro (complex reasoning)

Each provider implements:
- generate() - Non-streaming completion
- stream() - Streaming completion (async generator)
- health_check() - Provider availability check
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

# Configuration - centralized vLLM settings
# OPENAI_API_BASE is the single source of truth for vLLM endpoint
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", f"{OPENAI_API_BASE}/chat/completions")
VLLM_MODELS_ENDPOINT = f"{OPENAI_API_BASE}/models"
# DEFAULT_MODEL is used across the system - AWQ quantized Qwen 32B
VLLM_MODEL = os.getenv("VLLM_MODEL", os.getenv("DEFAULT_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ"))
VLLM_TIMEOUT_MS = int(os.getenv("VLLM_TIMEOUT_MS", "120000"))

GEMINI_API_KEY_FILE = os.getenv("GEMINI_API_KEY_FILE", os.path.expanduser("~/.gemini_api"))
GEMINI_MODEL_FLASH = os.getenv("GEMINI_MODEL_FLASH", "gemini-2.5-flash")
GEMINI_MODEL_PRO = os.getenv("GEMINI_MODEL_PRO", "gemini-2.5-pro")
GEMINI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "30000"))


def _load_gemini_api_key() -> Optional[str]:
    """Load Gemini API key from file or environment."""
    # Try environment first
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    # Try file
    key_path = Path(GEMINI_API_KEY_FILE)
    if key_path.exists():
        return key_path.read_text().strip()

    return None


@dataclass
class ProviderResponse:
    """Unified response from any provider."""
    content: str
    provider: str
    model: str
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None
    latency_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }


@dataclass
class HealthStatus:
    """Health check result."""
    healthy: bool
    provider: str
    model: str
    endpoint: str
    latency_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self.healthy else "unhealthy",
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class ChatLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (vllm, gemini-flash, gemini-pro)."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier."""
        pass

    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> ProviderResponse:
        """Generate a non-streaming completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming completion, yielding content chunks."""
        pass

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Check if the provider is available."""
        pass


class VLLMProvider(ChatLLMProvider):
    """Local vLLM provider (OpenAI-compatible API)."""

    def __init__(self, endpoint: str = VLLM_ENDPOINT, model: str = VLLM_MODEL):
        self._endpoint = endpoint
        self._model = model
        self._timeout = VLLM_TIMEOUT_MS / 1000

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> ProviderResponse:
        start = time.time()

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(self._endpoint, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]
        finish = data["choices"][0].get("finish_reason", "stop")
        usage = data.get("usage")

        return ProviderResponse(
            content=content,
            provider=self.name,
            model=self._model,
            finish_reason=finish,
            usage=usage,
            latency_ms=int((time.time() - start) * 1000),
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", self._endpoint, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue

    async def health_check(self) -> HealthStatus:
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                # Check 1: Models endpoint is reachable
                resp = await client.get(VLLM_MODELS_ENDPOINT)
                resp.raise_for_status()
                models_data = resp.json()

                # Check 2: Validate configured model is available
                available_models = [m.get("id", "") for m in models_data.get("data", [])]
                if self._model not in available_models:
                    return HealthStatus(
                        healthy=False,
                        provider=self.name,
                        model=self._model,
                        endpoint=self._endpoint,
                        latency_ms=int((time.time() - start) * 1000),
                        error=f"Model '{self._model}' not found. Available: {available_models}",
                    )

                # Check 3: Chat completions route exists (should return 405 for GET, not 404)
                chat_resp = await client.get(self._endpoint)
                if chat_resp.status_code == 404:
                    return HealthStatus(
                        healthy=False,
                        provider=self.name,
                        model=self._model,
                        endpoint=self._endpoint,
                        latency_ms=int((time.time() - start) * 1000),
                        error=f"Chat completions route not found (404). Check OPENAI_API_BASE: {OPENAI_API_BASE}",
                    )
                # 405 Method Not Allowed is expected for GET (route exists but needs POST)

            return HealthStatus(
                healthy=True,
                provider=self.name,
                model=self._model,
                endpoint=self._endpoint,
                latency_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                provider=self.name,
                model=self._model,
                endpoint=self._endpoint,
                latency_ms=int((time.time() - start) * 1000),
                error=str(e),
            )


class GeminiProvider(ChatLLMProvider):
    """Google Gemini provider base class."""

    def __init__(self, model_name: str, provider_name: str):
        self._model = model_name
        self._provider_name = provider_name
        self._api_key = _load_gemini_api_key()
        self._timeout = GEMINI_TIMEOUT_MS / 1000
        self._base_url = "https://generativelanguage.googleapis.com/v1beta"

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    @property
    def available(self) -> bool:
        """Check if API key is available."""
        return self._api_key is not None

    def _convert_messages(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Convert OpenAI-style messages to Gemini format."""
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})

        result = {"contents": contents}
        if system_instruction:
            result["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        return result

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> ProviderResponse:
        if not self._api_key:
            raise ValueError("Gemini API key not configured")

        start = time.time()

        url = f"{self._base_url}/models/{self._model}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self._api_key}

        payload = self._convert_messages(messages)
        payload["generationConfig"] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=headers, params=params, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Extract content from Gemini response
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("No candidates in Gemini response")

        # Gemini 2.5 models use internal thinking tokens which can exhaust max_tokens
        # before generating visible output, leaving "parts" empty or missing
        candidate = candidates[0]
        content_obj = candidate.get("content", {})
        parts = content_obj.get("parts", [])

        if not parts:
            finish_reason = candidate.get("finishReason", "UNKNOWN")
            if finish_reason == "MAX_TOKENS":
                raise ValueError(
                    f"Gemini {self._model} exhausted tokens during thinking. "
                    "Increase max_tokens or use gemini-flash for simpler tasks."
                )
            raise ValueError(f"No content parts in Gemini response (finishReason={finish_reason})")

        content = parts[0].get("text", "")
        finish_reason = candidate.get("finishReason", "STOP")

        # Extract usage
        usage_meta = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_meta.get("promptTokenCount", 0),
            "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
            "total_tokens": usage_meta.get("totalTokenCount", 0),
        }

        return ProviderResponse(
            content=content,
            provider=self.name,
            model=self._model,
            finish_reason=finish_reason.lower(),
            usage=usage,
            latency_ms=int((time.time() - start) * 1000),
        )

    async def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> AsyncIterator[str]:
        if not self._api_key:
            raise ValueError("Gemini API key not configured")

        url = f"{self._base_url}/models/{self._model}:streamGenerateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self._api_key, "alt": "sse"}

        payload = self._convert_messages(messages)
        payload["generationConfig"] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST", url, headers=headers, params=params, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                if parts and "text" in parts[0]:
                                    yield parts[0]["text"]
                        except json.JSONDecodeError:
                            continue

    async def health_check(self) -> HealthStatus:
        if not self._api_key:
            return HealthStatus(
                healthy=False,
                provider=self.name,
                model=self._model,
                endpoint=self._base_url,
                error="API key not configured",
            )

        start = time.time()
        try:
            # Quick test with minimal prompt
            url = f"{self._base_url}/models/{self._model}:generateContent"
            params = {"key": self._api_key}
            payload = {
                "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
                "generationConfig": {"maxOutputTokens": 5},
            }

            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(url, params=params, json=payload)
                resp.raise_for_status()

            return HealthStatus(
                healthy=True,
                provider=self.name,
                model=self._model,
                endpoint=self._base_url,
                latency_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                provider=self.name,
                model=self._model,
                endpoint=self._base_url,
                latency_ms=int((time.time() - start) * 1000),
                error=str(e),
            )


class GeminiFlashProvider(GeminiProvider):
    """Gemini 1.5 Flash - fast, cost-effective for simple queries."""

    def __init__(self):
        super().__init__(GEMINI_MODEL_FLASH, "gemini-flash")


class GeminiProProvider(GeminiProvider):
    """Gemini 1.5 Pro - better reasoning for complex queries."""

    def __init__(self):
        super().__init__(GEMINI_MODEL_PRO, "gemini-pro")


# Provider registry
_providers: Dict[str, ChatLLMProvider] = {}


def get_provider(name: str) -> ChatLLMProvider:
    """Get or create a provider by name."""
    global _providers

    if name not in _providers:
        if name == "vllm":
            _providers[name] = VLLMProvider()
        elif name == "gemini-flash":
            _providers[name] = GeminiFlashProvider()
        elif name == "gemini-pro":
            _providers[name] = GeminiProProvider()
        else:
            raise ValueError(f"Unknown provider: {name}")

    return _providers[name]


async def get_all_health() -> Dict[str, HealthStatus]:
    """Get health status for all providers."""
    providers = ["vllm", "gemini-flash", "gemini-pro"]
    results = await asyncio.gather(*[
        get_provider(p).health_check() for p in providers
    ], return_exceptions=True)

    health = {}
    for name, result in zip(providers, results):
        if isinstance(result, Exception):
            health[name] = HealthStatus(
                healthy=False,
                provider=name,
                model="unknown",
                endpoint="unknown",
                error=str(result),
            )
        else:
            health[name] = result

    return health


if __name__ == "__main__":
    import asyncio

    async def test_providers():
        print("Testing LLM providers...\n")

        # Test health
        health = await get_all_health()
        for name, status in health.items():
            print(f"{name}: {status.to_dict()}")

        print()

        # Test vLLM if healthy
        vllm = get_provider("vllm")
        vllm_health = await vllm.health_check()
        if vllm_health.healthy:
            print("Testing vLLM generation...")
            resp = await vllm.generate([
                {"role": "user", "content": "Say hello in 5 words."}
            ], max_tokens=50)
            print(f"Response: {resp.content}")
            print(f"Latency: {resp.latency_ms}ms")

    asyncio.run(test_providers())
