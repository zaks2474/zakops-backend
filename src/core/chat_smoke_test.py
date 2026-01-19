#!/usr/bin/env python3
"""
Chat API Smoke Test

Tests the chat backend endpoints:
- POST /api/chat (SSE streaming)
- POST /api/chat/complete (non-streaming)
- GET /api/chat/session/{id}

Usage:
    python3 chat_smoke_test.py
    python3 chat_smoke_test.py --api-url http://localhost:8090
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict, List

import httpx

# Default configuration
DEFAULT_API_URL = "http://localhost:8090"


class ChatSmokeTest:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.passed = 0
        self.failed = 0
        self.results: List[Dict[str, Any]] = []

    def log_result(self, name: str, passed: bool, details: str = ""):
        status = "PASS" if passed else "FAIL"
        icon = "\033[92m✓\033[0m" if passed else "\033[91m✗\033[0m"
        print(f"{icon} {name}: {status} {details}")

        if passed:
            self.passed += 1
        else:
            self.failed += 1

        self.results.append({
            "name": name,
            "passed": passed,
            "details": details,
        })

    async def test_health(self) -> bool:
        """Test that the API is healthy."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.api_url}/health")
                if resp.status_code == 200:
                    self.log_result("Health Check", True, f"HTTP {resp.status_code}")
                    return True
                else:
                    self.log_result("Health Check", False, f"HTTP {resp.status_code}")
                    return False
        except Exception as e:
            self.log_result("Health Check", False, str(e))
            return False

    async def test_chat_complete(self) -> bool:
        """Test non-streaming chat endpoint."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                payload = {
                    "query": "What is the status of DEAL-2025-001?",
                    "scope": {
                        "type": "deal",
                        "deal_id": "DEAL-2025-001"
                    }
                }

                start = time.time()
                resp = await client.post(
                    f"{self.api_url}/api/chat/complete",
                    json=payload
                )
                latency = int((time.time() - start) * 1000)

                if resp.status_code == 200:
                    data = resp.json()
                    has_content = bool(data.get("content"))
                    has_evidence = data.get("evidence_summary") is not None

                    if has_content:
                        self.log_result(
                            "Chat Complete",
                            True,
                            f"HTTP {resp.status_code}, {latency}ms, content_len={len(data['content'])}"
                        )
                        return True
                    else:
                        self.log_result("Chat Complete", False, "No content in response")
                        return False
                else:
                    self.log_result("Chat Complete", False, f"HTTP {resp.status_code}: {resp.text[:100]}")
                    return False
        except httpx.TimeoutException:
            self.log_result("Chat Complete", False, "Timeout (60s)")
            return False
        except Exception as e:
            self.log_result("Chat Complete", False, str(e))
            return False

    async def test_chat_stream(self) -> bool:
        """Test SSE streaming chat endpoint."""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                payload = {
                    "query": "Hello, what can you help me with?",
                    "scope": {"type": "global"}
                }

                start = time.time()
                async with client.stream(
                    "POST",
                    f"{self.api_url}/api/chat",
                    json=payload
                ) as resp:
                    if resp.status_code != 200:
                        self.log_result("Chat Stream", False, f"HTTP {resp.status_code}")
                        return False

                    events_received = 0
                    tokens_received = 0
                    got_done = False

                    async for line in resp.aiter_lines():
                        if line.startswith("event: "):
                            event_type = line[7:].strip()
                            events_received += 1
                        elif line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if "token" in data:
                                    tokens_received += 1
                            except json.JSONDecodeError:
                                pass
                        elif line == "" and events_received > 0:
                            # End of event
                            pass

                        # Check for done event
                        if "done" in line:
                            got_done = True

                latency = int((time.time() - start) * 1000)

                if events_received > 0:
                    self.log_result(
                        "Chat Stream",
                        True,
                        f"HTTP 200, {latency}ms, events={events_received}, tokens={tokens_received}"
                    )
                    return True
                else:
                    self.log_result("Chat Stream", False, "No events received")
                    return False

        except httpx.TimeoutException:
            self.log_result("Chat Stream", False, "Timeout (60s)")
            return False
        except Exception as e:
            self.log_result("Chat Stream", False, str(e))
            return False

    async def test_global_scope(self) -> bool:
        """Test chat with global scope."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                payload = {
                    "query": "How many deals are active?",
                    "scope": {"type": "global"}
                }

                resp = await client.post(
                    f"{self.api_url}/api/chat/complete",
                    json=payload
                )

                if resp.status_code == 200:
                    data = resp.json()
                    sources = data.get("evidence_summary", {}).get("sources_queried", [])
                    self.log_result(
                        "Global Scope",
                        True,
                        f"sources_queried={sources}"
                    )
                    return True
                else:
                    self.log_result("Global Scope", False, f"HTTP {resp.status_code}")
                    return False
        except Exception as e:
            self.log_result("Global Scope", False, str(e))
            return False

    async def test_deal_scope(self) -> bool:
        """Test chat with deal scope."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                payload = {
                    "query": "What events happened recently?",
                    "scope": {
                        "type": "deal",
                        "deal_id": "DEAL-2025-001"
                    }
                }

                resp = await client.post(
                    f"{self.api_url}/api/chat/complete",
                    json=payload
                )

                if resp.status_code == 200:
                    data = resp.json()
                    evidence = data.get("evidence_summary", {})
                    events_count = evidence.get("events", {}).get("count", 0)
                    registry_loaded = evidence.get("registry", {}).get("loaded", False)

                    self.log_result(
                        "Deal Scope",
                        True,
                        f"events={events_count}, registry_loaded={registry_loaded}"
                    )
                    return True
                else:
                    self.log_result("Deal Scope", False, f"HTTP {resp.status_code}")
                    return False
        except Exception as e:
            self.log_result("Deal Scope", False, str(e))
            return False

    async def run_all(self) -> bool:
        """Run all tests."""
        print("=" * 50)
        print("Chat API Smoke Test")
        print(f"API URL: {self.api_url}")
        print("=" * 50)
        print()

        # Check health first
        if not await self.test_health():
            print("\nAPI is not healthy, aborting remaining tests.")
            return False

        print()
        print("Testing Chat Endpoints")
        print("-" * 30)

        await self.test_chat_complete()
        await self.test_chat_stream()

        print()
        print("Testing Scope Modes")
        print("-" * 30)

        await self.test_global_scope()
        await self.test_deal_scope()

        # Summary
        print()
        print("=" * 50)
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print("=" * 50)

        return self.failed == 0


async def main():
    parser = argparse.ArgumentParser(description="Chat API Smoke Test")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"API base URL (default: {DEFAULT_API_URL})"
    )
    args = parser.parse_args()

    tester = ChatSmokeTest(args.api_url)
    success = await tester.run_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
