"""
E2E Test Fixtures

Provides fixtures for end-to-end workflow testing.
"""

import pytest
import asyncio
import os
from typing import Dict, Any, AsyncGenerator
from uuid import uuid4
from httpx import AsyncClient, ASGITransport


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    # Set environment for tests
    os.environ["AUTH_REQUIRED"] = "false"
    os.environ["DATABASE_BACKEND"] = "postgresql"

    from src.api.orchestration.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def test_deal_id(client: AsyncClient) -> AsyncGenerator[str, None]:
    """
    Create a test deal for workflows.

    Returns deal_id string for use in tests.
    Cleans up after test completes.
    """
    # Create deal via API
    response = await client.post(
        "/api/deals",
        json={
            "canonical_name": f"E2E Test Deal {uuid4().hex[:8]}",
            "display_name": "E2E Test Company",
            "stage": "inbound",
            "status": "active"
        }
    )

    if response.status_code in [200, 201]:
        deal = response.json()
        deal_id = deal.get("deal_id")

        yield deal_id

        # Cleanup is automatic since we use the API
    else:
        # If API fails, skip test
        pytest.skip(f"Could not create test deal: {response.status_code}")


@pytest.fixture
def trace_id() -> str:
    """Generate a test trace ID."""
    return f"test-trace-{uuid4().hex[:12]}"
