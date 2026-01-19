"""
Integration Test Fixtures
"""

import pytest
import asyncio
import os
from httpx import AsyncClient, ASGITransport


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def client():
    """Create async test client."""
    # Set environment for tests
    os.environ["AUTH_REQUIRED"] = "false"
    os.environ["DATABASE_BACKEND"] = "postgresql"

    from src.api.orchestration.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
