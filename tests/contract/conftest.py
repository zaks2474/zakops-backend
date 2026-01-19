"""
Contract Test Fixtures
"""

import pytest
import asyncio
import json
import os
from typing import Dict, Any
from httpx import AsyncClient, ASGITransport

# Load OpenAPI spec
SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "shared/openapi/zakops-api.json"
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def openapi_spec() -> Dict[str, Any]:
    """Load OpenAPI specification."""
    with open(SPEC_PATH) as f:
        return json.load(f)


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


@pytest.fixture
def spec_paths(openapi_spec) -> Dict[str, Any]:
    """Get paths from OpenAPI spec."""
    return openapi_spec.get("paths", {})


@pytest.fixture
def spec_schemas(openapi_spec) -> Dict[str, Any]:
    """Get schemas from OpenAPI spec."""
    return openapi_spec.get("components", {}).get("schemas", {})
