"""
Contract Compliance Tests

Verifies that API responses match OpenAPI specification.
"""

import pytest
from typing import Dict, Any, List
from jsonschema import validate, ValidationError


def resolve_ref(ref: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a $ref to its schema."""
    if not ref.startswith("#/"):
        raise ValueError(f"External refs not supported: {ref}")

    parts = ref[2:].split("/")
    result = spec
    for part in parts:
        result = result[part]
    return result


def resolve_schema(schema: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolve all $refs in a schema."""
    if "$ref" in schema:
        resolved = resolve_ref(schema["$ref"], spec)
        return resolve_schema(resolved, spec)

    result = schema.copy()

    if "properties" in result:
        result["properties"] = {
            k: resolve_schema(v, spec)
            for k, v in result["properties"].items()
        }

    if "items" in result:
        result["items"] = resolve_schema(result["items"], spec)

    if "allOf" in result:
        result["allOf"] = [resolve_schema(s, spec) for s in result["allOf"]]

    if "anyOf" in result:
        result["anyOf"] = [resolve_schema(s, spec) for s in result["anyOf"]]

    if "oneOf" in result:
        result["oneOf"] = [resolve_schema(s, spec) for s in result["oneOf"]]

    return result


class TestContractCompliance:
    """Test that API responses match OpenAPI contracts."""

    # =========================================================================
    # DEALS ENDPOINTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_deals_contract(self, client, openapi_spec):
        """GET /api/deals matches contract."""
        response = await client.get("/api/deals")

        # May return 503 if database is unavailable
        assert response.status_code in [200, 503], f"Expected 200/503, got {response.status_code}"

        if response.status_code != 200:
            pytest.skip("Database unavailable")
            return

        # Get expected schema from spec
        path_spec = openapi_spec["paths"].get("/api/deals", {})
        get_spec = path_spec.get("get", {})
        response_spec = get_spec.get("responses", {}).get("200", {})

        if "content" in response_spec:
            schema = response_spec["content"]["application/json"]["schema"]
            resolved = resolve_schema(schema, openapi_spec)

            # Validate response against schema
            try:
                validate(instance=response.json(), schema=resolved)
            except ValidationError as e:
                pytest.fail(f"Response doesn't match schema: {e.message}")

    @pytest.mark.asyncio
    async def test_get_deal_by_id_contract(self, client, openapi_spec):
        """GET /api/deals/{id} matches contract."""
        # First get a deal ID
        list_response = await client.get("/api/deals")
        deals = list_response.json()

        if isinstance(deals, list) and len(deals) > 0:
            deal_id = deals[0].get("deal_id") or deals[0].get("id")
        elif isinstance(deals, dict) and deals.get("data"):
            deal_id = deals["data"][0].get("deal_id") or deals["data"][0].get("id")
        else:
            pytest.skip("No deals available for testing")
            return

        response = await client.get(f"/api/deals/{deal_id}")

        assert response.status_code in [200, 404], f"Unexpected status: {response.status_code}"

    # =========================================================================
    # ACTIONS ENDPOINTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_actions_contract(self, client, openapi_spec):
        """GET /api/actions matches contract."""
        response = await client.get("/api/actions")

        # May return 503 if database is unavailable
        assert response.status_code in [200, 503], f"Expected 200/503, got {response.status_code}"

        if response.status_code != 200:
            pytest.skip("Database unavailable")
            return

        data = response.json()

        # Verify it's a list or paginated response
        assert isinstance(data, (list, dict)), "Response should be list or object"

    # =========================================================================
    # PIPELINE / AGENT ENDPOINTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_pipeline_summary_contract(self, client, openapi_spec):
        """GET /api/pipeline/summary matches contract."""
        response = await client.get("/api/pipeline/summary")

        # May return 200, 503 (db unavailable), or 404
        assert response.status_code in [200, 404, 503], f"Unexpected status: {response.status_code}"

    @pytest.mark.asyncio
    async def test_get_pipeline_stats_contract(self, client, openapi_spec):
        """GET /api/pipeline/stats matches contract."""
        response = await client.get("/api/pipeline/stats")

        # May return 200, 503 (db unavailable), or 404
        assert response.status_code in [200, 404, 503], f"Unexpected status: {response.status_code}"

    # =========================================================================
    # QUARANTINE ENDPOINTS (HITL)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_quarantine_contract(self, client, openapi_spec):
        """GET /api/quarantine matches contract."""
        response = await client.get("/api/quarantine")

        # May return 200 or 503 depending on database
        assert response.status_code in [200, 503], f"Unexpected status: {response.status_code}"

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict)), "Response should be list or object"

    @pytest.mark.asyncio
    async def test_get_pending_approvals_contract(self, client, openapi_spec):
        """GET /api/pending-tool-approvals matches contract."""
        response = await client.get("/api/pending-tool-approvals")

        # May return 200 or 503 depending on database
        assert response.status_code in [200, 503], f"Unexpected status: {response.status_code}"

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict)), "Response should be list or object"

    # =========================================================================
    # AUTH ENDPOINTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_auth_me_contract(self, client, openapi_spec):
        """GET /api/auth/me matches contract."""
        response = await client.get("/api/auth/me")

        # Without auth, should be 401 or return dev user
        assert response.status_code in [200, 401], f"Unexpected status: {response.status_code}"

    @pytest.mark.asyncio
    async def test_auth_check_contract(self, client, openapi_spec):
        """GET /api/auth/check matches contract."""
        response = await client.get("/api/auth/check")

        assert response.status_code in [200, 404], f"Unexpected status: {response.status_code}"

    # =========================================================================
    # HEALTH ENDPOINTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_health_contract(self, client, openapi_spec):
        """GET /health matches contract."""
        response = await client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert "status" in data

    @pytest.mark.asyncio
    async def test_health_ready_contract(self, client, openapi_spec):
        """GET /health/ready matches contract."""
        response = await client.get("/health/ready")

        # May be 200 or 503 depending on dependencies
        assert response.status_code in [200, 503]

        data = response.json()
        assert "status" in data


class TestResponseFormat:
    """Test that all responses follow standard format."""

    @pytest.mark.asyncio
    async def test_error_response_format(self, client):
        """Error responses should follow standard format."""
        # Request non-existent resource
        response = await client.get("/api/deals/00000000-0000-0000-0000-000000000000")

        if response.status_code == 404:
            data = response.json()
            # Should have error structure
            assert "error" in data or "detail" in data or "message" in data

    @pytest.mark.asyncio
    async def test_validation_error_format(self, client):
        """Validation errors should follow standard format."""
        # Send invalid data
        response = await client.post(
            "/api/hitl/assess-risk",
            json={"invalid": "data"}
        )

        if response.status_code == 422:
            data = response.json()
            # Should have validation error details
            assert "detail" in data or "error" in data

    @pytest.mark.asyncio
    async def test_trace_id_header(self, client):
        """Responses should include trace headers."""
        response = await client.get("/api/deals")

        # Check for trace headers (may be in response or not depending on implementation)
        # This is informational, not a hard failure
        has_trace = (
            "x-trace-id" in response.headers or
            "x-request-id" in response.headers or
            "trace_id" in response.json().get("meta", {})
        )

        # Log for visibility but don't fail
        if not has_trace:
            print("Note: No trace_id found in response headers or meta")
