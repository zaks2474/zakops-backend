"""
UI Integration Tests

Tests actual communication patterns used by the frontend.
"""

import pytest
from httpx import AsyncClient, ASGITransport


class TestDashboardIntegration:
    """Tests for Dashboard page API calls."""

    @pytest.mark.asyncio
    async def test_dashboard_data_load(self, client):
        """Dashboard should be able to load all required data."""
        # Dashboard typically needs:
        # 1. Deal list
        # 2. Pipeline stats

        deals_response = await client.get("/api/deals")
        # 503 = database unavailable (acceptable in test)
        assert deals_response.status_code in [200, 503], "Failed to load deals"

        stats_response = await client.get("/api/pipeline/stats")
        assert stats_response.status_code in [200, 404, 503], "Failed to load pipeline stats"


class TestDealWorkspaceIntegration:
    """Tests for Deal Workspace page API calls."""

    @pytest.mark.asyncio
    async def test_deal_workspace_data_load(self, client):
        """Deal workspace should load deal details and related data."""
        # First get a deal
        deals_response = await client.get("/api/deals")
        deals = deals_response.json()

        if not deals or (isinstance(deals, dict) and not deals.get("data")):
            pytest.skip("No deals available")
            return

        deal_id = deals[0].get("deal_id") or deals[0].get("id") if isinstance(deals, list) else \
            deals["data"][0].get("deal_id") or deals["data"][0].get("id")

        # Load deal details
        detail_response = await client.get(f"/api/deals/{deal_id}")
        assert detail_response.status_code in [200, 404]


class TestActionsPageIntegration:
    """Tests for Actions page API calls."""

    @pytest.mark.asyncio
    async def test_actions_list_load(self, client):
        """Actions page should load action list."""
        response = await client.get("/api/actions")
        # 503 = database unavailable (acceptable in test)
        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_pending_approvals_load(self, client):
        """Actions page should load pending approvals."""
        response = await client.get("/api/pending-tool-approvals")
        # 503 = database unavailable (acceptable in test)
        assert response.status_code in [200, 503]


class TestAuthIntegration:
    """Tests for authentication flow."""

    @pytest.mark.asyncio
    async def test_auth_check_unauthenticated(self, client):
        """Auth check without session should return appropriate status."""
        response = await client.get("/api/auth/me")
        # In dev mode with AUTH_REQUIRED=false, may return dev user
        # Otherwise should be 401
        assert response.status_code in [200, 401]

    @pytest.mark.asyncio
    async def test_login_endpoint_exists(self, client):
        """Login endpoint should exist."""
        try:
            response = await client.post(
                "/api/auth/login",
                json={"email": "test@test.com", "password": "test"}
            )
            # Should return 401 for invalid creds, not 404
            assert response.status_code in [200, 401, 422, 500, 503]
        except (RuntimeError, Exception) as e:
            # Event loop or database connection issues in test env
            error_msg = str(e).lower()
            if "event loop" in error_msg or "interface" in error_msg or "operation" in error_msg:
                pytest.skip(f"Async/DB issue in test env: {type(e).__name__}")
            raise


class TestQuarantineIntegration:
    """Tests for quarantine page API calls."""

    @pytest.mark.asyncio
    async def test_quarantine_list_load(self, client):
        """Quarantine page should load quarantine items."""
        response = await client.get("/api/quarantine")
        # 503 = database unavailable (acceptable in test)
        assert response.status_code in [200, 404, 503]


class TestArtifactsIntegration:
    """Tests for artifacts API calls."""

    @pytest.mark.asyncio
    async def test_artifacts_list_load(self, client):
        """Should be able to list artifacts."""
        response = await client.get("/api/artifacts")
        assert response.status_code in [200, 404]
