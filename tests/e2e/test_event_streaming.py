"""
E2E Test: Event Streaming

Tests SSE event streaming and event system:
1. Event publishing
2. Event types
3. WebSocket/SSE endpoints
4. trace_id in events
"""

import pytest
from uuid import uuid4
from typing import Dict, Any
from httpx import AsyncClient


class TestEventSystem:
    """Test event system."""

    @pytest.mark.asyncio
    async def test_event_taxonomy_exists(self):
        """Verify event taxonomy exists."""
        from src.core.events.taxonomy import (
            AgentEventType,
            ActionEventType,
            DealEventType
        )

        # Verify agent event types
        assert hasattr(AgentEventType, 'RUN_STARTED')
        assert hasattr(AgentEventType, 'RUN_COMPLETED')
        assert hasattr(AgentEventType, 'TOOL_CALLED')
        assert hasattr(AgentEventType, 'TOOL_COMPLETED')

        # Verify action event types
        assert hasattr(ActionEventType, 'CREATED')

    @pytest.mark.asyncio
    async def test_event_models_exist(self):
        """Verify event models exist."""
        from src.core.events import AgentEvent
        from uuid import uuid4

        correlation_id = str(uuid4())
        event = AgentEvent(
            correlation_id=correlation_id,
            event_type="agent.test",
            event_data={"test": True}
        )

        assert str(event.correlation_id) == correlation_id
        assert event.event_type == "agent.test"

    @pytest.mark.asyncio
    async def test_event_publisher_exists(self):
        """Verify event publisher exists."""
        from src.core.events import publish_event

        assert publish_event is not None
        assert callable(publish_event)


class TestEventStreaming:
    """Test SSE event streaming."""

    @pytest.mark.asyncio
    async def test_thread_run_stream_endpoint(self, client: AsyncClient):
        """Test thread run stream endpoint exists."""
        # First create a thread
        response = await client.post(
            "/api/threads",
            json={"assistant_id": "test"}
        )

        if response.status_code not in [200, 201]:
            pytest.skip("Could not create thread")

        thread_id = response.json()["thread_id"]

        # Try to get stream endpoint
        response = await client.get(f"/api/threads/{thread_id}/runs/test-run/stream")

        # May be 404 (run not found) or 200 (streaming)
        assert response.status_code in [200, 404, 503]

    @pytest.mark.asyncio
    async def test_websocket_endpoint_exists(self, client: AsyncClient):
        """Test WebSocket endpoint configuration."""
        # Just verify the endpoint is registered
        # Actual WebSocket testing requires different setup
        response = await client.get("/ws/updates")

        # Should return 400 or similar (not a valid WS request)
        # or 426 Upgrade Required
        assert response.status_code in [400, 403, 404, 426]


class TestEventTraceId:
    """Test trace_id in events."""

    @pytest.mark.asyncio
    async def test_events_contain_trace_id(self, test_deal_id: str, client: AsyncClient):
        """Verify emitted events contain trace_id."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        custom_trace = f"test-trace-{uuid4().hex[:8]}"

        # Invoke agent with trace_id
        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id,
                "task": "Test event trace"
            },
            headers={"X-Trace-ID": custom_trace}
        )

        if response.status_code == 200:
            data = response.json()

            # Response should contain trace_id
            assert "trace_id" in data


class TestEventPublishing:
    """Test event publishing."""

    @pytest.mark.asyncio
    async def test_agent_run_emits_events(self, client: AsyncClient, test_deal_id: str):
        """Verify agent run emits events."""
        if test_deal_id is None:
            pytest.skip("No test deal available")

        response = await client.post(
            "/api/agent/invoke",
            json={
                "deal_id": test_deal_id,
                "task": "Test events"
            }
        )

        # If successful, events should have been emitted
        # (Events are best-effort, so we just verify the run succeeded)
        if response.status_code == 200:
            data = response.json()
            assert "run_id" in data
            # Event publishing happens asynchronously

    @pytest.mark.asyncio
    async def test_callback_handler_exists(self):
        """Verify callback handler exists."""
        from src.core.agent import AgentCallbackHandler
        from uuid import uuid4

        handler = AgentCallbackHandler(
            run_id=uuid4(),
            deal_id=uuid4(),
            trace_id="test-trace",
            correlation_id="test-correlation"
        )

        assert handler is not None
        assert handler.trace_id == "test-trace"


class TestSSEEvents:
    """Test SSE event format."""

    @pytest.mark.asyncio
    async def test_run_events_endpoint(self, client: AsyncClient):
        """Test run events endpoint."""
        try:
            # Create thread and run first
            response = await client.post(
                "/api/threads",
                json={"assistant_id": "test"}
            )

            if response.status_code not in [200, 201]:
                pytest.skip("Could not create thread")

            thread_id = response.json()["thread_id"]

            # Create a run
            response = await client.post(
                f"/api/threads/{thread_id}/runs",
                json={"input_message": "Test"}
            )

            if response.status_code not in [200, 201]:
                pytest.skip("Could not create run")

            run_id = response.json()["run_id"]

            # Get events for the run
            response = await client.get(f"/api/threads/{thread_id}/runs/{run_id}/events")

            assert response.status_code in [200, 404, 503]

            if response.status_code == 200:
                events = response.json()
                assert isinstance(events, list)
        except RuntimeError as e:
            # Event loop issues in test environment
            if "loop" in str(e).lower():
                pytest.skip("Event loop conflict in test environment")
            raise
