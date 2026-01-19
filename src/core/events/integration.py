"""
Event Integration Helpers (Updated for Outbox - Phase 3)

Decorators and utilities to add event publishing to existing code
without modifying the core logic.

Supports both direct publishing (Phase 2) and outbox-based
reliable delivery (Phase 3).
"""

import functools
import logging
import os
from typing import Callable, Any
from uuid import UUID

from .publisher import publish_deal_event, publish_action_event
from .taxonomy import DealEventType, ActionEventType

logger = logging.getLogger(__name__)


def use_outbox() -> bool:
    """Check if outbox should be used for event publishing."""
    return os.getenv("OUTBOX_ENABLED", "true").lower() == "true"


def emit_deal_event(event_type: DealEventType):
    """
    Decorator to emit a deal event after a function completes.

    The decorated function must return a dict with 'deal_id' key,
    or the deal_id can be passed as a keyword argument.

    If OUTBOX_ENABLED=true (default), events go through the outbox
    for guaranteed delivery. Otherwise, direct publishing is used.

    Usage:
        @emit_deal_event(DealEventType.CREATED)
        async def create_deal(deal_data: dict) -> dict:
            deal = await db.insert(...)
            return {"deal_id": deal.id, "name": deal.name}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)

            # Extract deal_id from result or kwargs
            deal_id = None
            if isinstance(result, dict):
                deal_id = result.get("deal_id") or result.get("id")
            if deal_id is None:
                deal_id = kwargs.get("deal_id")

            if deal_id:
                # Convert to UUID if string
                if isinstance(deal_id, str):
                    try:
                        deal_id = UUID(deal_id)
                    except ValueError:
                        logger.warning(f"Invalid deal_id format: {deal_id}")
                        return result

                # Publish event
                try:
                    if use_outbox():
                        # Use outbox for reliable delivery
                        from ..outbox import get_outbox_writer
                        async with get_outbox_writer() as writer:
                            await writer.write(
                                correlation_id=deal_id,
                                event_type=event_type.value,
                                event_data=result if isinstance(result, dict) else {},
                                aggregate_type="deal",
                                aggregate_id=str(deal_id)
                            )
                    else:
                        # Direct publishing (Phase 2 behavior)
                        await publish_deal_event(
                            deal_id=deal_id,
                            event_type=event_type.value,
                            event_data=result if isinstance(result, dict) else {}
                        )
                except Exception as e:
                    # Log but don't fail the operation
                    logger.warning(f"Failed to emit deal event: {e}")

            return result
        return wrapper
    return decorator


def emit_action_event(event_type: ActionEventType):
    """
    Decorator to emit an action event after a function completes.

    If OUTBOX_ENABLED=true (default), events go through the outbox
    for guaranteed delivery. Otherwise, direct publishing is used.

    Usage:
        @emit_action_event(ActionEventType.APPROVED)
        async def approve_action(action_id: UUID, correlation_id: UUID) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            result = await func(*args, **kwargs)

            # Extract IDs from result or kwargs
            action_id = None
            correlation_id = None
            deal_id = None

            if isinstance(result, dict):
                action_id = result.get("action_id") or result.get("id")
                correlation_id = result.get("correlation_id")
                deal_id = result.get("deal_id")

            action_id = action_id or kwargs.get("action_id")
            correlation_id = correlation_id or kwargs.get("correlation_id")
            deal_id = deal_id or kwargs.get("deal_id")

            if action_id and correlation_id:
                # Convert to UUID if string
                try:
                    if isinstance(action_id, str):
                        action_id = UUID(action_id)
                    if isinstance(correlation_id, str):
                        correlation_id = UUID(correlation_id)
                    if isinstance(deal_id, str):
                        deal_id = UUID(deal_id)
                except ValueError as e:
                    logger.warning(f"Invalid UUID format: {e}")
                    return result

                try:
                    if use_outbox():
                        # Use outbox for reliable delivery
                        from ..outbox import get_outbox_writer
                        event_data = result if isinstance(result, dict) else {}
                        event_data["action_id"] = str(action_id)
                        if deal_id:
                            event_data["deal_id"] = str(deal_id)

                        async with get_outbox_writer() as writer:
                            await writer.write(
                                correlation_id=correlation_id,
                                event_type=event_type.value,
                                event_data=event_data,
                                aggregate_type="action",
                                aggregate_id=str(action_id)
                            )
                    else:
                        # Direct publishing (Phase 2 behavior)
                        await publish_action_event(
                            action_id=action_id,
                            correlation_id=correlation_id,
                            event_type=event_type.value,
                            event_data=result if isinstance(result, dict) else {},
                            deal_id=deal_id
                        )
                except Exception as e:
                    logger.warning(f"Failed to emit action event: {e}")

            return result
        return wrapper
    return decorator


# Convenience function for manual event emission via outbox
async def emit_via_outbox(
    correlation_id: UUID,
    event_type: str,
    event_data: dict,
    aggregate_type: str = "event",
    aggregate_id: str = None
):
    """
    Manually emit an event via the outbox.

    Use this when you need to emit events outside of decorators.

    Usage:
        await emit_via_outbox(
            correlation_id=deal_id,
            event_type="deal.custom_event",
            event_data={"key": "value"}
        )
    """
    if use_outbox():
        from ..outbox import get_outbox_writer
        async with get_outbox_writer() as writer:
            await writer.write(
                correlation_id=correlation_id,
                event_type=event_type,
                event_data=event_data,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id or str(correlation_id)
            )
    else:
        from .publisher import publish_event
        from .models import EventBase
        event = EventBase(
            correlation_id=correlation_id,
            event_type=event_type,
            event_data=event_data,
            source="manual"
        )
        await publish_event(event)
