"""
Inbox Pattern Implementation

Phase 3: Execution Hardening

Provides consumer-side deduplication for exactly-once processing.

Usage:
    from src.core.inbox import InboxGuard

    async with InboxGuard(event_id, consumer_id="my-consumer") as guard:
        if guard.should_process:
            # Process the event
            await do_something(event)
        else:
            # Already processed, skip
            pass
"""

from .guard import InboxGuard, is_processed, mark_processed

__all__ = [
    "InboxGuard",
    "is_processed",
    "mark_processed",
]
