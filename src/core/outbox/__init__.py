"""
Outbox Pattern Implementation

Provides transactional event publishing with guaranteed delivery.

Usage:
    from src.core.outbox import OutboxWriter, get_outbox_writer

    async with get_outbox_writer() as outbox:
        # This is atomic with your business transaction
        await outbox.write(
            correlation_id=deal_id,
            event_type="action.created",
            event_data={"action_id": action_id, "title": title}
        )
"""

from .writer import OutboxWriter, get_outbox_writer
from .processor import OutboxProcessor, start_outbox_processor, stop_outbox_processor
from .models import OutboxEntry, OutboxStatus
from .transactional import TransactionalPublisher, transactional_publish
from .dlq import DLQManager, DLQEntry, DLQAction, get_dlq_manager

__all__ = [
    "OutboxWriter",
    "get_outbox_writer",
    "OutboxProcessor",
    "start_outbox_processor",
    "stop_outbox_processor",
    "OutboxEntry",
    "OutboxStatus",
    "TransactionalPublisher",
    "transactional_publish",
    "DLQManager",
    "DLQEntry",
    "DLQAction",
    "get_dlq_manager",
]
