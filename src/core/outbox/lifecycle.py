"""
Outbox Lifecycle Management

Phase 3: Execution Hardening

Integrates outbox processor with FastAPI application lifecycle.
"""

import os
import logging
from contextlib import asynccontextmanager

from .processor import start_outbox_processor, stop_outbox_processor

logger = logging.getLogger(__name__)


def is_outbox_enabled() -> bool:
    """Check if outbox processing is enabled."""
    return os.getenv("OUTBOX_ENABLED", "true").lower() == "true"


def is_outbox_processor_enabled() -> bool:
    """
    Check if this instance should run the outbox processor.

    In multi-instance deployments, only one should process to avoid
    duplicate delivery attempts.
    """
    return os.getenv("OUTBOX_PROCESSOR_ENABLED", "true").lower() == "true"


@asynccontextmanager
async def outbox_lifespan():
    """
    Lifespan context manager for outbox processor.

    Usage in FastAPI:
        from src.core.outbox.lifecycle import outbox_lifespan

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with outbox_lifespan():
                yield

        app = FastAPI(lifespan=lifespan)
    """
    if is_outbox_enabled() and is_outbox_processor_enabled():
        logger.info("Starting outbox processor...")
        processor = await start_outbox_processor(
            poll_interval=float(os.getenv("OUTBOX_POLL_INTERVAL", "1.0")),
            batch_size=int(os.getenv("OUTBOX_BATCH_SIZE", "100"))
        )
        try:
            yield processor
        finally:
            logger.info("Stopping outbox processor...")
            await stop_outbox_processor()
    else:
        reason = []
        if not is_outbox_enabled():
            reason.append("OUTBOX_ENABLED=false")
        if not is_outbox_processor_enabled():
            reason.append("OUTBOX_PROCESSOR_ENABLED=false")
        logger.info(f"Outbox processor disabled: {', '.join(reason)}")
        yield None


async def start_outbox_on_startup():
    """
    Alternative startup hook for non-lifespan usage.

    Call this in your application startup:
        @app.on_event("startup")
        async def startup():
            await start_outbox_on_startup()
    """
    if is_outbox_enabled() and is_outbox_processor_enabled():
        logger.info("Starting outbox processor on startup...")
        await start_outbox_processor(
            poll_interval=float(os.getenv("OUTBOX_POLL_INTERVAL", "1.0")),
            batch_size=int(os.getenv("OUTBOX_BATCH_SIZE", "100"))
        )
    else:
        logger.info("Outbox processor disabled")


async def stop_outbox_on_shutdown():
    """
    Alternative shutdown hook for non-lifespan usage.

    Call this in your application shutdown:
        @app.on_event("shutdown")
        async def shutdown():
            await stop_outbox_on_shutdown()
    """
    logger.info("Stopping outbox processor on shutdown...")
    await stop_outbox_processor()
