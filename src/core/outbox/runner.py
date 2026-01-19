"""
Outbox Processor Runner

Phase 14: Deployment

Standalone script to run the outbox processor as a background service.
Designed to be run in a separate container for production deployments.

Usage:
    python -m src.core.outbox.runner

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (required)
    OUTBOX_POLL_INTERVAL: Polling interval in seconds (default: 1.0)
    OUTBOX_BATCH_SIZE: Batch size for processing (default: 100)
    OUTBOX_MAX_ATTEMPTS: Max delivery attempts (default: 5)
    LOG_LEVEL: Logging level (default: INFO)
"""

import os
import sys
import signal
import asyncio
import logging
from typing import Optional

from .processor import OutboxProcessor

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


class OutboxRunner:
    """
    Manages the outbox processor lifecycle with graceful shutdown.
    """

    def __init__(self):
        self.processor: Optional[OutboxProcessor] = None
        self._shutdown_event = asyncio.Event()
        self._shutdown_requested = False

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown_signal, sig)

    def _handle_shutdown_signal(self, sig: signal.Signals):
        """Handle shutdown signal."""
        if self._shutdown_requested:
            logger.warning(f"Received {sig.name} again, forcing exit")
            sys.exit(1)

        logger.info(f"Received {sig.name}, initiating graceful shutdown")
        self._shutdown_requested = True
        self._shutdown_event.set()

    async def run(self):
        """Run the outbox processor until shutdown is requested."""
        # Configuration from environment
        poll_interval = float(os.getenv("OUTBOX_POLL_INTERVAL", "1.0"))
        batch_size = int(os.getenv("OUTBOX_BATCH_SIZE", "100"))
        max_attempts = int(os.getenv("OUTBOX_MAX_ATTEMPTS", "5"))

        logger.info("Starting Outbox Processor Runner")
        logger.info(f"  Poll interval: {poll_interval}s")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max attempts: {max_attempts}")

        # Set up signal handlers
        self._setup_signal_handlers()

        # Create and start processor
        self.processor = OutboxProcessor(
            poll_interval=poll_interval,
            batch_size=batch_size,
            max_attempts=max_attempts
        )

        try:
            await self.processor.start()
            logger.info("Outbox Processor is running")

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except Exception as e:
            logger.error(f"Outbox Processor error: {e}", exc_info=True)
            raise
        finally:
            # Graceful shutdown
            logger.info("Stopping Outbox Processor")
            if self.processor:
                await self.processor.stop()
            logger.info("Outbox Processor stopped")

    async def health_check(self) -> dict:
        """Return health status for monitoring."""
        status = "healthy" if self.processor and self.processor._running else "unhealthy"
        return {
            "status": status,
            "running": self.processor._running if self.processor else False,
            "shutdown_requested": self._shutdown_requested
        }


async def main():
    """Main entry point."""
    # Verify DATABASE_URL is set
    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    runner = OutboxRunner()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
