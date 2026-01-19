"""
Database abstraction layer supporting SQLite and PostgreSQL.

This module provides a unified interface for database operations that works
with both SQLite (legacy) and PostgreSQL (target) backends, enabling a
zero-downtime migration strategy.

Usage:
    from core.database import get_database, DatabaseAdapter

    # Get the global database instance
    db = await get_database()

    # Execute queries (works with both backends)
    rows = await db.fetch("SELECT * FROM actions WHERE deal_id = $1", deal_id)
    await db.execute("UPDATE actions SET status = $1 WHERE id = $2", status, id)
"""

from .adapter import (
    DatabaseAdapter,
    DatabaseBackend,
    DatabaseConfig,
    get_database,
    close_database,
)

__all__ = [
    "DatabaseAdapter",
    "DatabaseBackend",
    "DatabaseConfig",
    "get_database",
    "close_database",
]
