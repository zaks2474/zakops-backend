"""
Database Adapter - Compatibility Layer

Supports both SQLite (legacy) and PostgreSQL (target) to enable
zero-downtime migration.

Features:
- Automatic query syntax translation (? to $1, $2, etc.)
- Dual-write mode for safe migration
- Connection pooling for PostgreSQL
- Unified interface for both backends
"""

from __future__ import annotations

import os
import re
import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# SQLite support
import sqlite3

# Async SQLite (optional but recommended)
try:
    import aiosqlite
    HAS_AIOSQLITE = True
except ImportError:
    HAS_AIOSQLITE = False

# PostgreSQL support
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

logger = logging.getLogger(__name__)


class DatabaseBackend(Enum):
    """Supported database backends."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class DatabaseConfig:
    """Database configuration from environment variables."""

    def __init__(self):
        self.backend = self._get_backend()
        self.sqlite_path = os.getenv(
            "SQLITE_PATH",
            os.getenv("ZAKOPS_STATE_DB", "/home/zaks/DataRoom/.deal-registry/ingest_state.db")
        )
        self.postgres_url = os.getenv(
            "DATABASE_URL",
            "postgresql://dealengine:changeme@localhost:5435/zakops"
        )
        # Dual-write mode: write to both databases during migration
        self.dual_write = os.getenv("DUAL_WRITE_ENABLED", "false").lower() == "true"
        # Read preference when dual-write is enabled
        self.read_from = os.getenv("READ_FROM", "sqlite").lower()

    def _get_backend(self) -> DatabaseBackend:
        """Determine which backend to use."""
        backend = os.getenv("DATABASE_BACKEND", "sqlite").lower()
        if backend == "postgresql":
            if not HAS_ASYNCPG:
                logger.warning("asyncpg not installed, falling back to SQLite")
                return DatabaseBackend.SQLITE
            return DatabaseBackend.POSTGRESQL
        return DatabaseBackend.SQLITE

    def __repr__(self) -> str:
        return (
            f"DatabaseConfig(backend={self.backend.value}, "
            f"dual_write={self.dual_write}, read_from={self.read_from})"
        )


class DatabaseAdapter:
    """
    Unified database adapter supporting both SQLite and PostgreSQL.

    Usage:
        db = DatabaseAdapter()
        await db.connect()

        # Queries work the same regardless of backend
        rows = await db.fetch("SELECT * FROM deals WHERE id = $1", deal_id)
        await db.execute("INSERT INTO deals (name) VALUES ($1)", name)

        await db.disconnect()

    Query Syntax:
        Use PostgreSQL-style $1, $2 placeholders. They will be automatically
        converted to ? for SQLite.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._pg_pool: Optional[asyncpg.Pool] = None
        self._sqlite_conn: Optional[Union[sqlite3.Connection, aiosqlite.Connection]] = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to the configured database backend(s)."""
        if self._connected:
            return

        logger.info(f"Connecting to database: {self.config}")

        # Connect to PostgreSQL if it's the primary or dual-write is enabled
        if self.config.backend == DatabaseBackend.POSTGRESQL or self.config.dual_write:
            if HAS_ASYNCPG:
                try:
                    self._pg_pool = await asyncpg.create_pool(
                        self.config.postgres_url,
                        min_size=2,
                        max_size=10,
                        command_timeout=60
                    )
                    logger.info("Connected to PostgreSQL")
                except Exception as e:
                    logger.error(f"Failed to connect to PostgreSQL: {e}")
                    if self.config.backend == DatabaseBackend.POSTGRESQL:
                        raise

        # Connect to SQLite if it's the primary or dual-write is enabled
        if self.config.backend == DatabaseBackend.SQLITE or self.config.dual_write:
            if HAS_AIOSQLITE:
                self._sqlite_conn = await aiosqlite.connect(
                    self.config.sqlite_path,
                    check_same_thread=False
                )
                self._sqlite_conn.row_factory = aiosqlite.Row
            else:
                self._sqlite_conn = sqlite3.connect(
                    self.config.sqlite_path,
                    check_same_thread=False
                )
                self._sqlite_conn.row_factory = sqlite3.Row
            logger.info(f"Connected to SQLite: {self.config.sqlite_path}")

        self._connected = True

    async def disconnect(self) -> None:
        """Disconnect from all databases."""
        if self._pg_pool:
            await self._pg_pool.close()
            self._pg_pool = None
            logger.info("Disconnected from PostgreSQL")

        if self._sqlite_conn:
            if HAS_AIOSQLITE and isinstance(self._sqlite_conn, aiosqlite.Connection):
                await self._sqlite_conn.close()
            else:
                self._sqlite_conn.close()
            self._sqlite_conn = None
            logger.info("Disconnected from SQLite")

        self._connected = False

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Fetch multiple rows.

        Args:
            query: SQL query with $1, $2, etc. placeholders
            *args: Query parameters

        Returns:
            List of dictionaries representing rows
        """
        if not self._connected:
            await self.connect()

        # Determine which backend to read from
        if self.config.dual_write:
            if self.config.read_from == "postgresql" and self._pg_pool:
                return await self._pg_fetch(query, *args)
            elif self._sqlite_conn:
                return await self._sqlite_fetch(query, *args)

        if self.config.backend == DatabaseBackend.POSTGRESQL and self._pg_pool:
            return await self._pg_fetch(query, *args)

        return await self._sqlite_fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Fetch a single row."""
        rows = await self.fetch(query, *args)
        return rows[0] if rows else None

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value from the first column of the first row."""
        row = await self.fetchrow(query, *args)
        if row:
            return list(row.values())[0]
        return None

    async def execute(self, query: str, *args) -> str:
        """
        Execute a query (INSERT, UPDATE, DELETE).

        If dual-write is enabled, writes to both databases.

        Args:
            query: SQL query with $1, $2, etc. placeholders
            *args: Query parameters

        Returns:
            Status string (e.g., "INSERT 0 1")
        """
        if not self._connected:
            await self.connect()

        result = ""

        # Primary backend
        if self.config.backend == DatabaseBackend.POSTGRESQL and self._pg_pool:
            result = await self._pg_execute(query, *args)
        elif self._sqlite_conn:
            result = await self._sqlite_execute(query, *args)

        # Dual-write to secondary backend
        if self.config.dual_write:
            try:
                if self.config.backend == DatabaseBackend.POSTGRESQL and self._sqlite_conn:
                    # Also write to SQLite
                    await self._sqlite_execute(query, *args)
                elif self.config.backend == DatabaseBackend.SQLITE and self._pg_pool:
                    # Also write to PostgreSQL
                    await self._pg_execute(query, *args)
            except Exception as e:
                logger.warning(f"Dual-write to secondary backend failed: {e}")

        return result

    async def executemany(self, query: str, args_list: List[tuple]) -> str:
        """Execute a query multiple times with different parameters."""
        if not self._connected:
            await self.connect()

        if self.config.backend == DatabaseBackend.POSTGRESQL and self._pg_pool:
            async with self._pg_pool.acquire() as conn:
                await conn.executemany(query, args_list)
            return "OK"

        sqlite_query = self._convert_to_sqlite(query)
        if HAS_AIOSQLITE and isinstance(self._sqlite_conn, aiosqlite.Connection):
            await self._sqlite_conn.executemany(sqlite_query, args_list)
            await self._sqlite_conn.commit()
        else:
            self._sqlite_conn.executemany(sqlite_query, args_list)
            self._sqlite_conn.commit()
        return "OK"

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for transactions.

        Usage:
            async with db.transaction():
                await db.execute("INSERT ...")
                await db.execute("UPDATE ...")
        """
        if self.config.backend == DatabaseBackend.POSTGRESQL and self._pg_pool:
            async with self._pg_pool.acquire() as conn:
                async with conn.transaction():
                    yield conn
        else:
            # SQLite transaction
            try:
                yield self._sqlite_conn
                if HAS_AIOSQLITE and isinstance(self._sqlite_conn, aiosqlite.Connection):
                    await self._sqlite_conn.commit()
                else:
                    self._sqlite_conn.commit()
            except Exception:
                if HAS_AIOSQLITE and isinstance(self._sqlite_conn, aiosqlite.Connection):
                    await self._sqlite_conn.rollback()
                else:
                    self._sqlite_conn.rollback()
                raise

    # PostgreSQL implementations
    async def _pg_fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch rows from PostgreSQL."""
        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def _pg_execute(self, query: str, *args) -> str:
        """Execute a query on PostgreSQL."""
        async with self._pg_pool.acquire() as conn:
            return await conn.execute(query, *args)

    # SQLite implementations
    async def _sqlite_fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch rows from SQLite."""
        sqlite_query = self._convert_to_sqlite(query)

        if HAS_AIOSQLITE and isinstance(self._sqlite_conn, aiosqlite.Connection):
            async with self._sqlite_conn.execute(sqlite_query, args) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        else:
            cursor = self._sqlite_conn.execute(sqlite_query, args)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    async def _sqlite_execute(self, query: str, *args) -> str:
        """Execute a query on SQLite."""
        sqlite_query = self._convert_to_sqlite(query)

        if HAS_AIOSQLITE and isinstance(self._sqlite_conn, aiosqlite.Connection):
            await self._sqlite_conn.execute(sqlite_query, args)
            await self._sqlite_conn.commit()
        else:
            self._sqlite_conn.execute(sqlite_query, args)
            self._sqlite_conn.commit()

        return "OK"

    def _convert_to_sqlite(self, query: str) -> str:
        """Convert PostgreSQL query syntax to SQLite."""
        # Convert $1, $2, etc. to ?
        return re.sub(r'\$\d+', '?', query)


# Global instance management
_db: Optional[DatabaseAdapter] = None


async def get_database() -> DatabaseAdapter:
    """Get the global database adapter instance."""
    global _db
    if _db is None:
        _db = DatabaseAdapter()
        await _db.connect()
    return _db


async def close_database() -> None:
    """Close the global database connection."""
    global _db
    if _db:
        await _db.disconnect()
        _db = None
