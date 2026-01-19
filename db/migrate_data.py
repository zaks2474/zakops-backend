#!/usr/bin/env python3
"""
Data Migration: SQLite → PostgreSQL

Migrates existing data from SQLite action engine to PostgreSQL while preserving
all records and relationships.

Usage:
    python -m db.migrate_data                # Run full migration
    python -m db.migrate_data --dry-run      # Preview without changes
    python -m db.migrate_data --table actions # Migrate specific table

Environment:
    SQLITE_PATH - Path to SQLite database
    DATABASE_URL - PostgreSQL connection string
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import asyncpg
except ImportError:
    print("ERROR: asyncpg not installed. Run: pip install asyncpg")
    exit(1)


SQLITE_PATH = os.getenv(
    "SQLITE_PATH",
    os.getenv("ZAKOPS_STATE_DB", "/home/zaks/DataRoom/.deal-registry/ingest_state.db")
)
POSTGRES_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://dealengine:changeme@localhost:5435/zakops"
)

# Table mapping: SQLite table -> PostgreSQL schema.table
TABLE_MAPPING = {
    "actions": "zakops.actions",
    "action_audit_events": "zakops.agent_events",  # Migrate audit to events
    "action_artifacts": "zakops.artifacts",
    "action_steps": None,  # TODO: Map to appropriate table
    "action_runner_leases": None,  # Runtime state, don't migrate
}

# Column mapping for actions table: SQLite column -> PostgreSQL column
ACTIONS_COLUMN_MAP = {
    "action_id": "action_id",
    "deal_id": "deal_id",
    "capability_id": "capability_id",
    "type": "action_type",
    "title": "title",
    "summary": "summary",
    "status": "status",
    "created_at": "created_at",
    "updated_at": "updated_at",
    "started_at": "started_at",
    "completed_at": "completed_at",
    "duration_seconds": "duration_seconds",
    "created_by": "created_by",
    "source": "source",
    "risk_level": "risk_level",
    "requires_human_review": "requires_human_review",
    "idempotency_key": "idempotency_key",
    "inputs": "inputs",
    "outputs": "outputs",
    "error": "error_message",
    "retry_count": "retry_count",
    "max_retries": "max_retries",
}


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except Exception:
        return None


def parse_json(value: Optional[str]) -> Any:
    """Parse JSON string to object."""
    if not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


class DataMigrator:
    """Migrates data from SQLite to PostgreSQL."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        self.pg_conn: Optional[asyncpg.Connection] = None
        self.stats: Dict[str, Dict[str, int]] = {}

    async def connect(self) -> None:
        """Connect to both databases."""
        print(f"SQLite: {SQLITE_PATH}")
        print(f"PostgreSQL: {POSTGRES_URL.split('@')[1] if '@' in POSTGRES_URL else POSTGRES_URL}")
        print()

        if not os.path.exists(SQLITE_PATH):
            raise FileNotFoundError(f"SQLite database not found: {SQLITE_PATH}")

        self.sqlite_conn = sqlite3.connect(SQLITE_PATH)
        self.sqlite_conn.row_factory = sqlite3.Row

        if not self.dry_run:
            self.pg_conn = await asyncpg.connect(POSTGRES_URL)

    async def disconnect(self) -> None:
        """Disconnect from databases."""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.pg_conn:
            await self.pg_conn.close()

    def get_sqlite_tables(self) -> List[str]:
        """Get list of tables in SQLite."""
        cursor = self.sqlite_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_row_count(self, table: str) -> int:
        """Get row count for a SQLite table."""
        cursor = self.sqlite_conn.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]

    async def migrate_actions(self) -> Dict[str, int]:
        """Migrate actions table from SQLite to PostgreSQL."""
        table = "actions"
        stats = {"total": 0, "migrated": 0, "skipped": 0, "errors": 0}

        cursor = self.sqlite_conn.execute("SELECT * FROM actions")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        stats["total"] = len(rows)
        print(f"  Found {stats['total']} rows in SQLite")

        if self.dry_run:
            print(f"  [DRY RUN] Would migrate {stats['total']} rows")
            return stats

        for row in rows:
            row_dict = dict(zip(columns, row))

            try:
                # Check if already exists
                existing = await self.pg_conn.fetchval(
                    "SELECT 1 FROM zakops.actions WHERE action_id = $1",
                    row_dict["action_id"]
                )

                if existing:
                    stats["skipped"] += 1
                    continue

                # Map columns and transform values
                pg_row = {
                    "action_id": row_dict["action_id"],
                    "deal_id": row_dict.get("deal_id"),
                    "capability_id": row_dict.get("capability_id") or row_dict.get("type", "UNKNOWN"),
                    "action_type": row_dict.get("type", "UNKNOWN"),
                    "title": row_dict.get("title", "Untitled"),
                    "summary": row_dict.get("summary"),
                    "status": self._map_status(row_dict.get("status", "PENDING_APPROVAL")),
                    "created_at": parse_timestamp(row_dict.get("created_at")) or datetime.utcnow(),
                    "updated_at": parse_timestamp(row_dict.get("updated_at")) or datetime.utcnow(),
                    "started_at": parse_timestamp(row_dict.get("started_at")),
                    "completed_at": parse_timestamp(row_dict.get("completed_at")),
                    "duration_seconds": row_dict.get("duration_seconds"),
                    "created_by": row_dict.get("created_by", "system"),
                    "source": row_dict.get("source", "system"),
                    "risk_level": row_dict.get("risk_level", "low"),
                    "requires_human_review": bool(row_dict.get("requires_human_review", True)),
                    "idempotency_key": row_dict.get("idempotency_key"),
                    "inputs": parse_json(row_dict.get("inputs")),
                    "outputs": parse_json(row_dict.get("outputs")),
                    "error_message": row_dict.get("error"),
                    "retry_count": row_dict.get("retry_count", 0),
                    "max_retries": row_dict.get("max_retries", 3),
                }

                # Insert into PostgreSQL
                await self.pg_conn.execute("""
                    INSERT INTO zakops.actions (
                        action_id, deal_id, capability_id, action_type, title, summary,
                        status, created_at, updated_at, started_at, completed_at,
                        duration_seconds, created_by, source, risk_level,
                        requires_human_review, idempotency_key, inputs, outputs,
                        error_message, retry_count, max_retries
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                        $15, $16, $17, $18, $19, $20, $21, $22
                    )
                    ON CONFLICT (action_id) DO NOTHING
                """,
                    pg_row["action_id"],
                    pg_row["deal_id"],
                    pg_row["capability_id"],
                    pg_row["action_type"],
                    pg_row["title"],
                    pg_row["summary"],
                    pg_row["status"],
                    pg_row["created_at"],
                    pg_row["updated_at"],
                    pg_row["started_at"],
                    pg_row["completed_at"],
                    pg_row["duration_seconds"],
                    pg_row["created_by"],
                    pg_row["source"],
                    pg_row["risk_level"],
                    pg_row["requires_human_review"],
                    pg_row["idempotency_key"],
                    json.dumps(pg_row["inputs"]),
                    json.dumps(pg_row["outputs"]) if pg_row["outputs"] else "{}",
                    pg_row["error_message"],
                    pg_row["retry_count"],
                    pg_row["max_retries"],
                )

                stats["migrated"] += 1

            except Exception as e:
                stats["errors"] += 1
                print(f"    Error migrating action {row_dict.get('action_id')}: {e}")

        return stats

    async def migrate_audit_events(self) -> Dict[str, int]:
        """Migrate action_audit_events to agent_events."""
        stats = {"total": 0, "migrated": 0, "skipped": 0, "errors": 0}

        try:
            cursor = self.sqlite_conn.execute("SELECT * FROM action_audit_events")
        except sqlite3.OperationalError:
            print("  Table action_audit_events not found, skipping")
            return stats

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        stats["total"] = len(rows)
        print(f"  Found {stats['total']} rows in SQLite")

        if self.dry_run:
            print(f"  [DRY RUN] Would migrate {stats['total']} rows")
            return stats

        for row in rows:
            row_dict = dict(zip(columns, row))

            try:
                # Generate a deterministic event_id from audit_id
                event_id = self._generate_uuid_from_string(row_dict["audit_id"])

                # Check if already exists
                existing = await self.pg_conn.fetchval(
                    "SELECT 1 FROM zakops.agent_events WHERE event_id = $1",
                    event_id
                )

                if existing:
                    stats["skipped"] += 1
                    continue

                # Map to agent_events schema
                event_type = f"action.{row_dict.get('event', 'unknown').lower()}"
                event_data = {
                    "action_id": row_dict.get("action_id"),
                    "actor": row_dict.get("actor"),
                    "details": parse_json(row_dict.get("details")),
                    "migrated_from": "action_audit_events",
                }

                await self.pg_conn.execute("""
                    INSERT INTO zakops.agent_events (
                        event_id, event_type, payload, source, created_at
                    ) VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT DO NOTHING
                """,
                    event_id,
                    event_type,
                    json.dumps(event_data),
                    "migration",
                    parse_timestamp(row_dict.get("timestamp")) or datetime.utcnow(),
                )

                stats["migrated"] += 1

            except Exception as e:
                stats["errors"] += 1
                print(f"    Error migrating audit event {row_dict.get('audit_id')}: {e}")

        return stats

    async def migrate_artifacts(self) -> Dict[str, int]:
        """Migrate action_artifacts to artifacts table."""
        stats = {"total": 0, "migrated": 0, "skipped": 0, "errors": 0}

        try:
            cursor = self.sqlite_conn.execute("SELECT * FROM action_artifacts")
        except sqlite3.OperationalError:
            print("  Table action_artifacts not found, skipping")
            return stats

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        stats["total"] = len(rows)
        print(f"  Found {stats['total']} rows in SQLite")

        if self.dry_run:
            print(f"  [DRY RUN] Would migrate {stats['total']} rows")
            return stats

        for row in rows:
            row_dict = dict(zip(columns, row))

            try:
                # Generate UUID from artifact_id
                artifact_uuid = self._generate_uuid_from_string(row_dict["artifact_id"])

                # Check if already exists
                existing = await self.pg_conn.fetchval(
                    "SELECT 1 FROM zakops.artifacts WHERE id = $1",
                    artifact_uuid
                )

                if existing:
                    stats["skipped"] += 1
                    continue

                await self.pg_conn.execute("""
                    INSERT INTO zakops.artifacts (
                        id, action_id, filename, file_path, mime_type,
                        file_size, sha256, created_at, storage_backend, correlation_id
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT DO NOTHING
                """,
                    artifact_uuid,
                    row_dict.get("action_id"),
                    row_dict.get("filename"),
                    row_dict.get("path"),
                    row_dict.get("mime_type"),
                    row_dict.get("size_bytes"),
                    row_dict.get("sha256"),
                    parse_timestamp(row_dict.get("created_at")) or datetime.utcnow(),
                    "local",
                    uuid.uuid4(),  # Generate correlation_id
                )

                stats["migrated"] += 1

            except Exception as e:
                stats["errors"] += 1
                print(f"    Error migrating artifact {row_dict.get('artifact_id')}: {e}")

        return stats

    def _map_status(self, status: str) -> str:
        """Map SQLite status to PostgreSQL status."""
        status_map = {
            "PENDING_APPROVAL": "PENDING_APPROVAL",
            "READY": "QUEUED",
            "PROCESSING": "RUNNING",
            "COMPLETED": "COMPLETED",
            "FAILED": "FAILED",
            "CANCELLED": "CANCELLED",
        }
        return status_map.get(status.upper(), "PENDING_APPROVAL")

    def _generate_uuid_from_string(self, s: str) -> uuid.UUID:
        """Generate a deterministic UUID from a string."""
        return uuid.uuid5(uuid.NAMESPACE_OID, s)

    async def run(self, tables: Optional[List[str]] = None) -> None:
        """Run the migration."""
        print("=" * 60)
        print("Data Migration: SQLite → PostgreSQL")
        print("=" * 60)
        if self.dry_run:
            print("[DRY RUN MODE - No changes will be made]")
        print()

        await self.connect()

        try:
            # Get available tables
            sqlite_tables = self.get_sqlite_tables()
            print(f"SQLite tables: {sqlite_tables}\n")

            # Migrate actions
            if not tables or "actions" in tables:
                print("Migrating: actions")
                self.stats["actions"] = await self.migrate_actions()
                print(f"  Result: {self.stats['actions']}\n")

            # Migrate audit events
            if not tables or "action_audit_events" in tables:
                print("Migrating: action_audit_events → agent_events")
                self.stats["action_audit_events"] = await self.migrate_audit_events()
                print(f"  Result: {self.stats['action_audit_events']}\n")

            # Migrate artifacts
            if not tables or "action_artifacts" in tables:
                print("Migrating: action_artifacts → artifacts")
                self.stats["action_artifacts"] = await self.migrate_artifacts()
                print(f"  Result: {self.stats['action_artifacts']}\n")

            # Summary
            print("=" * 60)
            print("Migration Summary")
            print("=" * 60)
            total_migrated = sum(s.get("migrated", 0) for s in self.stats.values())
            total_skipped = sum(s.get("skipped", 0) for s in self.stats.values())
            total_errors = sum(s.get("errors", 0) for s in self.stats.values())

            print(f"Total migrated: {total_migrated}")
            print(f"Total skipped (already exists): {total_skipped}")
            print(f"Total errors: {total_errors}")

            if total_errors > 0:
                print("\n⚠️  Some migrations had errors. Review output above.")
            elif not self.dry_run:
                print("\n✅ Migration complete!")

        finally:
            await self.disconnect()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate data from SQLite to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes"
    )
    parser.add_argument(
        "--table",
        action="append",
        dest="tables",
        help="Migrate specific table(s) only"
    )
    args = parser.parse_args()

    migrator = DataMigrator(dry_run=args.dry_run)
    await migrator.run(tables=args.tables)


if __name__ == "__main__":
    asyncio.run(main())
