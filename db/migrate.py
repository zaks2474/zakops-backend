#!/usr/bin/env python3
"""
Database Migration Runner

Usage:
    python -m db.migrate              # Run all pending migrations
    python -m db.migrate --status     # Show migration status
    python -m db.migrate --rollback   # Rollback last migration (interactive)

Environment:
    DATABASE_URL - PostgreSQL connection string
                   Default: postgresql://dealengine:changeme@localhost:5435/zakops
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

try:
    import asyncpg
except ImportError:
    print("ERROR: asyncpg not installed. Run: pip install asyncpg")
    sys.exit(1)


MIGRATIONS_DIR = Path(__file__).parent / "migrations"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://dealengine:changeme@localhost:5435/zakops"
)


async def get_connection() -> asyncpg.Connection:
    """Get a database connection."""
    return await asyncpg.connect(DATABASE_URL)


async def ensure_schema_exists(conn: asyncpg.Connection) -> None:
    """Ensure the zakops schema exists."""
    await conn.execute("CREATE SCHEMA IF NOT EXISTS zakops")


async def get_applied_migrations(conn: asyncpg.Connection) -> set:
    """Get set of already-applied migration versions."""
    try:
        rows = await conn.fetch("SELECT version FROM zakops.schema_migrations")
        return {row['version'] for row in rows}
    except asyncpg.UndefinedTableError:
        return set()


async def run_migration(conn: asyncpg.Connection, version: str, sql: str) -> None:
    """Run a single migration."""
    print(f"  Running migration {version}...")
    try:
        await conn.execute(sql)
        print(f"  ✅ Migration {version} complete")
    except Exception as e:
        print(f"  ❌ Migration {version} failed: {e}")
        raise


async def run_all_migrations() -> None:
    """Run all pending migrations."""
    print("=" * 60)
    print("ZakOps Database Migration Runner")
    print("=" * 60)
    print(f"\nDatabase: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")
    print(f"Migrations: {MIGRATIONS_DIR}\n")

    conn = await get_connection()

    try:
        await ensure_schema_exists(conn)
        applied = await get_applied_migrations(conn)

        # Find migration files (exclude rollback files)
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        migration_files = [f for f in migration_files if "rollback" not in f.name.lower()]

        pending = []
        for f in migration_files:
            version = f.stem.split("_")[0]
            if version not in applied:
                pending.append((version, f))

        if not pending:
            print("No pending migrations. Database is up to date.")
            return

        print(f"Found {len(pending)} pending migration(s):\n")

        for version, filepath in pending:
            print(f"  • {version}: {filepath.stem}")

        print()

        for version, filepath in pending:
            sql = filepath.read_text()
            await run_migration(conn, version, sql)

        print("\n" + "=" * 60)
        print("✅ All migrations complete!")
        print("=" * 60)

    finally:
        await conn.close()


async def show_status() -> None:
    """Show migration status."""
    print("=" * 60)
    print("Migration Status")
    print("=" * 60)
    print(f"\nDatabase: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else DATABASE_URL}")
    print(f"Migrations: {MIGRATIONS_DIR}\n")

    conn = await get_connection()

    try:
        applied = await get_applied_migrations(conn)

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        migration_files = [f for f in migration_files if "rollback" not in f.name.lower()]

        print("Migrations:")
        print("-" * 50)

        for f in migration_files:
            version = f.stem.split("_")[0]
            status = "✅ Applied" if version in applied else "⏳ Pending"
            print(f"  {version}: {f.stem}")
            print(f"      Status: {status}")

        # Also show tables
        print("\n" + "-" * 50)
        print("Tables in zakops schema:")
        tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'zakops' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        for t in tables:
            print(f"  • {t['table_name']}")

    finally:
        await conn.close()


async def run_rollback() -> None:
    """Rollback last migration (interactive)."""
    print("=" * 60)
    print("Migration Rollback")
    print("=" * 60)
    print("\n⚠️  WARNING: Rollback can cause data loss!")
    print("This feature requires manual confirmation.\n")

    conn = await get_connection()

    try:
        applied = await get_applied_migrations(conn)

        if not applied:
            print("No migrations to rollback.")
            return

        # Find the last applied migration
        last_version = sorted(applied)[-1]
        rollback_file = MIGRATIONS_DIR / f"{last_version}_*_rollback.sql"
        rollback_files = list(MIGRATIONS_DIR.glob(f"{last_version}_*_rollback.sql"))

        if not rollback_files:
            print(f"No rollback file found for migration {last_version}")
            return

        rollback_file = rollback_files[0]
        print(f"Last applied migration: {last_version}")
        print(f"Rollback file: {rollback_file.name}")
        print()
        print("To rollback, run the SQL manually:")
        print(f"  psql $DATABASE_URL -f {rollback_file}")

    finally:
        await conn.close()


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZakOps Database Migration Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m db.migrate              # Run pending migrations
  python -m db.migrate --status     # Show status
  python -m db.migrate --rollback   # Show rollback instructions
        """
    )
    parser.add_argument("--status", action="store_true", help="Show migration status")
    parser.add_argument("--rollback", action="store_true", help="Show rollback instructions")
    args = parser.parse_args()

    if args.status:
        await show_status()
    elif args.rollback:
        await run_rollback()
    else:
        await run_all_migrations()


if __name__ == "__main__":
    asyncio.run(main())
