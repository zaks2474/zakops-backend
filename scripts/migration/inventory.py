#!/usr/bin/env python3
"""
Migration Inventory Script

Scans for existing data sources and generates inventory report.

Usage:
    python scripts/migration/inventory.py
    python scripts/migration/inventory.py --output audit/migration-inventory.json
"""

import argparse
import asyncio
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class MigrationInventory:
    """Scan and inventory existing data sources."""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.inventory = {
            "timestamp": datetime.now().isoformat(),
            "base_path": str(self.base_path.absolute()),
            "sqlite_databases": [],
            "data_directories": [],
            "file_counts": {},
            "total_records": {},
            "recommendations": []
        }

    def scan_sqlite_databases(self) -> List[Dict[str, Any]]:
        """Find all SQLite databases."""
        databases = []

        # Common locations for SQLite files
        patterns = ["*.db", "*.sqlite", "*.sqlite3"]

        for pattern in patterns:
            for db_path in self.base_path.glob(pattern):
                if db_path.is_file():
                    db_info = self._analyze_sqlite(db_path)
                    if db_info:
                        databases.append(db_info)

        # Also check data directory
        data_dir = self.base_path / "data"
        if data_dir.exists():
            for db_path in data_dir.rglob("*.db"):
                db_info = self._analyze_sqlite(db_path)
                if db_info:
                    databases.append(db_info)

        self.inventory["sqlite_databases"] = databases
        return databases

    def _analyze_sqlite(self, db_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a SQLite database."""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # Get record counts
            table_counts = {}
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_counts[table] = count
                except:
                    table_counts[table] = "error"

            conn.close()

            return {
                "path": str(db_path),
                "size_bytes": db_path.stat().st_size,
                "tables": tables,
                "record_counts": table_counts,
                "total_records": sum(c for c in table_counts.values() if isinstance(c, int))
            }
        except Exception as e:
            return {
                "path": str(db_path),
                "error": str(e)
            }

    def scan_data_directories(self) -> List[Dict[str, Any]]:
        """Find data directories (DataRoom, uploads, etc.)."""
        directories = []

        # Common data directory names
        dir_names = ["DataRoom", "data", "uploads", "files", "artifacts", "documents"]

        for dir_name in dir_names:
            dir_path = self.base_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                dir_info = self._analyze_directory(dir_path)
                directories.append(dir_info)

        self.inventory["data_directories"] = directories
        return directories

    def _analyze_directory(self, dir_path: Path) -> Dict[str, Any]:
        """Analyze a data directory."""
        file_count = 0
        total_size = 0
        file_types = {}

        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                file_count += 1
                total_size += file_path.stat().st_size
                ext = file_path.suffix.lower() or "no_extension"
                file_types[ext] = file_types.get(ext, 0) + 1

        return {
            "path": str(dir_path),
            "file_count": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_types": file_types
        }

    async def scan_postgresql(self) -> Dict[str, Any]:
        """Check PostgreSQL for existing data."""
        try:
            import asyncpg

            # Get database URL from environment or use default
            db_url = os.environ.get(
                "DATABASE_URL",
                "postgresql://dealengine:changeme@localhost:5435/zakops"
            )

            conn = await asyncpg.connect(db_url)

            # Tables to check in zakops schema
            tables = [
                "zakops.deals",
                "zakops.actions",
                "zakops.artifacts",
                "zakops.agent_runs",
                "zakops.agent_threads",
                "zakops.deal_events",
                "zakops.outbox"
            ]
            counts = {}

            for table in tables:
                try:
                    result = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {table}")
                    counts[table] = result["count"] if result else 0
                except:
                    counts[table] = "table_not_found"

            # Check deal_id presence (deals use deal_id as primary identifier)
            deal_check = await conn.fetchrow(
                "SELECT COUNT(*) as total, COUNT(deal_id) as with_id FROM zakops.deals"
            )

            await conn.close()

            return {
                "connected": True,
                "record_counts": counts,
                "total_records": sum(c for c in counts.values() if isinstance(c, int)),
                "deal_id_coverage": {
                    "total_deals": deal_check["total"],
                    "with_deal_id": deal_check["with_id"]
                }
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e)
            }

    def generate_recommendations(self):
        """Generate migration recommendations based on inventory."""
        recommendations = []

        # Check for SQLite data
        sqlite_total = sum(
            db.get("total_records", 0)
            for db in self.inventory["sqlite_databases"]
            if isinstance(db.get("total_records"), int)
        )

        if sqlite_total > 0:
            recommendations.append({
                "priority": "high",
                "type": "sqlite_migration",
                "message": f"Found {sqlite_total} records in SQLite databases to migrate"
            })

        # Check for file data
        file_total = sum(
            d.get("file_count", 0)
            for d in self.inventory["data_directories"]
        )

        if file_total > 0:
            recommendations.append({
                "priority": "high",
                "type": "file_migration",
                "message": f"Found {file_total} files in data directories to migrate"
            })

        # Check PostgreSQL
        pg = self.inventory.get("postgresql", {})
        if pg.get("connected"):
            if pg.get("total_records", 0) > 0:
                recommendations.append({
                    "priority": "info",
                    "type": "existing_data",
                    "message": f"PostgreSQL already has {pg['total_records']} records"
                })

            # Check deal_id coverage
            deal_cov = pg.get("deal_id_coverage", {})
            if deal_cov.get("total_deals", 0) > 0:
                coverage = deal_cov.get("with_deal_id", 0) / deal_cov["total_deals"] * 100
                if coverage < 100:
                    recommendations.append({
                        "priority": "warn",
                        "type": "deal_id_gap",
                        "message": f"Only {coverage:.0f}% of deals have deal_id"
                    })
                else:
                    recommendations.append({
                        "priority": "info",
                        "type": "deal_id_complete",
                        "message": "All deals have deal_id assigned"
                    })

        # No data found
        if sqlite_total == 0 and file_total == 0:
            recommendations.append({
                "priority": "info",
                "type": "no_migration_needed",
                "message": "No legacy data found - migration not necessary"
            })

        self.inventory["recommendations"] = recommendations
        return recommendations

    def run_full_scan(self) -> Dict[str, Any]:
        """Run complete inventory scan (sync version for standalone use)."""
        print("Scanning SQLite databases...")
        self.scan_sqlite_databases()

        print("Scanning data directories...")
        self.scan_data_directories()

        print("Scanning PostgreSQL...")
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in an event loop, can't use asyncio.run()
            # This path is for when called from another async function
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(self._async_scan_pg())
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            asyncio.run(self._async_scan_pg())

        print("Generating recommendations...")
        self.generate_recommendations()

        return self.inventory

    async def run_full_scan_async(self) -> Dict[str, Any]:
        """Run complete inventory scan (async version for use within event loop)."""
        print("Scanning SQLite databases...")
        self.scan_sqlite_databases()

        print("Scanning data directories...")
        self.scan_data_directories()

        print("Scanning PostgreSQL...")
        result = await self.scan_postgresql()
        self.inventory["postgresql"] = result

        print("Generating recommendations...")
        self.generate_recommendations()

        return self.inventory

    async def _async_scan_pg(self):
        """Async wrapper for PostgreSQL scan."""
        result = await self.scan_postgresql()
        self.inventory["postgresql"] = result

    def print_summary(self):
        """Print inventory summary."""
        print("\n" + "=" * 60)
        print("MIGRATION INVENTORY SUMMARY")
        print("=" * 60)

        # SQLite
        sqlite_dbs = self.inventory["sqlite_databases"]
        if sqlite_dbs:
            print(f"\nSQLite Databases: {len(sqlite_dbs)}")
            for db in sqlite_dbs:
                if "error" not in db:
                    print(f"  - {db['path']}: {db['total_records']} records")
        else:
            print("\nSQLite Databases: None found")

        # Directories
        dirs = self.inventory["data_directories"]
        if dirs:
            print(f"\nData Directories: {len(dirs)}")
            for d in dirs:
                print(f"  - {d['path']}: {d['file_count']} files ({d['total_size_mb']} MB)")
        else:
            print("\nData Directories: None found")

        # PostgreSQL
        pg = self.inventory.get("postgresql", {})
        if pg.get("connected"):
            print(f"\nPostgreSQL: Connected")
            for table, count in pg.get("record_counts", {}).items():
                print(f"  - {table}: {count}")

            deal_cov = pg.get("deal_id_coverage", {})
            if deal_cov:
                print(f"\nDeal ID Coverage:")
                print(f"  - Deals with deal_id: {deal_cov.get('with_deal_id', 0)}/{deal_cov.get('total_deals', 0)}")
        else:
            print(f"\nPostgreSQL: Not connected - {pg.get('error', 'Unknown error')}")

        # Recommendations
        print("\nRecommendations:")
        for rec in self.inventory.get("recommendations", []):
            print(f"  [{rec['priority'].upper()}] {rec['message']}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Migration Inventory Scanner")
    parser.add_argument("--base", "-b", default=".", help="Base path to scan")
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()

    scanner = MigrationInventory(args.base)
    inventory = scanner.run_full_scan()
    scanner.print_summary()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(inventory, f, indent=2, default=str)
        print(f"\nInventory saved to: {args.output}")


if __name__ == "__main__":
    main()
