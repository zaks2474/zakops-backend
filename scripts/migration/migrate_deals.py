#!/usr/bin/env python3
"""
Deal Migration Script

Migrates deals from SQLite to PostgreSQL.

Usage:
    python scripts/migration/migrate_deals.py --source data/deals.db
    python scripts/migration/migrate_deals.py --dry-run --source legacy.db
"""

import argparse
import asyncio
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

sys.path.insert(0, ".")


def generate_deal_id(old_id: Optional[str] = None, sequence: int = 0) -> str:
    """Generate a deal_id in the expected format (DEAL-XXXXXX)."""
    if old_id and old_id.startswith("DEAL-"):
        return old_id
    # Generate a new ID
    return f"DEAL-{sequence + 1:06d}"


class DealMigrator:
    """Migrate deals from SQLite to PostgreSQL."""

    def __init__(self, source_db: str, dry_run: bool = False):
        self.source_db = source_db
        self.dry_run = dry_run
        self.migrated = 0
        self.skipped = 0
        self.errors = []
        self.db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://dealengine:changeme@localhost:5435/zakops"
        )

    def extract_deals(self) -> List[Dict[str, Any]]:
        """Extract deals from SQLite."""
        conn = sqlite3.connect(self.source_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Try to get deals - schema may vary
        try:
            cursor.execute("SELECT * FROM deals")
            deals = [dict(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Table might not exist or have different name
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Available tables: {tables}")
            deals = []

        conn.close()
        return deals

    def transform_deal(self, old_deal: Dict[str, Any], sequence: int) -> Dict[str, Any]:
        """Transform old deal schema to new zakops schema."""
        # Generate or preserve deal_id
        deal_id = old_deal.get("deal_id") or old_deal.get("id")
        if not deal_id or not str(deal_id).startswith("DEAL-"):
            deal_id = generate_deal_id(str(deal_id) if deal_id else None, sequence)
        else:
            deal_id = str(deal_id)

        # Extract company info
        company_info = old_deal.get("company_info", {})
        if isinstance(company_info, str):
            try:
                company_info = json.loads(company_info)
            except:
                company_info = {}

        # Map fields to zakops schema
        new_deal = {
            "deal_id": deal_id,
            "canonical_name": old_deal.get("canonical_name") or old_deal.get("name") or f"Deal {sequence}",
            "display_name": old_deal.get("display_name") or old_deal.get("title"),
            "folder_path": old_deal.get("folder_path"),
            "stage": old_deal.get("stage") or old_deal.get("current_stage") or "inbound",
            "status": old_deal.get("status") or "active",
            "identifiers": self._parse_json(old_deal.get("identifiers", {})),
            "company_info": company_info if company_info else {"company_name": old_deal.get("company_name", "")},
            "broker": self._parse_json(old_deal.get("broker", {})),
            "metadata": {
                "migrated_from": "sqlite",
                "original_id": str(old_deal.get("id", "")),
                "migration_date": datetime.now().isoformat(),
                **self._parse_json(old_deal.get("metadata", {}))
            },
            "email_thread_ids": old_deal.get("email_thread_ids", []) or [],
            "related_folders": old_deal.get("related_folders", []) or [],
            "deleted": old_deal.get("deleted", False) or False,
            "created_at": self._parse_date(old_deal.get("created_at")),
            "updated_at": datetime.now()
        }

        return new_deal

    def _parse_json(self, value: Any) -> Dict:
        """Parse JSON from various formats."""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return {}
        return {}

    def _parse_date(self, value: Any) -> datetime:
        """Parse date from various formats."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except:
                pass
        return datetime.now()

    async def load_deals(self, deals: List[Dict[str, Any]]) -> int:
        """Load deals into PostgreSQL."""
        if self.dry_run:
            print(f"[DRY RUN] Would insert {len(deals)} deals")
            for deal in deals[:5]:  # Show first 5
                print(f"  - {deal['deal_id']}: {deal['canonical_name']}")
            if len(deals) > 5:
                print(f"  ... and {len(deals) - 5} more")
            return len(deals)

        import asyncpg
        conn = await asyncpg.connect(self.db_url)

        for deal in deals:
            try:
                # Check if deal already exists
                existing = await conn.fetchrow(
                    "SELECT deal_id FROM zakops.deals WHERE deal_id = $1",
                    deal["deal_id"]
                )

                if existing:
                    print(f"  Skipping existing deal: {deal['deal_id']}")
                    self.skipped += 1
                    continue

                # Insert deal
                await conn.execute(
                    """
                    INSERT INTO zakops.deals (
                        deal_id, canonical_name, display_name, folder_path,
                        stage, status, identifiers, company_info, broker, metadata,
                        email_thread_ids, related_folders, deleted, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    """,
                    deal["deal_id"],
                    deal["canonical_name"],
                    deal["display_name"],
                    deal["folder_path"],
                    deal["stage"],
                    deal["status"],
                    json.dumps(deal["identifiers"]),
                    json.dumps(deal["company_info"]),
                    json.dumps(deal["broker"]),
                    json.dumps(deal["metadata"]),
                    deal["email_thread_ids"],
                    deal["related_folders"],
                    deal["deleted"],
                    deal["created_at"],
                    deal["updated_at"]
                )

                self.migrated += 1
                print(f"  Migrated: {deal['canonical_name']} ({deal['deal_id']})")

            except Exception as e:
                self.errors.append({"deal_id": deal["deal_id"], "error": str(e)})
                print(f"  Error: {deal['canonical_name']} - {e}")

        await conn.close()
        return self.migrated

    async def run(self) -> Dict[str, Any]:
        """Run the migration."""
        print(f"\nMigrating deals from: {self.source_db}")
        print(f"Dry run: {self.dry_run}\n")

        # Extract
        print("Extracting deals from SQLite...")
        old_deals = self.extract_deals()
        print(f"  Found {len(old_deals)} deals")

        if not old_deals:
            return {
                "status": "no_data",
                "source": self.source_db,
                "deals_found": 0
            }

        # Transform
        print("\nTransforming deals...")
        new_deals = [self.transform_deal(d, i) for i, d in enumerate(old_deals)]

        # Load
        print("\nLoading deals into PostgreSQL...")
        await self.load_deals(new_deals)

        # Summary
        result = {
            "status": "complete",
            "source": self.source_db,
            "dry_run": self.dry_run,
            "deals_found": len(old_deals),
            "deals_migrated": self.migrated,
            "deals_skipped": self.skipped,
            "errors": self.errors
        }

        print(f"\n{'='*40}")
        print("Migration Complete")
        print(f"{'='*40}")
        print(f"Found:    {len(old_deals)}")
        print(f"Migrated: {self.migrated}")
        print(f"Skipped:  {self.skipped}")
        print(f"Errors:   {len(self.errors)}")

        return result


def main():
    parser = argparse.ArgumentParser(description="Migrate deals from SQLite to PostgreSQL")
    parser.add_argument("--source", "-s", required=True, help="Source SQLite database")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Dry run (don't insert)")
    parser.add_argument("--output", "-o", help="Output report file")
    args = parser.parse_args()

    if not Path(args.source).exists():
        print(f"Source database not found: {args.source}")
        sys.exit(1)

    migrator = DealMigrator(args.source, args.dry_run)
    result = asyncio.run(migrator.run())

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
