#!/usr/bin/env python3
"""
Master Migration Script

Orchestrates the complete migration process.

Usage:
    python scripts/migration/run_migration.py
    python scripts/migration/run_migration.py --dry-run
    python scripts/migration/run_migration.py --skip-inventory
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import sys

# Ensure we can import from the migration directory
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, ".")

from inventory import MigrationInventory
from migrate_deals import DealMigrator
from migrate_artifacts import ArtifactMigrator
from validate_migration import MigrationValidator


async def run_migration(
    dry_run: bool = False,
    skip_inventory: bool = False,
    skip_validation: bool = False
):
    """Run the complete migration process."""

    print("=" * 60)
    print("ZakOps Data Migration")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Dry run: {dry_run}")
    print()

    results = {
        "started_at": datetime.now().isoformat(),
        "dry_run": dry_run,
        "steps": {}
    }

    # Ensure audit directory exists
    Path("audit").mkdir(exist_ok=True)

    # Step 1: Inventory
    if not skip_inventory:
        print("\n" + "=" * 40)
        print("STEP 1: INVENTORY")
        print("=" * 40)

        inventory = MigrationInventory()
        inv_result = await inventory.run_full_scan_async()
        inventory.print_summary()
        results["steps"]["inventory"] = inv_result

        # Save inventory
        with open("audit/migration-inventory.json", "w") as f:
            json.dump(inv_result, f, indent=2, default=str)
        print(f"\nInventory saved to: audit/migration-inventory.json")

    # Step 2: Deal Migration
    print("\n" + "=" * 40)
    print("STEP 2: DEAL MIGRATION")
    print("=" * 40)

    # Check for SQLite databases
    sqlite_files = list(Path(".").glob("*.db"))
    if Path("data").exists():
        sqlite_files.extend(Path("data").glob("*.db"))

    if sqlite_files:
        for db_file in sqlite_files:
            print(f"\nMigrating from: {db_file}")
            migrator = DealMigrator(str(db_file), dry_run)
            deal_result = await migrator.run()
            results["steps"][f"deals_{db_file.name}"] = deal_result
    else:
        print("No SQLite databases found to migrate")
        results["steps"]["deals"] = {"status": "skipped", "reason": "no_source"}

    # Step 3: Artifact Migration
    print("\n" + "=" * 40)
    print("STEP 3: ARTIFACT MIGRATION")
    print("=" * 40)

    data_dirs = ["DataRoom", "data", "uploads", "files"]
    found_data_dir = False

    for dir_name in data_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            # Check if directory has files (not just subdirs)
            has_files = any(f.is_file() for f in dir_path.rglob("*"))
            if has_files:
                print(f"\nMigrating from: {dir_name}")
                migrator = ArtifactMigrator(dir_name, dry_run)
                artifact_result = await migrator.run()
                results["steps"][f"artifacts_{dir_name}"] = artifact_result
                found_data_dir = True
                break

    if not found_data_dir:
        print("No data directories with files found to migrate")
        results["steps"]["artifacts"] = {"status": "skipped", "reason": "no_source"}

    # Step 4: Validation
    if not skip_validation:
        print("\n" + "=" * 40)
        print("STEP 4: VALIDATION")
        print("=" * 40)

        validator = MigrationValidator()
        val_result = await validator.run()
        results["steps"]["validation"] = val_result

        # Save validation report
        report = validator.generate_report()
        with open("audit/migration-validation.md", "w") as f:
            f.write(report)
        print(f"\nValidation report saved to: audit/migration-validation.md")

    # Summary
    results["completed_at"] = datetime.now().isoformat()

    print("\n" + "=" * 60)
    print("MIGRATION COMPLETE")
    print("=" * 60)
    print(f"Completed: {results['completed_at']}")

    # Determine overall status
    validation = results["steps"].get("validation", {})
    if validation.get("overall") == "FAIL":
        print("\nStatus: COMPLETED WITH FAILURES")
        print("Review validation report for details.")
    elif validation.get("warnings", 0) > 0:
        print("\nStatus: COMPLETED WITH WARNINGS")
        print("Review validation report for details.")
    else:
        print("\nStatus: SUCCESS")

    # Save full results
    with open("audit/migration-results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nResults saved to: audit/migration-results.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run complete data migration")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Dry run (no changes)")
    parser.add_argument("--skip-inventory", action="store_true", help="Skip inventory step")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation step")
    args = parser.parse_args()

    asyncio.run(run_migration(
        dry_run=args.dry_run,
        skip_inventory=args.skip_inventory,
        skip_validation=args.skip_validation
    ))


if __name__ == "__main__":
    main()
