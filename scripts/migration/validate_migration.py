#!/usr/bin/env python3
"""
Migration Validation Script

Validates data integrity after migration.

Usage:
    python scripts/migration/validate_migration.py
    python scripts/migration/validate_migration.py --report audit/migration-validation.md
"""

import argparse
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, List
import sys

sys.path.insert(0, ".")


class MigrationValidator:
    """Validate migration results."""

    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://dealengine:changeme@localhost:5435/zakops"
        )

    def add_check(self, name: str, status: str, message: str, details: Dict = None):
        """Add a check result."""
        self.checks.append({
            "name": name,
            "status": status,
            "message": message,
            "details": details or {}
        })

        if status == "pass":
            self.passed += 1
        elif status == "fail":
            self.failed += 1
        else:
            self.warnings += 1

    async def validate_deals(self, conn):
        """Validate deals table."""
        # Count deals
        result = await conn.fetchrow("SELECT COUNT(*) as count FROM zakops.deals")
        count = result["count"]

        self.add_check(
            "deals_count",
            "pass" if count >= 0 else "fail",
            f"Found {count} deals in PostgreSQL"
        )

        # Check all deals have required fields
        result = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM zakops.deals WHERE canonical_name IS NULL OR canonical_name = ''"
        )
        null_count = result["count"]

        self.add_check(
            "deals_canonical_name",
            "pass" if null_count == 0 else "warn",
            f"{null_count} deals without canonical_name" if null_count > 0 else "All deals have canonical_name"
        )

        # Check all deals have valid stage
        result = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM zakops.deals WHERE stage IS NULL OR stage = ''"
        )
        null_stage = result["count"]

        self.add_check(
            "deals_stage",
            "pass" if null_stage == 0 else "warn",
            f"{null_stage} deals without stage" if null_stage > 0 else "All deals have stage"
        )

        # Check for deleted deals
        result = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM zakops.deals WHERE deleted = TRUE"
        )
        deleted_count = result["count"]

        self.add_check(
            "deals_deleted",
            "pass",
            f"{deleted_count} deals marked as deleted",
            {"deleted_count": deleted_count}
        )

    async def validate_actions(self, conn):
        """Validate actions table."""
        # Count actions
        result = await conn.fetchrow("SELECT COUNT(*) as count FROM zakops.actions")
        count = result["count"]

        self.add_check(
            "actions_count",
            "pass" if count >= 0 else "fail",
            f"Found {count} actions in PostgreSQL"
        )

        # Check foreign keys to deals
        result = await conn.fetchrow(
            """SELECT COUNT(*) as count FROM zakops.actions a
               LEFT JOIN zakops.deals d ON a.deal_id = d.deal_id
               WHERE d.deal_id IS NULL AND a.deal_id IS NOT NULL"""
        )
        orphans = result["count"]

        self.add_check(
            "actions_fk_deals",
            "pass" if orphans == 0 else "fail",
            f"{orphans} orphaned actions (missing deal)" if orphans > 0 else "All action foreign keys valid"
        )

        # Check action statuses
        result = await conn.fetch(
            "SELECT status, COUNT(*) as count FROM zakops.actions GROUP BY status"
        )
        status_counts = {row["status"]: row["count"] for row in result}

        self.add_check(
            "actions_status_distribution",
            "pass",
            f"Action statuses: {status_counts}",
            {"status_counts": status_counts}
        )

    async def validate_artifacts(self, conn):
        """Validate artifacts table."""
        # Count artifacts
        result = await conn.fetchrow("SELECT COUNT(*) as count FROM zakops.artifacts")
        count = result["count"]

        self.add_check(
            "artifacts_count",
            "pass" if count >= 0 else "fail",
            f"Found {count} artifacts in PostgreSQL"
        )

        if count == 0:
            self.add_check(
                "artifacts_note",
                "pass",
                "No artifacts to validate (empty table)"
            )
            return

        # Check all artifacts have correlation_id
        result = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM zakops.artifacts WHERE correlation_id IS NULL"
        )
        null_corr = result["count"]

        self.add_check(
            "artifacts_correlation_id",
            "pass" if null_corr == 0 else "fail",
            f"{null_corr} artifacts without correlation_id" if null_corr > 0 else "All artifacts have correlation_id"
        )

        # Check foreign keys to deals
        result = await conn.fetchrow(
            """SELECT COUNT(*) as count FROM zakops.artifacts a
               LEFT JOIN zakops.deals d ON a.deal_id = d.deal_id
               WHERE d.deal_id IS NULL AND a.deal_id IS NOT NULL"""
        )
        orphans = result["count"]

        self.add_check(
            "artifacts_fk_deals",
            "pass" if orphans == 0 else "fail",
            f"{orphans} orphaned artifacts (missing deal)" if orphans > 0 else "All artifact foreign keys valid"
        )

    async def validate_outbox(self, conn):
        """Validate outbox table."""
        # Count total entries
        result = await conn.fetchrow("SELECT COUNT(*) as count FROM zakops.outbox")
        total = result["count"]

        # Check for stuck entries (pending for more than 1 hour)
        result = await conn.fetchrow(
            "SELECT COUNT(*) as count FROM zakops.outbox WHERE status = 'pending' AND created_at < NOW() - INTERVAL '1 hour'"
        )
        stuck = result["count"]

        self.add_check(
            "outbox_total",
            "pass",
            f"Outbox has {total} total entries"
        )

        self.add_check(
            "outbox_stuck",
            "pass" if stuck == 0 else "warn",
            f"{stuck} potentially stuck outbox entries" if stuck > 0 else "No stuck outbox entries"
        )

        # Check status distribution
        result = await conn.fetch(
            "SELECT status, COUNT(*) as count FROM zakops.outbox GROUP BY status"
        )
        status_counts = {row["status"]: row["count"] for row in result}

        if status_counts:
            self.add_check(
                "outbox_status_distribution",
                "pass",
                f"Outbox statuses: {status_counts}",
                {"status_counts": status_counts}
            )

    async def validate_agent_data(self, conn):
        """Validate agent-related tables."""
        # Agent threads
        result = await conn.fetchrow("SELECT COUNT(*) as count FROM zakops.agent_threads")
        threads = result["count"]

        # Agent runs
        result = await conn.fetchrow("SELECT COUNT(*) as count FROM zakops.agent_runs")
        runs = result["count"]

        # Agent events
        result = await conn.fetchrow("SELECT COUNT(*) as count FROM zakops.agent_events")
        events = result["count"]

        self.add_check(
            "agent_data",
            "pass",
            f"Agent data: {threads} threads, {runs} runs, {events} events",
            {"threads": threads, "runs": runs, "events": events}
        )

    async def validate_checkpoints(self, conn):
        """Validate execution checkpoints table."""
        result = await conn.fetchrow("SELECT COUNT(*) as count FROM zakops.execution_checkpoints")
        count = result["count"]

        self.add_check(
            "checkpoints_count",
            "pass",
            f"Found {count} execution checkpoints"
        )

    async def run(self) -> Dict[str, Any]:
        """Run all validations."""
        import asyncpg

        print("Running migration validation...\n")

        try:
            conn = await asyncpg.connect(self.db_url)

            await self.validate_deals(conn)
            await self.validate_actions(conn)
            await self.validate_artifacts(conn)
            await self.validate_outbox(conn)
            await self.validate_agent_data(conn)
            await self.validate_checkpoints(conn)

            await conn.close()

        except Exception as e:
            self.add_check(
                "database_connection",
                "fail",
                f"Failed to connect to database: {e}"
            )

        # Print results
        print("=" * 50)
        print("VALIDATION RESULTS")
        print("=" * 50)

        for check in self.checks:
            icon = {"pass": "PASS", "fail": "FAIL", "warn": "WARN"}[check["status"]]
            print(f"  [{icon}] {check['name']}: {check['message']}")

        print("\n" + "-" * 50)
        print(f"Passed:   {self.passed}")
        print(f"Failed:   {self.failed}")
        print(f"Warnings: {self.warnings}")
        print("-" * 50)

        overall = "PASS" if self.failed == 0 else "FAIL"
        print(f"\nOverall: {overall}")

        return {
            "timestamp": datetime.now().isoformat(),
            "overall": overall,
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
            "checks": self.checks
        }

    def generate_report(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Migration Validation Report",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Status**: {'PASSED' if self.failed == 0 else 'FAILED'}",
            "",
            "## Summary",
            "",
            f"- **Passed**: {self.passed}",
            f"- **Failed**: {self.failed}",
            f"- **Warnings**: {self.warnings}",
            "",
            "## Checks",
            "",
            "| Check | Status | Message |",
            "|-------|--------|---------|",
        ]

        for check in self.checks:
            status_icon = {"pass": "PASS", "fail": "FAIL", "warn": "WARN"}[check["status"]]
            lines.append(f"| {check['name']} | {status_icon} | {check['message']} |")

        lines.extend([
            "",
            "## Details",
            "",
        ])

        for check in self.checks:
            if check.get("details"):
                lines.append(f"### {check['name']}")
                lines.append("")
                lines.append("```json")
                import json
                lines.append(json.dumps(check["details"], indent=2))
                lines.append("```")
                lines.append("")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Validate migration results")
    parser.add_argument("--report", "-r", help="Output markdown report")
    args = parser.parse_args()

    validator = MigrationValidator()
    result = asyncio.run(validator.run())

    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        report = validator.generate_report()
        with open(args.report, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.report}")


if __name__ == "__main__":
    main()
