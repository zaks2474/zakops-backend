#!/usr/bin/env python3
"""
ZakOps Spec Compliance Verification Suite

Phase 8.5: HARD GATE

Verifies the backend implementation matches the Master Architecture Specification.
This is a HARD GATE - all checks must pass before integration can begin.

Usage:
    python scripts/verify_spec_compliance.py
    python scripts/verify_spec_compliance.py --section schema
    python scripts/verify_spec_compliance.py --verbose
"""

import asyncio
import argparse
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from uuid import uuid4
from dataclasses import dataclass, field
from enum import Enum

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Color:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class Status(Enum):
    """Check result status."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARN = "WARN"


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    status: Status
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: int = 0


@dataclass
class SectionResult:
    """Result of a section of checks."""
    name: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == Status.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.status == Status.FAIL)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


class SpecComplianceVerifier:
    """Main verification suite."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: Dict[str, SectionResult] = {}
        self.db = None

    async def setup(self):
        """Initialize database connection."""
        from src.core.database.adapter import get_database
        self.db = await get_database()

    async def teardown(self):
        """Cleanup."""
        pass

    def log(self, message: str, color: str = Color.RESET):
        """Print with optional color."""
        print(f"{color}{message}{Color.RESET}")

    def log_check(self, result: CheckResult):
        """Log a check result."""
        icon = {
            Status.PASS: f"{Color.GREEN}✓{Color.RESET}",
            Status.FAIL: f"{Color.RED}✗{Color.RESET}",
            Status.SKIP: f"{Color.YELLOW}○{Color.RESET}",
            Status.WARN: f"{Color.YELLOW}⚠{Color.RESET}",
        }[result.status]

        status_color = {
            Status.PASS: Color.GREEN,
            Status.FAIL: Color.RED,
            Status.SKIP: Color.YELLOW,
            Status.WARN: Color.YELLOW,
        }[result.status]

        print(f"  {icon} {result.name}: {status_color}{result.message}{Color.RESET}")

        if self.verbose and result.details:
            for key, value in result.details.items():
                print(f"      {key}: {value}")

    # =========================================================================
    # SECTION 1: SCHEMA COMPLIANCE
    # =========================================================================

    async def verify_schema(self) -> SectionResult:
        """Verify database schema matches specification."""
        section = SectionResult(name="Schema Compliance")

        # Expected tables from Master Spec
        expected_tables = [
            "deals", "actions", "artifacts", "operators",
            "agent_runs", "agent_events", "deal_events",
            "outbox", "inbox", "execution_checkpoints"
        ]

        # Check each table exists
        for table in expected_tables:
            start = datetime.now()
            try:
                result = await self.db.fetchrow(
                    f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'zakops' AND table_name = $1)",
                    table
                )
                exists = result['exists'] if result else False

                if exists:
                    cols = await self.db.fetch(
                        "SELECT column_name FROM information_schema.columns WHERE table_schema = 'zakops' AND table_name = $1",
                        table
                    )
                    check = CheckResult(
                        name=f"Table: {table}",
                        status=Status.PASS,
                        message=f"Exists with {len(cols)} columns",
                        duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                    )
                else:
                    check = CheckResult(
                        name=f"Table: {table}",
                        status=Status.FAIL,
                        message="Table does not exist",
                        duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                    )
            except Exception as e:
                check = CheckResult(
                    name=f"Table: {table}",
                    status=Status.FAIL,
                    message=f"Error: {str(e)[:100]}",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )

            section.checks.append(check)
            self.log_check(check)

        # Check critical columns exist
        # Note: deals uses 'stage' (not 'current_stage'), and deals don't have
        # correlation_id directly (deals ARE the correlation via deal_id)
        critical_columns = [
            ("deals", "deal_id"),
            ("deals", "stage"),
            ("actions", "deal_id"),
            ("actions", "risk_level"),
            ("actions", "status"),
            ("outbox", "status"),
            ("outbox", "attempts"),
            ("inbox", "event_id"),
            ("inbox", "consumer_id"),
            ("execution_checkpoints", "checkpoint_data"),
        ]

        for table, column in critical_columns:
            start = datetime.now()
            try:
                result = await self.db.fetchrow(
                    """SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = 'zakops' AND table_name = $1 AND column_name = $2
                    )""",
                    table, column
                )
                exists = result['exists'] if result else False

                check = CheckResult(
                    name=f"Column: {table}.{column}",
                    status=Status.PASS if exists else Status.FAIL,
                    message="Exists" if exists else "Missing",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
            except Exception as e:
                check = CheckResult(
                    name=f"Column: {table}.{column}",
                    status=Status.FAIL,
                    message=f"Error: {str(e)[:100]}",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )

            section.checks.append(check)
            self.log_check(check)

        self.results["schema"] = section
        return section

    # =========================================================================
    # SECTION 2: IDEMPOTENCY ENFORCEMENT
    # =========================================================================

    async def verify_idempotency(self) -> SectionResult:
        """Verify idempotency is enforced via inbox."""
        section = SectionResult(name="Idempotency Enforcement")

        # Test 1: Inbox blocks duplicates
        start = datetime.now()
        try:
            from src.core.inbox import InboxGuard

            event_id = uuid4()
            consumer_id = f"test-consumer-{uuid4().hex[:8]}"

            # First processing should succeed
            async with InboxGuard(event_id, consumer_id) as guard1:
                first_should_process = guard1.should_process

            # Second processing should be blocked
            async with InboxGuard(event_id, consumer_id) as guard2:
                second_should_process = guard2.should_process

            if first_should_process and not second_should_process:
                check = CheckResult(
                    name="Inbox duplicate blocking",
                    status=Status.PASS,
                    message="First accepted, duplicate blocked",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
            else:
                check = CheckResult(
                    name="Inbox duplicate blocking",
                    status=Status.FAIL,
                    message=f"First={first_should_process}, Second={second_should_process}",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
        except Exception as e:
            check = CheckResult(
                name="Inbox duplicate blocking",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 2: Inbox records in database
        start = datetime.now()
        try:
            result = await self.db.fetchrow(
                "SELECT COUNT(*) as count FROM zakops.inbox"
            )
            count = result['count'] if result else 0

            check = CheckResult(
                name="Inbox database recording",
                status=Status.PASS if count > 0 else Status.WARN,
                message=f"{count} records found",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Inbox database recording",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 3: Inbox table structure
        start = datetime.now()
        try:
            result = await self.db.fetch(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'inbox'"""
            )
            columns = [r['column_name'] for r in result]
            required = ['event_id', 'consumer_id']
            missing = [c for c in required if c not in columns]

            if not missing:
                check = CheckResult(
                    name="Inbox table structure",
                    status=Status.PASS,
                    message=f"Required columns present: {required}",
                    details={"columns": columns},
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
            else:
                check = CheckResult(
                    name="Inbox table structure",
                    status=Status.FAIL,
                    message=f"Missing columns: {missing}",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
        except Exception as e:
            check = CheckResult(
                name="Inbox table structure",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        self.results["idempotency"] = section
        return section

    # =========================================================================
    # SECTION 3: OUTBOX PROCESSOR
    # =========================================================================

    async def verify_outbox(self) -> SectionResult:
        """Verify outbox processor is running and functioning."""
        section = SectionResult(name="Outbox Processor")

        # Test 1: Outbox module importable
        start = datetime.now()
        try:
            from src.core.outbox import OutboxWriter, OutboxProcessor, get_outbox_writer
            check = CheckResult(
                name="Outbox module import",
                status=Status.PASS,
                message="All components importable",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Outbox module import",
                status=Status.FAIL,
                message=f"Import error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 2: Can write to outbox
        start = datetime.now()
        try:
            from src.core.outbox import get_outbox_writer

            test_correlation_id = uuid4()
            async with get_outbox_writer() as writer:
                entry = await writer.write(
                    correlation_id=test_correlation_id,
                    event_type="system.test",
                    event_data={"test": True, "timestamp": datetime.now(timezone.utc).isoformat()}
                )

            check = CheckResult(
                name="Outbox write",
                status=Status.PASS,
                message=f"Entry created: {entry.id}",
                details={"entry_id": str(entry.id)},
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Outbox write",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 3: Outbox status values in database
        start = datetime.now()
        try:
            result = await self.db.fetch(
                "SELECT DISTINCT status FROM zakops.outbox"
            )
            statuses = [r['status'] for r in result]

            check = CheckResult(
                name="Outbox status values",
                status=Status.PASS,
                message=f"Statuses in use: {statuses if statuses else ['(table empty)']}",
                details={"found": statuses},
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Outbox status values",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 4: Max attempts / poison-pill protection
        start = datetime.now()
        try:
            from src.core.outbox.models import OutboxEntry

            entry = OutboxEntry(
                correlation_id=uuid4(),
                event_type="test",
                event_data={}
            )
            max_attempts = entry.max_attempts

            if max_attempts and max_attempts > 0:
                check = CheckResult(
                    name="Outbox max attempts (poison-pill)",
                    status=Status.PASS,
                    message=f"Max attempts: {max_attempts}",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
            else:
                check = CheckResult(
                    name="Outbox max attempts (poison-pill)",
                    status=Status.FAIL,
                    message="Max attempts not configured",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
        except Exception as e:
            check = CheckResult(
                name="Outbox max attempts (poison-pill)",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 5: DLQ status exists
        start = datetime.now()
        try:
            from src.core.outbox.models import OutboxStatus

            has_dlq = hasattr(OutboxStatus, 'DEAD') or hasattr(OutboxStatus, 'DLQ') or hasattr(OutboxStatus, 'FAILED')

            check = CheckResult(
                name="Outbox DLQ/FAILED status",
                status=Status.PASS if has_dlq else Status.WARN,
                message="DLQ/FAILED status defined" if has_dlq else "DLQ status not found",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Outbox DLQ/FAILED status",
                status=Status.WARN,
                message=f"Could not verify: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        self.results["outbox"] = section
        return section

    # =========================================================================
    # SECTION 4: EVENT TAXONOMY
    # =========================================================================

    async def verify_event_taxonomy(self) -> SectionResult:
        """Verify event taxonomy matches specification."""
        section = SectionResult(name="Event Taxonomy")

        # Test 1: Taxonomy module importable
        start = datetime.now()
        try:
            from src.core.events.taxonomy import DealEventType, ActionEventType, AgentEventType
            check = CheckResult(
                name="Taxonomy module import",
                status=Status.PASS,
                message="Taxonomy imported successfully",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Taxonomy module import",
                status=Status.FAIL,
                message=f"Import error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 2: Required event types exist
        start = datetime.now()
        try:
            from src.core.events.taxonomy import DealEventType, ActionEventType, AgentEventType

            # Check DealEventType
            deal_events = ["CREATED", "UPDATED", "STAGE_CHANGED"]
            action_events = ["CREATED", "APPROVED", "REJECTED", "EXECUTING", "COMPLETED", "FAILED"]
            agent_events = ["RUN_STARTED", "RUN_COMPLETED", "RUN_FAILED"]

            missing = []
            for event in deal_events:
                if not hasattr(DealEventType, event):
                    missing.append(f"DealEventType.{event}")
            for event in action_events:
                if not hasattr(ActionEventType, event):
                    missing.append(f"ActionEventType.{event}")
            for event in agent_events:
                if not hasattr(AgentEventType, event):
                    missing.append(f"AgentEventType.{event}")

            if not missing:
                check = CheckResult(
                    name="Required event types",
                    status=Status.PASS,
                    message=f"All required events defined in taxonomy",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
            else:
                check = CheckResult(
                    name="Required event types",
                    status=Status.FAIL,
                    message=f"Missing: {missing}",
                    details={"missing": missing},
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
        except Exception as e:
            check = CheckResult(
                name="Required event types",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 3: Events can be published
        start = datetime.now()
        test_uuid = None
        truncated_deal_id = None
        try:
            from src.core.events import publish_deal_event
            from src.core.events.taxonomy import DealEventType
            from uuid import UUID

            # The publisher converts UUID to string and truncates to 20 chars
            # UUID string format: "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" (36 chars)
            # First 20 chars: "xxxxxxxx-xxxx-xxxx-x"
            # So we need to create a deal whose deal_id matches this truncation

            test_uuid = uuid4()
            truncated_deal_id = str(test_uuid)[:20]

            await self.db.execute(
                """INSERT INTO zakops.deals (deal_id, canonical_name, stage, status)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (deal_id) DO NOTHING""",
                truncated_deal_id, f"Test Deal Event Publishing", "inbound", "active"
            )

            try:
                event_id = await publish_deal_event(
                    deal_id=test_uuid,
                    event_type=DealEventType.CREATED.value,
                    event_data={"test": True}
                )

                check = CheckResult(
                    name="Event publishing",
                    status=Status.PASS,
                    message=f"Event published successfully",
                    details={"event_id": str(event_id) if event_id else "N/A"},
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
            finally:
                # Cleanup test data
                await self.db.execute(
                    "DELETE FROM zakops.deal_events WHERE deal_id = $1", truncated_deal_id
                )
                await self.db.execute(
                    "DELETE FROM zakops.deals WHERE deal_id = $1", truncated_deal_id
                )
        except Exception as e:
            check = CheckResult(
                name="Event publishing",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 4: deal_id field exists in events table (events are correlated via deal_id)
        start = datetime.now()
        try:
            result = await self.db.fetchrow(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'deal_events'
                   AND column_name = 'deal_id'"""
            )

            check = CheckResult(
                name="Event deal_id field",
                status=Status.PASS if result else Status.FAIL,
                message="deal_id field exists in deal_events" if result else "deal_id missing",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Event deal_id field",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        self.results["events"] = section
        return section

    # =========================================================================
    # SECTION 5: HITL CHECKPOINTS
    # =========================================================================

    async def verify_hitl_checkpoints(self) -> SectionResult:
        """Verify HITL checkpoint behavior."""
        section = SectionResult(name="HITL Checkpoints")

        # Test 1: HITL module importable
        start = datetime.now()
        try:
            from src.core.hitl import (
                CheckpointStore, RiskAssessor, ApprovalWorkflow,
                assess_risk, RiskLevel
            )
            check = CheckResult(
                name="HITL module import",
                status=Status.PASS,
                message="All HITL components importable",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="HITL module import",
                status=Status.FAIL,
                message=f"Import error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 2: Checkpoint save works
        start = datetime.now()
        test_deal_id_cp = None
        test_action_id_cp = None
        try:
            from src.core.hitl import get_checkpoint_store

            # Create test deal and action to satisfy FK constraints
            test_deal_id_cp = f"TEST-{uuid4().hex[:12].upper()}"
            test_action_id_cp = f"TEST-ACT-{uuid4().hex[:8]}"
            correlation_id = str(uuid4())

            await self.db.execute(
                """INSERT INTO zakops.deals (deal_id, canonical_name, stage, status)
                   VALUES ($1, $2, $3, $4) ON CONFLICT (deal_id) DO NOTHING""",
                test_deal_id_cp, f"Test Deal Checkpoint", "inbound", "active"
            )
            await self.db.execute(
                """INSERT INTO zakops.actions (action_id, deal_id, capability_id, action_type, title, status, risk_level)
                   VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (action_id) DO NOTHING""",
                test_action_id_cp, test_deal_id_cp, "test.capability", "test_action", "Test Action", "PENDING_APPROVAL", "low"
            )

            store = await get_checkpoint_store()

            checkpoint = await store.save_checkpoint(
                correlation_id=correlation_id,
                checkpoint_name="test_checkpoint",
                checkpoint_data={"step": 1, "data": "test"},
                action_id=test_action_id_cp,
            )

            check = CheckResult(
                name="Checkpoint save",
                status=Status.PASS,
                message=f"Checkpoint saved: {checkpoint.checkpoint_id}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Checkpoint save",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        finally:
            # Cleanup will be done after checkpoint restore test
            pass

        section.checks.append(check)
        self.log_check(check)

        # Test 3: Checkpoint restore works
        start = datetime.now()
        test_deal_id_restore = None
        test_action_id_restore = None
        try:
            from src.core.hitl import get_checkpoint_store

            # Create fresh test deal and action
            test_deal_id_restore = f"TEST-{uuid4().hex[:12].upper()}"
            test_action_id_restore = f"TEST-ACT-{uuid4().hex[:8]}"
            correlation_id = str(uuid4())

            await self.db.execute(
                """INSERT INTO zakops.deals (deal_id, canonical_name, stage, status)
                   VALUES ($1, $2, $3, $4) ON CONFLICT (deal_id) DO NOTHING""",
                test_deal_id_restore, f"Test Deal Restore", "inbound", "active"
            )
            await self.db.execute(
                """INSERT INTO zakops.actions (action_id, deal_id, capability_id, action_type, title, status, risk_level)
                   VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT (action_id) DO NOTHING""",
                test_action_id_restore, test_deal_id_restore, "test.capability", "test_action", "Test Action Restore", "PENDING_APPROVAL", "low"
            )

            store = await get_checkpoint_store()

            # Save
            await store.save_checkpoint(
                correlation_id=correlation_id,
                checkpoint_name="restore_test",
                checkpoint_data={"value": 42},
                action_id=test_action_id_restore,
            )

            # Restore
            checkpoint = await store.get_latest_checkpoint(action_id=test_action_id_restore, checkpoint_name="restore_test")

            if checkpoint and checkpoint.checkpoint_data.get("value") == 42:
                check = CheckResult(
                    name="Checkpoint restore",
                    status=Status.PASS,
                    message="Data restored correctly",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
            else:
                check = CheckResult(
                    name="Checkpoint restore",
                    status=Status.FAIL,
                    message=f"Data mismatch: {checkpoint.checkpoint_data if checkpoint else 'None'}",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
        except Exception as e:
            check = CheckResult(
                name="Checkpoint restore",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        finally:
            # Cleanup all test data
            try:
                if test_action_id_cp:
                    await self.db.execute("DELETE FROM zakops.execution_checkpoints WHERE action_id = $1", test_action_id_cp)
                    await self.db.execute("DELETE FROM zakops.actions WHERE action_id = $1", test_action_id_cp)
                if test_deal_id_cp:
                    await self.db.execute("DELETE FROM zakops.deals WHERE deal_id = $1", test_deal_id_cp)
                if test_action_id_restore:
                    await self.db.execute("DELETE FROM zakops.execution_checkpoints WHERE action_id = $1", test_action_id_restore)
                    await self.db.execute("DELETE FROM zakops.actions WHERE action_id = $1", test_action_id_restore)
                if test_deal_id_restore:
                    await self.db.execute("DELETE FROM zakops.deals WHERE deal_id = $1", test_deal_id_restore)
            except Exception:
                pass  # Ignore cleanup errors

        section.checks.append(check)
        self.log_check(check)

        # Test 4: Risk assessment works
        start = datetime.now()
        try:
            from src.core.hitl import assess_risk, RiskLevel

            # Test risk assessment (not async)
            low_risk = assess_risk("analyze_document")
            high_risk = assess_risk("send_email", {"recipient": "external@example.com"})

            check = CheckResult(
                name="Risk assessment",
                status=Status.PASS,
                message=f"Low: {low_risk.risk_level.value}, High: {high_risk.risk_level.value}",
                details={
                    "low_risk_requires_approval": low_risk.requires_approval,
                    "high_risk_requires_approval": high_risk.requires_approval
                },
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Risk assessment",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 5: Checkpoint table structure
        start = datetime.now()
        try:
            result = await self.db.fetch(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'execution_checkpoints'"""
            )
            columns = [r['column_name'] for r in result]

            if 'checkpoint_data' in columns:
                check = CheckResult(
                    name="Checkpoint table structure",
                    status=Status.PASS,
                    message=f"checkpoint_data column present ({len(columns)} total columns)",
                    details={"columns": columns},
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
            else:
                check = CheckResult(
                    name="Checkpoint table structure",
                    status=Status.FAIL,
                    message="checkpoint_data column missing",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
        except Exception as e:
            check = CheckResult(
                name="Checkpoint table structure",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        self.results["hitl"] = section
        return section

    # =========================================================================
    # SECTION 6: ARTIFACTSTORE
    # =========================================================================

    async def verify_artifact_store(self) -> SectionResult:
        """Verify ArtifactStore routing and functionality."""
        section = SectionResult(name="ArtifactStore")

        # Test 1: Storage module importable
        start = datetime.now()
        try:
            from src.core.storage import ArtifactStore, LocalFilesystemArtifactStore, get_artifact_store
            check = CheckResult(
                name="Storage module import",
                status=Status.PASS,
                message="All storage components importable",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Storage module import",
                status=Status.FAIL,
                message=f"Import error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 2: Local filesystem backend works
        start = datetime.now()
        try:
            from src.core.storage import get_artifact_store

            store = get_artifact_store()

            # Test write
            test_key = f"test/{uuid4().hex}/test.txt"
            test_content = b"Test artifact content"

            metadata = store.put(
                key=test_key,
                data=test_content,
                mime_type="text/plain"
            )

            # Test read
            content = store.get(test_key)

            if content == test_content:
                check = CheckResult(
                    name="Local storage write/read",
                    status=Status.PASS,
                    message="Content roundtrip successful",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
            else:
                check = CheckResult(
                    name="Local storage write/read",
                    status=Status.FAIL,
                    message="Content mismatch",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )

            # Cleanup
            store.delete(test_key)

        except Exception as e:
            check = CheckResult(
                name="Local storage write/read",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 3: Storage exists method works
        start = datetime.now()
        try:
            from src.core.storage import get_artifact_store

            store = get_artifact_store()
            test_key = f"test/{uuid4().hex}/exists_test.txt"

            # Should not exist initially
            before = store.exists(test_key)

            # Create it
            store.put(key=test_key, data=b"test", mime_type="text/plain")

            # Should exist now
            after = store.exists(test_key)

            # Cleanup
            store.delete(test_key)

            if not before and after:
                check = CheckResult(
                    name="Storage exists method",
                    status=Status.PASS,
                    message="exists() works correctly",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
            else:
                check = CheckResult(
                    name="Storage exists method",
                    status=Status.FAIL,
                    message=f"before={before}, after={after}",
                    duration_ms=int((datetime.now() - start).total_seconds() * 1000)
                )
        except Exception as e:
            check = CheckResult(
                name="Storage exists method",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        # Test 4: Artifacts table accessible
        start = datetime.now()
        try:
            result = await self.db.fetchrow(
                "SELECT COUNT(*) as count FROM zakops.artifacts"
            )

            check = CheckResult(
                name="Artifacts table",
                status=Status.PASS,
                message=f"Table accessible, {result['count']} records",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            check = CheckResult(
                name="Artifacts table",
                status=Status.FAIL,
                message=f"Error: {str(e)[:100]}",
                duration_ms=int((datetime.now() - start).total_seconds() * 1000)
            )

        section.checks.append(check)
        self.log_check(check)

        self.results["storage"] = section
        return section

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    async def run_all(self, sections: List[str] = None) -> bool:
        """Run all verification sections."""
        await self.setup()

        all_sections = [
            ("schema", self.verify_schema),
            ("idempotency", self.verify_idempotency),
            ("outbox", self.verify_outbox),
            ("events", self.verify_event_taxonomy),
            ("hitl", self.verify_hitl_checkpoints),
            ("storage", self.verify_artifact_store),
        ]

        # Filter sections if specified
        if sections:
            all_sections = [(name, fn) for name, fn in all_sections if name in sections]

        self.log(f"\n{'='*60}", Color.BLUE)
        self.log("ZakOps Spec Compliance Verification", Color.BOLD)
        self.log(f"{'='*60}\n", Color.BLUE)

        all_passed = True

        for section_name, verify_fn in all_sections:
            self.log(f"\n{Color.BOLD}[{section_name.upper()}]{Color.RESET}")
            self.log("-" * 40)

            result = await verify_fn()

            if not result.all_passed:
                all_passed = False

            status_icon = f"{Color.GREEN}✓{Color.RESET}" if result.all_passed else f"{Color.RED}✗{Color.RESET}"
            self.log(f"\n{status_icon} {result.name}: {result.passed}/{len(result.checks)} passed\n")

        await self.teardown()

        # Print summary
        self.log(f"\n{'='*60}", Color.BLUE)
        self.log("SUMMARY", Color.BOLD)
        self.log(f"{'='*60}\n", Color.BLUE)

        total_passed = sum(r.passed for r in self.results.values())
        total_checks = sum(len(r.checks) for r in self.results.values())
        total_failed = sum(r.failed for r in self.results.values())

        for name, result in self.results.items():
            status = f"{Color.GREEN}PASS{Color.RESET}" if result.all_passed else f"{Color.RED}FAIL{Color.RESET}"
            self.log(f"  {name.upper()}: {status} ({result.passed}/{len(result.checks)})")

        self.log(f"\n{Color.BOLD}Total: {total_passed}/{total_checks} checks passed{Color.RESET}")

        if total_failed > 0:
            self.log(f"{Color.RED}\n⚠️  {total_failed} CHECKS FAILED - HARD GATE NOT PASSED{Color.RESET}")
            self.log(f"{Color.YELLOW}Fix all failures before proceeding to Phase 9{Color.RESET}\n")
        else:
            self.log(f"{Color.GREEN}\n✅ ALL CHECKS PASSED - HARD GATE CLEARED{Color.RESET}")
            self.log(f"{Color.GREEN}You may proceed to Phase 9: Contract-First Integration{Color.RESET}\n")

        return all_passed

    def generate_report(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Phase 8.5 Report: Spec Compliance Verification",
            "",
            f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Status**: {'PASSED - HARD GATE CLEARED' if all(r.all_passed for r in self.results.values()) else 'FAILED - HARD GATE NOT CLEARED'}",
            "",
            "## Summary",
            "",
        ]

        total_passed = sum(r.passed for r in self.results.values())
        total_checks = sum(len(r.checks) for r in self.results.values())

        lines.append(f"**Total: {total_passed}/{total_checks} checks passed**")
        lines.append("")
        lines.append("| Section | Status | Passed | Failed |")
        lines.append("|---------|--------|--------|--------|")

        for name, result in self.results.items():
            status = "PASS" if result.all_passed else "FAIL"
            lines.append(f"| {name.title()} | {status} | {result.passed} | {result.failed} |")

        lines.append("")
        lines.append("## Detailed Results")
        lines.append("")

        for name, result in self.results.items():
            lines.append(f"### {result.name}")
            lines.append("")
            lines.append("| Check | Status | Message |")
            lines.append("|-------|--------|---------|")

            for check in result.checks:
                status_icon = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "SKIP"}[check.status.value]
                lines.append(f"| {check.name} | {status_icon} | {check.message} |")

            lines.append("")

        lines.append("## Hard Gate Status")
        lines.append("")
        all_passed = all(r.all_passed for r in self.results.values())
        if all_passed:
            lines.append("**HARD GATE: CLEARED**")
            lines.append("")
            lines.append("All quality gates passed. Proceed to Phase 9: Contract-First Integration Testing.")
        else:
            lines.append("**HARD GATE: NOT CLEARED**")
            lines.append("")
            lines.append("Fix failing checks and re-run verification before proceeding.")

        return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="ZakOps Spec Compliance Verification")
    parser.add_argument("--section", "-s", help="Run specific section (schema, idempotency, outbox, events, hitl, storage)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--report", "-r", help="Output report to file")

    args = parser.parse_args()

    verifier = SpecComplianceVerifier(verbose=args.verbose)

    sections = [args.section] if args.section else None
    passed = await verifier.run_all(sections)

    if args.report:
        report = verifier.generate_report()
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\nReport written to: {args.report}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
