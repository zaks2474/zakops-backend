# Current Database Schema Audit

**Audit Date**: 2026-01-19
**Repository**: zakops-backend
**Status**: Phase 0 Baseline Audit

---

## Database Technology

**Current**: SQLite (file-based)
**Spec Requirement**: PostgreSQL

**Database File**: `/home/zaks/DataRoom/.deal-registry/ingest_state.db`

**Gap**: The spec requires PostgreSQL with proper tables, indexes, and constraints. Current implementation uses SQLite for the action engine only.

---

## Existing Tables (SQLite - Action Engine)

### actions

```sql
CREATE TABLE IF NOT EXISTS actions (
  action_id TEXT PRIMARY KEY,
  deal_id TEXT,
  capability_id TEXT,
  type TEXT NOT NULL,
  title TEXT NOT NULL,
  summary TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  duration_seconds REAL,
  created_by TEXT NOT NULL,
  source TEXT NOT NULL,
  risk_level TEXT NOT NULL,
  requires_human_review INTEGER NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  inputs TEXT NOT NULL,
  outputs TEXT,
  error TEXT,
  retry_count INTEGER NOT NULL DEFAULT 0,
  max_retries INTEGER NOT NULL DEFAULT 3,
  next_attempt_at TEXT,
  runner_lock_owner TEXT,
  runner_lock_expires_at TEXT,
  runner_heartbeat_at TEXT,
  cancelled_by TEXT,
  cancelled_at TEXT,
  cancel_reason TEXT,
  hidden_from_quarantine INTEGER NOT NULL DEFAULT 0,
  quarantine_hidden_at TEXT,
  quarantine_hidden_by TEXT,
  quarantine_hidden_reason TEXT
);
```

**Indexes**:
- idx_actions_deal_id ON actions(deal_id)
- idx_actions_status ON actions(status)
- idx_actions_type ON actions(type)
- idx_actions_created_at ON actions(created_at DESC)

### action_audit_events

```sql
CREATE TABLE IF NOT EXISTS action_audit_events (
  audit_id TEXT PRIMARY KEY,
  action_id TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  event TEXT NOT NULL,
  actor TEXT NOT NULL,
  details TEXT
);
```

**Indexes**:
- idx_audit_action_id ON action_audit_events(action_id)
- idx_audit_timestamp ON action_audit_events(timestamp DESC)

### action_artifacts

```sql
CREATE TABLE IF NOT EXISTS action_artifacts (
  artifact_id TEXT PRIMARY KEY,
  action_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  path TEXT NOT NULL,
  size_bytes INTEGER,
  sha256 TEXT,
  created_at TEXT NOT NULL
);
```

**Indexes**:
- idx_artifacts_action_id ON action_artifacts(action_id)

### action_runner_leases

```sql
CREATE TABLE IF NOT EXISTS action_runner_leases (
  lease_id TEXT PRIMARY KEY,
  runner_id TEXT NOT NULL,
  action_id TEXT NOT NULL,
  acquired_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  released_at TEXT
);
```

### action_steps

```sql
CREATE TABLE IF NOT EXISTS action_steps (
  step_id TEXT PRIMARY KEY,
  action_id TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT,
  completed_at TEXT,
  result TEXT,
  error TEXT
);
```

### sender_history_scans (deal_backfill_sender_history.py)

```sql
CREATE TABLE IF NOT EXISTS sender_history_scans (
  scan_key TEXT PRIMARY KEY,
  deal_id TEXT NOT NULL,
  sender_email TEXT NOT NULL,
  lookback_days INTEGER NOT NULL,
  max_messages INTEGER NOT NULL,
  mode TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  status TEXT NOT NULL,
  error TEXT
);
```

### sender_history_processed

```sql
CREATE TABLE IF NOT EXISTS sender_history_processed (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scan_key TEXT NOT NULL,
  deal_id TEXT NOT NULL,
  message_id TEXT NOT NULL UNIQUE,
  thread_id TEXT NOT NULL,
  decision TEXT NOT NULL,
  confidence REAL NOT NULL,
  processed_at TEXT NOT NULL
);
```

---

## Spec-Required Tables (NOT Present)

### Missing Tables

| Table Name | Purpose | Priority |
|------------|---------|----------|
| operators | Operator/user accounts | P0 |
| deals | Central deal entity (PostgreSQL) | P0 |
| artifacts | File metadata with storage_uri | P0 |
| emails | Email records | P0 |
| agent_runs | Agent invocation tracking | P0 |
| agent_events | Append-only event timeline | P0 |
| run_checkpoints | HITL resume state | P1 |
| deal_threads | LangSmith thread mapping | P1 |
| outbox | Transactional outbox | P1 |
| inbox | Consumer deduplication | P1 |
| idempotency_keys | Operation deduplication | P1 |
| agent_events_archive | Archived events | P2 |

---

## Data Storage Analysis

### Deal Registry (JSON-based)

**Location**: `/home/zaks/DataRoom/.deal-registry/deal_registry.json`

The deal registry is currently a JSON file, not a PostgreSQL table. This is a significant gap.

### Correspondence/Materials

Stored as files in DataRoom directory structure, not in database.

---

## Column Gap Analysis (actions table)

### Present in Current Schema

| Column | Present | Notes |
|--------|---------|-------|
| action_id | Yes | Primary key |
| deal_id | Yes | |
| capability_id | Yes | |
| type | Yes | Called "type" not "action_type" |
| title | Yes | |
| summary | Yes | Called "summary" not "description" |
| status | Yes | |
| inputs | Yes | JSON string |
| outputs | Yes | JSON string |
| error | Yes | JSON string |
| created_at | Yes | ISO string |
| updated_at | Yes | ISO string |
| created_by | Yes | |
| idempotency_key | Yes | Unique |
| retry_count | Yes | |
| max_retries | Yes | |

### Missing from Spec (actions table)

| Column | Spec Requirement | Current |
|--------|------------------|---------|
| operator_id | REFERENCES operators(id) | Not present |
| run_id | REFERENCES agent_runs(id) | Not present |
| trace_id | UUID for tracing | Not present |
| correlation_id | UUID (= deal_id) | Not present |
| causation_id | UUID | Not present |
| requires_approval | BOOLEAN | Has requires_human_review |
| approved_by | UUID REFERENCES operators(id) | Not present (no FK) |
| approved_at | TIMESTAMPTZ | Not present |
| rejection_reason | TEXT | Not present |
| archived_at | TIMESTAMPTZ | Not present (uses hidden_from_quarantine) |

---

## Summary

| Category | Current State | Spec Requirement | Gap |
|----------|---------------|------------------|-----|
| Database Engine | SQLite | PostgreSQL | Major |
| Core Tables | actions + 5 support tables | 12+ tables | Major |
| Deal Storage | JSON file | PostgreSQL table | Major |
| Event System | action_audit_events only | agent_events (append-only) | Major |
| Correlation IDs | Not implemented | trace_id, correlation_id, causation_id | Major |
| Foreign Keys | None (SQLite) | Full referential integrity | Major |
| Operators Table | Not exists | Required | Major |
| Outbox/Inbox | Not exists | Required for reliability | Major |

---

## Recommendations

1. **Migrate to PostgreSQL** - Required for production reliability
2. **Create spec tables** - agent_runs, agent_events, etc.
3. **Add correlation IDs** - trace_id, correlation_id to existing tables
4. **Implement outbox pattern** - For reliable event publishing
5. **Migrate deal registry** - From JSON to PostgreSQL
