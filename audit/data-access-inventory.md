# Backend Data Access Inventory

**Audit Date**: 2026-01-19
**Phase**: 1 Pre-Phase Deliverable
**Purpose**: Document all SQLite usage for PostgreSQL migration

---

## Database Configuration

| Item | Current Value | Location |
|------|---------------|----------|
| Database Type | SQLite | Multiple files |
| Primary Connection | `sqlite3.connect()` | `src/actions/engine/store.py:150` |
| Connection String | File path | Environment variable `ZAKOPS_STATE_DB` |
| Session/Engine Setup | Direct sqlite3 connection | No ORM |

---

## Database Files

| File | Path | Size | Primary Tables |
|------|------|------|----------------|
| ingest_state.db | `/home/zaks/DataRoom/.deal-registry/ingest_state.db` | 2.5M | actions, action_audit_events, action_artifacts, action_steps |
| email_triage_state.db | `/home/zaks/DataRoom/.deal-registry/email_triage_state.db` | 56K | email_triage_messages |
| sender_history.db | `/home/zaks/DataRoom/.deal-registry/sender_history.db` | 116K | sender_history_scans, sender_history_processed |
| email_backfill_state.db | `/home/zaks/DataRoom/.deal-registry/email_backfill_state.db` | 32K | sender_history_scans, sender_history_processed |

---

## Data Access Patterns

### Primary Store Classes

| Class | File | Database | Tables Accessed |
|-------|------|----------|-----------------|
| `ActionStore` | `src/actions/engine/store.py:143` | ingest_state.db | actions, action_audit_events, action_artifacts, action_runner_leases, action_steps |
| `MemoryStore` | `src/actions/memory/store.py:110` | ingest_state.db | action_memory |
| `ChatSessionStore` | `src/core/chat_orchestrator.py` | SQLite persistence | chat sessions |

### Direct SQL in API Files

| File | Line | Query Type | Table(s) | Purpose |
|------|------|------------|----------|---------|
| `src/api/deal_lifecycle/main.py` | 1923 | SELECT | actions | Get action stats by status |
| `src/api/deal_lifecycle/main.py` | 1930 | SELECT | actions | Count completed in last 24h |
| `src/api/deal_lifecycle/main.py` | 1936 | SELECT | actions | Count failed in last 24h |
| `src/api/deal_lifecycle/main.py` | 1942 | SELECT | actions | Average duration by type |
| `src/api/deal_lifecycle/main.py` | 1949 | SELECT | actions | Get recent errors |
| `src/api/deal_lifecycle/main.py` | 2067 | SELECT | actions | Get errors for ops report |
| `src/api/deal_lifecycle/main.py` | 2136 | SELECT | actions | Count for statistics |
| `src/api/deal_lifecycle/main.py` | 2341 | SELECT | tool_invocations | Tool invocation logs |
| `src/api/deal_lifecycle/main.py` | 3031 | SELECT | actions | Get distinct action types |
| `src/api/deal_lifecycle/main.py` | 3038 | SELECT | actions | Get sample actions by type |
| `src/api/deal_lifecycle/main.py` | 3070 | SELECT | actions | Get recent actions |

### SQL in ActionStore (store.py)

| Method | Line | Query Type | Table | Description |
|--------|------|------------|-------|-------------|
| `_load_audit` | 234 | SELECT | action_audit_events | Load audit trail |
| `_load_artifacts` | 259 | SELECT | action_artifacts | Load artifacts |
| `_load_steps` | 280 | SELECT | action_steps | Load action steps |
| `get_artifact` | 321 | SELECT | action_artifacts | Get single artifact |
| `create` | 347 | INSERT | actions | Create new action |
| `get_by_idempotency_key` | 389 | SELECT | actions | Idempotency check |
| `get_by_id` | 404 | SELECT | actions | Get action by ID |
| `list_actions` | 445 | SELECT | actions | List with filters |
| `_record_audit` | 471 | INSERT | action_audit_events | Record audit event |
| `record_artifact` | 478 | INSERT | action_artifacts | Save artifact |
| `update_inputs` | 511 | UPDATE | actions | Update inputs |
| `mark_ready` | 530 | UPDATE | actions | Mark ready |
| `approve` | 554 | UPDATE | actions | Approve action |
| `reject` | 582 | UPDATE | actions | Reject action |
| `hide_from_quarantine` | 611 | UPDATE | actions | Hide from quarantine |
| `acquire_lease` | 656 | INSERT/UPDATE | action_runner_leases | Lock action |
| `release_lease` | 692 | UPDATE | action_runner_leases | Release lock |
| `refresh_lease` | 703 | UPDATE | action_runner_leases | Heartbeat |
| `get_lease` | 709 | SELECT | action_runner_leases | Get lease info |
| `mark_processing` | 733 | UPDATE | actions | Start processing |
| `mark_completed` | 752 | UPDATE | actions | Complete action |
| `mark_failed` | 765 | UPDATE | actions | Mark failed |
| `claim_ready` | 784 | SELECT | actions | Claim ready action |
| `claim_retryable` | 808 | SELECT | actions | Claim retryable action |
| `retry` | 853 | UPDATE | actions | Schedule retry |
| `give_up` | 888 | UPDATE | actions | Stop retrying |
| `claim_stale` | 928 | SELECT | actions | Find stale processing |
| `mark_stale_ready` | 959 | UPDATE | actions | Reset stale to ready |
| `cancel` | 996 | UPDATE | actions | Cancel action |
| `archive_completed` | 1042 | UPDATE | actions | Archive completed |
| `bulk_archive` | 1086 | SELECT/UPDATE | actions | Bulk archive |
| `delete` | 1144 | UPDATE | actions | Soft delete |
| `get_stats` | 1175-1192 | SELECT | actions | Get statistics |
| `create_step` | 1244 | INSERT | action_steps | Create step |
| `get_step` | 1294 | SELECT | action_steps | Get step |
| `update_step` | 1320 | UPDATE | action_steps | Update step |
| `approve_step` | 1378 | UPDATE | action_steps | Approve step |
| `reject_step` | 1397 | (not impl) | action_steps | Reject step |
| `list_steps` | 1432 | SELECT | action_steps | List steps |
| `get_pending_steps` | 1474 | SELECT | action_steps | Get pending steps |

---

## Current SQLite Schema

### actions table

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

### action_audit_events table

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

### action_artifacts table

```sql
CREATE TABLE IF NOT EXISTS action_artifacts (
  artifact_id TEXT PRIMARY KEY,
  action_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  mime_type TEXT NOT NULL,
  path TEXT NOT NULL,
  size_bytes INTEGER NOT NULL DEFAULT 0,
  sha256 TEXT,
  created_at TEXT NOT NULL
);
```

### action_runner_leases table

```sql
CREATE TABLE IF NOT EXISTS action_runner_leases (
  runner_name TEXT PRIMARY KEY,
  owner_id TEXT NOT NULL,
  lease_expires_at TEXT NOT NULL,
  heartbeat_at TEXT NOT NULL,
  pid INTEGER NOT NULL,
  host TEXT NOT NULL
);
```

### action_steps table

```sql
CREATE TABLE IF NOT EXISTS action_steps (
  step_id TEXT PRIMARY KEY,
  action_id TEXT NOT NULL,
  step_index INTEGER NOT NULL,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT,
  ended_at TEXT,
  output_ref TEXT,
  error TEXT,
  requires_approval INTEGER NOT NULL DEFAULT 0,
  approved_by TEXT,
  approved_at TEXT,
  metadata TEXT,
  UNIQUE(action_id, step_index)
);
```

### action_memory table (memory/store.py)

```sql
CREATE TABLE IF NOT EXISTS action_memory (
  memory_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  action_id TEXT,
  action_type TEXT NOT NULL,
  deal_id TEXT,
  inputs_fingerprint TEXT NOT NULL,
  status TEXT NOT NULL,
  summary_text TEXT,
  output_data TEXT,
  UNIQUE(action_type, deal_id, inputs_fingerprint)
);
```

---

## Hardcoded Paths Found

| File | Path | Type | Line |
|------|------|------|------|
| store.py | `/home/zaks/DataRoom/.deal-registry/ingest_state.db` | Database | 28 |
| deal_lifecycle/main.py | `/home/zaks/DataRoom/.deal-registry/ingest_state.db` | Database | 2336 |
| deal_backfill_sender_history.py | `/home/zaks/DataRoom/.deal-registry/email_triage_state.db` | Database | 610 |
| deal_backfill_sender_history.py | `/home/zaks/DataRoom/.deal-registry/email_backfill_state.db` | Database | 684 |

---

## PostgreSQL Already in Use (Orchestration API)

The orchestration API (`src/api/orchestration/`) already uses PostgreSQL:

| File | Usage | Tables |
|------|-------|--------|
| `agent_invocation.py:41` | DATABASE_URL | agent_threads, agent_runs, agent_tool_calls, agent_events |
| `main.py:25` | DATABASE_URL | deals, actions, quarantine_items, deal_events, deal_aliases |

**Note**: There's a separate PostgreSQL schema (`zakops.*`) used by the orchestration API that is different from the SQLite schema used by the action engine.

---

## Migration Risk Assessment

| Pattern | Count | Migration Complexity | Notes |
|---------|-------|---------------------|-------|
| Raw SQL strings | 50+ | HIGH | Needs review for PostgreSQL syntax |
| `?` placeholder params | 50+ | MEDIUM | Convert to `$1, $2` format |
| TEXT timestamps | 20+ | MEDIUM | Convert to TIMESTAMPTZ |
| INTEGER booleans | 5+ | LOW | Convert to BOOLEAN |
| TEXT JSON | 10+ | LOW | Convert to JSONB |
| TEXT UUIDs | 15+ | LOW | Convert to UUID |
| No FK constraints | All | LOW | Add constraints |

### SQL Syntax Differences to Address

| SQLite | PostgreSQL | Count |
|--------|------------|-------|
| `?` placeholder | `$1, $2, ...` | 50+ |
| `TEXT` for timestamps | `TIMESTAMPTZ` | 20+ |
| `INTEGER` for boolean | `BOOLEAN` | 5+ |
| `INSERT OR REPLACE` | `ON CONFLICT DO UPDATE` | 3 |
| `AUTOINCREMENT` | `SERIAL` or `GENERATED` | 0 |
| No `UUID` type | `UUID` type | 15+ |

---

## Recommendations

1. **Create DatabaseAdapter** - Abstract SQLite/PostgreSQL differences
2. **Use parameterized queries** - Already done, just change placeholder syntax
3. **Add migration for types** - TEXT → proper types (UUID, TIMESTAMPTZ, BOOLEAN)
4. **Dual-write during transition** - Write to both DBs, read from SQLite first
5. **Validate data after migration** - Row counts, checksums
6. **Keep SQLite backup** - 30 days after PostgreSQL cutover
