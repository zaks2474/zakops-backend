# Event System Audit

**Audit Date**: 2026-01-19
**Repository**: zakops-backend
**Status**: Phase 0 Baseline Audit

---

## Current Event System

### Existing Tables

| Table | Purpose | Append-Only? |
|-------|---------|--------------|
| action_audit_events | Action state changes | Yes |
| (none) | General agent events | N/A |

### action_audit_events Schema

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

**Events Captured**:
- Action state transitions (created, approved, started, completed, failed)
- Action cancellations
- Retry attempts

**Events NOT Captured**:
- Agent run start/complete
- Tool calls
- Deal events
- Email events
- Worker events

---

## Spec Requirements

### agent_events Table (Missing)

```sql
CREATE TABLE agent_events (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES agent_runs(id),

    -- Correlation
    trace_id UUID NOT NULL,
    correlation_id UUID NOT NULL,  -- = deal_id
    causation_id UUID,

    -- Event info
    event_type VARCHAR(100) NOT NULL,
    schema_version INT NOT NULL DEFAULT 1,
    payload JSONB NOT NULL,

    -- Context
    deal_id UUID,
    action_id UUID,

    -- Timestamp (immutable)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()

    -- NO updated_at - events are immutable
);
```

### Required Event Types

| Category | Event Type | Status |
|----------|------------|--------|
| **Deal Events** | deal.created | Missing |
| | deal.stage_changed | Missing |
| | deal.archived | Missing |
| | deal.scored | Missing |
| **Action Events** | action.created | Partial (audit) |
| | action.awaiting_approval | Partial (audit) |
| | action.approved | Partial (audit) |
| | action.rejected | Partial (audit) |
| | action.completed | Partial (audit) |
| | action.failed | Partial (audit) |
| **Run Events** | run.started | Missing |
| | run.tool_called | Missing |
| | run.tool_completed | Missing |
| | run.paused | Missing |
| | run.resumed | Missing |
| | run.completed | Missing |
| | run.failed | Missing |
| **Worker Events** | worker.job_queued | Missing |
| | worker.job_started | Missing |
| | worker.job_retrying | Missing |
| | worker.job_completed | Missing |
| | worker.job_failed | Missing |
| **Email Events** | email.ingested | Missing |
| | email.classified | Missing |
| | email.processed | Missing |

---

## Outbox/Inbox Pattern

### Outbox Table (Missing)

```sql
CREATE TABLE outbox (
    id UUID PRIMARY KEY,
    aggregate_type VARCHAR(50) NOT NULL,
    aggregate_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    schema_version INT NOT NULL DEFAULT 1,
    payload JSONB NOT NULL,
    trace_id UUID,
    correlation_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    attempts INT NOT NULL DEFAULT 0,
    last_error TEXT,
    is_dead BOOLEAN NOT NULL DEFAULT FALSE
);
```

**Status**: Not implemented

### Inbox Table (Missing)

```sql
CREATE TABLE inbox (
    event_id UUID PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Status**: Not implemented

---

## Correlation ID Implementation

### Spec Requirement

```python
# Every record should have:
trace_id     # New UUID per request
correlation_id  # = deal_id (always)
causation_id  # ID of event that caused this event
```

### Current Implementation

| Table | trace_id | correlation_id | causation_id |
|-------|----------|----------------|--------------|
| actions | Missing | Missing | Missing |
| action_audit_events | Missing | Missing | Missing |

**Gap**: No correlation IDs implemented anywhere in current system.

---

## Event Emission

### Current Pattern

Events are emitted by directly inserting into `action_audit_events`:

```python
# Current pattern (simplified)
def record_audit(action_id: str, event: str, actor: str, details: dict):
    db.execute("""
        INSERT INTO action_audit_events
        (audit_id, action_id, timestamp, event, actor, details)
        VALUES (?, ?, ?, ?, ?, ?)
    """, uuid4(), action_id, now(), event, actor, json.dumps(details))
```

### Spec Pattern

```python
# Spec pattern
def emit_event(event_type: str, payload: dict, run_id: UUID, trace_id: UUID):
    await db.execute("""
        INSERT INTO agent_events
        (run_id, trace_id, correlation_id, event_type, schema_version, payload)
        VALUES ($1, $2, $3, $4, $5, $6)
    """, run_id, trace_id, correlation_id, event_type, 1, json.dumps(payload))

    # Also write to outbox for reliable publishing
    await db.execute("""
        INSERT INTO outbox
        (aggregate_type, aggregate_id, event_type, payload, trace_id, correlation_id)
        VALUES ($1, $2, $3, $4, $5, $6)
    """, ...)
```

---

## Schema Versioning

### Spec Requirement

All events should include `schema_version` for forward compatibility.

### Current Implementation

- [x] action_audit_events has `details` (JSON) but no schema version
- [ ] No event payload versioning

---

## Event Archival

### Spec Requirement

```sql
CREATE TABLE agent_events_archive (
    -- Same schema as agent_events
    -- For archiving old events
);
```

### Current Implementation

Not implemented. No event archival strategy.

---

## Gap Summary

| Component | Spec | Current | Gap |
|-----------|------|---------|-----|
| Event table | agent_events (PostgreSQL) | action_audit_events (SQLite) | Major |
| Event types | 20+ types | ~6 action types | Major |
| Correlation IDs | trace/correlation/causation | None | Major |
| Outbox pattern | Required | Missing | Major |
| Inbox deduplication | Required | Missing | Major |
| Schema versioning | Required | Missing | Medium |
| Event archival | agent_events_archive | Missing | Low |

---

## Recommendations

1. **Create agent_events table** in PostgreSQL with full spec schema
2. **Add correlation IDs** (trace_id, correlation_id, causation_id) to all tables
3. **Implement outbox pattern** for reliable event publishing
4. **Add inbox table** for consumer-side deduplication
5. **Define event type registry** with versioned payload schemas
6. **Migrate action_audit_events** to new agent_events table
7. **Implement event archival** strategy for old events
