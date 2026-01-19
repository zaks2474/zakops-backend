# Phase 3: Execution Hardening (Outbox/Inbox) - Completion Report

**Date**: 2026-01-19
**Phase**: 3 of 8
**Status**: ✅ COMPLETE
**Dependencies**: Phase 1 ✅, Phase 2 ✅

---

## Executive Summary

Phase 3 implements the Outbox and Inbox patterns for reliable, exactly-once event delivery with transactional guarantees. This enhances the Phase 2 event system without breaking existing functionality.

**Key Deliverables:**
- ✅ OutboxWriter - Transactional event writing
- ✅ OutboxProcessor - Background delivery worker
- ✅ InboxGuard - Consumer-side deduplication
- ✅ TransactionalPublisher - Atomic business + events
- ✅ Lifecycle Integration - FastAPI lifespan hooks
- ✅ Backward Compatibility - Existing event publishing unchanged

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRODUCER (API/Worker)                        │
│                                                                  │
│  BEGIN TRANSACTION                                               │
│    1. Execute business logic (create action, update deal)       │
│    2. Write event to OUTBOX table (via OutboxWriter)            │
│  COMMIT TRANSACTION                                              │
│                                                                  │
│  ✅ Atomic: Either both succeed or both fail                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTBOX TABLE                                │
│                                                                  │
│  id | correlation_id | event_type | event_data | status         │
│  ---|----------------|------------|------------|--------        │
│  1  | deal-123       | action.created | {...}  | pending        │
│  2  | deal-456       | deal.updated   | {...}  | delivered      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTBOX PROCESSOR (Worker)                     │
│                                                                  │
│  POLL: SELECT * FROM outbox WHERE status = 'pending'            │
│                                                                  │
│  FOR EACH event:                                                │
│    1. Deliver to event system (Phase 2 publisher)               │
│    2. Mark as 'delivered' in outbox                             │
│                                                                  │
│  Retry failed deliveries with exponential backoff               │
│  (5s, 15s, 1m, 5m, 15m)                                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CONSUMER                                   │
│                                                                  │
│  BEFORE processing:                                             │
│    Check INBOX via InboxGuard                                   │
│    If processed → Skip (idempotent)                             │
│    If not → Process and record in INBOX                         │
│                                                                  │
│  ✅ Exactly-once semantics via deduplication                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files Created

### Outbox Module (`src/core/outbox/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `models.py` | OutboxEntry and OutboxStatus models |
| `writer.py` | OutboxWriter for transactional event writing |
| `processor.py` | OutboxProcessor background worker |
| `transactional.py` | TransactionalPublisher context manager |
| `lifecycle.py` | FastAPI lifespan integration |

### Inbox Module (`src/core/inbox/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `guard.py` | InboxGuard for consumer deduplication |

### Updated Files

| File | Changes |
|------|---------|
| `src/core/events/integration.py` | Added outbox support with backward compatibility |
| `.env.example` | Added outbox configuration options |

---

## Usage Examples

### Writing to Outbox

```python
from src.core.outbox import get_outbox_writer

async with get_outbox_writer() as writer:
    await writer.write(
        correlation_id=deal_id,
        event_type="action.created",
        event_data={"action_id": str(action_id), "title": title}
    )
```

### Transactional Publishing

```python
from src.core.outbox import transactional_publish

async with transactional_publish() as txn:
    # Business logic
    await txn.db.execute("INSERT INTO actions ...")

    # Event (same transaction)
    await txn.emit(
        correlation_id=deal_id,
        event_type="action.created",
        event_data={"action_id": str(action_id)}
    )
# Both commit together or both rollback
```

### Consumer Deduplication

```python
from src.core.inbox import InboxGuard

async with InboxGuard(event_id, "my-consumer") as guard:
    if guard.should_process:
        # Process the event
        await handle_event(event)
    else:
        # Already processed, skip
        pass
```

### Starting Outbox Processor

```python
from src.core.outbox.lifecycle import outbox_lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with outbox_lifespan():
        yield

app = FastAPI(lifespan=lifespan)
```

---

## Configuration Options

| Variable | Default | Purpose |
|----------|---------|---------|
| `OUTBOX_ENABLED` | `true` | Enable/disable outbox pattern |
| `OUTBOX_PROCESSOR_ENABLED` | `true` | Run processor on this instance |
| `OUTBOX_POLL_INTERVAL` | `1.0` | Polling interval (seconds) |
| `OUTBOX_BATCH_SIZE` | `100` | Entries per batch |
| `OUTBOX_MAX_ATTEMPTS` | `5` | Max retry attempts |

---

## Backward Compatibility

### Critical Constraint Met ✅

```
1. Existing event publishing (Phase 2) continues to work  ✅
2. Outbox is ADDITIVE — events still go to agent_events   ✅
3. Inbox deduplication is transparent to consumers         ✅
4. No breaking changes to API response shapes              ✅
5. Gradual rollout: outbox can be disabled via config      ✅
```

### How It Works

- When `OUTBOX_ENABLED=true` (default): Events go through outbox for reliability
- When `OUTBOX_ENABLED=false`: Events published directly (Phase 2 behavior)
- The `emit_deal_event` and `emit_action_event` decorators automatically use the appropriate method

---

## Outbox Entry Lifecycle

```
┌─────────┐    ┌────────────┐    ┌───────────┐    ┌────────────┐
│ PENDING │───▶│ PROCESSING │───▶│ DELIVERED │    │    DEAD    │
└─────────┘    └────────────┘    └───────────┘    └────────────┘
     ▲               │                                   ▲
     │               │ (failure)                         │
     └───────────────┘ (if attempts < max)               │
                       └─────────────────────────────────┘
                         (if attempts >= max)
```

### Retry Strategy

Exponential backoff intervals:
1. 5 seconds
2. 15 seconds
3. 1 minute
4. 5 minutes
5. 15 minutes

After 5 failed attempts, entry moves to `DEAD` status (dead letter queue).

---

## Database Tables Used

From Phase 1 migration:

### `zakops.outbox`
```sql
id UUID PRIMARY KEY
correlation_id UUID NOT NULL
aggregate_type VARCHAR(50)
aggregate_id VARCHAR(100)
event_type VARCHAR(100)
schema_version INTEGER DEFAULT 1
event_data JSONB
trace_id UUID
status VARCHAR(50) DEFAULT 'pending'
attempts INTEGER DEFAULT 0
max_attempts INTEGER DEFAULT 5
last_attempt_at TIMESTAMPTZ
next_attempt_at TIMESTAMPTZ
delivered_at TIMESTAMPTZ
error_message TEXT
created_at TIMESTAMPTZ
```

### `zakops.inbox`
```sql
id UUID PRIMARY KEY
event_id UUID NOT NULL
consumer_id VARCHAR(100) NOT NULL
processed_at TIMESTAMPTZ
UNIQUE(event_id, consumer_id)
```

---

## Quality Gates

| Gate | Status |
|------|--------|
| Outbox tables ready (Phase 1) | ✅ |
| Outbox writer works | ✅ |
| Outbox processor works | ✅ |
| Inbox deduplication works | ✅ |
| Existing events work | ✅ |
| All modules compile | ✅ |

---

## Next Steps

- **Phase 4**: ✅ Artifact Storage (Complete)
- **Phase 5**: API Stabilization
- **Phase 6**: HITL & Checkpoints

---

## Sign-off

- [x] All deliverables complete
- [x] Backward compatibility verified
- [x] Configuration documented
- [x] Code compiles cleanly
- [x] Ready for Phase 5

**Phase 3 Status: COMPLETE** ✅
