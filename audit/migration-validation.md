# Migration Validation Report

**Date**: 2026-01-19 15:27:39 UTC
**Status**: PASSED

## Summary

- **Passed**: 14
- **Failed**: 0
- **Warnings**: 0

## Checks

| Check | Status | Message |
|-------|--------|---------|
| deals_count | PASS | Found 5 deals in PostgreSQL |
| deals_canonical_name | PASS | All deals have canonical_name |
| deals_stage | PASS | All deals have stage |
| deals_deleted | PASS | 0 deals marked as deleted |
| actions_count | PASS | Found 83 actions in PostgreSQL |
| actions_fk_deals | PASS | All action foreign keys valid |
| actions_status_distribution | PASS | Action statuses: {'READY': 1, 'COMPLETED': 79, 'PENDING_APPROVAL': 3} |
| artifacts_count | PASS | Found 0 artifacts in PostgreSQL |
| artifacts_note | PASS | No artifacts to validate (empty table) |
| outbox_total | PASS | Outbox has 5 total entries |
| outbox_stuck | PASS | No stuck outbox entries |
| outbox_status_distribution | PASS | Outbox statuses: {'pending': 5} |
| agent_data | PASS | Agent data: 0 threads, 0 runs, 0 events |
| checkpoints_count | PASS | Found 0 execution checkpoints |

## Details

### deals_deleted

```json
{
  "deleted_count": 0
}
```

### actions_status_distribution

```json
{
  "status_counts": {
    "READY": 1,
    "COMPLETED": 79,
    "PENDING_APPROVAL": 3
  }
}
```

### outbox_status_distribution

```json
{
  "status_counts": {
    "pending": 5
  }
}
```

### agent_data

```json
{
  "threads": 0,
  "runs": 0,
  "events": 0
}
```
