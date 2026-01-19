# API Inventory

**Audit Date**: 2026-01-19
**Repository**: zakops-backend
**Status**: Phase 0 Baseline Audit

---

## FastAPI Routes (deal_lifecycle/main.py)

### Dashboard & Health Endpoints

| Method | Endpoint | Handler | Notes |
|--------|----------|---------|-------|
| GET | / | HTML response | Dashboard UI |
| GET | /dashboard | HTML response | Dashboard UI |
| GET | /health | health check | Returns OK |
| GET | /metrics | Prometheus metrics | Standard metrics |
| GET | /api/version | Version info | API version |
| GET | /api/diagnostics | System diagnostics | Debug info |
| GET | /api/debug/config | Config debug | Dev only |

### Deal Endpoints

| Method | Endpoint | Handler | Spec Compliant |
|--------|----------|---------|----------------|
| GET | /api/deals | List deals | Yes |
| GET | /api/deals/{deal_id} | Get deal detail | Yes |
| POST | /api/deals/{deal_id}/archive | Archive deal | Yes |
| POST | /api/deals/{deal_id}/restore | Restore deal | N/A (extra) |
| POST | /api/deals/bulk-archive | Bulk archive | Yes |
| GET | /api/deals/{deal_id}/pipeline-outputs | Pipeline outputs | N/A (extra) |
| GET | /api/deals/{deal_id}/events | Deal events | Yes |
| GET | /api/deals/{deal_id}/case-file | Case file | Yes |
| GET | /api/deals/{deal_id}/materials | Materials | Yes |
| GET | /api/deals/{deal_id}/enrichment | Enrichment data | N/A (extra) |
| POST | /api/deals/{deal_id}/transition | Stage transition | Yes |
| POST | /api/deals/{deal_id}/note | Add note | Yes |

**Missing from Spec**:
- GET /api/deals/:id/workspace (combined workspace data)
- POST /api/deals (create deal with idempotency)

### Action Endpoints (Kinetic Action Engine)

| Method | Endpoint | Handler | Spec Compliant |
|--------|----------|---------|----------------|
| GET | /api/actions | List actions | Yes |
| POST | /api/actions | Create action | Yes |
| GET | /api/actions/{action_id} | Get action | Yes |
| POST | /api/actions/{action_id}/approve | Approve | Yes |
| POST | /api/actions/{action_id}/execute | Execute | Yes |
| POST | /api/actions/{action_id}/cancel | Cancel | Yes |
| POST | /api/actions/{action_id}/update | Update inputs | N/A (extra) |
| POST | /api/actions/{action_id}/retry | Retry | N/A (extra) |
| GET | /api/actions/{action_id}/artifacts | List artifacts | N/A (extra) |
| GET | /api/actions/{action_id}/artifact/{artifact_id} | Download | N/A (extra) |
| GET | /api/actions/capabilities | List capabilities | N/A (extra) |
| GET | /api/actions/capabilities/{capability_id} | Get capability | N/A (extra) |
| GET | /api/actions/metrics | Action metrics | N/A (extra) |
| POST | /api/actions/plan | Plan actions | N/A (extra) |
| GET | /api/actions/runner-status | Runner status | N/A (extra) |

### Action Bulk Operations

| Method | Endpoint | Handler | Spec Compliant |
|--------|----------|---------|----------------|
| POST | /api/actions/bulk/archive | Bulk archive | Yes |
| POST | /api/actions/bulk/delete | Bulk delete | Yes |
| POST | /api/actions/clear-completed | Clear completed | Yes |
| GET | /api/actions/completed-count | Count completed | Yes |

### Quarantine Endpoints

| Method | Endpoint | Handler | Spec Compliant |
|--------|----------|---------|----------------|
| GET | /api/quarantine | List items | Yes |
| GET | /api/quarantine/health | Health status | N/A (extra) |
| GET | /api/quarantine/{id} | Get item | Yes |
| POST | /api/quarantine/{id}/resolve | Resolve | Custom |
| POST | /api/quarantine/{id}/delete | Delete/hide | N/A (extra) |
| POST | /api/quarantine/bulk-delete | Bulk delete | N/A (extra) |

**Note**: Quarantine uses action-based workflow (EMAIL_TRIAGE.REVIEW_EMAIL)

### Quarantine via Actions

| Method | Endpoint | Handler | Notes |
|--------|----------|---------|-------|
| GET | /api/actions/quarantine | List quarantine actions | Primary |
| GET | /api/actions/quarantine/{action_id} | Get quarantine action | |
| GET | /api/actions/quarantine/{action_id}/preview | Preview | |
| POST | /api/actions/quarantine/{action_id}/approve | Approve | |
| POST | /api/actions/quarantine/{action_id}/reject | Reject | |

### Legacy Deferred Actions

| Method | Endpoint | Handler | Notes |
|--------|----------|---------|-------|
| GET | /api/deferred-actions | List | Legacy |
| GET | /api/deferred-actions/due | Due actions | Legacy |
| POST | /api/deferred-actions/{action_id}/execute | Execute | Legacy |
| POST | /api/deferred-actions/{action_id}/cancel | Cancel | Legacy |

### Enrichment Endpoints

| Method | Endpoint | Handler | Notes |
|--------|----------|---------|-------|
| GET | /api/enrichment/audit | Enrichment audit | |
| GET | /api/enrichment/pending-links | Pending links | |
| POST | /api/enrichment/mark-link-fetched | Mark fetched | |

### Chat Endpoints

| Method | Endpoint | Handler | Spec Compliant |
|--------|----------|---------|----------------|
| POST | /api/chat | SSE streaming | Yes |
| POST | /api/chat/complete | Non-streaming | Yes |
| POST | /api/chat/execute-proposal | Execute proposal | Yes |
| GET | /api/chat/session/{session_id} | Get session | Yes |
| GET | /api/chat/llm-health | LLM health | N/A (extra) |

### Agent Endpoints

| Method | Endpoint | Handler | Spec Compliant |
|--------|----------|---------|----------------|
| POST | /api/agents/{agent_name}/invoke | Invoke agent | N/A |
| GET | /api/agents/{agent_name}/history | Agent history | N/A |

**Missing from Spec**:
- GET /api/agent/activity (event timeline)
- GET /api/agent/runs (run list)

### Tool Endpoints

| Method | Endpoint | Handler | Notes |
|--------|----------|---------|-------|
| GET | /api/tools | List tools | |
| GET | /api/tools/health | Tools health | |
| GET | /api/tools/{tool_id} | Get tool | |

### Pipeline & Metrics

| Method | Endpoint | Handler | Notes |
|--------|----------|---------|-------|
| GET | /api/pipeline | Pipeline summary | |
| GET | /api/alerts | System alerts | |
| GET | /api/checkpoints | Checkpoints | |
| GET | /api/metrics/classification | Classification metrics | |

### Gmail Integration

| Method | Endpoint | Handler | Notes |
|--------|----------|---------|-------|
| GET | /api/gmail/health | Gmail health | |

### Debug Endpoints

| Method | Endpoint | Handler | Notes |
|--------|----------|---------|-------|
| POST | /api/actions/{action_id}/unstick | Unstick action | Admin |
| POST | /api/actions/{action_id}/requeue | Requeue action | Admin |
| GET | /api/actions/{action_id}/debug | Debug info | Admin |
| GET | /api/actions/debug/missing-executors | Missing executors | Admin |
| GET | /api/actions/debug/capability-mismatches | Capability issues | Admin |

---

## Total Endpoint Count

| Category | Count |
|----------|-------|
| Health/Diagnostics | 7 |
| Deals | 12 |
| Actions | 21 |
| Quarantine | 11 |
| Enrichment | 3 |
| Chat | 5 |
| Agent | 2 |
| Tools | 3 |
| Pipeline/Metrics | 4 |
| Gmail | 1 |
| Debug | 5 |
| **Total** | **74** |

---

## Spec Gap Analysis

### Missing Endpoints (from Master Architecture Spec)

| Spec Endpoint | Status | Impact |
|---------------|--------|--------|
| GET /api/dashboard/overview | Missing | Dashboard uses multiple calls |
| GET /api/hq/stats | Missing | HQ uses /api/pipeline |
| GET /api/agent/activity | Missing | No agent_events table |
| GET /api/agent/runs | Missing | No agent_runs table |
| GET /api/deals/:id/workspace | Missing | Uses multiple calls |
| POST /api/quarantine/:id/approve | Different | Uses action-based workflow |
| POST /api/quarantine/:id/reject | Different | Uses action-based workflow |

### Idempotency Headers

| Endpoint | Spec Requirement | Current Status |
|----------|------------------|----------------|
| POST /api/actions | Idempotency-Key required | Uses idempotency_key in body |
| POST /api/deals | Idempotency-Key required | No create endpoint exists |
| POST /api/actions/:id/approve | Idempotency-Key required | Not implemented |

---

## Authentication

**Current**: No authentication middleware visible in main.py
**Spec Requirement**: Session-based authentication with operator_id from session

**Gap**: Authentication/authorization not implemented. All endpoints appear to be unprotected.
