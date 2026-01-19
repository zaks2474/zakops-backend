# Artifact Storage Audit

**Audit Date**: 2026-01-19
**Repository**: zakops-backend
**Status**: Phase 0 Baseline Audit

---

## Current Artifact Storage Configuration

### Base Path

**Development/Lab Path**: `/home/zaks/DataRoom`

**Environment Variables** (from .env.example):
```bash
DATAROOM_ROOT=/home/zaks/DataRoom
ARTIFACT_STORAGE_PATH=/var/lib/zakops/dataroom
```

### Configuration Method

- [x] Environment variable (DATAROOM_ROOT)
- [x] Environment variable (ARTIFACT_STORAGE_PATH)
- [x] Hardcoded fallback paths exist in code

---

## DataRoom Directory Structure

```
/home/zaks/DataRoom/
├── .deal-registry/          # Registry and state databases
│   ├── deal_registry.json   # Main deal registry (JSON)
│   └── ingest_state.db      # SQLite action engine database
├── .intake-dropzone/        # Email intake processing
├── .triage_stats.json       # Triage statistics
├── 00-PIPELINE/             # Pipeline configuration
├── 01-ACTIVE-DEALS/         # Active deal folders
├── 02-PORTFOLIO/            # Portfolio data
├── 03-DEAL-SOURCES/         # Deal sources
├── 04-FRAMEWORKS/           # Framework templates
├── 05-ADVISORS/             # Advisor information
├── 06-KNOWLEDGE-BASE/       # Knowledge base documents
├── 07-FINANCE/              # Financial data
├── 08-ARCHIVE/              # Archived items
└── [Documentation files]    # Various MD files
```

---

## Artifact References in Database

### action_artifacts Table (SQLite)

| Column | Type | Contains Path? | Example |
|--------|------|----------------|---------|
| path | TEXT | Yes | /home/zaks/DataRoom/... |
| filename | TEXT | No | document.pdf |

**Storage Pattern**: Artifacts are stored with full filesystem paths in the `path` column.

---

## Storage Abstraction Status

### Spec Requirements

- [ ] ArtifactStore interface exists
- [ ] LocalFilesystemArtifactStore exists
- [ ] S3ArtifactStore exists
- [ ] Storage backend is configurable
- [ ] storage_uri column in artifacts table
- [ ] storage_key column in artifacts table

### Current Implementation

The current implementation uses **direct filesystem paths** stored in the database. There is no abstraction layer.

**Code References** (from store.py):
```python
# Artifact paths are stored as full filesystem paths
path TEXT NOT NULL,  # e.g., "/home/zaks/DataRoom/deals/ABC/artifact.pdf"
```

**Code References** (from executors):
```python
# Direct Path usage in executor code
registry_path = Path("/home/zaks/DataRoom/.deal-registry/deal_registry.json")
```

---

## Gap Analysis

### Missing from Spec

| Requirement | Status | Notes |
|-------------|--------|-------|
| ArtifactStore abstract interface | Missing | Not implemented |
| LocalFilesystemArtifactStore | Missing | Uses direct paths |
| S3ArtifactStore | Missing | Not implemented |
| get_artifact_store() factory | Missing | Not implemented |
| storage_uri column | Missing | Uses `path` instead |
| storage_key column | Missing | Uses `path` instead |
| Configurable storage backend | Partial | Has env vars but no abstraction |

### Hardcoded Paths Found

| File | Path | Type |
|------|------|------|
| store.py | /home/zaks/DataRoom/.deal-registry/ingest_state.db | Database |
| test_e2e_actions.py | /home/zaks/DataRoom/.deal-registry/deal_registry.json | Registry |

---

## Artifact Key Structure Analysis

### Spec Requirement

```
{deal_id}/
├── emails/
│   └── {message_id}.eml
├── documents/
│   ├── cim/
│   ├── teasers/
│   └── financials/
├── generated/
│   ├── loi/
│   └── responses/
└── extracted/
```

### Current Structure

The current structure uses a flat folder hierarchy:
```
01-ACTIVE-DEALS/
├── {deal_folder_name}/
│   └── [files directly in deal folder]
```

**Gap**: The current structure does not follow the spec's organized artifact key structure with typed subdirectories.

---

## Migration Considerations

### For S3/MinIO Compatibility

1. **Add storage_uri and storage_key columns** to artifacts table
2. **Implement ArtifactStore interface** with:
   - put(key, data) -> storage_uri
   - get(key) -> data
   - delete(key) -> bool
   - exists(key) -> bool
   - get_url(key) -> signed URL
   - list(prefix) -> keys
3. **Backfill existing artifacts** with proper storage keys
4. **Keep backward compatibility** for existing filesystem paths

### Database Changes Required

```sql
-- Add new columns to artifacts table (when migrated to PostgreSQL)
ALTER TABLE artifacts ADD COLUMN storage_uri VARCHAR(1000);
ALTER TABLE artifacts ADD COLUMN storage_key VARCHAR(500);

-- Backfill with file:// URIs for existing paths
UPDATE artifacts SET
  storage_uri = 'file://' || path,
  storage_key = path;
```

---

## Recommendations

1. **Implement ArtifactStore abstraction** before any production deployment
2. **Add storage_uri/storage_key columns** to PostgreSQL artifacts table
3. **Migrate to structured artifact keys** following spec pattern
4. **Remove hardcoded paths** from code, use environment variables
5. **Consider S3-compatible storage** (MinIO) for HA deployments

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Hardcoded paths break portability | High | Extract to config |
| No storage abstraction limits scaling | Medium | Implement ArtifactStore |
| Direct filesystem paths in DB | Medium | Add storage_uri column |
| No S3 support for production | High | Implement S3ArtifactStore |
