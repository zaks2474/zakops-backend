-- Rollback: 001_foundation_tables
-- WARNING: This will DROP all tables created in this migration
-- Use with caution - data loss will occur

-- Drop new tables
DROP TABLE IF EXISTS zakops.inbox CASCADE;
DROP TABLE IF EXISTS zakops.outbox CASCADE;
DROP TABLE IF EXISTS zakops.idempotency_keys CASCADE;
DROP TABLE IF EXISTS zakops.execution_checkpoints CASCADE;
DROP TABLE IF EXISTS zakops.artifacts CASCADE;
DROP TABLE IF EXISTS zakops.operators CASCADE;

-- Note: We do NOT drop columns from existing tables to avoid data loss
-- If you need to remove added columns, do so manually:
--
-- ALTER TABLE zakops.actions DROP COLUMN IF EXISTS trace_id;
-- ALTER TABLE zakops.actions DROP COLUMN IF EXISTS correlation_id;
-- ALTER TABLE zakops.actions DROP COLUMN IF EXISTS causation_id;
-- ALTER TABLE zakops.actions DROP COLUMN IF EXISTS approved_by;
-- ALTER TABLE zakops.actions DROP COLUMN IF EXISTS approved_at;
-- ALTER TABLE zakops.actions DROP COLUMN IF EXISTS operator_id;
-- ALTER TABLE zakops.actions DROP COLUMN IF EXISTS run_id;
--
-- ALTER TABLE zakops.agent_events DROP COLUMN IF EXISTS correlation_id;
-- ALTER TABLE zakops.agent_events DROP COLUMN IF EXISTS trace_id;
-- ALTER TABLE zakops.agent_events DROP COLUMN IF EXISTS causation_id;

-- Remove migration record
DELETE FROM zakops.schema_migrations WHERE version = '001';

-- Note: schema_migrations table is NOT dropped to preserve other migration history
