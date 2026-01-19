-- Migration: 001_foundation_tables
-- Description: Add foundation tables for ZakOps Master Architecture compliance
-- Date: 2026-01-19
-- Compatibility: ADDITIVE ONLY - does not modify existing tables
-- Rollback: See 001_foundation_tables_rollback.sql

-- ============================================================================
-- SCHEMA MIGRATIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS zakops.schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    description TEXT,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================================
-- OPERATORS TABLE (User/Approver Identity)
-- ============================================================================
-- Spec Section: Authentication & Security

CREATE TABLE IF NOT EXISTS zakops.operators (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'operator',
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT operators_role_check CHECK (role IN ('admin', 'operator', 'viewer'))
);

CREATE INDEX IF NOT EXISTS idx_operators_email ON zakops.operators(email);

-- ============================================================================
-- ARTIFACTS TABLE (Document/File Storage)
-- ============================================================================
-- Spec Section: Storage Abstraction

CREATE TABLE IF NOT EXISTS zakops.artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id UUID NOT NULL,
    deal_id VARCHAR(20) REFERENCES zakops.deals(deal_id) ON DELETE SET NULL,
    action_id VARCHAR(50) REFERENCES zakops.actions(action_id) ON DELETE SET NULL,

    -- File info
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    file_type VARCHAR(100),
    file_size BIGINT,
    mime_type VARCHAR(100),
    sha256 VARCHAR(64),

    -- Metadata
    category VARCHAR(100),
    extracted_text TEXT,
    metadata JSONB DEFAULT '{}',

    -- Storage abstraction
    storage_backend VARCHAR(50) DEFAULT 'local',
    storage_uri VARCHAR(1024),
    storage_key VARCHAR(500),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_artifacts_correlation_id ON zakops.artifacts(correlation_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_deal_id ON zakops.artifacts(deal_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_action_id ON zakops.artifacts(action_id);

-- ============================================================================
-- EXECUTION CHECKPOINTS TABLE (Durable Execution)
-- ============================================================================
-- Spec Section: HITL & Checkpoints

CREATE TABLE IF NOT EXISTS zakops.execution_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id UUID NOT NULL,
    action_id VARCHAR(50) REFERENCES zakops.actions(action_id) ON DELETE CASCADE,
    run_id UUID,

    -- Checkpoint info
    checkpoint_name VARCHAR(255) NOT NULL,
    checkpoint_type VARCHAR(50) DEFAULT 'state',
    checkpoint_data JSONB NOT NULL,

    -- Sequence for ordering
    sequence_number INTEGER NOT NULL,

    -- State
    status VARCHAR(50) DEFAULT 'active',
    expires_at TIMESTAMPTZ,
    resumed_at TIMESTAMPTZ,
    resumed_by UUID,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(action_id, checkpoint_name, sequence_number),
    CONSTRAINT checkpoints_status_check CHECK (status IN ('active', 'resumed', 'expired', 'cancelled'))
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_action_id ON zakops.execution_checkpoints(action_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_status ON zakops.execution_checkpoints(status);

-- ============================================================================
-- IDEMPOTENCY KEYS TABLE (Request Deduplication)
-- ============================================================================
-- Spec Section: Execution Hardening

CREATE TABLE IF NOT EXISTS zakops.idempotency_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Key
    idempotency_key VARCHAR(255) UNIQUE NOT NULL,

    -- Request identity
    request_hash VARCHAR(64) NOT NULL,
    endpoint VARCHAR(255),
    method VARCHAR(10),

    -- Cached response
    response_status INTEGER,
    response_body JSONB,

    -- Timing
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '24 hours'),

    -- Processing state
    is_processing BOOLEAN DEFAULT FALSE,
    processing_started_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_idempotency_key ON zakops.idempotency_keys(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON zakops.idempotency_keys(expires_at);

-- ============================================================================
-- OUTBOX TABLE (Reliable Event Delivery)
-- ============================================================================
-- Spec Section: Event System

CREATE TABLE IF NOT EXISTS zakops.outbox (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id UUID NOT NULL,

    -- Event info
    aggregate_type VARCHAR(50) NOT NULL,
    aggregate_id VARCHAR(100) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    event_data JSONB NOT NULL,

    -- Tracing
    trace_id UUID,

    -- Delivery tracking
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 5,
    last_attempt_at TIMESTAMPTZ,
    next_attempt_at TIMESTAMPTZ,
    delivered_at TIMESTAMPTZ,
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT outbox_status_check CHECK (status IN ('pending', 'processing', 'delivered', 'failed', 'dead'))
);

CREATE INDEX IF NOT EXISTS idx_outbox_status ON zakops.outbox(status);
CREATE INDEX IF NOT EXISTS idx_outbox_next_attempt ON zakops.outbox(next_attempt_at) WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_outbox_created_at ON zakops.outbox(created_at);

-- ============================================================================
-- INBOX TABLE (Consumer-Side Deduplication)
-- ============================================================================
-- Spec Section: Event System

CREATE TABLE IF NOT EXISTS zakops.inbox (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Event identity
    event_id UUID NOT NULL,
    consumer_id VARCHAR(100) NOT NULL,

    -- Processing info
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique constraint for deduplication
    UNIQUE(event_id, consumer_id)
);

CREATE INDEX IF NOT EXISTS idx_inbox_event_id ON zakops.inbox(event_id);

-- ============================================================================
-- ADD COLUMNS TO EXISTING TABLES (ADDITIVE ONLY)
-- ============================================================================

-- Add trace_id and correlation_id to actions if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'actions' AND column_name = 'trace_id') THEN
        ALTER TABLE zakops.actions ADD COLUMN trace_id UUID;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'actions' AND column_name = 'correlation_id') THEN
        ALTER TABLE zakops.actions ADD COLUMN correlation_id UUID;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'actions' AND column_name = 'causation_id') THEN
        ALTER TABLE zakops.actions ADD COLUMN causation_id UUID;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'actions' AND column_name = 'approved_by') THEN
        ALTER TABLE zakops.actions ADD COLUMN approved_by UUID;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'actions' AND column_name = 'approved_at') THEN
        ALTER TABLE zakops.actions ADD COLUMN approved_at TIMESTAMPTZ;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'actions' AND column_name = 'operator_id') THEN
        ALTER TABLE zakops.actions ADD COLUMN operator_id UUID;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'actions' AND column_name = 'run_id') THEN
        ALTER TABLE zakops.actions ADD COLUMN run_id UUID;
    END IF;
END $$;

-- Add correlation_id to agent_events if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'agent_events' AND column_name = 'correlation_id') THEN
        ALTER TABLE zakops.agent_events ADD COLUMN correlation_id UUID;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'agent_events' AND column_name = 'trace_id') THEN
        ALTER TABLE zakops.agent_events ADD COLUMN trace_id UUID;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_schema = 'zakops' AND table_name = 'agent_events' AND column_name = 'causation_id') THEN
        ALTER TABLE zakops.agent_events ADD COLUMN causation_id UUID;
    END IF;
END $$;

-- ============================================================================
-- CREATE INDEXES FOR NEW COLUMNS
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_actions_trace_id ON zakops.actions(trace_id);
CREATE INDEX IF NOT EXISTS idx_actions_correlation_id ON zakops.actions(correlation_id);
CREATE INDEX IF NOT EXISTS idx_actions_run_id ON zakops.actions(run_id);

CREATE INDEX IF NOT EXISTS idx_agent_events_correlation_id ON zakops.agent_events(correlation_id);
CREATE INDEX IF NOT EXISTS idx_agent_events_trace_id ON zakops.agent_events(trace_id);

-- ============================================================================
-- RECORD MIGRATION
-- ============================================================================

INSERT INTO zakops.schema_migrations (version, description)
VALUES ('001', 'Foundation tables for ZakOps Master Architecture')
ON CONFLICT (version) DO NOTHING;
