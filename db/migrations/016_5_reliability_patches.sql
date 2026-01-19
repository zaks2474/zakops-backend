-- Migration: 016_5_reliability_patches
-- Description: Add reliability columns for idempotency, pagination, and audit
-- Phase: 16.5 - Reliability Patches

-- 1. Add idempotency_key to deal_events for tracking duplicate transitions
ALTER TABLE zakops.deal_events
ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(64);

-- Index for idempotency lookup (recent 24h)
CREATE INDEX IF NOT EXISTS idx_deal_events_idempotency
ON zakops.deal_events (deal_id, idempotency_key, created_at)
WHERE idempotency_key IS NOT NULL;

-- 2. Add sequence_number to deal_events for ordering
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'zakops'
        AND table_name = 'deal_events'
        AND column_name = 'sequence_number'
    ) THEN
        -- Add sequence column
        ALTER TABLE zakops.deal_events ADD COLUMN sequence_number BIGINT;

        -- Create sequence
        CREATE SEQUENCE IF NOT EXISTS zakops.deal_events_sequence_seq START 1;

        -- Backfill existing events
        UPDATE zakops.deal_events SET sequence_number = nextval('zakops.deal_events_sequence_seq')
        WHERE sequence_number IS NULL;

        -- Set default for future rows
        ALTER TABLE zakops.deal_events
        ALTER COLUMN sequence_number SET DEFAULT nextval('zakops.deal_events_sequence_seq');
    END IF;
END $$;

-- 3. Add actor columns to deal_events for audit
ALTER TABLE zakops.deal_events
ADD COLUMN IF NOT EXISTS actor_id VARCHAR(64);

ALTER TABLE zakops.deal_events
ADD COLUMN IF NOT EXISTS actor_type VARCHAR(32) DEFAULT 'system';

-- 4. Index for event ordering by sequence
CREATE INDEX IF NOT EXISTS idx_deal_events_sequence
ON zakops.deal_events (sequence_number);

-- Index for correlation + sequence (for SSE replay)
CREATE INDEX IF NOT EXISTS idx_deal_events_deal_sequence
ON zakops.deal_events (deal_id, sequence_number);

-- 5. Add composite index for cursor pagination on deals
CREATE INDEX IF NOT EXISTS idx_deals_cursor_pagination
ON zakops.deals (updated_at DESC, deal_id DESC);

-- 6. Add composite index for cursor pagination on actions
CREATE INDEX IF NOT EXISTS idx_actions_cursor_pagination
ON zakops.actions (created_at DESC, action_id DESC);

-- 7. Add idempotency_key to actions table for action idempotency
ALTER TABLE zakops.actions
ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(64);

CREATE INDEX IF NOT EXISTS idx_actions_idempotency
ON zakops.actions (idempotency_key)
WHERE idempotency_key IS NOT NULL;

-- Verify migration
DO $$
BEGIN
    -- Check idempotency_key exists in deal_events
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'zakops'
        AND table_name = 'deal_events'
        AND column_name = 'idempotency_key'
    ) THEN
        RAISE EXCEPTION 'Migration failed: idempotency_key not added to deal_events';
    END IF;

    -- Check actor_type exists in deal_events
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'zakops'
        AND table_name = 'deal_events'
        AND column_name = 'actor_type'
    ) THEN
        RAISE EXCEPTION 'Migration failed: actor_type not added to deal_events';
    END IF;

    RAISE NOTICE 'Migration 016_5 completed successfully';
END $$;
