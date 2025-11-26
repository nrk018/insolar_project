-- Migration: Add last_notified_date column to workers table
-- This column tracks the date when a worker was last notified (email/SMS)
-- Prevents duplicate notifications on the same day

-- Add the column if it doesn't exist
ALTER TABLE workers 
ADD COLUMN IF NOT EXISTS last_notified_date DATE;

-- Add a comment to document the column
COMMENT ON COLUMN workers.last_notified_date IS 'Date when worker was last notified (prevents duplicate daily notifications)';

-- Create an index for faster lookups
CREATE INDEX IF NOT EXISTS idx_workers_last_notified_date 
ON workers(worker_id, last_notified_date);

