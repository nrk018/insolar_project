-- Migration: Add snapshot_path column to detection_events table
-- This column stores the path to the snapshot image with annotations (face + PPE boxes)

-- Add snapshot_path column if it doesn't exist
ALTER TABLE detection_events 
ADD COLUMN IF NOT EXISTS snapshot_path TEXT;

-- Add a comment to document the column
COMMENT ON COLUMN detection_events.snapshot_path IS 'Path to snapshot image with face recognition and PPE detection annotations, saved when person is detected';

