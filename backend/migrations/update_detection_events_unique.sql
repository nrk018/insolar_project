-- Migration: Add UNIQUE constraint to worker_name in detection_events table
-- This ensures only one entry per person, updating timestamp on new detections

-- Add UNIQUE constraint to worker_name
-- Note: If you have duplicate entries, you may need to clean them up first
ALTER TABLE detection_events 
ADD CONSTRAINT detection_events_worker_name_unique UNIQUE (worker_name);

-- If the above fails due to duplicates, run this first to keep only the latest entry per person:
-- DELETE FROM detection_events 
-- WHERE id NOT IN (
--     SELECT DISTINCT ON (worker_name) id 
--     FROM detection_events 
--     ORDER BY worker_name, detected_at DESC
-- );

