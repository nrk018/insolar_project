-- Migration script to add PPE detection columns to attendance table
-- Run this in your Supabase SQL editor if the table already exists

ALTER TABLE attendance 
ADD COLUMN IF NOT EXISTS ppe_compliant BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS ppe_items_detected JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS ppe_detection_confidence DECIMAL(5,2) DEFAULT 0.00;

-- Add comment for documentation
COMMENT ON COLUMN attendance.ppe_compliant IS 'Whether all required PPE items were detected';
COMMENT ON COLUMN attendance.ppe_items_detected IS 'JSON object with detected PPE items: {"helmet": true, "gloves": true, "boots": true, "jacket": true}';
COMMENT ON COLUMN attendance.ppe_detection_confidence IS 'Average confidence score of PPE detections (0.00 to 1.00)';

