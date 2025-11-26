-- Migration: Add detection_events table
-- This table stores real-time detection events from the camera feed
-- Shows when employees were recognized with timestamps

-- Create the table if it doesn't exist
CREATE TABLE IF NOT EXISTS detection_events (
    id SERIAL PRIMARY KEY,
    worker_id TEXT,
    worker_name TEXT NOT NULL UNIQUE, -- UNIQUE: Only one entry per person, updates timestamp on new detection
    confidence FLOAT,
    ppe_compliant BOOLEAN,
    ppe_items JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    camera_source TEXT -- 'rtsp' or 'webcam'
);

-- Create indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_detection_events_worker_id ON detection_events(worker_id);
CREATE INDEX IF NOT EXISTS idx_detection_events_detected_at ON detection_events(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_detection_events_worker_name ON detection_events(worker_name);

-- Add a comment to document the table
COMMENT ON TABLE detection_events IS 'Stores real-time face recognition detection events from camera feed';

