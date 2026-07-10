CREATE TABLE users (
    id SERIAL PRIMARY KEY,  
    employee_id VARCHAR(50) UNIQUE NOT NULL,      
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL, 
    phone VARCHAR(20) NOT NULL,
    department VARCHAR(100) NOT NULL,
    designation VARCHAR(100) NOT NULL,
    profilePhoto TEXT,            
    password TEXT NOT NULL,   
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
);

CREATE TABLE attendance (
    id SERIAL PRIMARY KEY,  
    employee_id VARCHAR(50) NOT NULL,
    date DATE NOT NULL,
    check_in_time TIME,
    check_out_time TIME,
    status VARCHAR(10) NOT NULL CHECK (status IN ('Present', 'Absent', 'Late')),
    ppe_compliant BOOLEAN DEFAULT FALSE,
    ppe_items_detected JSONB DEFAULT '{}',
    ppe_detection_confidence DECIMAL(5,2) DEFAULT 0.00,
    FOREIGN KEY (employee_id) REFERENCES users(employee_id) ON DELETE CASCADE
);

-- Recent detections from live camera feed (one row per worker, updated on each sighting)
CREATE TABLE IF NOT EXISTS detection_events (
    id SERIAL PRIMARY KEY,
    worker_id TEXT,
    worker_name TEXT NOT NULL UNIQUE,
    confidence FLOAT,
    ppe_compliant BOOLEAN,
    ppe_items JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    camera_source TEXT,
    snapshot_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_detection_events_worker_id ON detection_events(worker_id);
CREATE INDEX IF NOT EXISTS idx_detection_events_detected_at ON detection_events(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_detection_events_worker_name ON detection_events(worker_name);
