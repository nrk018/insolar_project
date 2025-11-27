import { useState, useEffect, useRef } from 'react';
import flaskAxios from '../utils/flaskAxios';
import axios from '../utils/axiosConfig';

const getTimeAgo = (date) => {
  // Ensure date is a proper Date object
  const detectionDate = date instanceof Date ? date : new Date(date);
  const now = new Date();
  
  // Calculate difference in milliseconds
  const diffMs = now - detectionDate;
  const seconds = Math.floor(diffMs / 1000);
  
  if (seconds < 0) return 'just now'; // Handle future dates
  if (seconds < 60) return `${seconds}s ago`;
  
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
};

const CameraMonitor = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [cameraSource, setCameraSource] = useState('rtsp'); // 'rtsp' or 'webcam'
  const [error, setError] = useState(null);
  const [recognizedWorkers, setRecognizedWorkers] = useState([]);
  const [flaskConnected, setFlaskConnected] = useState(false);
  const [recentDetections, setRecentDetections] = useState([]);
  const videoRef = useRef(null);

  const FLASK_URL = import.meta.env.VITE_FLASK_URL || 'http://localhost:5000';

  useEffect(() => {
    // Check camera status on mount
    checkCameraStatus();
    
    // Set up video stream placeholder (will update when camera starts)
    if (videoRef.current) {
      videoRef.current.src = `${FLASK_URL}/video_feed`;
    }

    // Fetch recent detections
    fetchRecentDetections();
    
    // Set up polling to refresh recent detections every 5 seconds
    const interval = setInterval(() => {
      fetchRecentDetections();
    }, 5000);

    return () => {
      // Cleanup: stop camera when component unmounts
      if (isRunning) {
        stopCamera();
      }
      clearInterval(interval);
    };
  }, []);

  const fetchRecentDetections = async () => {
    try {
      const response = await axios.get('/api/detections/recent?limit=50');
      if (response.data) {
        // Group by worker_name and keep only the latest detection per person
        const grouped = response.data.reduce((acc, detection) => {
          const name = detection.worker_name;
          if (!acc[name] || new Date(detection.detected_at) > new Date(acc[name].detected_at)) {
            acc[name] = detection;
          }
          return acc;
        }, {});
        
        // Convert to array and sort by detected_at (most recent first)
        const uniqueDetections = Object.values(grouped).sort((a, b) => 
          new Date(b.detected_at) - new Date(a.detected_at)
        );
        
        setRecentDetections(uniqueDetections);
        console.log(`[RECENT DETECTIONS] Fetched ${response.data.length} detections, showing ${uniqueDetections.length} unique`);
      }
    } catch (err) {
      console.error('Error fetching recent detections:', err);
      // If error, set empty array to avoid showing stale data
      if (err.response?.status === 500) {
        setRecentDetections([]);
      }
    }
  };

  const checkCameraStatus = async () => {
    try {
      // First check if Flask server is running
      const healthResponse = await flaskAxios.get(`${FLASK_URL}/api/health`, { timeout: 3000 });
      console.log('Flask server health:', healthResponse.data);
      setFlaskConnected(true);
      setError(null);
      
      // Then check camera status
      const response = await flaskAxios.get(`${FLASK_URL}/api/camera/status`);
      setIsRunning(response.data.running);
      setCameraSource(response.data.source || 'rtsp');
    } catch (err) {
      console.error('Error checking camera status:', err);
      setFlaskConnected(false);
      if (err.code === 'ECONNREFUSED' || err.message?.includes('Failed to fetch') || err.message?.includes('NetworkError')) {
        setError('Cannot connect to Flask server. Make sure videoServer.py is running on port 5000. Run: cd flaskServer && python videoServer.py');
      } else {
        setError(`Connection error: ${err.message}`);
      }
    }
  };

  const startCamera = async (source = 'rtsp') => {
    try {
      setError(null);
      const response = await flaskAxios.post(`${FLASK_URL}/api/camera/start`, {
        source: source
      });
      
      if (response.data.status === 'started') {
        setIsRunning(true);
        setCameraSource(source);
        // Refresh video feed with timestamp to force reload
        if (videoRef.current) {
          videoRef.current.src = `${FLASK_URL}/video_feed?t=${Date.now()}`;
        }
      } else {
        setError(response.data.message || 'Failed to start camera');
      }
    } catch (err) {
      const errorMessage = err.response?.data?.message || err.message || 'Failed to start camera. Make sure Flask server is running on port 5000.';
      setError(errorMessage);
      console.error('Error starting camera:', err);
    }
  };

  const stopCamera = async () => {
    try {
      await flaskAxios.post(`${FLASK_URL}/api/camera/stop`);
      setIsRunning(false);
    } catch (err) {
      console.error('Error stopping camera:', err);
    }
  };

  const switchCamera = async (newSource) => {
    try {
      setError(null);
      const response = await flaskAxios.post(`${FLASK_URL}/api/camera/switch`, {
        source: newSource
      });
      
      if (response.data.status === 'switched') {
        setCameraSource(newSource);
        // Refresh video feed
        if (videoRef.current) {
          videoRef.current.src = `${FLASK_URL}/video_feed?t=${Date.now()}`;
        }
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to switch camera');
      console.error('Error switching camera:', err);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Camera Monitor</h1>
        <p className="text-muted-foreground">Live face recognition and PPE detection using RTSP camera or laptop camera</p>
      </div>

      {/* Camera Controls */}
      <div className="rounded-lg border bg-card p-4 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => startCamera('rtsp')}
              disabled={isRunning && cameraSource === 'rtsp' || !flaskConnected}
              className={`px-4 py-2 rounded-md font-medium ${
                isRunning && cameraSource === 'rtsp'
                  ? 'bg-green-500 text-white cursor-not-allowed'
                  : !flaskConnected
                  ? 'bg-gray-400 text-white cursor-not-allowed'
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              {isRunning && cameraSource === 'rtsp' ? 'RTSP Running' : 'Start RTSP'}
            </button>

            <button
              onClick={() => startCamera('webcam')}
              disabled={isRunning && cameraSource === 'webcam' || !flaskConnected}
              className={`px-4 py-2 rounded-md font-medium ${
                isRunning && cameraSource === 'webcam'
                  ? 'bg-green-500 text-white cursor-not-allowed'
                  : !flaskConnected
                  ? 'bg-gray-400 text-white cursor-not-allowed'
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              {isRunning && cameraSource === 'webcam' ? 'Laptop Camera Running' : 'Start Laptop Camera'}
            </button>

            {isRunning && (
              <>
                <button
                  onClick={() => switchCamera(cameraSource === 'rtsp' ? 'webcam' : 'rtsp')}
                  className="px-4 py-2 rounded-md font-medium bg-yellow-500 text-white hover:bg-yellow-600"
                >
                  Switch to {cameraSource === 'rtsp' ? 'Laptop Camera' : 'RTSP'}
                </button>
                <button
                  onClick={stopCamera}
                  className="px-4 py-2 rounded-md font-medium bg-red-500 text-white hover:bg-red-600"
                >
                  Stop Camera
                </button>
              </>
            )}
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${flaskConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-muted-foreground">
                Flask: {flaskConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-green-500' : 'bg-gray-400'}`}></div>
              <span className="text-sm text-muted-foreground">
                Camera: {isRunning ? `Active (${cameraSource})` : 'Inactive'}
              </span>
            </div>
          </div>
        </div>

        {error && (
          <div className="rounded-md bg-red-50 border border-red-200 p-3">
            <p className="text-sm text-red-800 font-medium">Error:</p>
            <p className="text-sm text-red-700 mt-1">{error}</p>
            {!flaskConnected && (
              <div className="mt-2 text-xs text-red-600">
                <p>To start the Flask server:</p>
                <code className="block mt-1 p-2 bg-red-100 rounded">
                  cd InsolareSafetySystem/flaskServer<br />
                  python videoServer.py
                </code>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Video Feed */}
      <div className="rounded-lg border bg-card p-4">
        <h2 className="text-xl font-semibold mb-4">Live Feed</h2>
        <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: '16/9' }}>
          {isRunning ? (
            <img
              ref={videoRef}
              src={`${FLASK_URL}/video_feed`}
              alt="Camera Feed"
              className="w-full h-full object-contain"
              onError={(e) => {
                console.error('Video feed error');
                setError('Failed to load video feed. Make sure Flask server is running on port 5000.');
              }}
            />
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              <div className="text-center">
                <p className="text-lg mb-2">Camera not started</p>
                <p className="text-sm">Click "Start RTSP" or "Start Laptop Camera" to begin</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Recent Detections */}
      <div className="rounded-lg border bg-card p-4">
        <h2 className="text-xl font-semibold mb-4">Recent Detections</h2>
        {recentDetections.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No detections yet. Workers detected in the live feed will appear here.
          </p>
        ) : (
          <div className="space-y-2">
            {recentDetections.map((detection, idx) => {
              // Parse the detected_at timestamp (handles both ISO string and Date objects)
              // The ISO string from backend is in UTC, new Date() automatically converts to local time
              let detectedAt;
              try {
                detectedAt = detection.detected_at 
                  ? new Date(detection.detected_at) 
                  : new Date();
                
                // Validate the date
                if (isNaN(detectedAt.getTime())) {
                  console.error('Invalid date:', detection.detected_at);
                  return null;
                }
              } catch (e) {
                console.error('Error parsing date:', detection.detected_at, e);
                return null;
              }
              
              const timeAgo = getTimeAgo(detectedAt);
              
              // Format time in India/Delhi timezone (Asia/Kolkata - IST)
              const timeString = detectedAt.toLocaleTimeString('en-IN', { 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit',
                hour12: true,
                timeZone: 'Asia/Kolkata'
              });
              
              const dateString = detectedAt.toLocaleDateString('en-IN', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                timeZone: 'Asia/Kolkata'
              });
              
              const FLASK_URL = import.meta.env.VITE_FLASK_URL || 'http://localhost:5000';
              // Construct snapshot URL - snapshot_path is like "detection_snapshots/filename.jpg"
              const snapshotUrl = detection.snapshot_path 
                ? `${FLASK_URL}/${detection.snapshot_path}`
                : null;
              
              // Debug logging
              if (detection.snapshot_path) {
                console.log(`[RECENT DETECTIONS] Snapshot path: ${detection.snapshot_path}, URL: ${snapshotUrl}`);
              }
              
              return (
                <div
                  key={idx}
                  className="flex items-center gap-4 p-3 rounded-md border bg-muted/30 hover:bg-muted/50 transition-colors"
                >
                  {/* Snapshot Image */}
                  {snapshotUrl ? (
                    <div className="flex-shrink-0">
                      <img
                        src={snapshotUrl}
                        alt={`${detection.worker_name} detection`}
                        className="w-24 h-24 object-cover rounded border"
                        onError={(e) => {
                          console.error(`[SNAPSHOT ERROR] Failed to load image: ${snapshotUrl}`);
                          e.target.style.display = 'none';
                          // Show error indicator
                          const parent = e.target.parentElement;
                          if (parent) {
                            parent.innerHTML = '<div class="w-24 h-24 bg-red-100 rounded border flex items-center justify-center"><span class="text-xs text-red-600">Error</span></div>';
                          }
                        }}
                        onLoad={() => {
                          console.log(`[SNAPSHOT] Successfully loaded: ${snapshotUrl}`);
                        }}
                      />
                    </div>
                  ) : (
                    <div className="flex-shrink-0 w-24 h-24 bg-muted rounded border flex items-center justify-center">
                      <span className="text-xs text-muted-foreground">No image</span>
                    </div>
                  )}
                  
                  {/* Detection Info */}
                  <div className="flex-1 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-3 h-3 rounded-full ${
                        detection.ppe_compliant ? 'bg-green-500' : 'bg-red-500'
                      }`}></div>
                      <div>
                        <p className="font-medium">{detection.worker_name}</p>
                        <p className="text-xs text-muted-foreground">
                          Confidence: {(detection.confidence * 100).toFixed(1)}% | 
                          {detection.camera_source && ` ${detection.camera_source.toUpperCase()}`}
                        </p>
                        {/* PPE Items Status */}
                        {detection.ppe_items && (
                          <div className="flex gap-2 mt-1 text-xs">
                            {Object.entries(detection.ppe_items).map(([item, detected]) => (
                              <span
                                key={item}
                                className={`px-1.5 py-0.5 rounded ${
                                  detected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                                }`}
                              >
                                {item.charAt(0).toUpperCase() + item.slice(1)}: {detected ? '✓' : '✗'}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">{timeAgo}</p>
                      <p className="text-xs text-muted-foreground">
                        {timeString}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {dateString}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraMonitor;

