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
  const [sendingEmails, setSendingEmails] = useState({}); // Track which emails are being sent
  const [previewDetection, setPreviewDetection] = useState(null);
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

    // Stop camera on page unload (logout, close tab, etc.)
    const handleBeforeUnload = () => {
      // Use sendBeacon for more reliable cleanup on page unload
      try {
        navigator.sendBeacon(`${FLASK_URL}/api/camera/stop`, '');
      } catch (e) {
        // Fallback if sendBeacon fails
        console.warn('Could not send camera stop beacon:', e);
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      // Don't stop camera on component unmount - let it run across pages
      // Only stop on logout or page close
      clearInterval(interval);
      window.removeEventListener('beforeunload', handleBeforeUnload);
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

  const handleSendEmail = async (detection) => {
    const detectionKey = `${detection.worker_name}_${detection.detected_at}`;
    
    // Prevent multiple clicks
    if (sendingEmails[detectionKey]) {
      return;
    }

    setSendingEmails(prev => ({ ...prev, [detectionKey]: true }));

    try {
      const response = await axios.post('/api/ppe/send-email', {
        worker_name: detection.worker_name,
        ppe_items: detection.ppe_items || {}
      });

      if (response.data.email_sent) {
        alert(`Email sent successfully to ${detection.worker_name}`);
      } else {
        alert(`Failed to send email: ${response.data.error || 'Unknown error'}`);
      }
    } catch (err) {
      console.error('Error sending email:', err);
      alert(`Error sending email: ${err.response?.data?.error || err.message || 'Unknown error'}`);
    } finally {
      setSendingEmails(prev => {
        const newState = { ...prev };
        delete newState[detectionKey];
        return newState;
      });
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

      {/* Main Content: Camera on Left, Recent Detections on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video Feed - Left Side */}
        <div className="rounded-lg border bg-card p-4">
          <h2 className="text-xl font-semibold mb-4">Live Feed</h2>
          <div className="relative bg-black rounded-lg overflow-hidden" style={{ aspectRatio: '16/9', maxHeight: '500px' }}>
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

        {/* Recent Detections - Right Side */}
        <div className="rounded-lg border bg-card p-4 flex flex-col">
          <h2 className="text-xl font-semibold mb-4">Recent Detections</h2>
          {recentDetections.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No detections yet. Workers detected in the live feed will appear here.
            </p>
          ) : (
            <div className="flex-1 overflow-y-auto space-y-2 pr-2" style={{ maxHeight: '500px' }}>
              {recentDetections.map((detection, idx) => {
              // Parse the detected_at timestamp
              // Backend sends UTC ISO string - convert to local time (IST/Asia/Kolkata)
              let detectedAt;
              try {
                if (detection.detected_at) {
                  const timestampStr = detection.detected_at;
                  
                  // Backend always sends UTC ISO string (ends with 'Z')
                  // Parse it explicitly as UTC
                  if (typeof timestampStr === 'string') {
                    // Ensure it's treated as UTC - add 'Z' if missing
                    const utcString = timestampStr.endsWith('Z') ? timestampStr : timestampStr + 'Z';
                    detectedAt = new Date(utcString);
                  } else if (timestampStr instanceof Date) {
                    detectedAt = timestampStr;
                  } else {
                    detectedAt = new Date(timestampStr);
                  }
                  
                  // Validate the date
                  if (isNaN(detectedAt.getTime())) {
                    console.error('Invalid date:', detection.detected_at);
                    return null;
                  }
                  
                } else {
                  detectedAt = new Date();
                }
              } catch (e) {
                console.error('Error parsing date:', detection.detected_at, e);
                return null;
              }
              
              // Calculate time ago (using the Date object directly)
              const timeAgo = getTimeAgo(detectedAt);
              
              // Format time in IST (Asia/Kolkata) - India Standard Time
              // The Date object is in UTC, so toLocaleString with timeZone will convert correctly
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
              
              // Full date-time string for display in IST
              // Format: "28 Nov 2025, 01:25:48 PM" (IST)
              const fullDateTimeString = detectedAt.toLocaleString('en-IN', {
                month: 'short',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: true,
                timeZone: 'Asia/Kolkata'
              });
              
              // Debug: Log the conversion for troubleshooting
              if (idx === 0) { // Only log for first item to avoid spam
                console.log('[TIME CONVERSION] Original timestamp:', detection.detected_at);
                console.log('[TIME CONVERSION] Parsed Date (UTC):', detectedAt.toISOString());
                console.log('[TIME CONVERSION] Current local time:', new Date().toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' }));
                console.log('[TIME CONVERSION] Display time (IST):', fullDateTimeString);
              }
              
              const FLASK_URL = import.meta.env.VITE_FLASK_URL || 'http://localhost:5000';
              // Construct snapshot URL - snapshot_path is like "detection_snapshots/filename.jpg"
              const snapshotUrl = detection.snapshot_path 
                ? `${FLASK_URL}/${detection.snapshot_path}`
                : null;
              
              // Debug logging
              if (detection.snapshot_path) {
                console.log(`[RECENT DETECTIONS] Snapshot path: ${detection.snapshot_path}, URL: ${snapshotUrl}`);
              }
              
              const detectionKey = `${detection.worker_name}_${detection.detected_at}`;
              const isSendingEmail = sendingEmails[detectionKey];

              return (
                <div
                  key={idx}
                  className="flex items-center gap-3 p-3 rounded-md border bg-muted/30 hover:bg-muted/50 transition-colors"
                >
                  {/* Snapshot Image */}
                  {snapshotUrl ? (
                    <div className="flex-shrink-0 cursor-pointer" onClick={() => setPreviewDetection(detection)}>
                      <img
                        src={snapshotUrl}
                        alt={`${detection.worker_name} detection`}
                        className="w-20 h-20 object-cover rounded border hover:opacity-80 transition-opacity"
                        onError={(e) => {
                          console.error(`[SNAPSHOT ERROR] Failed to load image: ${snapshotUrl}`);
                          e.target.style.display = 'none';
                          // Show error indicator
                          const parent = e.target.parentElement;
                          if (parent) {
                            parent.innerHTML = '<div class="w-20 h-20 bg-red-100 rounded border flex items-center justify-center"><span class="text-xs text-red-600">Error</span></div>';
                          }
                        }}
                        onLoad={() => {
                          console.log(`[SNAPSHOT] Successfully loaded: ${snapshotUrl}`);
                        }}
                      />
                    </div>
                  ) : (
                    <div className="flex-shrink-0 w-20 h-20 bg-muted rounded border flex items-center justify-center">
                      <span className="text-xs text-muted-foreground">No image</span>
                    </div>
                  )}
                  
                  {/* Detection Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                        detection.ppe_compliant ? 'bg-green-500' : 'bg-red-500'
                      }`}></div>
                      <p className="font-medium text-sm truncate">{detection.worker_name}</p>
                    </div>
                    <p className="text-xs text-muted-foreground mb-1">
                      {(detection.confidence * 100).toFixed(1)}% | {timeAgo}
                    </p>
                    <p className="text-xs text-muted-foreground mb-1">
                      Recognized at: {fullDateTimeString}
                    </p>
                    {/* PPE Items Status */}
                    {detection.ppe_items && (
                      <div className="flex flex-wrap gap-1 mb-2 text-xs">
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
                    {/* Send Email Button */}
                    <button
                      onClick={() => handleSendEmail(detection)}
                      disabled={isSendingEmail}
                      className={`text-xs px-3 py-1.5 rounded-md font-medium transition-colors ${
                        isSendingEmail
                          ? 'bg-gray-400 text-white cursor-not-allowed'
                          : 'bg-blue-500 text-white hover:bg-blue-600'
                      }`}
                    >
                      {isSendingEmail ? 'Sending...' : 'Send Email'}
                    </button>
                  </div>
                </div>
              );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Image Preview Modal */}
      {previewDetection && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ backdropFilter: 'blur(8px)', backgroundColor: 'rgba(0, 0, 0, 0.5)' }}
          onClick={() => setPreviewDetection(null)}
        >
          <div 
            className="bg-white rounded-lg shadow-xl w-full max-w-6xl max-h-[95vh] overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header with Close Button */}
            <div className="flex items-center justify-between p-4 border-b">
              <h3 className="text-xl font-semibold">Detection Preview</h3>
              <button
                onClick={() => setPreviewDetection(null)}
                className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                aria-label="Close"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Content: Image on Left, Details on Right */}
            <div className="flex flex-col lg:flex-row flex-1 overflow-hidden">
              {/* Image Section - Left */}
              <div className="lg:w-3/5 p-8 flex items-center justify-center bg-gray-50">
                {previewDetection.snapshot_path ? (
                  <img
                    src={`${FLASK_URL}/${previewDetection.snapshot_path}`}
                    alt={previewDetection.worker_name}
                    className="w-full h-full max-h-[85vh] rounded-lg shadow-lg object-contain"
                    onError={(e) => {
                      e.target.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="300"%3E%3Crect fill="%23ddd" width="400" height="300"/%3E%3Ctext fill="%23999" font-family="sans-serif" font-size="20" dy="10.5" font-weight="bold" x="50%25" y="50%25" text-anchor="middle"%3EImage not available%3C/text%3E%3C/svg%3E';
                    }}
                  />
                ) : (
                  <div className="w-full h-64 flex items-center justify-center bg-gray-200 rounded-lg">
                    <p className="text-gray-500">No image available</p>
                  </div>
                )}
              </div>

              {/* Details Section - Right */}
              <div className="lg:w-2/5 p-8 overflow-y-auto">
                <div className="space-y-6">
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-2">Worker Name</h4>
                    <p className="text-xl font-semibold">{previewDetection.worker_name || "—"}</p>
                  </div>

                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-2">Detection Status</h4>
                    {previewDetection.ppe_compliant ? (
                      <span className="inline-flex items-center px-4 py-2 rounded-full text-base font-medium bg-green-100 text-green-800">
                        Compliant
                      </span>
                    ) : (
                      <span className="inline-flex items-center px-4 py-2 rounded-full text-base font-medium bg-red-100 text-red-800">
                        Violation
                      </span>
                    )}
                  </div>

                  {previewDetection.confidence && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-500 mb-2">Recognition Confidence</h4>
                      <p className="text-lg">{(previewDetection.confidence * 100).toFixed(1)}%</p>
                    </div>
                  )}

                  {(() => {
                    if (!previewDetection.detected_at) return null;
                    try {
                      const timestampStr = previewDetection.detected_at;
                      const utcString = timestampStr.endsWith('Z') ? timestampStr : timestampStr + 'Z';
                      const detectedAt = new Date(utcString);
                      if (!isNaN(detectedAt.getTime())) {
                        const timeString = detectedAt.toLocaleTimeString('en-IN', { 
                          hour: '2-digit', 
                          minute: '2-digit',
                          second: '2-digit',
                          hour12: true,
                          timeZone: 'Asia/Kolkata'
                        });
                        const dateString = detectedAt.toLocaleDateString('en-IN', {
                          weekday: 'long',
                          year: 'numeric',
                          month: 'long',
                          day: 'numeric',
                          timeZone: 'Asia/Kolkata'
                        });
                        return (
                          <div>
                            <h4 className="text-sm font-medium text-gray-500 mb-2">Detected At</h4>
                            <p className="text-lg">{dateString}</p>
                            <p className="text-lg text-gray-600">{timeString} IST</p>
                            <p className="text-sm text-gray-500 mt-1">{getTimeAgo(detectedAt)}</p>
                          </div>
                        );
                      }
                    } catch (e) {
                      return null;
                    }
                    return null;
                  })()}

                  {previewDetection.ppe_items && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-500 mb-3">PPE Items</h4>
                      <div className="grid grid-cols-2 gap-3">
                        <div className={`p-3 rounded-lg ${previewDetection.ppe_items.helmet ? 'bg-green-50 text-green-800 border border-green-200' : 'bg-red-50 text-red-800 border border-red-200'}`}>
                          <span className="font-semibold">Helmet:</span> {previewDetection.ppe_items.helmet ? '✓ Detected' : '✗ Missing'}
                        </div>
                        <div className={`p-3 rounded-lg ${previewDetection.ppe_items.gloves ? 'bg-green-50 text-green-800 border border-green-200' : 'bg-red-50 text-red-800 border border-red-200'}`}>
                          <span className="font-semibold">Gloves:</span> {previewDetection.ppe_items.gloves ? '✓ Detected' : '✗ Missing'}
                        </div>
                        <div className={`p-3 rounded-lg ${previewDetection.ppe_items.boots ? 'bg-green-50 text-green-800 border border-green-200' : 'bg-red-50 text-red-800 border border-red-200'}`}>
                          <span className="font-semibold">Boots:</span> {previewDetection.ppe_items.boots ? '✓ Detected' : '✗ Missing'}
                        </div>
                        <div className={`p-3 rounded-lg ${previewDetection.ppe_items.jacket ? 'bg-green-50 text-green-800 border border-green-200' : 'bg-red-50 text-red-800 border border-red-200'}`}>
                          <span className="font-semibold">Jacket:</span> {previewDetection.ppe_items.jacket ? '✓ Detected' : '✗ Missing'}
                        </div>
                      </div>
                    </div>
                  )}

                  {previewDetection.camera_source && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-500 mb-2">Camera Source</h4>
                      <p className="text-lg">{previewDetection.camera_source === 'rtsp' ? 'RTSP Camera' : 'Webcam'}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CameraMonitor;

