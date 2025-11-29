import { useEffect, useState } from "react";
import axios from '../utils/axiosConfig';
import flaskAxios from '../utils/flaskAxios';
import { stopCamera } from '../utils/cameraUtils';

const getTimeAgo = (date) => {
  const detectionDate = date instanceof Date ? date : new Date(date);
  const now = new Date();
  const diffMs = now - detectionDate;
  const seconds = Math.floor(diffMs / 1000);
  
  if (seconds < 0) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
};

const Dashboard = () => {
  const [recentDetections, setRecentDetections] = useState([]);
  const [todayViolations, setTodayViolations] = useState([]);
  const [violationStats, setViolationStats] = useState({
    total: 0,
    uniqueWorkers: 0,
    violationsByDay: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [sendingEmails, setSendingEmails] = useState({});
  const [isRunning, setIsRunning] = useState(false);
  const [cameraSource, setCameraSource] = useState('rtsp');
  const [flaskConnected, setFlaskConnected] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [previewDetection, setPreviewDetection] = useState(null);

  const FLASK_URL = import.meta.env.VITE_FLASK_URL || 'http://localhost:5000';

  useEffect(() => {
    fetchDashboardData();
    checkCameraStatus();
    
    // Refresh data every 10 seconds
    const dataInterval = setInterval(() => {
      fetchDashboardData();
    }, 10000);

    // Check camera status every 5 seconds
    const cameraInterval = setInterval(() => {
      checkCameraStatus();
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
      clearInterval(dataInterval);
      clearInterval(cameraInterval);
      // Don't stop camera on component unmount - let it run across pages
      // Only stop on logout or page close
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  const fetchDashboardData = async () => {
    try {
      setError(null);
      
      // Fetch recent detections
      const detectionsRes = await axios.get('/api/detections/recent?limit=20');
      if (detectionsRes.data) {
        const grouped = detectionsRes.data.reduce((acc, detection) => {
          const name = detection.worker_name;
          if (!acc[name] || new Date(detection.detected_at) > new Date(acc[name].detected_at)) {
            acc[name] = detection;
          }
          return acc;
        }, {});
        const uniqueDetections = Object.values(grouped).sort((a, b) => 
          new Date(b.detected_at) - new Date(a.detected_at)
        );
        setRecentDetections(uniqueDetections);
      }

      // Fetch today's violations
      const today = new Date().toISOString().split('T')[0];
      const violationsRes = await axios.get(`/api/ppe/defaulters?date=${today}`);
      const violations = Array.isArray(violationsRes.data) ? violationsRes.data : [];
      setTodayViolations(violations);

      // Fetch violation stats for graph (last 7 days)
      await fetchViolationStatsWithViolations(violations);
      
    } catch (e) {
      console.error('Error fetching dashboard data:', e);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchViolationStatsWithViolations = async (todayViolationsData) => {
    try {
      const stats = [];
      const today = new Date();
      
      for (let i = 6; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        const dateStr = date.toISOString().split('T')[0];
        
        try {
          const res = await axios.get(`/api/ppe/defaulters?date=${dateStr}`);
          const violations = Array.isArray(res.data) ? res.data : [];
          stats.push({
            date: dateStr,
            day: date.toLocaleDateString('en-IN', { weekday: 'short' }),
            count: violations.length,
            uniqueWorkers: new Set(violations.map(v => v.worker_id)).size
          });
        } catch (err) {
          stats.push({
            date: dateStr,
            day: date.toLocaleDateString('en-IN', { weekday: 'short' }),
            count: 0,
            uniqueWorkers: 0
          });
        }
      }
      
      const totalViolations = stats.reduce((sum, s) => sum + s.count, 0);
      
      setViolationStats({
        total: totalViolations,
        uniqueWorkers: todayViolationsData.length > 0 ? new Set(todayViolationsData.map(v => v.worker_id)).size : 0,
        violationsByDay: stats
      });
    } catch (err) {
      console.error('Error fetching violation stats:', err);
    }
  };

  const checkCameraStatus = async () => {
    try {
      const healthResponse = await flaskAxios.get(`${FLASK_URL}/api/health`, { timeout: 3000 });
      setFlaskConnected(true);
      setCameraError(null);
      
      const response = await flaskAxios.get(`${FLASK_URL}/api/camera/status`);
      setIsRunning(response.data.running);
      setCameraSource(response.data.source || 'rtsp');
    } catch (err) {
      setFlaskConnected(false);
      if (err.code === 'ECONNREFUSED' || err.message?.includes('Failed to fetch') || err.message?.includes('NetworkError')) {
        setCameraError('Flask server not connected');
      } else {
        setCameraError(`Connection error: ${err.message}`);
      }
    }
  };

  const startCamera = async (source = 'rtsp') => {
    try {
      setCameraError(null);
      const response = await flaskAxios.post(`${FLASK_URL}/api/camera/start`, {
        source: source
      });
      
      if (response.data.status === 'started') {
        setIsRunning(true);
        setCameraSource(source);
      } else {
        setCameraError(response.data.message || 'Failed to start camera');
      }
    } catch (err) {
      const errorMessage = err.response?.data?.message || err.message || 'Failed to start camera';
      setCameraError(errorMessage);
      console.error('Error starting camera:', err);
    }
  };

  const stopCamera = async () => {
    try {
      await flaskAxios.post(`${FLASK_URL}/api/camera/stop`);
      setIsRunning(false);
      setCameraError(null);
    } catch (err) {
      console.error('Error stopping camera:', err);
      setCameraError('Failed to stop camera');
    }
  };

  const switchCamera = async (newSource) => {
    try {
      setCameraError(null);
      const response = await flaskAxios.post(`${FLASK_URL}/api/camera/switch`, {
        source: newSource
      });
      
      if (response.data.status === 'switched') {
        setCameraSource(newSource);
      }
    } catch (err) {
      setCameraError(err.response?.data?.message || 'Failed to switch camera');
      console.error('Error switching camera:', err);
    }
  };

  const handleSendEmail = async (detection) => {
    const detectionKey = `${detection.worker_name}_${detection.detected_at}`;
    setSendingEmails(prev => ({ ...prev, [detectionKey]: true }));

    try {
      const ppeItems = {
        helmet: detection.ppe_items?.helmet === true,
        gloves: detection.ppe_items?.gloves === true,
        boots: detection.ppe_items?.boots === true,
        jacket: detection.ppe_items?.jacket === true,
      };

      const response = await axios.post('/api/ppe/send-email', {
        worker_name: detection.worker_name,
        ppe_items: ppeItems,
      });

      if (response.data.email_sent) {
        alert(`Email sent successfully to ${detection.worker_name}`);
      } else {
        throw new Error(response.data.error || 'Failed to send email');
      }
    } catch (err) {
      console.error('Error sending email:', err);
      alert(`Failed to send email: ${err.response?.data?.error || err.message}`);
    } finally {
      setSendingEmails(prev => ({ ...prev, [detectionKey]: false }));
    }
  };

  const maxViolations = Math.max(...violationStats.violationsByDay.map(s => s.count), 1);

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Safety Dashboard</h1>
        <p className="text-muted-foreground">
          Live overview of PPE compliance and recent violations
        </p>
      </div>

      {/* Camera Status and Controls */}
      <div className="rounded-lg border bg-card p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">Camera Status</h2>
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
                Camera: {isRunning ? `Active (${cameraSource === 'rtsp' ? 'RTSP' : 'Webcam'})` : 'Inactive'}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 flex-wrap">
          <button
            onClick={() => startCamera('rtsp')}
            disabled={isRunning && cameraSource === 'rtsp' || !flaskConnected}
            className={`px-4 py-2 rounded-md font-medium text-sm ${
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
            className={`px-4 py-2 rounded-md font-medium text-sm ${
              isRunning && cameraSource === 'webcam'
                ? 'bg-green-500 text-white cursor-not-allowed'
                : !flaskConnected
                ? 'bg-gray-400 text-white cursor-not-allowed'
                : 'bg-blue-500 text-white hover:bg-blue-600'
            }`}
          >
            {isRunning && cameraSource === 'webcam' ? 'Webcam Running' : 'Start Webcam'}
          </button>

          {isRunning && (
            <>
              <button
                onClick={() => switchCamera(cameraSource === 'rtsp' ? 'webcam' : 'rtsp')}
                className="px-4 py-2 rounded-md font-medium text-sm bg-yellow-500 text-white hover:bg-yellow-600"
              >
                Switch to {cameraSource === 'rtsp' ? 'Webcam' : 'RTSP'}
              </button>
              <button
                onClick={stopCamera}
                className="px-4 py-2 rounded-md font-medium text-sm bg-red-500 text-white hover:bg-red-600"
              >
                Stop Camera
              </button>
            </>
          )}
        </div>

        {cameraError && (
          <div className="mt-4 rounded-md bg-red-50 border border-red-200 p-3">
            <p className="text-sm text-red-800 font-medium">Camera Error:</p>
            <p className="text-sm text-red-700 mt-1">{cameraError}</p>
          </div>
        )}
      </div>

      {loading ? (
        <div className="rounded-lg border bg-card p-12 text-center">
          <p className="text-muted-foreground">Loading dashboard data...</p>
        </div>
      ) : error ? (
        <div className="rounded-lg border border-destructive bg-destructive/10 p-4">
          <p className="text-destructive">{error}</p>
        </div>
      ) : (
        <>
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-lg border bg-card p-6 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Today's Violations</p>
                  <p className="text-3xl font-bold mt-2">{todayViolations.length}</p>
                </div>
                <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center">
                  <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="rounded-lg border bg-card p-6 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Unique Workers</p>
                  <p className="text-3xl font-bold mt-2">
                    {new Set(todayViolations.map(v => v.worker_id)).size}
                  </p>
                </div>
                <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center">
                  <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                </div>
              </div>
            </div>

            <div className="rounded-lg border bg-card p-6 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Recent Detections</p>
                  <p className="text-3xl font-bold mt-2">{recentDetections.length}</p>
                </div>
                <div className="w-12 h-12 rounded-full bg-green-100 flex items-center justify-center">
                  <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content - Single Column */}
          <div className="space-y-6">
            {/* Today's Violations Box */}
            <div className="rounded-lg border bg-card p-6 shadow-sm">
              <h2 className="text-xl font-semibold mb-4">Today's Violations</h2>
              {todayViolations.length === 0 ? (
                <p className="text-sm text-muted-foreground">No violations today. Great job! 🎉</p>
              ) : (
                <div className="space-y-2">
                  <p className="text-lg font-medium text-red-600 mb-3">
                    {todayViolations.length} violation{todayViolations.length !== 1 ? 's' : ''} from {new Set(todayViolations.map(v => v.worker_id)).size} worker{new Set(todayViolations.map(v => v.worker_id)).size !== 1 ? 's' : ''} today
                  </p>
                  <div className="max-h-48 overflow-y-auto space-y-2">
                    {todayViolations.map((v, idx) => {
                      const violationItems = [];
                      if (v.helmet_status === "No") violationItems.push("Helmet");
                      if (v.gloves_status === "No") violationItems.push("Gloves");
                      if (v.vests_status === "No") violationItems.push("Jacket");
                      if (v.boots_status === "No") violationItems.push("Boots");
                      
                      return (
                        <div key={idx} className="flex items-center justify-between p-3 rounded-md bg-muted/30">
                          <div>
                            <p className="font-medium">{v.name || v.worker_id}</p>
                            <p className="text-sm text-muted-foreground">
                              Missing: {violationItems.length > 0 ? violationItems.join(", ") : "Unknown"}
                            </p>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-semibold text-red-600">Streak: {v.streak || 0}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
            {/* Violations Graph */}
            <div className="rounded-lg border bg-card p-6 shadow-sm">
              <h2 className="text-xl font-semibold mb-4">Violations Trend (Last 7 Days)</h2>
              {violationStats.violationsByDay.length === 0 ? (
                <p className="text-sm text-muted-foreground">No data available</p>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-end justify-between gap-2 h-64">
                    {violationStats.violationsByDay.map((stat, idx) => (
                      <div key={idx} className="flex-1 flex flex-col items-center gap-2">
                        <div className="w-full flex flex-col items-center justify-end" style={{ height: '200px' }}>
                          <div
                            className="w-full bg-red-500 rounded-t transition-all hover:bg-red-600"
                            style={{
                              height: `${(stat.count / maxViolations) * 100}%`,
                              minHeight: stat.count > 0 ? '4px' : '0',
                            }}
                            title={`${stat.count} violations on ${stat.day}`}
                          />
                        </div>
                        <div className="text-xs text-muted-foreground text-center mt-2">
                          <div className="font-medium">{stat.day}</div>
                          <div className="text-red-600 font-semibold">{stat.count}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Recent Detections */}
            <div className="rounded-lg border bg-card p-6 shadow-sm flex flex-col">
              <h2 className="text-xl font-semibold mb-4">Recent Detections</h2>
              {recentDetections.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No recent detections. Workers detected in the live feed will appear here.
                </p>
              ) : (
                <div className="flex-1 overflow-y-auto space-y-3 pr-2" style={{ maxHeight: '500px' }}>
                  {recentDetections.map((detection, idx) => {
                    let detectedAt;
                    try {
                      if (detection.detected_at) {
                        const timestampStr = detection.detected_at;
                        const utcString = timestampStr.endsWith('Z') ? timestampStr : timestampStr + 'Z';
                        detectedAt = new Date(utcString);
                      } else {
                        detectedAt = new Date();
                      }
                      
                      if (isNaN(detectedAt.getTime())) {
                        return null;
                      }
                    } catch (e) {
                      return null;
                    }
                    
                    const timeAgo = getTimeAgo(detectedAt);
                    const timeString = detectedAt.toLocaleTimeString('en-IN', { 
                      hour: '2-digit', 
                      minute: '2-digit',
                      timeZone: 'Asia/Kolkata'
                    });
                    
                    const FLASK_URL = import.meta.env.VITE_FLASK_URL || 'http://localhost:5000';
                    const snapshotUrl = detection.snapshot_path 
                      ? `${FLASK_URL}/${detection.snapshot_path}`
                      : null;
                    
                    const detectionKey = `${detection.worker_name}_${detection.detected_at}`;
                    const isSendingEmail = sendingEmails[detectionKey];
                    const isCompliant = detection.ppe_compliant === true;

                    return (
                      <div
                        key={idx}
                        className="flex items-center gap-3 p-3 rounded-md border bg-muted/30 hover:bg-muted/50 transition-colors"
                      >
                        {snapshotUrl ? (
                          <div className="flex-shrink-0 cursor-pointer" onClick={() => setPreviewDetection(detection)}>
                            <img
                              src={snapshotUrl}
                              alt={detection.worker_name}
                              className="w-16 h-16 rounded-md object-cover hover:opacity-80 transition-opacity"
                              onError={(e) => {
                                e.target.style.display = 'none';
                              }}
                            />
                          </div>
                        ) : (
                          <div className="w-16 h-16 rounded-md bg-muted flex items-center justify-center">
                            <svg className="w-8 h-8 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                            </svg>
                          </div>
                        )}
                        
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <p className="font-medium truncate">{detection.worker_name}</p>
                            {isCompliant ? (
                              <span className="px-2 py-0.5 text-xs rounded-full bg-green-100 text-green-800">
                                Compliant
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 text-xs rounded-full bg-red-100 text-red-800">
                                Violation
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {timeString} • {timeAgo}
                          </p>
                          {detection.confidence && (
                            <p className="text-xs text-muted-foreground">
                              Confidence: {(detection.confidence * 100).toFixed(1)}%
                            </p>
                          )}
                        </div>
                        
                        <button
                          onClick={() => handleSendEmail(detection)}
                          disabled={isSendingEmail || isCompliant}
                          className="px-3 py-1.5 text-xs bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400 disabled:cursor-not-allowed whitespace-nowrap"
                        >
                          {isSendingEmail ? "Sending..." : "Send Email"}
                        </button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </>
      )}

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

export default Dashboard;
