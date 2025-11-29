import { useEffect, useState } from 'react';
import flaskAxios from '../utils/flaskAxios';
import { stopCamera } from '../utils/cameraUtils';

const CameraControls = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [cameraSource, setCameraSource] = useState('rtsp');
  const [flaskConnected, setFlaskConnected] = useState(false);
  const [cameraError, setCameraError] = useState(null);

  const FLASK_URL = import.meta.env.VITE_FLASK_URL || 'http://localhost:5000';

  useEffect(() => {
    checkCameraStatus();
    
    // Check camera status every 5 seconds
    const cameraInterval = setInterval(() => {
      checkCameraStatus();
    }, 5000);

    // Stop camera on page unload
    const handleBeforeUnload = () => {
      try {
        navigator.sendBeacon(`${FLASK_URL}/api/camera/stop`, '');
      } catch (e) {
        console.warn('Could not send camera stop beacon:', e);
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      clearInterval(cameraInterval);
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

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

  const handleStopCamera = async () => {
    try {
      await stopCamera();
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

  return (
    <div className="rounded-lg border bg-card p-4 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Camera Status</h2>
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

      <div className="flex items-center gap-3 flex-wrap">
        <button
          onClick={() => startCamera('rtsp')}
          disabled={isRunning && cameraSource === 'rtsp' || !flaskConnected}
          className={`px-3 py-1.5 rounded-md font-medium text-xs ${
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
          className={`px-3 py-1.5 rounded-md font-medium text-xs ${
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
              className="px-3 py-1.5 rounded-md font-medium text-xs bg-yellow-500 text-white hover:bg-yellow-600"
            >
              Switch to {cameraSource === 'rtsp' ? 'Webcam' : 'RTSP'}
            </button>
            <button
              onClick={handleStopCamera}
              className="px-3 py-1.5 rounded-md font-medium text-xs bg-red-500 text-white hover:bg-red-600"
            >
              Stop Camera
            </button>
          </>
        )}
      </div>

      {cameraError && (
        <div className="mt-3 rounded-md bg-red-50 border border-red-200 p-2">
          <p className="text-xs text-red-800">{cameraError}</p>
        </div>
      )}
    </div>
  );
};

export default CameraControls;

