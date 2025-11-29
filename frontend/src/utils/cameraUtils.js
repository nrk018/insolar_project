import flaskAxios from './flaskAxios';

const FLASK_URL = import.meta.env.VITE_FLASK_URL || 'http://localhost:5000';

/**
 * Stops the camera on the Flask server
 * This can be called from anywhere in the app, including logout handlers
 * Uses multiple methods to ensure camera stops even if one fails
 */
export const stopCamera = async () => {
  try {
    // Try regular POST request first
    await flaskAxios.post(`${FLASK_URL}/api/camera/stop`, {}, { timeout: 2000 });
    console.log('[CAMERA UTILS] Camera stopped successfully');
    return true;
  } catch (err) {
    // If regular request fails, try sendBeacon as fallback
    try {
      if (navigator.sendBeacon) {
        navigator.sendBeacon(`${FLASK_URL}/api/camera/stop`, '');
        console.log('[CAMERA UTILS] Camera stop sent via beacon');
      }
    } catch (beaconErr) {
      console.warn('[CAMERA UTILS] Beacon also failed:', beaconErr);
    }
    
    // If Flask server is not running or camera is already stopped, that's okay
    console.warn('[CAMERA UTILS] Could not stop camera (may already be stopped):', err.message);
    return false;
  }
};

