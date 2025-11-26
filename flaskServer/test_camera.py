"""
Quick test script to verify camera connection
Run this to test if RTSP or webcam can be opened
"""
import cv2
import sys

RTSP_URL = "rtsp://admin:Test%401122@192.168.1.216:554/101?rtsp_transport=tcp"

def test_rtsp():
    print("Testing RTSP connection...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if cap.isOpened():
        print("✓ RTSP camera opened")
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✓ Successfully read frame: {frame.shape}")
            cap.release()
            return True
        else:
            print("✗ Camera opened but cannot read frames")
            cap.release()
            return False
    else:
        print("✗ Failed to open RTSP camera")
        return False

def test_webcam():
    print("\nTesting webcam connection...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✓ Webcam opened")
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✓ Successfully read frame: {frame.shape}")
            cap.release()
            return True
        else:
            print("✗ Webcam opened but cannot read frames")
            cap.release()
            return False
    else:
        print("✗ Failed to open webcam")
        return False

if __name__ == '__main__':
    print("="*60)
    print("Camera Connection Test")
    print("="*60)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'webcam':
        test_webcam()
    else:
        test_rtsp()
        test_webcam()
    
    print("\n" + "="*60)

