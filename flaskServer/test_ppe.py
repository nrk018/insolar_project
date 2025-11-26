"""
Quick test script to verify PPE detection is working
Run this before using the full face recognition system
"""

import cv2
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ppeDetection import load_ppe_model, detect_ppe_items, check_ppe_compliance, get_ppe_status_string

def test_ppe_detection():
    """Test PPE detection on camera feed"""
    print("=" * 50)
    print("PPE Detection Test")
    print("=" * 50)
    
    # Load model
    print("\n[1/3] Loading PPE detection model...")
    model = load_ppe_model()
    
    if model is None:
        print("❌ Failed to load PPE model!")
        print("   Make sure 'ultralytics' is installed: pip install ultralytics")
        return False
    
    print("✅ Model loaded successfully")
    
    # Open camera
    print("\n[2/3] Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return False
    
    print("✅ Camera opened")
    print("\n[3/3] Starting detection test...")
    print("   - Press 'q' to quit")
    print("   - PPE status will be displayed on screen")
    print("   - Check console for detailed detection info\n")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run PPE detection every 5 frames (for performance)
            if frame_count % 5 == 0:
                detections = detect_ppe_items(frame, model=model)
                is_compliant, missing, compliance_dict = check_ppe_compliance(detections)
                
                # Print status
                status = "✅ COMPLIANT" if is_compliant else "⚠️ NON-COMPLIANT"
                print(f"\rFrame {frame_count}: {status} | Missing: {', '.join(missing) if missing else 'None'}", end="", flush=True)
                
                # Draw on frame
                status_color = (0, 255, 0) if is_compliant else (0, 0, 255)
                status_text = f"PPE: {status}"
                cv2.putText(frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                
                # Show detailed status
                if compliance_dict:
                    y_offset = 60
                    for item, data in compliance_dict.items():
                        detected = data["detected"]
                        conf = data["confidence"]
                        item_text = f"{item.capitalize()}: {'✓' if detected else '✗'} ({conf:.2f})"
                        color = (0, 255, 0) if detected else (0, 0, 255)
                        cv2.putText(frame, item_text, (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        y_offset += 30
            
            cv2.imshow("PPE Detection Test", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        print("\n\n✅ Test completed successfully!")
        print("   PPE detection is working correctly.")
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n❌ Error during test: {e}")
        return False
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    success = test_ppe_detection()
    sys.exit(0 if success else 1)

