"""Quick test to see what the model detects"""
from ultralytics import YOLO
import cv2
import os

# Load model
model_path = 'runs/detect/ppe_detection/weights/best.pt'
model = YOLO(model_path)

print(f"Model classes: {model.names}")
print(f"Number of classes: {len(model.names)}\n")

# Test on a few images
test_img_dir = 'datasets/images/test'
img_files = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]

print(f"Testing on {len(img_files)} images...\n")

for img_file in img_files:
    img_path = os.path.join(test_img_dir, img_file)
    img = cv2.imread(img_path)
    
    if img is None:
        continue
    
    # Run detection with lower confidence threshold
    results = model(img, conf=0.25, verbose=False)
    
    num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"Image: {img_file}")
    print(f"  Detections: {num_detections}")
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for i in range(min(10, len(results[0].boxes))):
            cls = int(results[0].boxes.cls[i])
            conf = float(results[0].boxes.conf[i])
            class_name = model.names[cls]
            print(f"    - {class_name} (class {cls}): {conf:.3f}")
    else:
        print("    - No detections")
    print()

