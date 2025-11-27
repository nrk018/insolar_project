"""
Flask server for video streaming (RTSP/webcam) and image analysis
Serves MJPEG stream and handles single image uploads for employee + PPE detection
"""
import os
import sys
import cv2
import numpy as np
import torch
import pandas as pd
import time
import threading
from collections import deque
from queue import Queue
from flask import Flask, Response, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from ppeDetection import load_ppe_model, detect_ppe_items, check_ppe_compliance, get_ppe_status_string, draw_annotated_image
import base64
from datetime import datetime
from predict import AntiSpoofPredict
from PIL import Image
import requests

app = Flask(__name__)
# Configure CORS to allow frontend origin (no credentials needed for Flask server)
CORS(app, 
     origins=["http://localhost:5173", "http://127.0.0.1:5173"],
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])

# Configuration
UPLOADS_FOLDER = '../backend/uploads'
API_URL = "http://localhost:3000/api/ppe/event"
THRESHOLD = 0.65
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
SAVED_IMAGES_FOLDER = 'saved_detections'  # Folder to save annotated images from video feed
DETECTION_SNAPSHOTS_FOLDER = 'detection_snapshots'  # Folder to save detection snapshots for Recent Detections

# Create saved images folders if they don't exist
os.makedirs(SAVED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(DETECTION_SNAPSHOTS_FOLDER, exist_ok=True)

# Global camera state
camera = None
camera_lock = threading.Lock()
camera_source = None  # 'rtsp' or 'webcam'
is_running = False

# Frame processing queue and state (for async processing)
frame_queue = Queue(maxsize=2)  # Small queue to avoid lag
latest_processed_frame = None
latest_processed_data = {}  # Store latest detection results
processing_lock = threading.Lock()

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[VIDEO SERVER] Using device: {device}")

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
anti_spoof = AntiSpoofPredict(device_id=0)
# Load trained PPE model if available, otherwise use pre-trained
TRAINED_PPE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "runs", "detect", "ppe_detection", "weights", "best.pt")
if os.path.exists(TRAINED_PPE_MODEL_PATH):
    print(f"[VIDEO SERVER] Loading trained PPE model from: {TRAINED_PPE_MODEL_PATH}")
    ppe_model = load_ppe_model(model_path=TRAINED_PPE_MODEL_PATH, use_pretrained=False)
else:
    print("[VIDEO SERVER] Trained PPE model not found, using pre-trained model")
    print("[VIDEO SERVER] To train the model, run: python train_ppe_model.py")
    ppe_model = load_ppe_model()

# Load embeddings
all_embeddings = []
all_names = []

for person_folder in os.listdir(UPLOADS_FOLDER):
    full_path = os.path.join(UPLOADS_FOLDER, person_folder)
    csv_path = os.path.join(full_path, 'embeddings.csv')
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path, header=None)
    for _, row in df.iterrows():
        known_embedding = row.values.astype(float)
        known_embedding = normalize([known_embedding])[0]
        all_embeddings.append(known_embedding)
        all_names.append(person_folder)

if all_embeddings:
    all_embeddings = np.array(all_embeddings).astype('float32')
    index = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute')
    index.fit(all_embeddings)
    print(f"[VIDEO SERVER] Loaded {len(all_embeddings)} embeddings from {len(set(all_names))} person(s)")
else:
    index = None
    print("[VIDEO SERVER] WARNING: No embeddings found!")

# RTSP URL - password contains @ which needs URL encoding: @ becomes %40
# RTSP URL - password contains @ which needs URL encoding: @ becomes %40
# Format: rtsp://username:password@ip:port/path?rtsp_transport=tcp
RTSP_URL = os.getenv("RTSP_URL", "rtsp://admin:Test%401122@192.168.1.216:554/101?rtsp_transport=tcp")
print(f"[VIDEO SERVER] RTSP URL configured: rtsp://admin:***@192.168.1.216:554/101")

def detect_and_encode(image):
    with torch.no_grad():
        boxes, _ = mtcnn.detect(image)
        if boxes is not None:
            embeddings = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = image[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face = cv2.resize(face, (160, 160))
                face = np.transpose(face, (2, 0, 1)).astype(np.float32) / 255.0
                face_tensor = torch.tensor(face).unsqueeze(0).to(device)
                encoding = resnet(face_tensor).cpu().detach().numpy().flatten()
                embedding = normalize([encoding])[0]
                embeddings.append((embedding, box))
            return embeddings
    return []

def match_embedding(input_embedding):
    if index is None:
        return "Unknown", 0.0
    input_embedding = np.array(input_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.kneighbors(input_embedding, n_neighbors=1)
    cosine_distance = distances[0][0]
    best_similarity = 1.0 - cosine_distance
    best_name = all_names[indices[0][0]]
    if best_similarity >= THRESHOLD:
        return best_name, best_similarity
    return "Unknown", best_similarity

def is_real_face(frame, box, anti_spoof):
    x1, y1, x2, y2 = map(int, box)
    face_img = frame[y1:y2, x1:x2]
    if face_img.size == 0:
        return False
    face_img_resized = cv2.resize(face_img, (80, 80))
    face_img_rgb = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(face_img_rgb)
    model_path = os.path.join(
        os.path.dirname(__file__),
        "Silent-Face-Anti-Spoofing",
        "resources",
        "anti_spoof_models",
        "2.7_80x80_MiniFASNetV2.pth",
    )
    prediction = anti_spoof.predict(pil_image, model_path)
    label = np.argmax(prediction)
    return bool(label == 1)  # Convert to Python bool for JSON serialization

def open_camera(source_type='rtsp', max_retries=5):
    """Open camera stream (RTSP or webcam) with retry logic and Windows-specific fixes"""
    global camera, camera_source
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        
        for attempt in range(max_retries):
            try:
                if source_type == 'rtsp':
                    print(f"[VIDEO SERVER] Attempting RTSP connection (attempt {attempt + 1}/{max_retries})...")
                    print(f"[VIDEO SERVER] RTSP URL: {RTSP_URL}")
                    
                    # On Windows, try different backends for better RTSP support
                    if sys.platform == 'win32':
                        # Try multiple backends on Windows
                        backends = [
                            (cv2.CAP_FFMPEG, "FFMPEG"),
                            (cv2.CAP_ANY, "ANY"),
                        ]
                        
                        for backend_id, backend_name in backends:
                            try:
                                print(f"[VIDEO SERVER] Trying backend: {backend_name}")
                                camera = cv2.VideoCapture(RTSP_URL, backend_id)
                                
                                if camera.isOpened():
                                    # Set timeout properties for Windows RTSP
                                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                    # Give more time for RTSP connection on Windows
                                    time.sleep(2)
                                    
                                    # Try reading a test frame with multiple attempts
                                    ret = False
                                    test_frame = None
                                    for read_attempt in range(5):
                                        ret, test_frame = camera.read()
                                        if ret and test_frame is not None and test_frame.size > 0:
                                            print(f"[VIDEO SERVER] RTSP camera opened successfully with {backend_name} backend!")
                                            print(f"[VIDEO SERVER] Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                                            camera_source = 'rtsp'
                                            return True
                                        time.sleep(1)
                                    
                                    if not ret:
                                        print(f"[VIDEO SERVER] {backend_name} backend opened but cannot read frames, trying next backend...")
                                        camera.release()
                                        camera = None
                                        continue
                                else:
                                    print(f"[VIDEO SERVER] {backend_name} backend failed to open camera")
                                    if camera:
                                        camera.release()
                                        camera = None
                                    continue
                                    
                            except Exception as backend_err:
                                print(f"[VIDEO SERVER] Error with {backend_name} backend: {backend_err}")
                                if camera:
                                    camera.release()
                                    camera = None
                                continue
                        
                        # If all backends failed, try one more time with default
                        print(f"[VIDEO SERVER] All backends failed, trying default method...")
                        camera = cv2.VideoCapture(RTSP_URL)
                        time.sleep(2)
                    else:
                        # On macOS/Linux, use FFMPEG backend
                        camera = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        time.sleep(1)
                    
                    camera_source = 'rtsp'
                    
                    if camera.isOpened():
                        # Try reading a test frame
                        ret, test_frame = camera.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            print(f"[VIDEO SERVER] RTSP camera opened successfully!")
                            print(f"[VIDEO SERVER] Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
                            return True
                        else:
                            print(f"[VIDEO SERVER] Camera opened but cannot read frames, retrying...")
                            if camera:
                                camera.release()
                                camera = None
                    else:
                        print(f"[VIDEO SERVER] Failed to open RTSP camera")
                        if camera:
                            camera.release()
                            camera = None
                        
                else:  # webcam (laptop camera)
                    print(f"[VIDEO SERVER] Attempting laptop camera connection (attempt {attempt + 1}/{max_retries})...")
                    # Try different camera indices (0, 1, 2) in case default doesn't work
                    # Index 0 is typically the built-in laptop camera
                    webcam_index = attempt  # Try 0, 1, 2
                    try:
                        camera = cv2.VideoCapture(webcam_index)
                        # Set backend to V4L2 on Linux, or default on Mac/Windows
                        if sys.platform != 'win32':
                            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        camera_source = 'webcam'
                    except Exception as webcam_err:
                        print(f"[VIDEO SERVER] Error creating VideoCapture for laptop camera (index {webcam_index}): {webcam_err}")
                        if camera:
                            camera.release()
                            camera = None
                        continue
                    
                    if camera.isOpened():
                        # Optimize camera settings for lower latency and better performance
                        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to reduce latency
                        camera.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
                        time.sleep(0.5)  # Give camera time to initialize
                        
                        # Try reading a test frame
                        ret, test_frame = camera.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            print(f"[VIDEO SERVER] Camera opened successfully: {source_type}")
                            return True
                        else:
                            print(f"[VIDEO SERVER] Camera opened but cannot read frames, retrying...")
                            if camera:
                                camera.release()
                                camera = None
                    else:
                        print(f"[VIDEO SERVER] Failed to open camera, retrying...")
                        if camera:
                            camera.release()
                            camera = None
                
                if attempt < max_retries - 1:
                    wait_time = 2 if source_type == 'rtsp' else 1
                    print(f"[VIDEO SERVER] Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                print(f"[VIDEO SERVER] Error opening camera: {e}")
                import traceback
                traceback.print_exc()
                if camera:
                    camera.release()
                    camera = None
                if attempt < max_retries - 1:
                    wait_time = 2 if source_type == 'rtsp' else 1
                    time.sleep(wait_time)
        
        error_msg = f"[VIDEO SERVER] Failed to open {source_type} camera after {max_retries} attempts"
        if source_type == 'rtsp':
            error_msg += "\n  - Verify RTSP URL is correct and camera is accessible"
            error_msg += f"\n  - Current RTSP URL: {RTSP_URL}"
            error_msg += "\n  - Check network connectivity: ping the camera IP"
            error_msg += "\n  - Verify RTSP port 554 is open"
            error_msg += "\n  - Test RTSP URL in VLC Media Player to confirm it works"
            error_msg += "\n  - Check camera credentials (username/password)"
            if sys.platform == 'win32':
                error_msg += "\n  - On Windows: Ensure OpenCV was built with FFMPEG support"
                error_msg += "\n  - Try installing opencv-python-headless or opencv-contrib-python"
        elif source_type == 'webcam':
            error_msg += "\n  - Make sure laptop camera is available and not in use by another app"
            error_msg += "\n  - Close other apps using the camera (Zoom, Teams, Photo Booth, etc.)"
            if sys.platform == 'darwin':
                error_msg += "\n  - On macOS: System Settings → Privacy & Security → Camera → Enable for Terminal/Python"
            elif sys.platform == 'win32':
                error_msg += "\n  - On Windows: Settings → Privacy → Camera → Allow apps to access camera"
        print(error_msg)
        return False

def process_frame_async(frame, frame_count):
    """Process frame in background thread (face detection, PPE detection)"""
    global latest_processed_frame, latest_processed_data
    
    try:
        # Process face recognition and PPE detection (expensive operations)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embeddings_with_boxes = detect_and_encode(rgb_frame)
        
        # PPE detection
        ppe_detections = None
        ppe_compliant = False
        ppe_compliance_dict = {}
        
        if ppe_model is not None:
            try:
                ppe_detections = detect_ppe_items(frame, model=ppe_model)
                ppe_compliant, _, ppe_compliance_dict = check_ppe_compliance(ppe_detections)
            except Exception as e:
                print(f"[PPE ERROR] {e}")
        
        # Process detected faces
        current_time = time.time()
        face_results_for_annotation = []
        
        for embedding, box in embeddings_with_boxes:
            name, confidence = match_embedding(embedding)
            x1, y1, x2, y2 = map(int, box)
            
            is_real = is_real_face(frame, box, anti_spoof)
            
            if not is_real:
                name = "Spoof Detected"
            else:
                # Prepare PPE items dict
                ppe_items_dict = {}
                ppe_avg_confidence = 0.0
                if ppe_compliance_dict:
                    for item, data in ppe_compliance_dict.items():
                        ppe_items_dict[item] = bool(data["detected"])
                    confidences = [data["confidence"] for data in ppe_compliance_dict.values() if data["detected"]]
                    ppe_avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            face_results_for_annotation.append({
                "name": name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "is_real": is_real,
                "ppe_compliant": ppe_compliant if is_real and name != "Unknown" else False,
                "ppe_items_dict": ppe_items_dict if is_real and name != "Unknown" else {}
            })
        
        # Update latest processed data
        with processing_lock:
            latest_processed_frame = frame.copy()
            latest_processed_data = {
                "embeddings_with_boxes": embeddings_with_boxes,
                "ppe_detections": ppe_detections,
                "ppe_compliant": ppe_compliant,
                "ppe_compliance_dict": ppe_compliance_dict,
                "face_results": face_results_for_annotation,
                "frame_count": frame_count
            }
    except Exception as e:
        print(f"[PROCESSING ERROR] {e}")


def generate_frames():
    """Generator function for video streaming - displays in web app, not OpenCV window"""
    global camera, is_running, latest_processed_data
    marked_once = set()  # Track who has been sent PPE event (once per session)
    last_detection_time = {}  # Track last detection time per person to avoid spam
    last_image_save_time = {}  # Track last image save time per person to avoid saving too many images
    last_snapshot_path = {}  # Track last snapshot path per person to delete old snapshots
    frame_count = 0
    process_every_n_frames = 8  # Process every 8th frame (reduced frequency for better performance)
    DETECTION_COOLDOWN = 2.0  # Minimum seconds between detection events for same person
    IMAGE_SAVE_COOLDOWN = 5.0  # Minimum seconds between saving images for same person (5 seconds)
    
    # Start background processing thread
    def background_processor():
        global is_running
        while True:
            try:
                if not is_running:
                    time.sleep(0.1)
                    continue
                if not frame_queue.empty():
                    frame, fc = frame_queue.get(timeout=0.1)
                    process_frame_async(frame, fc)
                else:
                    time.sleep(0.01)  # Small sleep if queue is empty
            except:
                time.sleep(0.1)
    
    processor_thread = None
    
    # Keep generating frames even when not running (show placeholder)
    while True:
        with camera_lock:
            if not is_running or camera is None or not camera.isOpened():
                # Stop processor thread if camera stops
                if processor_thread and processor_thread.is_alive():
                    processor_thread = None
                
                # Send a placeholder frame when camera is not available
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                if not is_running:
                    text = "Camera not started - Click Start RTSP or Start Laptop Camera"
                else:
                    text = "Camera connection lost - Reconnecting..."
                cv2.putText(blank_frame, text, (50, 220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(blank_frame, "Waiting for camera...", (150, 260), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                ret, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.5)
                continue
            
            # Start processor thread if not running
            if processor_thread is None or not processor_thread.is_alive():
                processor_thread = threading.Thread(target=background_processor, daemon=True)
                processor_thread.start()
            
            # Read frame without grabbing multiple times (causes stuttering)
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)  # Short sleep on read failure
                continue
        
        frame_count += 1
        should_process = (frame_count % process_every_n_frames == 0)
        
        # Queue frame for processing (non-blocking)
        if should_process and not frame_queue.full():
            try:
                frame_queue.put_nowait((frame.copy(), frame_count))
            except:
                pass  # Skip if queue is full (don't block)
        
        # Get latest processed data (non-blocking)
        processed_data = None
        with processing_lock:
            if latest_processed_data:
                # Use latest processed data if available (don't check frame_count too strictly)
                processed_data = latest_processed_data.copy()
        
        # Draw on current frame using latest processed data
        if processed_data:
            embeddings_with_boxes = processed_data.get("embeddings_with_boxes", [])
            ppe_detections = processed_data.get("ppe_detections")
            ppe_compliant = processed_data.get("ppe_compliant", False)
            ppe_compliance_dict = processed_data.get("ppe_compliance_dict", {})
            face_results = processed_data.get("face_results", [])
            current_time = time.time()
            
            # Debug: Log if we have face results
            if len(face_results) > 0:
                print(f"[DEBUG] Processing {len(face_results)} face(s) in frame {frame_count}")
            
            # Draw face boxes and process events
            for result in face_results:
                name = result["name"]
                confidence = result["confidence"]
                bbox = result["bbox"]
                is_real = result["is_real"]
                ppe_compliant_person = result.get("ppe_compliant", False)
                ppe_items_dict = result.get("ppe_items_dict", {})
                
                x1, y1, x2, y2 = bbox
                
                if not is_real:
                    color = (0, 165, 255)  # Orange for spoof
                else:
                    color = (0, 255, 0) if name != 'Unknown' else (0, 0, 255)
                
                # Send PPE event to backend (only once per person per session)
                if name != "Unknown" and name not in marked_once:
                    try:
                        ppe_avg_confidence = 0.0
                        if ppe_items_dict:
                            confidences = [ppe_compliance_dict.get(item, {}).get("confidence", 0) 
                                         for item, detected in ppe_items_dict.items() if detected]
                            ppe_avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                        
                        requests.post(API_URL, json={
                            "name": name,
                            "ppe_compliant": bool(ppe_compliant_person),
                            "ppe_items": ppe_items_dict,
                            "ppe_confidence": float(ppe_avg_confidence)
                        }, timeout=0.5)
                        marked_once.add(name)
                    except Exception as e:
                        print(f"[PPE EVENT ERROR] {e}")
                
                # Store detection event for recent detections
                if name != "Unknown" and name != "Spoof Detected":
                    should_send_detection = False
                    if name not in last_detection_time:
                        should_send_detection = True
                        print(f"[DEBUG] First detection for {name}, will send event")
                    elif (current_time - last_detection_time[name]) >= DETECTION_COOLDOWN:
                        should_send_detection = True
                        print(f"[DEBUG] Cooldown passed for {name}, will send event")
                    
                    if should_send_detection:
                        print(f"[DEBUG] Sending detection event for {name} (confidence: {confidence:.2f})")
                        try:
                            # Delete previous snapshot for this person if it exists
                            if name in last_snapshot_path:
                                old_snapshot_path = last_snapshot_path[name]
                                old_snapshot_filepath = os.path.join(DETECTION_SNAPSHOTS_FOLDER, os.path.basename(old_snapshot_path))
                                try:
                                    if os.path.exists(old_snapshot_filepath):
                                        os.remove(old_snapshot_filepath)
                                        print(f"[SNAPSHOT] Deleted previous snapshot: {os.path.basename(old_snapshot_path)}")
                                except Exception as e:
                                    print(f"[SNAPSHOT DELETE ERROR] Failed to delete old snapshot for {name}: {e}")
                            
                            # Create snapshot with annotations for Recent Detections
                            snapshot_path = None
                            annotated_snapshot = None
                            try:
                                # Use the processed frame that matches the detection data
                                snapshot_frame = None
                                with processing_lock:
                                    if latest_processed_frame is not None:
                                        snapshot_frame = latest_processed_frame.copy()
                                
                                # Fallback to current frame if processed frame not available
                                if snapshot_frame is None:
                                    snapshot_frame = frame.copy()
                                    print(f"[SNAPSHOT WARNING] Using current frame instead of processed frame")
                                
                                annotated_snapshot = draw_annotated_image(
                                    snapshot_frame, 
                                    ppe_detections, 
                                    face_results, 
                                    use_display_thresholds=True
                                )
                                print(f"[SNAPSHOT] Created annotated image for {name}, frame shape: {annotated_snapshot.shape}")
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                                safe_name = name.replace(" ", "_").replace("/", "_")
                                snapshot_filename = f"detection_{safe_name}_{timestamp}.jpg"
                                snapshot_filepath = os.path.join(DETECTION_SNAPSHOTS_FOLDER, snapshot_filename)
                                success = cv2.imwrite(snapshot_filepath, annotated_snapshot)
                                if not success:
                                    print(f"[SNAPSHOT ERROR] Failed to write image to {snapshot_filepath}")
                                else:
                                    # Store relative path for backend (will be served as static file)
                                    snapshot_path = f"detection_snapshots/{snapshot_filename}"
                                    # Update last snapshot path for this person
                                    last_snapshot_path[name] = snapshot_path
                                    print(f"[SNAPSHOT] ✅ Saved detection snapshot: {snapshot_filename} at {snapshot_filepath}")
                                    print(f"[SNAPSHOT] 📍 Snapshot path stored: {snapshot_path}")
                            except Exception as e:
                                print(f"[SNAPSHOT ERROR] Failed to save snapshot for {name}: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            detection_event_url = "http://localhost:3000/api/detections/event"
                            response = None
                            try:
                                payload = {
                                    "worker_name": name,
                                    "confidence": float(confidence),
                                    "ppe_compliant": bool(ppe_compliant_person),
                                    "ppe_items": ppe_items_dict,
                                    "camera_source": camera_source or "webcam",
                                    "snapshot_path": snapshot_path  # Include snapshot path
                                }
                                print(f"[DETECTION EVENT] 📤 Sending to backend: {name}, confidence: {confidence:.2f}, snapshot_path: {snapshot_path}")
                                response = requests.post(detection_event_url, json=payload, timeout=3.0)
                                if response.status_code == 200:
                                    last_detection_time[name] = current_time
                                    print(f"[DETECTION EVENT] ✅ Successfully stored: {name} (confidence: {confidence:.2f}, PPE: {ppe_compliant_person}, snapshot: {snapshot_path})")
                                    
                                    # Also save to saved_detections folder (with cooldown) - only if detection was successful
                                    should_save_image = False
                                    if name not in last_image_save_time:
                                        should_save_image = True
                                    elif (current_time - last_image_save_time[name]) >= IMAGE_SAVE_COOLDOWN:
                                        should_save_image = True
                                    
                                    if should_save_image:
                                        try:
                                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            ppe_status = "compliant" if ppe_compliant_person else "non_compliant"
                                            safe_name = name.replace(" ", "_").replace("/", "_")
                                            filename = f"{safe_name}_{timestamp}_{ppe_status}.jpg"
                                            filepath = os.path.join(SAVED_IMAGES_FOLDER, filename)
                                            # Use annotated snapshot if available, otherwise use processed frame
                                            save_frame = annotated_snapshot if annotated_snapshot is not None else frame.copy()
                                            cv2.imwrite(filepath, save_frame)
                                            last_image_save_time[name] = current_time
                                            print(f"[IMAGE SAVED] Saved annotated image: {filename}")
                                        except Exception as e:
                                            print(f"[IMAGE SAVE ERROR] Failed to save image for {name}: {e}")
                                            import traceback
                                            traceback.print_exc()
                                else:
                                    print(f"[DETECTION EVENT] ❌ Backend returned {response.status_code}: {response.text}")
                            except requests.exceptions.ConnectionError as conn_err:
                                print(f"[DETECTION EVENT] ❌ Connection error - Is backend running on port 3000? {conn_err}")
                            except requests.exceptions.Timeout as timeout_err:
                                print(f"[DETECTION EVENT] ❌ Timeout error: {timeout_err}")
                            except requests.exceptions.RequestException as req_err:
                                print(f"[DETECTION EVENT] ❌ Request error: {req_err}")
                            except Exception as req_e:
                                print(f"[DETECTION EVENT] ❌ Unexpected error: {req_e}")
                                import traceback
                                traceback.print_exc()
                        except Exception as e:
                            print(f"[DETECTION EVENT ERROR] Exception: {e}")
                            import traceback
                            traceback.print_exc()
                            pass
                
                # Don't draw face boxes on live feed (user requested removal)
                # Face recognition is still processed in background, just not displayed on stream
            
            # Don't draw PPE boxes on live feed (user requested removal)
            # PPE detection is still processed in background, just not displayed on stream
            # All annotations (face + PPE) will be shown in the snapshot saved for Recent Detections
        
        # Encode frame as JPEG with lower quality for faster transmission
        # Lower quality = smaller file = faster transmission = smoother video
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            time.sleep(0.01)
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # No sleep - let it stream as fast as possible

@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start camera stream and face recognition processing (RTSP or webcam)"""
    global is_running, camera
    data = request.json or {}
    source_type = data.get('source', 'rtsp')  # 'rtsp' or 'webcam'
    
    # If camera is already running, stop it first
    if is_running:
        is_running = False
        time.sleep(0.5)
    
    try:
        if open_camera(source_type):
            is_running = True
            print(f"[VIDEO SERVER] Camera started: {source_type}")
            return jsonify({"status": "started", "source": source_type})
        else:
            error_msg = f"Failed to open {source_type} camera. "
            if source_type == 'rtsp':
                error_msg += "Troubleshooting steps:\n"
                error_msg += f"1. Verify RTSP URL: {RTSP_URL}\n"
                error_msg += "2. Test camera connectivity: ping the camera IP address\n"
                error_msg += "3. Test RTSP stream in VLC Media Player\n"
                error_msg += "4. Check camera credentials (username/password)\n"
                error_msg += "5. Verify port 554 is open and accessible\n"
                if sys.platform == 'win32':
                    error_msg += "6. On Windows: Ensure OpenCV has FFMPEG support (try: pip install opencv-python-headless)\n"
            else:
                error_msg += "Check if laptop camera is available. Close other apps using the camera (Zoom, Teams, etc.). "
                if sys.platform == 'darwin':
                    error_msg += "On macOS, grant camera permissions in System Settings."
                elif sys.platform == 'win32':
                    error_msg += "On Windows: Settings → Privacy → Camera → Allow apps to access camera."
            return jsonify({"status": "error", "message": error_msg}), 500
    except Exception as e:
        print(f"[VIDEO SERVER] Exception in start_camera: {e}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"}), 500

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop camera stream and face recognition processing"""
    global is_running, camera
    is_running = False
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    print("[VIDEO SERVER] Camera stopped")
    return jsonify({"status": "stopped"})

@app.route('/api/camera/switch', methods=['POST'])
def switch_camera():
    """Switch between RTSP and webcam while keeping processing running"""
    global is_running
    data = request.json or {}
    source_type = data.get('source', 'rtsp')
    
    was_running = is_running
    if was_running:
        is_running = False
        time.sleep(0.5)  # Give time for current frame processing to finish
    
    if open_camera(source_type):
        is_running = was_running  # Restore previous running state
        print(f"[VIDEO SERVER] Camera switched to: {source_type}")
        return jsonify({"status": "switched", "source": source_type})
    return jsonify({"status": "error", "message": "Failed to switch camera"}), 500

@app.route('/api/camera/status', methods=['GET'])
def camera_status():
    """Get current camera status"""
    global is_running, camera, camera_source
    with camera_lock:
        is_open = camera is not None and camera.isOpened()
    return jsonify({
        "running": is_running,
        "open": is_open,
        "source": camera_source
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify Flask server is running"""
    return jsonify({
        "status": "ok",
        "message": "Flask video server is running",
        "models_loaded": {
            "face_recognition": index is not None,
            "ppe_detection": ppe_model is not None,
            "embeddings_count": len(all_embeddings)
        }
    })

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for employee recognition and PPE detection"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400
        
        # Read image
        try:
            file_bytes = file.read()
            if len(file_bytes) == 0:
                return jsonify({"error": "Image file is empty"}), 400
            
            # Convert bytes to numpy array - use frombuffer (not frombuffer)
            nparr = np.frombuffer(file_bytes, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"error": "Failed to decode image. Please ensure the file is a valid image (PNG, JPG, JPEG)"}), 400
            
            if frame.size == 0:
                return jsonify({"error": "Decoded image is empty"}), 400
                
        except Exception as e:
            print(f"[ANALYZE ERROR] Failed to read/decode image: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 400
        
        # Convert to RGB for face detection
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[ANALYZE ERROR] Failed to convert color: {e}")
            return jsonify({"error": f"Image color conversion failed: {str(e)}"}), 400
        
        # Detect faces and generate embeddings
        try:
            embeddings_with_boxes = detect_and_encode(rgb_frame)
        except Exception as e:
            print(f"[ANALYZE ERROR] Face detection failed: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Face detection failed: {str(e)}"}), 500
        
        results = []
        
        # PPE detection
        ppe_detections = None
        ppe_compliant = False
        ppe_compliance_dict = {}
        
        if ppe_model is not None:
            try:
                # For image analysis, use less strict spatial validation
                # (single images may not have person detection, but PPE is still visible)
                ppe_detections = detect_ppe_items(frame, model=ppe_model, strict_spatial_validation=False)
                ppe_compliant, _, ppe_compliance_dict = check_ppe_compliance(ppe_detections)
            except Exception as e:
                print(f"[PPE ERROR] {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[ANALYZE WARNING] PPE model not loaded")
        
        # Process each detected face
        for embedding, box in embeddings_with_boxes:
            try:
                name, confidence = match_embedding(embedding)
                x1, y1, x2, y2 = map(int, box)
                
                is_real = is_real_face(frame, box, anti_spoof)
                
                ppe_items_dict = {}
                ppe_items_accuracy = {}  # Store accuracy/confidence for each item
                ppe_avg_confidence = 0.0
                if ppe_compliance_dict:
                    for item, data in ppe_compliance_dict.items():
                        # Ensure boolean values are Python bool, not NumPy bool_
                        ppe_items_dict[item] = bool(data["detected"])
                        ppe_items_accuracy[item] = float(data["confidence"])  # Store confidence as accuracy
                    confidences = [data["confidence"] for data in ppe_compliance_dict.values() if data["detected"]]
                    ppe_avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                results.append({
                    "name": name,
                    "confidence": float(confidence),
                    "is_real": bool(is_real),  # Convert to Python bool
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "ppe_compliant": bool(ppe_compliant),  # Convert to Python bool
                    "ppe_items": ppe_items_dict,
                    "ppe_items_accuracy": ppe_items_accuracy,  # Add accuracy metrics per item
                    "ppe_confidence": float(ppe_avg_confidence),
                    "helmet_detected": bool(ppe_items_dict.get("helmet", False))  # Convert to Python bool
                })
            except Exception as e:
                print(f"[ANALYZE ERROR] Error processing face: {e}")
                continue
        
        # Prepare PPE items dict for response (even if no faces)
        ppe_items_dict = {}
        ppe_items_accuracy = {}
        if ppe_compliance_dict:
            for item, data in ppe_compliance_dict.items():
                # Ensure boolean values are Python bool, not NumPy bool_
                ppe_items_dict[item] = bool(data["detected"])
                ppe_items_accuracy[item] = float(data["confidence"])  # Store confidence as accuracy
        
        # Create annotated image with PPE boxes and person labels
        annotated_frame = draw_annotated_image(frame, ppe_detections, results, use_display_thresholds=True)
        
        # Encode annotated image as base64
        try:
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ret:
                annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')
                annotated_image_url = f"data:image/jpeg;base64,{annotated_image_base64}"
            else:
                annotated_image_url = None
        except Exception as e:
            print(f"[ANALYZE ERROR] Failed to encode annotated image: {e}")
            annotated_image_url = None
        
        if len(results) == 0:
            return jsonify({
                "message": "No faces detected in image",
                "ppe_compliant": bool(ppe_compliant),  # Convert to Python bool
                "ppe_items": ppe_items_dict,
                "ppe_items_accuracy": ppe_items_accuracy,  # Include accuracy metrics
                "annotated_image": annotated_image_url  # Include annotated image
            })
        
        return jsonify({
            "results": results,
            "ppe_compliant": bool(ppe_compliant),  # Convert to Python bool
            "ppe_items": ppe_items_dict,
            "ppe_items_accuracy": ppe_items_accuracy,  # Include accuracy metrics
            "annotated_image": annotated_image_url  # Include annotated image
        })
        
    except Exception as e:
        print(f"[ANALYZE ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error during analysis: {str(e)}"}), 500

@app.route('/detection_snapshots/<path:filename>')
def serve_snapshot(filename):
    """Serve detection snapshot images"""
    try:
        # Decode URL-encoded filename
        from urllib.parse import unquote
        filename = unquote(filename)
        filepath = os.path.join(DETECTION_SNAPSHOTS_FOLDER, filename)
        
        print(f"[SNAPSHOT SERVE] Requested: {filename}")
        print(f"[SNAPSHOT SERVE] Full path: {filepath}")
        print(f"[SNAPSHOT SERVE] Exists: {os.path.exists(filepath)}")
        
        if not os.path.exists(filepath):
            print(f"[SNAPSHOT SERVE ERROR] File not found: {filepath}")
            # List available files for debugging
            if os.path.exists(DETECTION_SNAPSHOTS_FOLDER):
                available_files = os.listdir(DETECTION_SNAPSHOTS_FOLDER)
                print(f"[SNAPSHOT SERVE] Available files: {available_files[:5]}...")
            return jsonify({"error": "Snapshot not found"}), 404
        
        print(f"[SNAPSHOT SERVE] ✅ Serving: {filename}")
        return send_from_directory(DETECTION_SNAPSHOTS_FOLDER, filename)
    except Exception as e:
        print(f"[SNAPSHOT SERVE ERROR] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Snapshot not found"}), 404

if __name__ == '__main__':
    print("\n" + "="*60)
    print("[VIDEO SERVER] Starting Flask video server...")
    print(f"[VIDEO SERVER] Models loaded:")
    print(f"  - Face Recognition: {'✓' if index is not None else '✗'} ({len(all_embeddings)} embeddings)")
    print(f"  - PPE Detection: {'✓' if ppe_model is not None else '✗'}")
    print(f"[VIDEO SERVER] RTSP URL: rtsp://admin:***@192.168.1.216:554/101")
    print(f"[VIDEO SERVER] Server will run on: http://0.0.0.0:5000")
    print(f"[VIDEO SERVER] Camera will NOT auto-start - use /api/camera/start from web app")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)

