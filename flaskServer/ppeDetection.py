"""
PPE (Personal Protective Equipment) Detection Module
Uses YOLOv12 (or YOLOv11 fallback) for detecting safety equipment: helmet, gloves, boots, jacket
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Required PPE items
REQUIRED_PPE = {
    "helmet": True,
    "gloves": True,
    "boots": True,
    "jacket": True,  # Also detects "vest" as jacket
}

# Confidence threshold for PPE detection
# Balanced thresholds to detect all 4 items while reducing false positives
PPE_CONFIDENCE_THRESHOLD = 0.65  # Lower base threshold to catch more detections
PPE_CLASS_THRESHOLDS = {
    "helmet": 0.75,   # Balanced threshold: 0.75 to reduce false positives but still detect real helmets
    "gloves": 0.65,   # Lowered from 0.75 to detect more gloves
    "boots": 0.65,    # Lowered from 0.75 to detect more boots
    "jacket": 0.65,   # Lowered from 0.75 to detect more jackets
}
# For drawing/display: use slightly higher thresholds to reduce false positives in UI
PPE_DISPLAY_THRESHOLDS = {
    "helmet": 0.78,   # Slightly higher for display: 0.78 to reduce false positives in UI
    "gloves": 0.70,   # Medium threshold for display
    "boots": 0.70,    # Medium threshold for display
    "jacket": 0.70,   # Medium threshold for display
}

# Class mappings for PPE items (YOLO class names may vary)
PPE_CLASS_MAPPINGS = {
    "helmet": ["helmet", "hard-hat", "hardhat", "safety-helmet"],
    "gloves": ["gloves", "glove", "safety-gloves"],
    "boots": ["boots", "boot", "safety-boots", "safety-shoes"],
    "jacket": ["jacket", "vest", "safety-vest", "safety-jacket", "reflective-vest"],
}

# Global model variable
ppe_model = None


def load_ppe_model(model_path=None, use_pretrained=True):
    """
    Load YOLO model for PPE detection.
    
    Args:
        model_path: Path to custom trained model (.pt file). If None, uses pre-trained.
        use_pretrained: If True, uses pre-trained YOLOv11. If False, requires model_path.
    
    Returns:
        YOLO model instance
    """
    global ppe_model
    
    if ppe_model is not None:
        return ppe_model
    
    try:
        if model_path and os.path.exists(model_path):
            print(f"[PPE] Loading custom model from: {model_path}")
            ppe_model = YOLO(model_path)
        elif use_pretrained:
            print("[PPE] Loading pre-trained YOLO model...")
            # Try YOLOv12 first (latest, best accuracy), then YOLOv11, then YOLOv8
            try:
                ppe_model = YOLO("yolo12n.pt")  # Latest version, better accuracy
                print("[PPE] YOLOv12 loaded successfully (best accuracy)")
            except:
                try:
                    ppe_model = YOLO("yolo11n.pt")  # Fallback to YOLOv11
                    print("[PPE] YOLOv11 loaded successfully")
                except:
                    print("[PPE] YOLOv11 not available, trying YOLOv8...")
                    ppe_model = YOLO("yolo8n.pt")
                    print("[PPE] YOLOv8 loaded successfully")
        else:
            raise ValueError("No model path provided and use_pretrained is False")
        
        # Move model to appropriate device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[PPE] Model loaded on device: {device}")
        
        return ppe_model
    except Exception as e:
        print(f"[PPE ERROR] Failed to load model: {e}")
        print("[PPE] Attempting to download pre-trained model...")
        try:
            # Try YOLOv12 first, then fallback
            try:
                ppe_model = YOLO("yolo12n.pt")
                return ppe_model
            except:
                ppe_model = YOLO("yolo11n.pt")
                return ppe_model
        except:
            print("[PPE ERROR] Could not load PPE model. PPE detection will be disabled.")
            return None


def normalize_class_name(class_name):
    """Normalize class name to match our PPE categories."""
    class_name_lower = class_name.lower().replace("_", "-").replace(" ", "-")
    
    # Check each PPE category
    for ppe_item, variants in PPE_CLASS_MAPPINGS.items():
        for variant in variants:
            if variant in class_name_lower or class_name_lower in variant:
                return ppe_item
    
    return None


def detect_ppe_items(frame, model=None, confidence_threshold=PPE_CONFIDENCE_THRESHOLD, strict_spatial_validation=True):
    """
    Detect PPE items in a frame.
    
    Args:
        frame: Input frame (BGR format, OpenCV format)
        model: YOLO model instance (if None, uses global model)
        confidence_threshold: Minimum confidence for detection
        strict_spatial_validation: If True, requires PPE to be near person (for video).
                                   If False, more lenient for single image analysis.
    
    Returns:
        dict: {
            "helmet": (detected: bool, confidence: float),
            "gloves": (detected: bool, confidence: float),
            "boots": (detected: bool, confidence: float),
            "jacket": (detected: bool, confidence: float),
            "all_detections": list of all detections
        }
    """
    if model is None:
        model = ppe_model
        if model is None:
            model = load_ppe_model()
    
    if model is None:
        # Return default (all False) if model not available
        return {
            "helmet": (False, 0.0),
            "gloves": (False, 0.0),
            "boots": (False, 0.0),
            "jacket": (False, 0.0),
            "all_detections": [],
        }
    
    try:
        # Run YOLO inference with lower initial threshold to catch more detections
        # Use lower threshold for YOLO (0.5), then apply class-specific filtering
        results = model(frame, conf=0.5, verbose=False)  # Lower initial threshold to catch all items
        
        # Initialize detection results
        detections = {
            "helmet": (False, 0.0),
            "gloves": (False, 0.0),
            "boots": (False, 0.0),
            "jacket": (False, 0.0),
            "all_detections": [],
        }
        
        # Find person detections for spatial validation
        person_boxes = []
        frame_height, frame_width = frame.shape[:2]
        
        # Process detections
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # First pass: collect person detections and all PPE detections
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                class_name = model.names[cls] if hasattr(model, 'names') else str(cls)
                
                # Collect person detections for spatial validation
                if class_name.lower() == "person" and conf >= 0.5:
                    if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                        try:
                            box_tensor = boxes.xyxy[i]
                            if hasattr(box_tensor, 'cpu'):
                                person_box = box_tensor.cpu().numpy().tolist()
                            else:
                                person_box = box_tensor.tolist() if hasattr(box_tensor, 'tolist') else list(box_tensor)
                            person_boxes.append(person_box)
                        except:
                            pass
            
            # Second pass: process PPE detections with stricter filtering
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                class_name = model.names[cls] if hasattr(model, 'names') else str(cls)
                
                # Normalize class name to our PPE categories
                ppe_item = normalize_class_name(class_name)
                
                if ppe_item and ppe_item in detections:
                    # Get class-specific threshold
                    item_threshold = PPE_CLASS_THRESHOLDS.get(ppe_item, confidence_threshold)
                    
                    # Apply stricter threshold
                    if conf < item_threshold:
                        continue  # Skip low-confidence detections
                    
                    # Spatial validation: check if PPE is near a person
                    is_near_person = False
                    helmet_position_valid = True  # For helmets, check if in upper portion of person
                    
                    if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                        try:
                            box_tensor = boxes.xyxy[i]
                            if hasattr(box_tensor, 'cpu'):
                                ppe_box = box_tensor.cpu().numpy().tolist()
                            else:
                                ppe_box = box_tensor.tolist() if hasattr(box_tensor, 'tolist') else list(box_tensor)
                            
                            # Check if PPE box overlaps or is near any person box
                            ppe_center_x = (ppe_box[0] + ppe_box[2]) / 2
                            ppe_center_y = (ppe_box[1] + ppe_box[3]) / 2
                            
                            for person_box in person_boxes:
                                person_center_x = (person_box[0] + person_box[2]) / 2
                                person_center_y = (person_box[1] + person_box[3]) / 2
                                
                                # Calculate distance between centers
                                distance = ((ppe_center_x - person_center_x)**2 + (ppe_center_y - person_center_y)**2)**0.5
                                
                                # Calculate person box size for relative distance
                                person_width = person_box[2] - person_box[0]
                                person_height = person_box[3] - person_box[1]
                                person_size = (person_width + person_height) / 2
                                
                                # PPE should be within 1.5x person size (reasonable distance)
                                if distance < person_size * 1.5:
                                    is_near_person = True
                                    
                                    # Additional validation for helmets: must be in upper 50% of person's bounding box
                                    if ppe_item == "helmet":
                                        person_top = person_box[1]
                                        person_bottom = person_box[3]
                                        person_upper_50_percent = person_top + (person_bottom - person_top) * 0.5
                                        
                                        # Helmet center should be in upper portion, and helmet bottom should not be too low
                                        # More lenient: allow helmet if center is in upper 50% OR if confidence is high enough
                                        if ppe_center_y > person_upper_50_percent:
                                            # If position is questionable, require higher confidence (0.80 instead of 0.75)
                                            if conf < 0.80:
                                                helmet_position_valid = False
                                            else:
                                                helmet_position_valid = True  # High confidence overrides position
                                        else:
                                            helmet_position_valid = True  # Position is good
                                    
                                    break
                        except:
                            # If spatial validation fails, allow detection if confidence is very high
                            is_near_person = conf >= 0.85
                            if ppe_item == "helmet":
                                helmet_position_valid = conf >= 0.90  # Even stricter for helmets
                    
                    # Size validation: check if detection box is reasonable
                    box_valid = True
                    if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                        try:
                            box_tensor = boxes.xyxy[i]
                            if hasattr(box_tensor, 'cpu'):
                                ppe_box = box_tensor.cpu().numpy().tolist()
                            else:
                                ppe_box = box_tensor.tolist() if hasattr(box_tensor, 'tolist') else list(box_tensor)
                            
                            box_width = ppe_box[2] - ppe_box[0]
                            box_height = ppe_box[3] - ppe_box[1]
                            box_area = box_width * box_height
                            frame_area = frame_width * frame_height
                            
                            # Box should be at least 0.1% of frame and at most 50% of frame
                            if box_area < frame_area * 0.001 or box_area > frame_area * 0.5:
                                box_valid = False
                        except:
                            pass
                    
                    # Get box coordinates for validation and storage
                    box_coords = None
                    if hasattr(boxes, 'xyxy') and boxes.xyxy is not None:
                        try:
                            box_tensor = boxes.xyxy[i]
                            if hasattr(box_tensor, 'cpu'):
                                box_coords = box_tensor.cpu().numpy().tolist()
                            else:
                                box_coords = box_tensor.tolist() if hasattr(box_tensor, 'tolist') else list(box_tensor)
                        except Exception as e:
                            print(f"[PPE] Error extracting box coordinates: {e}")
                            box_coords = None
                    
                    # For image analysis, be more lenient (don't require person detection)
                    # For video, use strict spatial validation
                    if strict_spatial_validation:
                        # Strict: require near person OR high confidence
                        # For helmets, also check position but be more lenient
                        if ppe_item == "helmet":
                            # Helmet must be near person AND (in correct position OR high confidence >= 0.80)
                            # OR have very high confidence (>= 0.85) regardless of position
                            # This allows high-confidence detections even if position is slightly off
                            accept_condition = box_valid and ((is_near_person and (helmet_position_valid or conf >= 0.80)) or conf >= 0.85)
                        else:
                            accept_condition = box_valid and (is_near_person or conf >= 0.75)
                    else:
                        # Lenient: accept if meets threshold and valid box (for single images)
                        # For helmets, use threshold but allow slightly lower if position is good
                        if ppe_item == "helmet":
                            accept_condition = box_valid and (conf >= item_threshold or (helmet_position_valid and conf >= item_threshold * 0.95))
                        else:
                            accept_condition = box_valid and conf >= (item_threshold * 0.9)  # 10% lower for other items
                    
                    # Only accept if conditions are met
                    if accept_condition and box_coords:
                        current_conf = detections[ppe_item][1]
                        if conf > current_conf:
                            detections[ppe_item] = (True, conf)
                            # Store only valid detections for visualization
                            detections["all_detections"].append({
                                "class": class_name,
                                "ppe_item": ppe_item,
                                "confidence": conf,
                                "box": box_coords
                            })
        
        return detections
        
    except Exception as e:
        print(f"[PPE ERROR] Detection failed: {e}")
        return {
            "helmet": (False, 0.0),
            "gloves": (False, 0.0),
            "boots": (False, 0.0),
            "jacket": (False, 0.0),
            "all_detections": [],
        }


def check_ppe_compliance(detections, required_items=None, confidence_threshold=PPE_CONFIDENCE_THRESHOLD):
    """
    Check if all required PPE items are detected.
    
    Args:
        detections: Output from detect_ppe_items()
        required_items: Dict of required items (default: REQUIRED_PPE)
        confidence_threshold: Minimum confidence to consider item detected
    
    Returns:
        tuple: (is_compliant: bool, missing_items: list, compliance_dict: dict)
    """
    if required_items is None:
        required_items = REQUIRED_PPE
    
    compliance_dict = {}
    missing_items = []
    
    for item, required in required_items.items():
        if required:
            detected, confidence = detections.get(item, (False, 0.0))
            
            # Check if detected with sufficient confidence
            is_detected = bool(detected and confidence >= confidence_threshold)  # Convert to Python bool
            compliance_dict[item] = {
                "detected": is_detected,
                "confidence": float(confidence)  # Ensure float for JSON
            }
            
            if not is_detected:
                missing_items.append(item)
    
    is_compliant = bool(len(missing_items) == 0)  # Convert to Python bool for JSON serialization
    
    return is_compliant, missing_items, compliance_dict


def get_ppe_status_string(detections, compliance_dict):
    """Get a human-readable string of PPE status."""
    status_parts = []
    
    for item in ["helmet", "gloves", "boots", "jacket"]:
        if item in compliance_dict:
            detected = compliance_dict[item]["detected"]
            conf = compliance_dict[item]["confidence"]
            status = "✓" if detected else "✗"
            status_parts.append(f"{status} {item.capitalize()}({conf:.2f})")
    
    return " | ".join(status_parts)


def draw_annotated_image(frame, ppe_detections, face_results=None, use_display_thresholds=True):
    """
    Draw annotated image with PPE boxes and person labels.
    
    Args:
        frame: Input frame (BGR format)
        ppe_detections: Output from detect_ppe_items()
        face_results: List of dicts with face recognition results:
            [{"name": str, "confidence": float, "bbox": [x1, y1, x2, y2], "is_real": bool}]
        use_display_thresholds: If True, uses higher thresholds for display (reduces false positives in UI)
    
    Returns:
        Annotated frame (BGR format)
    """
    annotated_frame = frame.copy()
    frame_height, frame_width = annotated_frame.shape[:2]
    
    # Color mapping for each PPE item
    ppe_colors = {
        "helmet": (0, 255, 0),      # Green
        "gloves": (0, 255, 255),    # Yellow/Cyan
        "boots": (255, 255, 0),     # Cyan
        "jacket": (255, 0, 0),      # Blue
        "vest": (255, 0, 0),        # Blue (same as jacket)
    }
    
    # Collect PPE detection summary for top display
    ppe_summary = []
    
    # Draw PPE detection boxes (without labels above boxes)
    if ppe_detections and "all_detections" in ppe_detections:
        thresholds = PPE_DISPLAY_THRESHOLDS if use_display_thresholds else PPE_CLASS_THRESHOLDS
        
        # Track best detection for each PPE type
        ppe_best_detections = {}
        
        for detection in ppe_detections["all_detections"]:
            ppe_item = detection.get("ppe_item")
            box = detection.get("box")
            conf = detection.get("confidence", 0.0)
            class_name = detection.get("class", "")
            
            if not ppe_item or not box:
                continue
            
            # Get threshold for this item
            item_threshold = thresholds.get(ppe_item, PPE_CONFIDENCE_THRESHOLD)
            
            # Only process if above threshold
            if ppe_item in ["helmet", "gloves", "boots", "jacket"] and conf >= item_threshold:
                # Keep track of best confidence for each item
                if ppe_item not in ppe_best_detections or conf > ppe_best_detections[ppe_item]["confidence"]:
                    ppe_best_detections[ppe_item] = {
                        "box": box,
                        "confidence": conf,
                        "color": ppe_colors.get(ppe_item, (255, 255, 255))
                    }
        
        # Draw boxes and collect summary
        for ppe_item, detection_data in ppe_best_detections.items():
            x1, y1, x2, y2 = map(int, detection_data["box"])
            color = detection_data["color"]
            conf = detection_data["confidence"]
            
            # Draw bounding box only (no labels)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Add to summary for top display (convert to percentage)
            ppe_summary.append(f"{ppe_item.capitalize()}: {conf*100:.1f}%")
    
    # Draw PPE summary text at the top of the image (each item on separate line)
    if ppe_summary:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # Very small font
        thickness = 1
        line_spacing = 15
        
        # Calculate maximum width needed
        max_width = 0
        total_height = 0
        for item_text in ppe_summary:
            (text_width, text_height), baseline = cv2.getTextSize(item_text, font, font_scale, thickness)
            max_width = max(max_width, text_width)
            total_height += text_height + line_spacing
        
        # Position at top left with padding
        text_x = 10
        start_y = 15
        
        # Draw semi-transparent background rectangle
        overlay = annotated_frame.copy()
        bg_padding = 5
        cv2.rectangle(overlay, 
                     (bg_padding, bg_padding), 
                     (max_width + bg_padding * 2, total_height + bg_padding * 2), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        # Draw each item on a separate line
        current_y = start_y
        for item_text in ppe_summary:
            (text_width, text_height), baseline = cv2.getTextSize(item_text, font, font_scale, thickness)
            cv2.putText(annotated_frame, item_text, (text_x, current_y), 
                      font, font_scale, (255, 255, 255), thickness)
            current_y += text_height + line_spacing
    
    # Draw face recognition boxes and labels
    if face_results:
        # Track label positions to avoid overlaps
        used_label_positions = []
        label_spacing = 30  # Minimum spacing between labels
        
        for result in face_results:
            name = result.get("name", "Unknown")
            confidence = result.get("confidence", 0.0)
            bbox = result.get("bbox", [])
            is_real = result.get("is_real", True)
            
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on recognition status
            if not is_real:
                color = (0, 165, 255)  # Orange for spoof
                name = "Spoof Detected"
            elif name.startswith("Unknown"):
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for recognized
            
            # Draw face bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label - position outside the box to avoid covering face
            label = f"{name} ({confidence:.2f})"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_height = label_size[1] + baseline
            label_width = label_size[0]
            
            frame_height, frame_width = annotated_frame.shape[:2]
            
            # Find a position that doesn't overlap with other labels
            label_x = None
            label_y = None
            
            # Try positions in order: above, right, left, below
            positions_to_try = [
                (x1, y1 - 10, "above"),  # Above box
                (x2 + 10, y1 + label_height, "right"),  # Right of box
                (x1 - label_width - 10, y1 + label_height, "left"),  # Left of box
                (x1, y2 + label_height + 10, "below")  # Below box
            ]
            
            for pos_x, pos_y, position_name in positions_to_try:
                # Check bounds
                if pos_x < 0 or pos_x + label_width > frame_width:
                    continue
                if pos_y < label_height or pos_y > frame_height:
                    continue
                
                # Check if this position overlaps with any existing label
                overlaps = False
                for used_x, used_y, used_w, used_h in used_label_positions:
                    # Check if rectangles overlap
                    if not (pos_x + label_width < used_x or pos_x > used_x + used_w or
                           pos_y - label_height > used_y or pos_y < used_y - used_h):
                        overlaps = True
                        break
                
                if not overlaps:
                    label_x = pos_x
                    label_y = pos_y
                    break
            
            # If no non-overlapping position found, use a position far from others
            if label_x is None:
                # Find a position with maximum distance from other labels
                best_x = x1
                best_y = y1 - 10 if y1 > label_height + 15 else y2 + label_height + 10
                max_min_distance = 0
                
                # Try positions around the face box
                for offset_x in range(-label_width - 50, label_width + 50, 20):
                    for offset_y in range(-label_height - 50, label_height + 50, 20):
                        test_x = x1 + offset_x
                        test_y = y1 + offset_y
                        
                        if test_x < 0 or test_x + label_width > frame_width:
                            continue
                        if test_y < label_height or test_y > frame_height:
                            continue
                        
                        # Calculate minimum distance to all used positions
                        min_distance = float('inf')
                        for used_x, used_y, used_w, used_h in used_label_positions:
                            used_center_x = used_x + used_w / 2
                            used_center_y = used_y - used_h / 2
                            distance = ((test_x + label_width/2 - used_center_x)**2 + 
                                       (test_y - label_height/2 - used_center_y)**2)**0.5
                            min_distance = min(min_distance, distance)
                        
                        if min_distance > max_min_distance:
                            max_min_distance = min_distance
                            best_x = test_x
                            best_y = test_y
                
                label_x = max(0, min(best_x, frame_width - label_width))
                label_y = max(label_height, min(best_y, frame_height))
            
            # Record this label position
            used_label_positions.append((label_x, label_y, label_width, label_height))
            
            # Draw background rectangle for text
            bg_x1 = label_x - 5
            bg_y1 = label_y - label_height - 5
            bg_x2 = label_x + label_width + 5
            bg_y2 = label_y + 5
            
            # Make sure background doesn't go outside image bounds
            bg_x1 = max(0, bg_x1)
            bg_y1 = max(0, bg_y1)
            bg_x2 = min(frame_width, bg_x2)
            bg_y2 = min(frame_height, bg_y2)
            
            cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, label, (label_x, label_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated_frame


# Initialize model on import (lazy loading - only loads when first used)
def initialize_ppe_model():
    """Initialize PPE model. Call this before first use."""
    if ppe_model is None:
        load_ppe_model()
    return ppe_model is not None

