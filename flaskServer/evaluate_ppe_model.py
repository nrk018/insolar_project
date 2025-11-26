"""
Evaluate PPE Model Accuracy
Tests the trained model on test dataset and calculates accuracy per PPE item
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import json
from collections import defaultdict

# Paths
DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "images", "test")
TEST_LABELS_DIR = os.path.join(DATASET_DIR, "labels", "test")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "runs", "detect", "ppe_detection", "weights", "best.pt")

# PPE class mappings (from data.yaml)
PPE_CLASSES = {
    0: "helmet",
    1: "gloves",
    2: "vest",  # Also maps to "jacket" in our system
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots"
}

# Our PPE items of interest
OUR_PPE_ITEMS = {
    "helmet": [0],  # helmet
    "gloves": [1],  # gloves
    "boots": [3],   # boots
    "jacket": [2],  # vest
}

CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


def parse_yolo_label(label_path):
    """Parse YOLO format label file."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
    return boxes


def yolo_to_xyxy(box, img_width, img_height):
    """Convert YOLO format (center, width, height) to (x1, y1, x2, y2)."""
    x_center = box['x_center'] * img_width
    y_center = box['y_center'] * img_height
    width = box['width'] * img_width
    height = box['height'] * img_height
    
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    
    return [x1, y1, x2, y2]


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def match_detections(ground_truth, predictions, iou_threshold=IOU_THRESHOLD):
    """Match ground truth boxes with predictions based on IoU."""
    matched = []
    used_pred = set()
    
    for gt_box in ground_truth:
        best_iou = 0
        best_pred_idx = -1
        
        for pred_idx, pred_box in enumerate(predictions):
            if pred_idx in used_pred:
                continue
            
            if gt_box['class_id'] == pred_box['class_id']:
                iou = calculate_iou(gt_box['bbox'], pred_box['bbox'])
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = pred_idx
        
        if best_pred_idx >= 0:
            matched.append({
                'gt': gt_box,
                'pred': predictions[best_pred_idx],
                'iou': best_iou
            })
            used_pred.add(best_pred_idx)
        else:
            matched.append({
                'gt': gt_box,
                'pred': None,
                'iou': 0
            })
    
    # Unmatched predictions (false positives)
    for pred_idx, pred_box in enumerate(predictions):
        if pred_idx not in used_pred:
            matched.append({
                'gt': None,
                'pred': pred_box,
                'iou': 0
            })
    
    return matched


def evaluate_model(model_path, test_images_dir, test_labels_dir):
    """Evaluate the trained model on test dataset."""
    
    print("=" * 80)
    print("PPE Model Evaluation")
    print("=" * 80)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using train_ppe_model.py")
        return None
    
    print(f"\n[1/4] Loading model from: {model_path}")
    model = YOLO(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Model loaded on device: {device}\n")
    
    # Get test images
    print(f"[2/4] Loading test dataset from: {test_images_dir}")
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(Path(test_images_dir).glob(ext))
    
    if not test_images:
        print(f"❌ No test images found in {test_images_dir}")
        return None
    
    print(f"✅ Found {len(test_images)} test images\n")
    
    # Statistics per PPE item
    stats = {
        'helmet': {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0},
        'gloves': {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0},
        'boots': {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0},
        'jacket': {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0},
    }
    
    print(f"[3/4] Evaluating on {len(test_images)} images...")
    print("Progress: ", end="", flush=True)
    
    for idx, img_path in enumerate(test_images):
        if (idx + 1) % 20 == 0:
            print(f"{idx + 1}/{len(test_images)}... ", end="", flush=True)
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        img_height, img_width = img.shape[:2]
        
        # Load ground truth
        label_path = os.path.join(test_labels_dir, img_path.stem + '.txt')
        gt_boxes = parse_yolo_label(label_path)
        
        # Convert GT boxes to xyxy format
        gt_boxes_xyxy = []
        for box in gt_boxes:
            bbox = yolo_to_xyxy(box, img_width, img_height)
            gt_boxes_xyxy.append({
                'class_id': box['class_id'],
                'bbox': bbox
            })
        
        # Run model prediction
        results = model(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Parse predictions
        pred_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                pred_boxes.append({
                    'class_id': cls,
                    'bbox': bbox,
                    'confidence': conf
                })
        
        # Match predictions with ground truth for each PPE item
        for ppe_item, class_ids in OUR_PPE_ITEMS.items():
            # Filter GT boxes for this PPE item
            gt_for_item = [b for b in gt_boxes_xyxy if b['class_id'] in class_ids]
            pred_for_item = [b for b in pred_boxes if b['class_id'] in class_ids]
            
            stats[ppe_item]['total_gt'] += len(gt_for_item)
            
            if len(gt_for_item) == 0 and len(pred_for_item) == 0:
                continue  # No GT and no predictions - skip
            
            # Match detections
            matched = match_detections(gt_for_item, pred_for_item, IOU_THRESHOLD)
            
            for match in matched:
                if match['gt'] is not None and match['pred'] is not None:
                    # True Positive
                    stats[ppe_item]['tp'] += 1
                elif match['gt'] is not None and match['pred'] is None:
                    # False Negative (missed detection)
                    stats[ppe_item]['fn'] += 1
                elif match['gt'] is None and match['pred'] is not None:
                    # False Positive (wrong detection)
                    stats[ppe_item]['fp'] += 1
    
    print(f"\n✅ Evaluation complete!\n")
    
    # Calculate metrics
    print(f"[4/4] Calculating metrics...\n")
    print("=" * 80)
    print("ACCURACY RESULTS PER PPE ITEM")
    print("=" * 80)
    print(f"{'PPE Item':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}")
    print("-" * 80)
    
    results = {}
    for ppe_item, stat in stats.items():
        tp = stat['tp']
        fp = stat['fp']
        fn = stat['fn']
        total_gt = stat['total_gt']
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy = TP / (TP + FP + FN) - percentage of correct detections
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        
        results[ppe_item] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_gt': total_gt
        }
        
        print(f"{ppe_item.capitalize():<15} {precision*100:>10.2f}% {recall*100:>10.2f}% {f1*100:>10.2f}% {accuracy*100:>10.2f}%")
    
    print("=" * 80)
    print(f"\nDetailed Statistics:")
    print("-" * 80)
    for ppe_item, metrics in results.items():
        print(f"\n{ppe_item.upper()}:")
        print(f"  True Positives (TP):  {metrics['tp']}")
        print(f"  False Positives (FP): {metrics['fp']}")
        print(f"  False Negatives (FN): {metrics['fn']}")
        print(f"  Total Ground Truth:   {metrics['total_gt']}")
        print(f"  Precision:            {metrics['precision']*100:.2f}%")
        print(f"  Recall:               {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:             {metrics['f1_score']*100:.2f}%")
        print(f"  Accuracy:             {metrics['accuracy']*100:.2f}%")
    
    # Save results to JSON
    results_file = os.path.join(os.path.dirname(__file__), "ppe_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = evaluate_model(MODEL_PATH, TEST_IMAGES_DIR, TEST_LABELS_DIR)
    if results:
        print("\n" + "=" * 80)
        print("Evaluation Complete!")
        print("=" * 80)



