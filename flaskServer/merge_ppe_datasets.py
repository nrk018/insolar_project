"""
Script to merge additional PPE datasets with existing Construction-PPE dataset
Focuses on improving gloves and boots detection
"""

import os
import shutil
from pathlib import Path
import yaml

# Paths
BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# Class mappings - adjust based on your dataset format
CLASS_MAPPINGS = {
    # Your current classes
    "helmet": 0,
    "gloves": 1,
    "vest": 2,  # Also maps to jacket
    "boots": 3,
    "goggles": 4,
    "none": 5,
    "Person": 6,
    "no_helmet": 7,
    "no_goggle": 8,
    "no_gloves": 9,
    "no_boots": 10,
    
    # Common variations in other datasets
    "safety_helmet": 0,
    "hard_hat": 0,
    "hardhat": 0,
    "glove": 1,
    "safety_gloves": 1,
    "safety_shoes": 3,
    "safety_boots": 3,
    "boot": 3,
    "shoe": 3,
    "safety_vest": 2,
    "vest": 2,
    "jacket": 2,
    "reflective_vest": 2,
}


def map_class_name(class_name):
    """Map class name from external dataset to our class ID."""
    class_name_lower = class_name.lower().replace("_", "-").replace(" ", "-")
    
    for key, class_id in CLASS_MAPPINGS.items():
        if key.lower() == class_name_lower or key.lower() in class_name_lower:
            return class_id
    
    return None  # Unknown class, skip


def convert_yolo_label(label_path, output_path, class_mapping_func):
    """Convert YOLO label file with class mapping."""
    if not os.path.exists(label_path):
        return False
    
    converted_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                old_class_id = int(parts[0])
                # If dataset uses class names instead of IDs, you'll need to handle that
                # For now, assuming numeric IDs
                # You may need to adjust based on the dataset format
                converted_lines.append(line)
    
    if converted_lines:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(converted_lines))
        return True
    return False


def merge_dataset(external_dataset_path, split='train', focus_classes=['gloves', 'boots']):
    """
    Merge external dataset with existing dataset.
    
    Args:
        external_dataset_path: Path to external dataset (should have images/ and labels/ folders)
        split: 'train', 'val', or 'test'
        focus_classes: List of classes to prioritize (e.g., ['gloves', 'boots'])
    """
    external_images_dir = os.path.join(external_dataset_path, "images", split)
    external_labels_dir = os.path.join(external_dataset_path, "labels", split)
    
    if not os.path.exists(external_images_dir):
        print(f"❌ External dataset images not found: {external_images_dir}")
        return False
    
    target_images_dir = os.path.join(IMAGES_DIR, split)
    target_labels_dir = os.path.join(LABELS_DIR, split)
    
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)
    
    # Get existing image count to avoid conflicts
    existing_images = set(os.listdir(target_images_dir)) if os.path.exists(target_images_dir) else set()
    
    copied_count = 0
    skipped_count = 0
    
    print(f"\n[INFO] Merging {split} split from: {external_dataset_path}")
    print(f"[INFO] Focus classes: {focus_classes}")
    
    for img_file in os.listdir(external_images_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # Check if image already exists
        if img_file in existing_images:
            skipped_count += 1
            continue
        
        img_path = os.path.join(external_images_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(external_labels_dir, label_file)
        
        # Check if label file exists and contains focus classes
        has_focus_class = False
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Check if this is a focus class (you may need to adjust based on dataset)
                        # For now, copy all images that have labels
                        has_focus_class = True
                        break
        
        # Copy image
        target_img_path = os.path.join(target_images_dir, img_file)
        shutil.copy2(img_path, target_img_path)
        
        # Copy and convert label if exists
        if os.path.exists(label_path):
            target_label_path = os.path.join(target_labels_dir, label_file)
            convert_yolo_label(label_path, target_label_path, map_class_name)
            copied_count += 1
        else:
            skipped_count += 1
    
    print(f"✅ Copied {copied_count} images with labels")
    print(f"⚠️  Skipped {skipped_count} images (duplicates or no labels)")
    
    return True


def download_roboflow_dataset(dataset_name, api_key=None):
    """
    Instructions for downloading Roboflow dataset.
    
    Args:
        dataset_name: Name of the dataset on Roboflow Universe
        api_key: Optional Roboflow API key
    """
    print("\n" + "="*80)
    print("HOW TO DOWNLOAD ROBOFLOW DATASET")
    print("="*80)
    print(f"\n1. Go to: https://universe.roboflow.com/{dataset_name}")
    print("2. Sign up for free Roboflow account (if needed)")
    print("3. Click 'Download' button")
    print("4. Select format: 'YOLO'")
    print("5. Select split: 'Train', 'Valid', 'Test' (or 'Train/Val/Test')")
    print("6. Download the ZIP file")
    print("7. Extract to a folder (e.g., 'external_datasets/roboflow_ppe')")
    print("8. Run this script to merge:")
    print(f"   python merge_ppe_datasets.py --dataset external_datasets/roboflow_ppe")
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge external PPE datasets")
    parser.add_argument("--dataset", type=str, help="Path to external dataset folder")
    parser.add_argument("--split", type=str, default="train", choices=['train', 'val', 'test'],
                       help="Dataset split to merge")
    parser.add_argument("--focus", nargs='+', default=['gloves', 'boots'],
                       help="Classes to focus on (default: gloves boots)")
    parser.add_argument("--roboflow", type=str, help="Roboflow dataset name (shows download instructions)")
    
    args = parser.parse_args()
    
    if args.roboflow:
        download_roboflow_dataset(args.roboflow)
    elif args.dataset:
        if os.path.exists(args.dataset):
            merge_dataset(args.dataset, args.split, args.focus)
        else:
            print(f"❌ Dataset path not found: {args.dataset}")
    else:
        print("="*80)
        print("PPE Dataset Merger - Improve Gloves & Boots Detection")
        print("="*80)
        print("\nUsage:")
        print("  python merge_ppe_datasets.py --dataset <path_to_dataset>")
        print("  python merge_ppe_datasets.py --roboflow <dataset_name>")
        print("\nExample:")
        print("  python merge_ppe_datasets.py --dataset external_datasets/roboflow_ppe")
        print("  python merge_ppe_datasets.py --roboflow mohamed-traore-2ekkp/ppe-detection-l80fg")
        print("\nRecommended datasets:")
        print("  1. Roboflow: mohamed-traore-2ekkp/ppe-detection-l80fg")
        print("  2. Roboflow: siabar/ppe-dataset-for-workplace")
        print("  3. SH17 Dataset (from arXiv paper)")

