"""
Improved PPE Model Training Script with YOLOv12
Addresses issues with false positives by:
- Using YOLOv12 (latest, 1.2% better mAP than YOLOv11)
- Extended epochs (200 instead of 50) - with early stopping
- Better data augmentation
- Higher image resolution
- Better learning rate schedule
- Early stopping prevents overfitting
"""

from ultralytics import YOLO
import os
import torch

# Path to dataset YAML file
dataset_yaml = os.path.join(os.path.dirname(__file__), "datasets", "data.yaml")

print("=" * 60)
print("Improved PPE Model Training")
print("=" * 60)
print(f"\nDataset YAML: {dataset_yaml}")
print("\nImprovements:")
print("  - More epochs: 200 (vs 50) - with early stopping")
print("  - Higher resolution: 640x640")
print("  - Better augmentation")
print("  - Optimized for reducing false positives")
print("  - Early stopping: Stops automatically if no improvement")
print("\nThis will take 3-5 hours on CPU, 1.5-3 hours on GPU...")
print("(Early stopping may finish earlier if model plateaus)")
print("Press Ctrl+C to stop training early\n")

try:
    # Load pre-trained YOLOv12 nano model (latest, better accuracy)
    print("[1/3] Loading YOLOv12n model...")
    print("   YOLOv12 has 1.2% better mAP than YOLOv11")
    model = YOLO("yolo12n.pt")
    print("✅ Model loaded\n")
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   - Device: {device}")
    if device == 'cpu':
        print("   ⚠️  Using CPU - training will be slower.")
        print("   💡 Consider using Google Colab for faster training!")
    print()
    
    # Train the model with improved parameters
    print("[2/3] Starting improved training...")
    print("   - Epochs: 200 (extended training for maximum accuracy)")
    print("   - Image size: 640")
    print("   - Batch size: 16 (reduced for stability)")
    print("   - Learning rate: 0.001 (lower for fine-tuning)")
    print("   - Augmentation: Enhanced")
    print("   - Early stopping: 40 epochs patience (stops if no improvement)")
    print("\nTraining progress:\n")
    
    results = model.train(
        data=dataset_yaml,
        epochs=200,  # Extended training - early stopping will prevent overfitting
        imgsz=640,
        batch=16,  # Smaller batch for stability
        device=device,
        project="runs/detect",
        name="ppe_detection_improved",
        exist_ok=True,
        patience=40,  # Early stopping patience - stops if no improvement for 40 epochs
        save=True,
        plots=True,
        workers=4,
        cache=True,
        amp=True,
        # Improved augmentation to reduce overfitting
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,  # Slight rotation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,  # Mixup augmentation
        copy_paste=0.1,  # Copy-paste augmentation
        # Learning rate
        lr0=0.001,  # Lower initial learning rate
        lrf=0.01,  # Lower final learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )
    
    print("\n" + "=" * 60)
    print("[3/3] Training Complete!")
    print("=" * 60)
    print(f"\n✅ Model saved to: runs/detect/ppe_detection_improved/weights/best.pt")
    print(f"✅ Model saved to: runs/detect/ppe_detection_improved/weights/last.pt")
    print("\nTo use the improved model, update videoServer.py:")
    print("   TRAINED_PPE_MODEL_PATH = 'runs/detect/ppe_detection_improved/weights/best.pt'")
    print("\n" + "=" * 60)
    
except KeyboardInterrupt:
    print("\n\n⚠️ Training interrupted by user")
    print("Partial model may be saved in runs/detect/ppe_detection_improved/weights/")
except Exception as e:
    print(f"\n\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()

