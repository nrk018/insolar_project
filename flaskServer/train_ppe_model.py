"""
Train YOLOv11 model on Construction-PPE dataset
"""

from ultralytics import YOLO
import os

# Path to dataset YAML file
dataset_yaml = os.path.join(os.path.dirname(__file__), "datasets", "data.yaml")

print("=" * 60)
print("Training YOLOv11 on Construction-PPE Dataset")
print("=" * 60)
print(f"\nDataset YAML: {dataset_yaml}")
print("\nThis will take some time (1-3 hours on CPU, 30-60 min on GPU)...")
print("Press Ctrl+C to stop training early\n")

try:
    # Load pre-trained YOLOv11 nano model (smallest, fastest)
    print("[1/3] Loading YOLOv11n model...")
    model = YOLO("yolo11n.pt")
    print("✅ Model loaded\n")
    
    # Train the model
    print("[2/3] Starting training...")
    print("   - Epochs: 50 (reduced for faster training)")
    print("   - Image size: 640")
    print("   - Batch size: 32 (larger for speed)")
    print("\nTraining progress:\n")
    
    # Check for GPU
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   - Device: {device}")
    if device == 'cpu':
        print("   ⚠️  Using CPU - training will be slower. Consider using GPU for faster training.")
    print()
    
    results = model.train(
        data=dataset_yaml,
        epochs=50,  # Reduced from 100 for faster training
        imgsz=640,
        batch=32,  # Larger batch for faster training
        device=device,
        project="runs/detect",
        name="ppe_detection",
        exist_ok=True,
        patience=20,  # Early stopping - stop if no improvement for 20 epochs
        save=True,
        plots=True,
        workers=4,  # Parallel data loading
        cache=True,  # Cache images in RAM for faster training
        amp=True,  # Automatic Mixed Precision for faster training
    )
    
    print("\n" + "=" * 60)
    print("[3/3] Training Complete!")
    print("=" * 60)
    print(f"\n✅ Model saved to: runs/detect/ppe_detection/weights/best.pt")
    print(f"✅ Model saved to: runs/detect/ppe_detection/weights/last.pt")
    print("\nTo use the trained model, update faceRecognition.py:")
    print("   ppe_model = load_ppe_model(model_path='runs/detect/ppe_detection/weights/best.pt')")
    print("\n" + "=" * 60)
    
except KeyboardInterrupt:
    print("\n\n⚠️ Training interrupted by user")
    print("Partial model may be saved in runs/detect/ppe_detection/weights/")
except Exception as e:
    print(f"\n\n❌ Training error: {e}")
    import traceback
    traceback.print_exc()

