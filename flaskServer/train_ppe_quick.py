"""
Quick training script - 10 epochs only for fast testing
Or use pre-trained model directly without training
"""

from ultralytics import YOLO
import os

# Path to dataset YAML file
dataset_yaml = os.path.join(os.path.dirname(__file__), "datasets", "data.yaml")

print("=" * 60)
print("Quick PPE Model Training (10 epochs only)")
print("=" * 60)
print("\nOPTION 1: Use pre-trained YOLOv11 directly (RECOMMENDED)")
print("   - No training needed")
print("   - Works immediately")
print("   - Good accuracy for general PPE detection")
print("\nOPTION 2: Quick fine-tune (10 epochs, ~10-15 minutes)")
print("   - Better accuracy for construction PPE")
print("   - Still fast enough")
print("\n" + "=" * 60)

choice = input("\nChoose: (1) Use pre-trained [default] or (2) Quick train? [1/2]: ").strip()

if choice == "2":
    try:
        print("\n[1/3] Loading YOLOv11n model...")
        model = YOLO("yolo11n.pt")
        print("✅ Model loaded\n")
        
        print("[2/3] Starting QUICK training (10 epochs only)...")
        print("   - Epochs: 10 (very fast)")
        print("   - Image size: 640")
        print("   - Batch size: 32")
        
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   - Device: {device}\n")
        
        results = model.train(
            data=dataset_yaml,
            epochs=10,  # Very few epochs for speed
            imgsz=640,
            batch=32,
            device=device,
            project="runs/detect",
            name="ppe_detection_quick",
            exist_ok=True,
            patience=5,  # Early stopping
            save=True,
            workers=4,
            cache=True,
            amp=True,
        )
        
        model_path = "runs/detect/ppe_detection_quick/weights/best.pt"
        print(f"\n✅ Quick training complete!")
        print(f"✅ Model saved to: {model_path}")
        print(f"\nUpdate faceRecognition.py to use:")
        print(f"   ppe_model = load_ppe_model(model_path='{model_path}')")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n✅ Using pre-trained YOLOv11 model (no training needed)")
    print("\nThe system will automatically use the pre-trained model.")
    print("It's already configured in ppeDetection.py - just run faceRecognition.py!")
    print("\nIf you want better accuracy later, you can:")
    print("  1. Run this script again and choose option 2")
    print("  2. Or run the full training: python train_ppe_model.py")

print("\n" + "=" * 60)

