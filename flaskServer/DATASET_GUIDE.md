# Guide to Improve Gloves & Boots Detection

## Current Model Performance
- **Gloves**: 65.88% accuracy (68.71% recall - missing 31% of gloves)
- **Boots**: 58.72% accuracy (65.40% recall - missing 35% of boots)

## Recommended Datasets

### 1. Roboflow Universe (Easiest to Use) ⭐

#### Option A: PPE Detection by Mohamed Traore
- **URL**: https://universe.roboflow.com/mohamed-traore-2ekkp/ppe-detection-l80fg
- **Includes**: Gloves, shoes, helmets, goggles, masks
- **Format**: YOLO-ready
- **Steps**:
  1. Sign up for free Roboflow account
  2. Go to the URL above
  3. Click "Download" → Select "YOLO" format
  4. Download Train/Val/Test splits
  5. Extract to `external_datasets/roboflow_ppe_traore/`
  6. Run: `python merge_ppe_datasets.py --dataset external_datasets/roboflow_ppe_traore`

#### Option B: PPE Dataset for Workplace Safety by SiaBar
- **URL**: https://universe.roboflow.com/siabar/ppe-dataset-for-workplace
- **Includes**: Boots, gloves, helmets
- **Format**: YOLO-ready
- **Steps**: Same as Option A

### 2. SH17 Dataset (Largest Dataset) ⭐⭐⭐
- **Paper**: https://arxiv.org/abs/2407.04590
- **Size**: 8,099 images, 75,994 instances
- **Classes**: 17 classes including gloves and safety shoes
- **How to get**: 
  - Check the paper for download links
  - May require contacting authors or checking GitHub
- **Best for**: Large-scale improvement

### 3. Construction-PPE Dataset (Ultralytics)
- **Source**: Ultralytics documentation
- **Includes**: Helmets, vests, gloves, boots, goggles
- **How to get**: Check Ultralytics docs or GitHub repositories

## Quick Start: Using Roboflow Dataset

### Step 1: Download Dataset
```bash
# Go to Roboflow and download one of these:
# 1. https://universe.roboflow.com/mohamed-traore-2ekkp/ppe-detection-l80fg
# 2. https://universe.roboflow.com/siabar/ppe-dataset-for-workplace

# Extract to:
mkdir -p external_datasets
# Extract downloaded ZIP to external_datasets/roboflow_ppe/
```

### Step 2: Merge with Existing Dataset
```bash
cd InsolareSafetySystem/flaskServer
python merge_ppe_datasets.py --dataset external_datasets/roboflow_ppe
```

### Step 3: Update data.yaml
Make sure your `datasets/data.yaml` has the correct class mappings.

### Step 4: Retrain Model
```bash
# Using improved training script
python train_ppe_model_improved.py

# Or use Google Colab (recommended for GPU)
# Upload train_ppe_colab.ipynb to Colab
```

## Expected Improvements

After merging additional datasets:
- **Gloves**: Expected to improve from 65.88% → 75-80% accuracy
- **Boots**: Expected to improve from 58.72% → 70-75% accuracy

## Tips

1. **Focus on Train Split**: Merge external datasets into your train split
2. **Keep Validation Separate**: Don't merge external validation sets to avoid data leakage
3. **Class Mapping**: Adjust `merge_ppe_datasets.py` if class names differ
4. **Balance Classes**: Ensure you have enough images for gloves and boots
5. **Augmentation**: Use data augmentation to further improve performance

## Manual Dataset Addition

If you have your own images:
1. Annotate using LabelImg or Roboflow
2. Save in YOLO format
3. Place in `datasets/images/train/` and `datasets/labels/train/`
4. Update `datasets/data.yaml` if needed
5. Retrain the model

## Next Steps

1. Download at least one Roboflow dataset
2. Merge with existing dataset
3. Retrain with YOLOv12 (200 epochs)
4. Re-evaluate accuracy
5. Iterate if needed

