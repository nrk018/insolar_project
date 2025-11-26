# Quick Start: Train PPE Model on Google Colab

## 🚀 5-Minute Setup

### Step 1: Prepare Dataset (on your Mac)
```bash
cd /Users/storage/projects/update/InsolareSafetySystem/flaskServer
./prepare_dataset_for_colab.sh
```

This creates `ppe_dataset.zip` (~100-200 MB)

### Step 2: Open Google Colab
1. Go to: https://colab.research.google.com/
2. Sign in with Google account
3. Click **File → Upload notebook**
4. Upload: `train_ppe_colab.ipynb`

### Step 3: Enable GPU
1. Click **Runtime → Change runtime type**
2. Set **Hardware accelerator** → **GPU**
3. Click **Save**

### Step 4: Run Training
1. Run all cells in the notebook (click ▶️ on each cell)
2. When prompted, upload `ppe_dataset.zip`
3. Wait 30-60 minutes for training
4. Download `best.pt` when done

### Step 5: Install Model (on your Mac)
```bash
# Copy downloaded best.pt to:
cp ~/Downloads/best.pt \
   /Users/storage/projects/update/InsolareSafetySystem/flaskServer/runs/detect/ppe_detection/weights/best.pt

# Restart server
cd /Users/storage/projects/update/InsolareSafetySystem/flaskServer
python videoServer.py
```

## ✅ Done!

Your model is now trained and ready to detect:
- ✅ Helmet
- ✅ Gloves  
- ✅ Boots
- ✅ Vest/Jacket
- ✅ Person

## 📊 Expected Results

After training, you should see:
- **Model accuracy**: 80-90%+
- **File size**: ~15-20 MB
- **Training time**: 30-60 min on T4 GPU

## 🆘 Troubleshooting

**"Out of memory" error?**
- Edit training cell: Change `batch=32` to `batch=16`

**"Dataset not found" error?**
- Make sure you uploaded `ppe_dataset.zip`
- Check that extraction completed

**Training stops early?**
- Check if early stopping triggered (this is normal if model is good enough)
- Verify `best.pt` file size is ~15MB+

## 📚 Full Guide

See `COLAB_TRAINING_GUIDE.md` for detailed instructions.

