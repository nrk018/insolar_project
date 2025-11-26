# Training PPE Detection Model on Google Colab

## Why Google Colab?
- **Free GPU**: T4 GPU (16GB) available for free
- **Fast Training**: 30-60 minutes vs 1-3 hours on CPU
- **No Setup**: Everything runs in the browser
- **Easy Download**: Download trained model directly to your Mac

## Step-by-Step Procedure

### Step 1: Prepare Your Dataset

Your dataset is already ready at:
```
InsolareSafetySystem/flaskServer/datasets/
```

You need to create a ZIP file of the dataset folder.

**Option A: Using Terminal (Mac)**
```bash
cd /Users/storage/projects/update/InsolareSafetySystem/flaskServer
zip -r ppe_dataset.zip datasets/
```

**Option B: Using Finder**
1. Right-click on the `datasets` folder
2. Select "Compress datasets"
3. Rename to `ppe_dataset.zip`

### Step 2: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Sign in with your Google account
3. Click **File → New Notebook**

### Step 3: Enable GPU

1. Click **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Select **T4 GPU** (free tier)
4. Click **Save**

### Step 4: Upload the Training Notebook

1. Upload the `train_ppe_colab.ipynb` file to Colab
   - Or copy-paste the code from the notebook
2. The notebook will guide you through the process

### Step 5: Upload Your Dataset

**In the Colab notebook, run the first cell** which will:
- Install required packages
- Create a file upload widget
- Upload your `ppe_dataset.zip` file

**OR manually upload:**
1. Click the folder icon (📁) on the left sidebar
2. Click the upload button (⬆️)
3. Select `ppe_dataset.zip`
4. Wait for upload to complete

### Step 6: Extract Dataset

The notebook will automatically:
- Extract the ZIP file
- Set up the correct paths
- Verify the dataset structure

### Step 7: Start Training

1. Run all cells in the notebook
2. Training will start automatically
3. You'll see progress bars and metrics
4. **Training time: 30-60 minutes on T4 GPU**

### Step 8: Monitor Training

Watch for:
- **Epoch progress**: Shows current epoch/total epochs
- **Loss values**: Should decrease over time
- **mAP (mean Average Precision)**: Should increase
- **Best model**: Saved automatically when performance improves

### Step 9: Download Trained Model

After training completes:

**Option A: Direct Download (Recommended)**
1. The notebook will create a download link
2. Click the link to download `best.pt`
3. Save to your Mac

**Option B: Manual Download**
1. Right-click on `best.pt` in Colab file browser
2. Select "Download"
3. Save to your Mac

### Step 10: Install Model on Your Mac

1. Copy the downloaded `best.pt` to:
   ```
   /Users/storage/projects/update/InsolareSafetySystem/flaskServer/runs/detect/ppe_detection/weights/
   ```

2. Replace the existing `best.pt` file (backup the old one first if needed)

3. Restart your Flask server:
   ```bash
   cd /Users/storage/projects/update/InsolareSafetySystem/flaskServer
   python videoServer.py
   ```

## Expected Results

After proper training, you should see:
- **Training accuracy**: 85-95%+
- **Validation accuracy**: 80-90%+
- **Model file size**: ~15-20 MB
- **Training epochs**: 20-50 (depends on early stopping)

## Troubleshooting

### "Out of Memory" Error
- Reduce batch size from 32 to 16 or 8
- Edit the notebook: `batch=16` instead of `batch=32`

### "Dataset not found" Error
- Make sure you uploaded `ppe_dataset.zip`
- Check that extraction completed successfully
- Verify paths in `data.yaml` are correct

### Training Stops Early
- Check if early stopping triggered (patience=20)
- Model might be good enough already
- Check `best.pt` file size (should be ~15MB+)

### Slow Training
- Make sure GPU is enabled (Runtime → Change runtime type → GPU)
- Check GPU usage: Runtime → Manage sessions → Check GPU usage

## Tips

1. **Keep Colab Tab Open**: Don't close the browser tab during training
2. **Colab Timeout**: Free Colab sessions timeout after ~12 hours of inactivity
3. **Save Progress**: Colab auto-saves, but download `best.pt` periodically
4. **Monitor GPU**: Use `!nvidia-smi` to check GPU usage
5. **Multiple Runs**: You can train multiple times and compare results

## What Gets Trained?

The model learns to detect:
- ✅ **Helmet** (class 0)
- ✅ **Gloves** (class 1)  
- ✅ **Vest/Jacket** (class 2)
- ✅ **Boots** (class 3)
- ✅ **Goggles** (class 4, optional)
- ✅ **Person** (class 6)
- ✅ **No-PPE classes** (classes 7-10)

## Next Steps After Training

1. **Evaluate the model**:
   ```bash
   python evaluate_ppe_model.py
   ```

2. **Test on real images**:
   ```bash
   python test_ppe.py
   ```

3. **Use in production**:
   - The model will auto-load in `videoServer.py`
   - No code changes needed!

## Support

If you encounter issues:
1. Check Colab console for error messages
2. Verify dataset structure matches `data.yaml`
3. Ensure GPU is enabled and working
4. Try reducing batch size if memory issues occur

