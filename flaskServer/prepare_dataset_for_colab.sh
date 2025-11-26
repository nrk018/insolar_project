#!/bin/bash
# Script to prepare dataset for Google Colab upload
# Run this on your Mac before uploading to Colab

echo "=========================================="
echo "Preparing PPE Dataset for Google Colab"
echo "=========================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if datasets folder exists
if [ ! -d "datasets" ]; then
    echo "❌ Error: 'datasets' folder not found!"
    echo "   Make sure you're in the flaskServer directory"
    exit 1
fi

# Create ZIP file
ZIP_NAME="ppe_dataset.zip"
echo ""
echo "📦 Creating ZIP file: $ZIP_NAME"
echo "   This may take a minute..."

# Remove old ZIP if exists
if [ -f "$ZIP_NAME" ]; then
    echo "   Removing old ZIP file..."
    rm "$ZIP_NAME"
fi

# Create ZIP (excluding cache files)
zip -r "$ZIP_NAME" datasets/ -x "*.cache" "*.cache/*" "*/__pycache__/*" "*.pyc"

# Check if ZIP was created
if [ -f "$ZIP_NAME" ]; then
    SIZE=$(du -h "$ZIP_NAME" | cut -f1)
    echo ""
    echo "✅ ZIP file created successfully!"
    echo "   File: $ZIP_NAME"
    echo "   Size: $SIZE"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Go to Google Colab: https://colab.research.google.com/"
    echo "   2. Upload train_ppe_colab.ipynb"
    echo "   3. Upload $ZIP_NAME when prompted"
    echo "   4. Run all cells to start training"
    echo ""
else
    echo "❌ Error: Failed to create ZIP file!"
    exit 1
fi

