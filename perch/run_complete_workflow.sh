#!/bin/bash
# Complete workflow script for Perch-based bird detection
# Usage: ./run_complete_workflow.sh

set -e  # Exit on error

echo "========================================"
echo "European Bee-eater Detection - Complete Workflow"
echo "========================================"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating virtual environment..."
    source .venvperch/bin/activate
fi

# Check if dependencies are installed
echo ""
echo "Step 1: Checking dependencies..."
python -c "from perch_hoplite.zoo import model_configs; import librosa; import sklearn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ All dependencies installed"
else
    echo "Installing dependencies..."
    pip install -q -r requirements.txt
    echo "✓ Dependencies installed"
fi

# Train the model
echo ""
echo "========================================"
echo "Step 2: Training Model"
echo "========================================"
if [ -f "../models/perch_beeeater_classifier.pkl" ]; then
    read -p "Model already exists. Retrain? (y/N): " retrain
    if [[ $retrain == "y" || $retrain == "Y" ]]; then
        python train_perch.py
    else
        echo "Using existing model"
    fi
else
    python train_perch.py
fi

# Run predictions
echo ""
echo "========================================"
echo "Step 3: Generating Predictions"
echo "========================================"

# Ask for threshold
echo ""
echo "Select detection threshold:"
echo "  1) High Recall (0.3) - Find all bee-eaters, more false positives"
echo "  2) Balanced (0.5) - Default balanced approach"
echo "  3) High Precision (0.7) - Fewer false positives, might miss some"
echo "  4) Custom"
read -p "Choice (1-4, default=1): " choice

case $choice in
    2)
        THRESHOLD=0.5
        MIN_DUR=0.5
        HOP=1.0
        ;;
    3)
        THRESHOLD=0.7
        MIN_DUR=1.0
        HOP=1.0
        ;;
    4)
        read -p "Enter threshold (0-1): " THRESHOLD
        read -p "Enter min duration (seconds): " MIN_DUR
        read -p "Enter hop size (seconds): " HOP
        ;;
    *)
        THRESHOLD=0.3
        MIN_DUR=0.3
        HOP=0.5
        ;;
esac

echo ""
echo "Running predictions with:"
echo "  Threshold: $THRESHOLD"
echo "  Min duration: ${MIN_DUR}s"
echo "  Hop size: ${HOP}s"

python predict_perch.py \
    --input-dir ../dataset/parc_audios/data \
    --output ../predictions_perch_.json \
    --threshold $THRESHOLD \
    --min-duration $MIN_DUR \
    --hop $HOP

# Summary
echo ""
echo "========================================"
echo "Workflow Complete!"
echo "========================================"
echo ""
echo "Results saved to: ../predictions_perch.json"
echo ""
echo "Next steps:"
echo "  1. Review predictions in predictions_perch.json"
echo "  2. Upload to scoring platform"
echo "  3. Adjust threshold if needed and re-run step 3"
echo ""
echo "To re-run predictions with different threshold:"
echo "  python predict_perch.py --threshold 0.4 --input-dir ../dataset_validation --output ../predictions_perch.json"
