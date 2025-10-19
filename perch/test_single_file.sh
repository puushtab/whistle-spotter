#!/bin/bash
# Standalone shell script to test Perch classifier on a single audio file
# Usage: ./test_single_file.sh [audio_file] [threshold]

set -e  # Exit on error

# Default parameters
AUDIO_FILE="${1:-../dataset_validation/audio_1.ogg}"
THRESHOLD="${2:-0.4}"
MIN_DURATION="${3:-0.3}"
MERGE_GAP="${4:-2.0}"
HOP="${5:-1.0}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Perch Bee-eater Classifier - Single File Test"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Check if virtual environment exists
if [ ! -d ".venvperch" ]; then
    echo "ERROR: Virtual environment .venvperch not found!"
    echo "Please create it first:"
    echo "  python3 -m venv .venvperch"
    echo "  source .venvperch/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venvperch/bin/activate

# Check if model exists
MODEL_FILE="../models/perch_beeeater_classifier.pkl"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Trained model not found: $MODEL_FILE"
    echo "Please train the model first:"
    echo "  python train_perch.py"
    exit 1
fi

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "ERROR: Audio file not found: $AUDIO_FILE"
    echo ""
    echo "Available validation files:"
    ls -1 ../dataset_validation/*.ogg | head -10
    exit 1
fi

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Generate output filename based on input
AUDIO_BASENAME=$(basename "$AUDIO_FILE" .ogg)
OUTPUT_FILE="test_${AUDIO_BASENAME}_results.json"

echo "Test Parameters:"
echo "  Audio file: $AUDIO_FILE"
echo "  Model: $MODEL_FILE"
echo "  Threshold: $THRESHOLD"
echo "  Min duration: ${MIN_DURATION}s"
echo "  Merge gap: ${MERGE_GAP}s"
echo "  Hop size: ${HOP}s"
echo "  Output: $OUTPUT_FILE"
echo ""
echo "========================================"
echo "Running test..."
echo "========================================"
echo ""

# Run the test
python test_single_file.py \
    --audio "$AUDIO_FILE" \
    --model "$MODEL_FILE" \
    --output "$OUTPUT_FILE" \
    --threshold "$THRESHOLD" \
    --min-duration "$MIN_DURATION" \
    --merge-gap "$MERGE_GAP" \
    --hop "$HOP"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Test Complete!"
echo "========================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Results saved to: $OUTPUT_FILE"
    echo ""
    echo "To view results:"
    echo "  cat $OUTPUT_FILE"
    echo ""
    echo "To test another file:"
    echo "  ./test_single_file.sh ../dataset_validation/audio_5.ogg 0.3"
else
    echo ""
    echo "✗ ERROR: Test failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
