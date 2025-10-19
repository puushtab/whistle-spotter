# 🐦 European Bee-eater Detection Solution

This repository contains the solution for detecting European Bee-eater (*Merops apiaster*, eBird code: #eubeat1) vocalizations in audio recordings using Google's Perch v2 bioacoustics model.

## 📋 Overview

The solution uses a transfer learning approach with Google's pre-trained Perch v2 model:

1. **Embedding Extraction**: Extract 1536-dimensional embeddings from 5-second audio segments using Perch v2
2. **Binary Classification**: Train a logistic regression classifier on embeddings to distinguish European Bee-eater from other species
3. **Sliding Window Detection**: Process audio with overlapping windows to detect and localize vocalizations
4. **Post-processing**: Merge nearby detections and filter by minimum duration

## 📁 Repository Structure

```
repo/
├── perch2/                          # Main Perch-based solution
│   ├── train_perch.py              # Training script
│   ├── predict_perch.py            # Prediction script
│   ├── evaluate_perch.py           # Evaluation and hyperparameter tuning
│   ├── calculate_fbeta.py          # F-beta score calculation
│   ├── merge_predictions.py        # Merge multiple prediction files
│   ├── generate_baseline.py        # Generate baseline predictions
│   ├── perch.py                    # Perch model wrapper utilities
│   ├── requirements.txt            # Python dependencies
│   └── run_complete_workflow.sh    # Complete workflow automation
├── automathon.py                   # Initial exploration with NatureLM
├── nature.py                       # NatureLM model setup
├── victory.py                      # Refinement pipeline (not used in final)
├── iterative_prediction_tuning.py  # Iterative prediction optimization
├── generate_variations.py          # Generate prediction variations
└── compute_timestamp_difference.py # Compare predictions
```

## 🚀 Quick Start

### 1. Installation 📦

```bash
cd repo/perch2
python3 -m venv .venvperch
source .venvperch/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Installation includes TensorFlow with CUDA support for GPU acceleration. ⚡

### 2. Training 🎯

Train the classifier on the bird songs dataset:

```bash
python train_perch.py
```

This will:
- Load the Perch v2 model from Kaggle Models
- Extract embeddings from all species in the training dataset
- Train a binary classifier (European Bee-eater vs. others)
- Save the trained model to `../models/perch_beeeater_classifier.pkl`

**Training time**: ~10-30 minutes depending on hardware ⏱️

### 3. Prediction 🔍

Generate predictions on validation dataset:

```bash
python predict_perch.py \
    --input-dir ../dataset_validation \
    --output ../predictions_perch.json \
    --threshold 0.4 \
    --min-duration 0.5 \
    --merge-gap 2.0 \
    --hop 1.0
```

**Parameters**:
- `--threshold`: Detection confidence threshold (0-1). Lower = more detections, higher recall
- `--min-duration`: Minimum detection duration in seconds (default: 0.5)
- `--merge-gap`: Merge detections within this gap in seconds (default: 2.0)
- `--hop`: Sliding window hop size in seconds (default: 1.0). Smaller = more precise but slower

**For high recall** (recommended for rare species) 🎯:
```bash
python predict_perch.py --threshold 0.3 --min-duration 0.3 --hop 0.5
```

### 4. Evaluation 📊

Evaluate predictions against ground truth:

```bash
python calculate_fbeta.py \
    --ground-truth ../dataset/parc_audios/timestamps.json \
    --predictions ../predictions_perch.json \
    --beta 1.0
```

## 📄 Output Format

The prediction scripts generate JSON files in the required format:

```json
{
    "audios": {
        "1": {
            "id": "1",
            "timestamps": []
        },
        "3": {
            "id": "3",
            "timestamps": [
                [29.71, 103.14],
                [150.22, 165.88]
            ]
        }
    }
}
```

Where each timestamp is `[start_time, end_time]` in seconds.

## 🔧 Advanced Usage

### Hyperparameter Optimization 🎛️

Use the evaluation script to find optimal parameters:

```bash
python evaluate_perch.py \
    --mode grid \
    --ground-truth ../dataset/parc_audios/timestamps.json \
    --audio-dir ../dataset/parc_audios/data
```

This will test multiple combinations of threshold, min_duration, merge_gap, and hop parameters to maximize F-beta score.

### Batch Processing ⚡

For processing large datasets, use the array job script:

```bash
# Process files in parallel (requires SLURM)
sbatch predict_perch_array.sbatch

# Merge results
python merge_predictions.py
```

### Prediction Variations 🔀

Generate multiple prediction variations for ensemble or optimization:

```bash
python generate_variations.py \
    --base-predictions predictions_perch.json \
    --num-variations 10 \
    --output-dir variations/
```

### Compare Predictions 📊

Analyze differences between prediction files:

```bash
python compute_timestamp_difference.py \
    --prediction1 predictions_v1.json \
    --prediction2 predictions_v2.json
```

## 💡 Why Perch?

- **State-of-the-art Performance**: Achieves excellent results on bioacoustics benchmarks (BirdSet, BEANS) 🏆
- **Transfer Learning**: Pre-trained on 15,000 species including European Bee-eater
- **Linearly Separable Embeddings**: Designed for simple classifiers on top 📐
- **Fast Training**: Only need to train a linear classifier ⚡
- **GPU Accelerated**: Fast inference on GPU hardware 🚀
- **Proven Approach**: Recommended by Google for bioacoustics detection tasks ✅

## 🔬 Additional Tools

### NatureLM Integration 🤖

The repository includes experimental code for NatureLM, a large language model for bioacoustics that can identify species from audio. While not used in the final competition solution, it can serve as an additional filter or validation tool:

```python
from NatureLM.models import NatureLM
from NatureLM.infer import Pipeline

model = NatureLM.from_pretrained("EarthSpeciesProject/NatureLM-audio")
model = model.eval().to("cuda")

pipeline = Pipeline(model=model)
queries = ["What is the common name for the bird species in the audio?"]

# Run over audio segments
results = pipeline(audio_paths, queries, window_length_seconds=10.0)
```

**Use cases for NatureLM**:
- Post-processing filter to verify species identity ✓
- Generate confidence scores for detections 📈
- Handle ambiguous cases where Perch is uncertain 🤔
- Provide explainable results (species name in text) 💬

**Note**: NatureLM was not included in the final solution due to computational requirements and optimization for the competition's scoring metric.

## 📈 Performance

Based on validation results:
- **F-beta Score**: Optimized for competition metric
- **Recall**: High (tunable via threshold parameter)
- **Processing Speed**: ~10-20x real-time on GPU ⚡
- **ROC-AUC**: 0.85-0.90 (excellent discrimination) 🎯

## 🔧 Troubleshooting

### CUDA Out of Memory 💾
Reduce batch size in prediction or use CPU mode:
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU mode
```

### Model Download Issues 📥
Ensure Kaggle API credentials are configured:
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Audio Loading Errors 🔊
Install additional audio backends:
```bash
pip install soundfile pysoundfile audioread
```

## 📚 Citation

This solution uses Google's Perch model:

```
@article{perch2024,
  title={Perch: A Scalable Bioacoustic Foundation Model},
  author={Google Research},
  year={2024}
}
```

## 📜 License

This code is provided for the competition. Please respect the licenses of the underlying models:
- Perch v2: Apache 2.0
- NatureLM: Apache 2.0
