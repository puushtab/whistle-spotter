#!/usr/bin/env python3
"""
Test script for Perch bee-eater classifier on a single audio file.

This script demonstrates how the prediction works on one file and shows
the detection results in detail.

Usage:
    python test_single_file.py [--audio AUDIO_FILE] [--threshold THRESHOLD]
"""

import os
import json
import pickle
import argparse
import numpy as np
import librosa
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Perch model imports
from perch_hoplite.zoo import model_configs


def load_model(model_path):
    """Load the trained model."""
    print(f"Loading trained model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    scaler = model_data['scaler']
    classifier = model_data['classifier']
    sample_rate = model_data['sample_rate']
    segment_length = model_data['segment_length']
    
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Segment length: {segment_length} s")
    
    return scaler, classifier, sample_rate, segment_length


def load_audio(audio_path, sample_rate):
    """Load and preprocess audio file."""
    print(f"\nLoading audio: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    
    # Peak normalize to 0.25 (recommended by Perch)
    if np.max(np.abs(audio)) > 0:
        audio = audio * (0.25 / np.max(np.abs(audio)))
    
    duration = len(audio) / sample_rate
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Samples: {len(audio)}")
    
    return audio, duration


def extract_embeddings(audio, perch_model, sample_rate, segment_length, hop_seconds=1.0):
    """Extract embeddings using sliding window."""
    segment_samples = int(sample_rate * segment_length)
    hop_length = int(sample_rate * hop_seconds)
    
    embeddings = []
    timestamps = []
    
    print(f"\nExtracting embeddings...")
    print(f"  Segment length: {segment_length}s")
    print(f"  Hop size: {hop_seconds}s")
    
    # Slide windows across audio
    num_windows = 0
    for start in range(0, len(audio) - segment_samples + 1, hop_length):
        segment = audio[start:start + segment_samples]
        
        if len(segment) == segment_samples:
            # Extract embedding
            outputs = perch_model.embed(segment)
            embedding = np.array(outputs.embeddings).flatten()
            embeddings.append(embedding)
            
            # Store start time in seconds
            timestamps.append(start / sample_rate)
            num_windows += 1
    
    # Handle last segment if audio doesn't divide evenly
    if len(audio) >= segment_samples:
        last_start = len(audio) - segment_samples
        if len(timestamps) == 0 or timestamps[-1] * sample_rate < last_start:
            segment = audio[last_start:]
            outputs = perch_model.embed(segment)
            embedding = np.array(outputs.embeddings).flatten()
            embeddings.append(embedding)
            timestamps.append(last_start / sample_rate)
            num_windows += 1
    
    print(f"  Extracted {num_windows} windows")
    print(f"  Embedding shape: {np.array(embeddings).shape}")
    
    return np.array(embeddings), np.array(timestamps)


def predict_probabilities(embeddings, scaler, classifier):
    """Predict probabilities for each embedding."""
    print(f"\nRunning classifier...")
    
    # Scale embeddings
    embeddings_scaled = scaler.transform(embeddings)
    
    # Predict probabilities
    probabilities = classifier.predict_proba(embeddings_scaled)[:, 1]
    
    print(f"  Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    print(f"  Mean probability: {probabilities.mean():.3f}")
    
    return probabilities


def probabilities_to_timestamps(probabilities, timestamps, segment_length, 
                                threshold=0.5, min_duration=0.5, merge_gap=2.0):
    """Convert probability sequence to timestamp intervals."""
    print(f"\nDetecting bee-eaters...")
    print(f"  Threshold: {threshold}")
    print(f"  Min duration: {min_duration}s")
    print(f"  Merge gap: {merge_gap}s")
    
    # Apply threshold
    detections = probabilities >= threshold
    num_detected = np.sum(detections)
    
    print(f"  Windows above threshold: {num_detected}/{len(detections)}")
    
    if not np.any(detections):
        print("  No detections found!")
        return []
    
    # Find continuous detection regions
    intervals = []
    in_detection = False
    start_idx = 0
    
    for i, detected in enumerate(detections):
        if detected and not in_detection:
            # Start of detection
            start_idx = i
            in_detection = True
        elif not detected and in_detection:
            # End of detection
            end_idx = i - 1
            intervals.append((start_idx, end_idx))
            in_detection = False
    
    # Handle case where detection continues to the end
    if in_detection:
        intervals.append((start_idx, len(detections) - 1))
    
    print(f"  Found {len(intervals)} continuous regions")
    
    # Convert indices to timestamps
    timestamp_intervals = []
    for start_idx, end_idx in intervals:
        start_time = timestamps[start_idx]
        end_time = timestamps[end_idx] + segment_length
        
        # Filter by minimum duration
        duration = end_time - start_time
        if duration >= min_duration:
            timestamp_intervals.append([start_time, end_time])
            print(f"    Region: [{start_time:.2f}s - {end_time:.2f}s] (duration: {duration:.2f}s)")
    
    # Merge nearby detections
    if len(timestamp_intervals) > 1:
        merged = [timestamp_intervals[0]]
        for current in timestamp_intervals[1:]:
            previous = merged[-1]
            
            # If gap is small, merge
            if current[0] - previous[1] <= merge_gap:
                merged[-1] = [previous[0], max(previous[1], current[1])]
            else:
                merged.append(current)
        
        if len(merged) < len(timestamp_intervals):
            print(f"  After merging: {len(merged)} regions")
            for start_time, end_time in merged:
                print(f"    Merged: [{start_time:.2f}s - {end_time:.2f}s] (duration: {end_time-start_time:.2f}s)")
        
        timestamp_intervals = merged
    
    return timestamp_intervals


def save_results(audio_id, timestamps, output_path):
    """Save results in required JSON format."""
    results = {
        "audios": {
            audio_id: {
                "id": audio_id,
                "timestamps": timestamps
            }
        }
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test Perch classifier on a single audio file'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default='../dataset_validation/audio_1.ogg',
        help='Path to audio file to test'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='../models/perch_beeeater_classifier.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_results.json',
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.4,
        help='Detection threshold (0-1)'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=0.3,
        help='Minimum detection duration in seconds'
    )
    parser.add_argument(
        '--merge-gap',
        type=float,
        default=2.0,
        help='Merge detections within this gap (seconds)'
    )
    parser.add_argument(
        '--hop',
        type=float,
        default=1.0,
        help='Hop size for sliding window in seconds'
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    audio_path = script_dir / args.audio
    model_path = script_dir / args.model
    output_path = script_dir / args.output
    
    print("="*70)
    print("Perch Bee-eater Classifier - Single File Test")
    print("="*70)
    
    # Load model
    scaler, classifier, sample_rate, segment_length = load_model(model_path)
    
    # Load Perch model
    print("\nLoading Perch v2 model...")
    perch_model = model_configs.load_model_by_name('perch_v2')
    print("Perch model loaded!")
    
    # Load audio
    audio, duration = load_audio(audio_path, sample_rate)
    
    # Extract embeddings
    embeddings, timestamps = extract_embeddings(
        audio, perch_model, sample_rate, segment_length, args.hop
    )
    
    # Predict
    probabilities = predict_probabilities(embeddings, scaler, classifier)
    
    # Show probability distribution
    print("\nProbability distribution:")
    for i, (ts, prob) in enumerate(zip(timestamps, probabilities)):
        if i < 5 or prob >= args.threshold:  # Show first 5 and all detections
            detection_mark = " ← DETECTION!" if prob >= args.threshold else ""
            print(f"  Window {i:3d}: time={ts:6.2f}s, probability={prob:.4f}{detection_mark}")
        elif i == 5:
            print(f"  ... ({len(timestamps) - 10} more windows) ...")
    
    # Convert to timestamps
    detection_intervals = probabilities_to_timestamps(
        probabilities, timestamps, segment_length,
        threshold=args.threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap
    )
    
    # Extract audio ID from filename (e.g., "audio_1.ogg" -> "1")
    audio_id = Path(audio_path).stem.split('_')[-1]
    
    # Save results
    save_results(audio_id, detection_intervals, output_path)
    
    # Summary
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)
    print(f"\nAudio file: {audio_path.name}")
    print(f"Audio ID: {audio_id}")
    print(f"Duration: {duration:.2f}s")
    print(f"Detections: {len(detection_intervals)} intervals")
    
    if detection_intervals:
        total_detection_time = sum(end - start for start, end in detection_intervals)
        print(f"Total detection time: {total_detection_time:.2f}s ({total_detection_time/duration*100:.1f}% of audio)")
    else:
        print("No European Bee-eater detected in this audio file.")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
