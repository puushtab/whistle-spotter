#!/usr/bin/env python3
"""
Prediction script for European Bee-eater detection using trained Perch model.

This script:
1. Loads the trained classifier
2. Processes validation audio files with sliding windows
3. Detects European Bee-eater vocalizations
4. Outputs predictions in the required timestamps.json format

Usage:
    python predict_perch.py [--input-dir INPUT_DIR] [--output OUTPUT] [--threshold THRESHOLD]
"""

import os
import json
import pickle
import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import maximum_filter1d
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Perch model imports
from perch_hoplite.zoo import model_configs


class PerchBeeEaterPredictor:
    """Predict European Bee-eater presence in audio files."""
    
    def __init__(self, model_path, threshold=0.5, min_duration=0.5, merge_gap=2.0):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model pickle file
            threshold: Probability threshold for detection (0-1)
            min_duration: Minimum duration for a detection (seconds)
            merge_gap: Merge detections within this gap (seconds)
        """
        self.threshold = threshold
        self.min_duration = min_duration
        self.merge_gap = merge_gap
        
        # Load trained model
        print(f"Loading trained model from: {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.scaler = model_data['scaler']
        self.classifier = model_data['classifier']
        self.sample_rate = model_data['sample_rate']
        self.segment_length = model_data['segment_length']
        self.segment_samples = int(self.sample_rate * self.segment_length)
        
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Segment length: {self.segment_length} s")
        
        # Load Perch model
        print("Loading Perch v2 model...")
        self.perch_model = model_configs.load_model_by_name('perch_v2')
        print("Perch model loaded successfully!")
        
    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        try:
            # Load audio at target sample rate
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Peak normalize to 0.25 (recommended by Perch)
            if np.max(np.abs(audio)) > 0:
                audio = audio * (0.25 / np.max(np.abs(audio)))
            
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def extract_embeddings_sliding_window(self, audio, hop_length=None):
        """
        Extract Perch embeddings using sliding window.
        
        Args:
            audio: Audio waveform
            hop_length: Hop size in samples (default: 1 second)
            
        Returns:
            embeddings: Array of embeddings (n_windows, embedding_dim)
            timestamps: Array of window start times in seconds
        """
        if hop_length is None:
            hop_length = self.sample_rate  # 1 second hop by default
        
        embeddings = []
        timestamps = []
        
        # Slide windows across audio
        for start in range(0, len(audio) - self.segment_samples + 1, hop_length):
            segment = audio[start:start + self.segment_samples]
            
            if len(segment) == self.segment_samples:
                # Extract embedding
                outputs = self.perch_model.embed(segment)
                # Flatten embedding to 1D vector
                embedding = np.array(outputs.embeddings).flatten()
                embeddings.append(embedding)
                
                # Store start time in seconds
                timestamps.append(start / self.sample_rate)
        
        # Handle last segment if audio doesn't divide evenly
        if len(audio) >= self.segment_samples:
            last_start = len(audio) - self.segment_samples
            if len(timestamps) == 0 or timestamps[-1] * self.sample_rate < last_start:
                segment = audio[last_start:]
                outputs = self.perch_model.embed(segment)
                embedding = np.array(outputs.embeddings).flatten()
                embeddings.append(embedding)
                timestamps.append(last_start / self.sample_rate)
        
        return np.array(embeddings), np.array(timestamps)
    
    def predict_probabilities(self, embeddings):
        """Predict probabilities for each embedding."""
        # Scale embeddings
        embeddings_scaled = self.scaler.transform(embeddings)
        
        # Predict probabilities
        probabilities = self.classifier.predict_proba(embeddings_scaled)[:, 1]
        
        return probabilities
    
    def probabilities_to_timestamps(self, probabilities, timestamps):
        """
        Convert probability sequence to timestamp intervals.
        
        Args:
            probabilities: Array of detection probabilities
            timestamps: Array of window start times
            
        Returns:
            List of [start, end] timestamp pairs
        """
        # Apply threshold
        detections = probabilities >= self.threshold
        
        if not np.any(detections):
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
        
        # Convert indices to timestamps
        timestamp_intervals = []
        for start_idx, end_idx in intervals:
            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx] + self.segment_length
            
            # Filter by minimum duration
            duration = end_time - start_time
            if duration >= self.min_duration:
                timestamp_intervals.append([start_time, end_time])
        
        # Merge nearby detections
        timestamp_intervals = self.merge_close_intervals(timestamp_intervals)
        
        return timestamp_intervals
    
    def merge_close_intervals(self, intervals):
        """Merge intervals that are close together."""
        if not intervals:
            return []
        
        # Sort by start time
        intervals = sorted(intervals, key=lambda x: x[0])
        
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            previous = merged[-1]
            
            # If gap is small, merge
            if current[0] - previous[1] <= self.merge_gap:
                merged[-1] = [previous[0], max(previous[1], current[1])]
            else:
                merged.append(current)
        
        return merged
    
    def predict_file(self, audio_path, hop_length=None):
        """
        Predict European Bee-eater presence in an audio file.
        
        Args:
            audio_path: Path to audio file
            hop_length: Hop size in samples
            
        Returns:
            List of [start, end] timestamp pairs
        """
        # Load audio
        audio = self.load_audio(audio_path)
        if audio is None:
            return []
        
        # Extract embeddings
        embeddings, timestamps = self.extract_embeddings_sliding_window(audio, hop_length)
        
        if len(embeddings) == 0:
            return []
        
        # Predict
        probabilities = self.predict_probabilities(embeddings)
        
        # Convert to timestamps
        detection_intervals = self.probabilities_to_timestamps(probabilities, timestamps)
        
        return detection_intervals
    
    def predict_directory(self, input_dir, output_path, hop_length=None, save_every=5, begin=None, end=None):
        """
        Predict on all audio files in a directory and save results.
        
        Args:
            input_dir: Directory containing audio files
            output_path: Path to save timestamps.json
            hop_length: Hop size in samples
            save_every: Save progress every N files (default: 5)
            begin: Start index for audio files (0-based, inclusive)
            end: End index for audio files (0-based, exclusive)
        """
        input_dir = Path(input_dir)
        audio_files = sorted(list(input_dir.glob('*.ogg')) + list(input_dir.glob('*.wav')))

        # Apply begin/end slice if specified
        total_files = len(audio_files)
        if begin is not None or end is not None:
            begin = begin if begin is not None else 0
            end = end if end is not None else total_files
            audio_files = audio_files[begin:end]
            print(f"\nProcessing files {begin} to {end-1} (out of {total_files} total files)")
        
        print(f"\nProcessing {len(audio_files)} audio files...")
        print(f"Threshold: {self.threshold}")
        print(f"Min duration: {self.min_duration}s")
        print(f"Merge gap: {self.merge_gap}s")
        print(f"Progressive save: every {save_every} files")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Load existing results if file exists (for resuming)
        if os.path.exists(output_path):
            print(f"\nFound existing results file. Loading to resume...")
            with open(output_path, 'r') as f:
                results = json.load(f)
            processed_ids = set(results['audios'].keys())
            print(f"Already processed: {len(processed_ids)} files")
        else:
            results = {"audios": {}}
            processed_ids = set()
        
        files_processed = 0
        
        for i, audio_file in enumerate(tqdm(audio_files, desc="Processing")):
            # Extract audio ID from filename (e.g., "audio_1.ogg" -> "1")
            audio_id = audio_file.stem.split('_')[-1]
            
            # Skip if already processed
            if audio_id in processed_ids:
                continue
            
            # Predict
            timestamps = self.predict_file(audio_file, hop_length)
            
            # Store results
            results["audios"][audio_id] = {
                "id": audio_id,
                "timestamps": timestamps
            }
            
            files_processed += 1
            
            # Progressive save
            if files_processed % save_every == 0:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=4)
                tqdm.write(f"  → Saved progress: {files_processed}/{len(audio_files)} files")
        
        # Final save
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to: {output_path}")
        
        # Print summary
        total_detections = sum(len(v['timestamps']) for v in results['audios'].values())
        files_with_detections = sum(1 for v in results['audios'].values() if len(v['timestamps']) > 0)
        
        print(f"\nSummary:")
        print(f"  Total files: {len(audio_files)}")
        print(f"  Files with detections: {files_with_detections}")
        print(f"  Total detection intervals: {total_detections}")


def main():
    """Main prediction pipeline."""
    parser = argparse.ArgumentParser(
        description='Predict European Bee-eater presence in audio files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../dataset_validation',
        help='Directory containing audio files to process'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for predictions JSON (default: auto-generated with date and range)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='../models/perch_beeeater_classifier.pkl',
        help='Path to trained model'
    )
    parser.add_argument(
        '--begin',
        type=int,
        default=None,
        help='Start index for audio files (0-based, inclusive)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End index for audio files (0-based, exclusive)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Detection threshold (0-1, higher = more conservative)'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=0.5,
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
        help='Hop size for sliding window in seconds (smaller = more detections but slower)'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=5,
        help='Save progress every N files (default: 5, useful for long runs)'
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    model_path = script_dir / args.model
    
    # Generate output filename with range if not specified (no timestamp)
    if args.output is None:
        begin_str = f"{args.begin}" if args.begin is not None else "0"
        end_str = f"{args.end}" if args.end is not None else "all"
        output_filename = f"predictions_perch_files_{begin_str}_to_{end_str}.json"
        # Output to results folder
        results_dir = script_dir / '..' / 'results'
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / output_filename
    else:
        output_path = script_dir / args.output
    
    print("="*60)
    print("European Bee-eater Detection - Prediction Pipeline")
    print("="*60)
    print(f"\nInput directory: {input_dir}")
    print(f"Output file: {output_path}")
    print(f"Model: {model_path}")
    if args.begin is not None or args.end is not None:
        print(f"File range: {args.begin if args.begin is not None else 0} to {args.end if args.end is not None else 'end'}")
    
    # Initialize predictor
    predictor = PerchBeeEaterPredictor(
        model_path=model_path,
        threshold=args.threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap
    )
    
    # Run predictions
    hop_length = int(args.hop * predictor.sample_rate)
    predictor.predict_directory(
        input_dir, 
        output_path, 
        hop_length, 
        save_every=args.save_every,
        begin=args.begin,
        end=args.end
    )
    
    print("\n" + "="*60)
    print("Prediction Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
