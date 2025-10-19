#!/usr/bin/env python3
"""
Generate various baseline predictions for testing and comparison.

This script can generate different baseline strategies:
- full: Mark entire duration of each audio (100% recall baseline)
- empty: Mark nothing (0% recall, 100% precision baseline)
- first-N: Mark first N seconds of each audio
- last-N: Mark last N seconds of each audio
- random: Random intervals for testing

Usage:
    python generate_baseline.py --strategy full
    python generate_baseline.py --strategy empty
    python generate_baseline.py --strategy first-30
    python generate_baseline.py --strategy random --num-intervals 3
"""

import os
import json
import argparse
import librosa
import random
from pathlib import Path
from tqdm import tqdm


def get_audio_duration(audio_path, sample_rate=32000):
    """Get duration of audio file."""
    try:
        duration = librosa.get_duration(path=audio_path, sr=sample_rate)
        return duration
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return None


def generate_full_duration(duration):
    """Full duration timestamp."""
    return [[0.0, duration]]


def generate_empty():
    """No detections."""
    return []


def generate_first_n_seconds(duration, n_seconds):
    """First N seconds."""
    if duration > n_seconds:
        return [[0.0, n_seconds]]
    else:
        return [[0.0, duration]]


def generate_last_n_seconds(duration, n_seconds):
    """Last N seconds."""
    if duration > n_seconds:
        return [[duration - n_seconds, duration]]
    else:
        return [[0.0, duration]]


def generate_random_intervals(duration, num_intervals=3, min_length=5.0, max_length=30.0):
    """Random intervals within the audio."""
    if duration < min_length:
        return [[0.0, duration]]
    
    intervals = []
    for _ in range(num_intervals):
        # Random length
        length = random.uniform(min_length, min(max_length, duration))
        
        # Random start position (ensuring interval fits)
        max_start = max(0, duration - length)
        start = random.uniform(0, max_start)
        end = start + length
        
        intervals.append([start, end])
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    return intervals


def generate_baseline(input_dir, output_path, strategy='full', **kwargs):
    """
    Generate baseline predictions.
    
    Args:
        input_dir: Directory containing audio files
        output_path: Path to save output JSON
        strategy: Baseline strategy ('full', 'empty', 'first-N', 'last-N', 'random')
        **kwargs: Additional parameters for strategies
    """
    input_dir = Path(input_dir)
    audio_files = sorted(input_dir.glob('*.ogg'))
    
    print(f"\nProcessing {len(audio_files)} audio files...")
    print(f"Strategy: {strategy}")
    
    results = {"audios": {}}
    
    for audio_file in tqdm(audio_files, desc="Generating baseline"):
        # Extract audio ID
        audio_id = audio_file.stem.split('_')[-1]
        
        # Get duration
        duration = get_audio_duration(audio_file)
        
        if duration is None:
            results["audios"][audio_id] = {
                "id": audio_id,
                "timestamps": []
            }
            continue
        
        # Generate timestamps based on strategy
        if strategy == 'full':
            timestamps = generate_full_duration(duration)
        elif strategy == 'empty':
            timestamps = generate_empty()
        elif strategy.startswith('first-'):
            n_seconds = float(strategy.split('-')[1])
            timestamps = generate_first_n_seconds(duration, n_seconds)
        elif strategy.startswith('last-'):
            n_seconds = float(strategy.split('-')[1])
            timestamps = generate_last_n_seconds(duration, n_seconds)
        elif strategy == 'random':
            num_intervals = kwargs.get('num_intervals', 3)
            min_length = kwargs.get('min_length', 5.0)
            max_length = kwargs.get('max_length', 30.0)
            timestamps = generate_random_intervals(duration, num_intervals, min_length, max_length)
        else:
            print(f"Unknown strategy: {strategy}")
            timestamps = []
        
        results["audios"][audio_id] = {
            "id": audio_id,
            "timestamps": timestamps
        }
    
    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    files_with_timestamps = sum(1 for v in results['audios'].values() if len(v['timestamps']) > 0)
    total_intervals = sum(len(v['timestamps']) for v in results['audios'].values())
    total_duration = sum(
        ts[1] - ts[0] 
        for v in results['audios'].values() 
        for ts in v['timestamps']
    )
    
    print(f"\nSummary:")
    print(f"  Total files: {len(audio_files)}")
    print(f"  Files with detections: {files_with_timestamps}")
    print(f"  Total intervals: {total_intervals}")
    print(f"  Total marked duration: {total_duration:.2f}s ({total_duration/60:.2f}min)")


def main():
    parser = argparse.ArgumentParser(
        description='Generate baseline predictions for testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full duration (100%% recall baseline):
    python generate_baseline.py --strategy full
    
  Empty (0%% recall, 100%% precision baseline):
    python generate_baseline.py --strategy empty
    
  First 30 seconds of each file:
    python generate_baseline.py --strategy first-30
    
  Last 60 seconds of each file:
    python generate_baseline.py --strategy last-60
    
  Random intervals:
    python generate_baseline.py --strategy random --num-intervals 5
        """
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='../dataset_validation',
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for baseline JSON (auto-generated if not specified)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='full',
        choices=['full', 'empty', 'random'],
        help='Baseline strategy (or use first-N, last-N with number)'
    )
    parser.add_argument(
        '--num-intervals',
        type=int,
        default=3,
        help='Number of random intervals per file (for random strategy)'
    )
    parser.add_argument(
        '--min-length',
        type=float,
        default=5.0,
        help='Minimum interval length for random strategy'
    )
    parser.add_argument(
        '--max-length',
        type=float,
        default=30.0,
        help='Maximum interval length for random strategy'
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    
    # Auto-generate output filename if not specified
    if args.output is None:
        output_filename = f"baseline_{args.strategy.replace('-', '_')}.json"
        output_path = script_dir.parent / output_filename
    else:
        output_path = script_dir / args.output
    
    print("="*60)
    print("Baseline Prediction Generator")
    print("="*60)
    print(f"\nInput directory: {input_dir}")
    print(f"Output file: {output_path}")
    print(f"Strategy: {args.strategy}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"\nERROR: Input directory not found: {input_dir}")
        return 1
    
    # Generate baseline
    kwargs = {
        'num_intervals': args.num_intervals,
        'min_length': args.min_length,
        'max_length': args.max_length
    }
    generate_baseline(input_dir, output_path, args.strategy, **kwargs)
    
    print("\n" + "="*60)
    print("Baseline Generation Complete!")
    print("="*60)
    
    strategy_explanations = {
        'full': "100% recall baseline - everything is marked as target species",
        'empty': "0% recall baseline - nothing is marked (useful for precision testing)",
        'random': "Random intervals - useful for format validation"
    }
    
    if args.strategy in strategy_explanations:
        print(f"\n{strategy_explanations[args.strategy]}")
    elif args.strategy.startswith('first-'):
        print(f"\nMarks first {args.strategy.split('-')[1]} seconds of each file")
    elif args.strategy.startswith('last-'):
        print(f"\nMarks last {args.strategy.split('-')[1]} seconds of each file")
    
    return 0


if __name__ == '__main__':
    exit(main())
