#!/usr/bin/env python3
"""
Baseline script to generate timestamps JSON with full duration for each audio file.

This creates a "detect everything" baseline where each audio file is marked
as having the target species present for its entire duration. Useful for:
- Testing the JSON format
- Understanding the scoring system
- Creating a maximum recall baseline

Usage:
    python generate_full_duration_baseline.py [--input-dir INPUT_DIR] [--output OUTPUT]
"""

import os
import json
import argparse
import librosa
from pathlib import Path
from tqdm import tqdm


def get_audio_duration(audio_path, sample_rate=32000):
    """
    Get the duration of an audio file.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Sample rate (default: 32000 for Perch)
        
    Returns:
        Duration in seconds
    """
    try:
        # Get duration without loading entire file (faster)
        duration = librosa.get_duration(path=audio_path, sr=sample_rate)
        return duration
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return None


def generate_full_duration_baseline(input_dir, output_path):
    """
    Generate baseline JSON with full duration timestamps for all audio files.
    
    Args:
        input_dir: Directory containing audio files
        output_path: Path to save output JSON
    """
    input_dir = Path(input_dir)
    audio_files = sorted(input_dir.glob('*.ogg'))
    
    print(f"Processing {len(audio_files)} audio files...")
    print(f"Generating full-duration baseline timestamps...")
    
    results = {"audios": {}}
    
    for audio_file in tqdm(audio_files, desc="Reading durations"):
        # Extract audio ID from filename (e.g., "audio_1.ogg" -> "1")
        audio_id = audio_file.stem.split('_')[-1]
        
        # Get duration
        duration = get_audio_duration(audio_file)
        
        if duration is not None:
            # Create timestamp for entire duration [0.0, duration]
            results["audios"][audio_id] = {
                "id": audio_id,
                "timestamps": [[0.0, duration]]
            }
        else:
            # If duration couldn't be read, create empty entry
            results["audios"][audio_id] = {
                "id": audio_id,
                "timestamps": []
            }
    
    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print summary
    files_with_timestamps = sum(1 for v in results['audios'].values() if len(v['timestamps']) > 0)
    total_duration = sum(
        ts[1] - ts[0] 
        for v in results['audios'].values() 
        for ts in v['timestamps']
    )
    
    print(f"\nSummary:")
    print(f"  Total files: {len(audio_files)}")
    print(f"  Files with timestamps: {files_with_timestamps}")
    print(f"  Total marked duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"\nThis baseline marks 100% of all audio as containing the target species.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate full-duration baseline timestamps for audio files'
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
        default='../baseline_full_duration.json',
        help='Output path for baseline JSON'
    )
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    input_dir = script_dir / args.input_dir
    output_path = script_dir / args.output
    
    print("="*60)
    print("Full Duration Baseline Generator")
    print("="*60)
    print(f"\nInput directory: {input_dir}")
    print(f"Output file: {output_path}")
    print("\nThis creates a 'detect everything' baseline where each")
    print("audio file is marked as containing the target species")
    print("for its entire duration.")
    print("="*60)
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"\nERROR: Input directory not found: {input_dir}")
        return 1
    
    # Generate baseline
    generate_full_duration_baseline(input_dir, output_path)
    
    print("\n" + "="*60)
    print("Baseline Generation Complete!")
    print("="*60)
    print(f"\nYou can now:")
    print(f"  1. Submit this baseline to understand the scoring system")
    print(f"  2. Compare your model predictions against this baseline")
    print(f"  3. Verify the JSON format is correct")
    
    return 0


if __name__ == '__main__':
    exit(main())
