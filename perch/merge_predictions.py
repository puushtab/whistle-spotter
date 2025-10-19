#!/usr/bin/env python3
"""
Merge prediction results from multiple array jobs into a single file.

This script merges all prediction JSON files created by the array job
into a single predictions file.

Usage:
    python merge_predictions.py [--pattern PATTERN] [--output OUTPUT]
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def merge_predictions(results_dir, pattern="predictions_perch_*.json", output="predictions_merged.json", 
                      original_file=None, new_files=None):
    """
    Merge prediction files. Supports two modes:
    1. Pattern mode: Merge all files matching a glob pattern
    2. Explicit mode: Merge specific new files into an original file
    
    Args:
        results_dir: Directory containing prediction files
        pattern: Glob pattern to match prediction files (used in pattern mode)
        output: Output filename for merged predictions
        original_file: Path to original predictions file (used in explicit mode)
        new_files: List of paths to new prediction files to merge (used in explicit mode)
    """
    results_dir = Path(results_dir)
    
    # Determine which mode to use
    if new_files:
        # Explicit mode: merge specific files
        prediction_files = [Path(f) for f in new_files]
        print(f"Explicit merge mode: merging {len(prediction_files)} new file(s)")
        if original_file:
            original_path = Path(original_file)
            print(f"Original file: {original_path}")
    else:
        # Pattern mode: find all files matching pattern
        prediction_files = sorted(results_dir.glob(pattern))
        print(f"Pattern merge mode: searching for '{pattern}'")
        original_file = None
    
    if not prediction_files:
        print(f"No prediction files found matching pattern: {pattern}")
        print(f"Searched in: {results_dir}")
        return
    
    print(f"\nFound {len(prediction_files)} prediction file(s) to merge:")
    for f in prediction_files:
        print(f"  - {f.name}")
    print()
    
    # Load original file if specified
    merged_audios = {}
    if original_file:
        original_path = Path(original_file)
        if original_path.exists():
            print(f"Loading original file: {original_path.name}...")
            with open(original_path, 'r') as f:
                original_data = json.load(f)
            merged_audios = original_data.get('audios', {})
            print(f"  Loaded {len(merged_audios)} audio entries from original file")
            print()
        else:
            print(f"Warning: Original file not found: {original_path}")
            print("Starting with empty predictions")
            print()
    
    # Track statistics
    updated_count = 0
    new_count = 0
    
    # Merge predictions from new files
    for pred_file in prediction_files:
        print(f"Processing: {pred_file.name}...")
        with open(pred_file, 'r') as f:
            data = json.load(f)
        
        # Add or update audios from this file
        for audio_id, audio_data in data.get('audios', {}).items():
            if audio_id in merged_audios:
                # Update existing entry
                old_timestamps = len(merged_audios[audio_id].get('timestamps', []))
                new_timestamps = len(audio_data.get('timestamps', []))
                if old_timestamps != new_timestamps:
                    print(f"  Updating audio_id {audio_id}: {old_timestamps} -> {new_timestamps} timestamps")
                    updated_count += 1
                merged_audios[audio_id] = audio_data
            else:
                # Add new entry
                new_count += 1
                merged_audios[audio_id] = audio_data
    
    print(f"\nMerge statistics:")
    print(f"  New audio IDs added: {new_count}")
    print(f"  Existing audio IDs updated: {updated_count}")
    
    # Create merged results
    merged_results = {
        "audios": merged_audios
    }
    
    # Save merged results
    output_path = results_dir / output if not Path(output).is_absolute() else Path(output)
    with open(output_path, 'w') as f:
        json.dump(merged_results, f, indent=4)
    
    print(f"\nMerged results saved to: {output_path}")
    
    # Print summary
    total_detections = sum(len(v.get('timestamps', [])) for v in merged_audios.values())
    files_with_detections = sum(1 for v in merged_audios.values() if len(v.get('timestamps', [])) > 0)
    
    print(f"\nFinal Summary:")
    print(f"  Total audio files: {len(merged_audios)}")
    print(f"  Files with detections: {files_with_detections}")
    print(f"  Total detection intervals: {total_detections}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge prediction results from array jobs or specific files',
        epilog="""
Examples:
  # Pattern mode: merge all files matching pattern
  python merge_predictions.py --pattern "predictions_perch_*.json"
  
  # Explicit mode: merge specific new files into original
  python merge_predictions.py --original predictions_perch.json --new-files file1.json file2.json
  
  # Merge into existing file (updates in place)
  python merge_predictions.py --original predictions_perch.json --new-files file1.json --output predictions_perch.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='../results',
        help='Directory containing prediction files'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='predictions_perch_*.json',
        help='Glob pattern to match prediction files (default: predictions_perch_*.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions_merged.json',
        help='Output filename for merged predictions (default: predictions_merged.json)'
    )
    parser.add_argument(
        '--original',
        type=str,
        help='Path to original predictions file (for explicit merge mode)'
    )
    parser.add_argument(
        '--new-files',
        type=str,
        nargs='+',
        help='List of new prediction files to merge (for explicit merge mode)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir
    
    # Handle original file path
    original_file = None
    if args.original:
        original_path = Path(args.original)
        if not original_path.is_absolute():
            original_file = results_dir / original_path
        else:
            original_file = original_path
    
    # Handle new files paths
    new_files = None
    if args.new_files:
        new_files = []
        for nf in args.new_files:
            nf_path = Path(nf)
            if not nf_path.is_absolute():
                new_files.append(results_dir / nf_path)
            else:
                new_files.append(nf_path)
    
    print("="*60)
    print("Merging Prediction Results")
    print("="*60)
    print(f"Results directory: {results_dir}")
    if new_files:
        print(f"Mode: Explicit file merge")
        if original_file:
            print(f"Original file: {original_file}")
        print(f"New files: {[f.name for f in new_files]}")
    else:
        print(f"Mode: Pattern-based merge")
        print(f"File pattern: {args.pattern}")
    print(f"Output file: {args.output}")
    print()
    
    merge_predictions(results_dir, args.pattern, args.output, original_file, new_files)
    
    print("\n" + "="*60)
    print("Merge Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
