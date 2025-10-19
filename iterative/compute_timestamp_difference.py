#!/usr/bin/env python3
"""
Compute the difference between two timestamp JSON files.

This script finds all intervals in timestamps2 that are NOT covered by timestamps1.
Useful for comparing predictions or finding what's new in one file vs another.

Usage:
    python compute_timestamp_difference.py file1.json file2.json [--output diff.json]
"""

import json
import argparse
from pathlib import Path


def load_timestamps(filepath):
    """Load timestamps from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('audios', {})


def merge_intervals(intervals):
    """Merge overlapping intervals."""
    if not intervals:
        return []
    
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        previous = merged[-1]
        if current[0] <= previous[1]:
            merged[-1] = [previous[0], max(previous[1], current[1])]
        else:
            merged.append(current)
    
    return merged


def subtract_intervals(intervals_a, intervals_b):
    """
    Subtract intervals_b from intervals_a.
    
    Returns all parts of intervals_a that are NOT covered by intervals_b.
    
    Args:
        intervals_a: List of [start, end] intervals (to subtract from)
        intervals_b: List of [start, end] intervals (to subtract)
        
    Returns:
        List of [start, end] intervals representing A - B
    """
    if not intervals_a:
        return []
    
    if not intervals_b:
        return intervals_a
    
    # Merge overlapping intervals first
    intervals_a = merge_intervals(intervals_a)
    intervals_b = merge_intervals(intervals_b)
    
    result = []
    
    for a_start, a_end in intervals_a:
        # Start with the full interval
        remaining = [[a_start, a_end]]
        
        # Subtract each interval from B
        for b_start, b_end in intervals_b:
            new_remaining = []
            
            for r_start, r_end in remaining:
                # No overlap
                if b_end <= r_start or b_start >= r_end:
                    new_remaining.append([r_start, r_end])
                else:
                    # Partial overlap - keep non-overlapping parts
                    if r_start < b_start:
                        new_remaining.append([r_start, min(r_end, b_start)])
                    if r_end > b_end:
                        new_remaining.append([max(r_start, b_end), r_end])
            
            remaining = new_remaining
            if not remaining:
                break
        
        result.extend(remaining)
    
    return result


def compute_difference(timestamps1, timestamps2, min_duration=0.0):
    """
    Compute timestamps2 - timestamps1 for all audio files.
    
    Returns intervals in timestamps2 that are NOT covered by timestamps1.
    
    Args:
        timestamps1: First timestamps dict (to subtract)
        timestamps2: Second timestamps dict (to find differences)
        min_duration: Minimum duration for difference intervals (seconds)
        
    Returns:
        Dictionary with difference intervals for each audio
    """
    # Get all audio IDs from both files
    all_ids = set(timestamps1.keys()) | set(timestamps2.keys())
    
    differences = {}
    stats = {
        'total_files': len(all_ids),
        'files_with_differences': 0,
        'total_intervals_ts1': 0,
        'total_intervals_ts2': 0,
        'total_difference_intervals': 0,
        'only_in_ts1': 0,  # Files only in timestamps1
        'only_in_ts2': 0,  # Files only in timestamps2
        'in_both': 0       # Files in both
    }
    
    for audio_id in sorted(all_ids, key=lambda x: int(x) if x.isdigit() else x):
        ts1_intervals = timestamps1.get(audio_id, {}).get('timestamps', [])
        ts2_intervals = timestamps2.get(audio_id, {}).get('timestamps', [])
        
        stats['total_intervals_ts1'] += len(ts1_intervals)
        stats['total_intervals_ts2'] += len(ts2_intervals)
        
        # Track file presence
        if audio_id in timestamps1 and audio_id in timestamps2:
            stats['in_both'] += 1
        elif audio_id in timestamps1:
            stats['only_in_ts1'] += 1
        else:
            stats['only_in_ts2'] += 1
        
        # Compute difference: ts2 - ts1
        diff_intervals = subtract_intervals(ts2_intervals, ts1_intervals)
        
        # Filter by minimum duration
        if min_duration > 0:
            diff_intervals = [
                interval for interval in diff_intervals
                if (interval[1] - interval[0]) >= min_duration
            ]
        
        if diff_intervals:
            stats['files_with_differences'] += 1
            stats['total_difference_intervals'] += len(diff_intervals)
        
        differences[audio_id] = {
            'id': audio_id,
            'timestamps_1': ts1_intervals,
            'timestamps_2': ts2_intervals,
            'difference': diff_intervals,  # Intervals in ts2 NOT in ts1
            'num_intervals_ts1': len(ts1_intervals),
            'num_intervals_ts2': len(ts2_intervals),
            'num_difference': len(diff_intervals)
        }
    
    return differences, stats


def print_summary(stats, file1, file2):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("TIMESTAMP DIFFERENCE SUMMARY")
    print("="*70)
    print(f"\nFile 1 (to subtract): {file1}")
    print(f"File 2 (to find diff): {file2}")
    print(f"\nTotal audio files: {stats['total_files']}")
    print(f"  - In both files: {stats['in_both']}")
    print(f"  - Only in file 1: {stats['only_in_ts1']}")
    print(f"  - Only in file 2: {stats['only_in_ts2']}")
    print(f"\nTotal intervals in file 1: {stats['total_intervals_ts1']}")
    print(f"Total intervals in file 2: {stats['total_intervals_ts2']}")
    print(f"\nFiles with differences: {stats['files_with_differences']}")
    print(f"Total difference intervals: {stats['total_difference_intervals']}")
    print(f"  (intervals in file 2 NOT covered by file 1)")


def print_detailed_differences(differences, max_files=10):
    """Print detailed differences for files that have them."""
    files_with_diff = {
        audio_id: data for audio_id, data in differences.items()
        if data['num_difference'] > 0
    }
    
    if not files_with_diff:
        print("\n✅ No differences found! Files are equivalent.")
        return
    
    print("\n" + "="*70)
    print(f"DETAILED DIFFERENCES (showing up to {max_files} files)")
    print("="*70)
    
    for i, (audio_id, data) in enumerate(sorted(files_with_diff.items(), 
                                                  key=lambda x: int(x[0]) if x[0].isdigit() else x[0])):
        if i >= max_files:
            remaining = len(files_with_diff) - max_files
            print(f"\n... and {remaining} more files with differences")
            break
        
        print(f"\n📄 Audio ID: {audio_id}")
        print(f"   Intervals in file 1: {data['num_intervals_ts1']}")
        print(f"   Intervals in file 2: {data['num_intervals_ts2']}")
        print(f"   New intervals (in 2, not in 1): {data['num_difference']}")
        
        if data['difference']:
            print(f"   Difference intervals:")
            for start, end in data['difference']:
                duration = end - start
                print(f"      [{start:.2f}s - {end:.2f}s] (duration: {duration:.2f}s)")


def main():
    parser = argparse.ArgumentParser(
        description='Compute difference between two timestamp JSON files'
    )
    parser.add_argument(
        '--file1',
        required=True,
        help='First timestamps file (intervals to subtract)'
    )
    parser.add_argument(
        '--file2',
        required=True,
        help='Second timestamps file (to find differences from)'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='timestamp_difference.json',
        help='Output file for differences (default: timestamp_difference.json)'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=0.0,
        help='Minimum duration for difference intervals (seconds)'
    )
    parser.add_argument(
        '--max-display',
        type=int,
        default=10,
        help='Maximum number of files to display in detail (default: 10)'
    )
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Only save to file, no console output'
    )
    
    args = parser.parse_args()
    
    # Load timestamps
    timestamps1 = load_timestamps(args.file1)
    timestamps2 = load_timestamps(args.file2)
    
    # Compute difference
    differences, stats = compute_difference(timestamps1, timestamps2, args.min_duration)
    
    # Save results in standard timestamp format (audios only, no metadata)
    output_data = {
        'audios': {
            audio_id: {
                'timestamps': data['difference']
            }
            for audio_id, data in differences.items()
            if data['num_difference'] > 0  # Only include files with differences
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    if not args.quiet:
        print_summary(stats, args.file1, args.file2)
        print_detailed_differences(differences, max_files=args.max_display)
        print(f"\n💾 Detailed results saved to: {args.output}")
    else:
        print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
