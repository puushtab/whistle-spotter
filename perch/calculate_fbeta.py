#!/usr/bin/env python3
"""
Calculate F-beta score for bird detection predictions.

This script compares ground truth timestamps with predicted timestamps
and calculates precision, recall, and F-beta score (with beta=4 to
emphasize recall over precision).

Metrics are calculated based on time intervals:
- TP (True Positive): Time where both GT and prediction have detection
- TN (True Negative): Time where both GT and prediction have no detection
- FP (False Positive): Time where prediction detects but GT doesn't
- FN (False Negative): Time where GT has detection but prediction doesn't

Usage:
    python calculate_fbeta.py --ground-truth GT.json --predictions PRED.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_timestamps_json(filepath):
    """
    Load timestamps JSON file.
    
    Returns:
        Dictionary mapping audio_id to list of [start, end] intervals
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    timestamps = {}
    for audio_id, audio_data in data['audios'].items():
        timestamps[audio_id] = audio_data['timestamps']
    
    return timestamps


def merge_overlapping_intervals(intervals):
    """
    Merge overlapping intervals.
    
    Args:
        intervals: List of [start, end] pairs
        
    Returns:
        List of merged non-overlapping intervals
    """
    if not intervals:
        return []
    
    # Sort by start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    
    merged = [sorted_intervals[0]]
    
    for current in sorted_intervals[1:]:
        previous = merged[-1]
        
        # Check if intervals overlap or are adjacent
        if current[0] <= previous[1]:
            # Merge
            merged[-1] = [previous[0], max(previous[1], current[1])]
        else:
            # No overlap, add as new interval
            merged.append(current)
    
    return merged


def get_total_duration(intervals):
    """Calculate total duration covered by intervals."""
    merged = merge_overlapping_intervals(intervals)
    return sum(end - start for start, end in merged)


def calculate_interval_intersection(intervals_a, intervals_b):
    """
    Calculate the intersection duration between two sets of intervals.
    
    Args:
        intervals_a: List of [start, end] pairs
        intervals_b: List of [start, end] pairs
        
    Returns:
        Total duration of intersection
    """
    if not intervals_a or not intervals_b:
        return 0.0
    
    # Merge overlapping intervals first
    intervals_a = merge_overlapping_intervals(intervals_a)
    intervals_b = merge_overlapping_intervals(intervals_b)
    
    intersection_duration = 0.0
    
    # For each interval in A, find overlaps with intervals in B
    for start_a, end_a in intervals_a:
        for start_b, end_b in intervals_b:
            # Calculate overlap
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            
            if overlap_start < overlap_end:
                intersection_duration += (overlap_end - overlap_start)
    
    return intersection_duration


def get_audio_duration_from_timestamps(gt_intervals, pred_intervals):
    """
    Estimate total audio duration from the intervals.
    We take the maximum end time from both GT and predictions.
    """
    max_time = 0.0
    
    for intervals in [gt_intervals, pred_intervals]:
        if intervals:
            for start, end in intervals:
                max_time = max(max_time, end)
    
    return max_time


def calculate_metrics_for_audio(gt_intervals, pred_intervals, audio_duration=None):
    """
    Calculate TP, TN, FP, FN for a single audio file.
    
    Args:
        gt_intervals: Ground truth intervals [[start, end], ...]
        pred_intervals: Predicted intervals [[start, end], ...]
        audio_duration: Total duration of audio (estimated if None)
        
    Returns:
        Dictionary with tp, tn, fp, fn (in seconds)
    """
    # Estimate audio duration if not provided
    if audio_duration is None:
        audio_duration = get_audio_duration_from_timestamps(gt_intervals, pred_intervals)
    
    # Calculate durations
    gt_duration = get_total_duration(gt_intervals)
    pred_duration = get_total_duration(pred_intervals)
    
    # TP: Intersection of GT and prediction (both have detection)
    tp = calculate_interval_intersection(gt_intervals, pred_intervals)
    
    # FN: GT has detection but prediction doesn't (GT - intersection)
    fn = gt_duration - tp
    
    # FP: Prediction has detection but GT doesn't (Pred - intersection)
    fp = pred_duration - tp
    
    # TN: Neither has detection (Total - GT - Pred + intersection)
    # This is the time where both agree there's no detection
    tn = audio_duration - gt_duration - pred_duration + tp
    
    # Ensure non-negative (due to floating point errors)
    tp = max(0.0, tp)
    tn = max(0.0, tn)
    fp = max(0.0, fp)
    fn = max(0.0, fn)
    
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'audio_duration': audio_duration,
        'gt_duration': gt_duration,
        'pred_duration': pred_duration
    }


def calculate_fbeta_score(tp, fp, fn, beta=4.0):
    """
    Calculate F-beta score.
    
    F-beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    
    With beta=4, recall is weighted 16x more than precision.
    
    Args:
        tp: True positives (seconds)
        fp: False positives (seconds)
        fn: False negatives (seconds)
        beta: Beta parameter (default: 4.0)
        
    Returns:
        Dictionary with precision, recall, and f_beta
    """
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Calculate F-beta
    beta_squared = beta ** 2
    
    if (precision + recall) > 0:
        f_beta = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    else:
        f_beta = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f_beta': f_beta
    }


def evaluate_predictions(gt_file, pred_file, verbose=True):
    """
    Evaluate predictions against ground truth.
    
    Args:
        gt_file: Path to ground truth JSON
        pred_file: Path to predictions JSON
        verbose: Print detailed results
        
    Returns:
        Dictionary with overall metrics
    """
    # Load data
    if verbose:
        print(f"Loading ground truth: {gt_file}")
    gt_timestamps = load_timestamps_json(gt_file)
    
    if verbose:
        print(f"Loading predictions: {pred_file}")
    pred_timestamps = load_timestamps_json(pred_file)
    
    # Get all audio IDs (union of both sets)
    all_audio_ids = set(gt_timestamps.keys()) | set(pred_timestamps.keys())
    
    if verbose:
        print(f"\nTotal audio files: {len(all_audio_ids)}")
        print(f"  In ground truth: {len(gt_timestamps)}")
        print(f"  In predictions: {len(pred_timestamps)}")
    
    # Calculate metrics for each audio
    total_tp = 0.0
    total_tn = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_audio_duration = 0.0
    
    per_audio_results = {}
    
    for audio_id in sorted(all_audio_ids):
        gt_intervals = gt_timestamps.get(audio_id, [])
        pred_intervals = pred_timestamps.get(audio_id, [])
        
        metrics = calculate_metrics_for_audio(gt_intervals, pred_intervals)
        
        total_tp += metrics['tp']
        total_tn += metrics['tn']
        total_fp += metrics['fp']
        total_fn += metrics['fn']
        total_audio_duration += metrics['audio_duration']
        
        per_audio_results[audio_id] = metrics
    
    # Calculate overall F-beta score
    overall_scores = calculate_fbeta_score(total_tp, total_fp, total_fn, beta=4.0)
    
    if verbose:
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        print(f"\nTotal Audio Duration: {total_audio_duration:.2f} seconds ({total_audio_duration/60:.2f} minutes)")
        
        print(f"\nConfusion Matrix (in seconds):")
        print(f"  True Positives (TP):  {total_tp:10.2f}s  ({total_tp/total_audio_duration*100:5.2f}%)")
        print(f"  True Negatives (TN):  {total_tn:10.2f}s  ({total_tn/total_audio_duration*100:5.2f}%)")
        print(f"  False Positives (FP): {total_fp:10.2f}s  ({total_fp/total_audio_duration*100:5.2f}%)")
        print(f"  False Negatives (FN): {total_fn:10.2f}s  ({total_fn/total_audio_duration*100:5.2f}%)")
        
        print(f"\nPerformance Metrics:")
        print(f"  Precision: {overall_scores['precision']:.4f} ({overall_scores['precision']*100:.2f}%)")
        print(f"  Recall:    {overall_scores['recall']:.4f} ({overall_scores['recall']*100:.2f}%)")
        print(f"  F-4 Score: {overall_scores['f_beta']:.4f} ({overall_scores['f_beta']*100:.2f}%)")
        
        print(f"\nNote: F-4 score emphasizes recall (16x weight) over precision.")
        print(f"      This is appropriate for detecting rare species.")
    
    return {
        'tp': total_tp,
        'tn': total_tn,
        'fp': total_fp,
        'fn': total_fn,
        'total_duration': total_audio_duration,
        'precision': overall_scores['precision'],
        'recall': overall_scores['recall'],
        'f_beta': overall_scores['f_beta'],
        'per_audio': per_audio_results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate F-beta score for bird detection predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python calculate_fbeta.py \\
      --ground-truth ../dataset/parc_audios/timestamps.json \\
      --predictions ../predictions_perch.json
      
The F-beta score (with beta=4) emphasizes recall over precision,
which is appropriate for detecting rare species where missing a
detection (false negative) is worse than a false alarm.
        """
    )
    
    parser.add_argument(
        '--ground-truth',
        type=str,
        required=True,
        help='Path to ground truth timestamps JSON'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions timestamps JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to JSON file (optional)'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=4.0,
        help='Beta parameter for F-beta score (default: 4.0)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only print final F-beta score'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    gt_file = Path(args.ground_truth).resolve()
    pred_file = Path(args.predictions).resolve()
    
    if not gt_file.exists():
        print(f"ERROR: Ground truth file not found: {gt_file}")
        return 1
    
    if not pred_file.exists():
        print(f"ERROR: Predictions file not found: {pred_file}")
        return 1
    
    if not args.quiet:
        print("="*70)
        print("F-BETA SCORE CALCULATOR")
        print("="*70)
    
    # Evaluate
    results = evaluate_predictions(gt_file, pred_file, verbose=not args.quiet)
    
    if args.quiet:
        print(f"F-{args.beta} Score: {results['f_beta']:.4f}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump({
                'ground_truth': str(gt_file),
                'predictions': str(pred_file),
                'beta': args.beta,
                'metrics': {
                    'tp': results['tp'],
                    'tn': results['tn'],
                    'fp': results['fp'],
                    'fn': results['fn'],
                    'total_duration': results['total_duration'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f_beta': results['f_beta']
                }
            }, f, indent=4)
        
        if not args.quiet:
            print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
