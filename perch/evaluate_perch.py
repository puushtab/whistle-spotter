#!/usr/bin/env python3
"""
Evaluation script for European Bee-eater detection using trained Perch model.

This script evaluates the model on training/validation data with ground truth labels,
allowing hyperparameter tuning for optimal F1/F4 scores.

Features:
- Comprehensive metrics: Precision, Recall, F1, F2, F4, IoU
- Hyperparameter grid search (threshold, min_duration, merge_gap, hop_size)
- Detailed per-file and aggregate statistics
- GPU-accelerated inference support

Usage:
    python evaluate_perch.py --mode single     # Single hyperparameter set
    python evaluate_perch.py --mode grid       # Grid search
    python evaluate_perch.py --mode custom     # Custom parameter ranges
"""

import os
import json
import pickle
import argparse
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optuna for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Use 'pip install optuna' for optimization mode.")

# Perch model imports
from perch_hoplite.zoo import model_configs


class BeeEaterEvaluator:
    """Evaluate European Bee-eater detection with comprehensive metrics."""
    
    def __init__(self, model_path, ground_truth_path):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to trained model pickle file
            ground_truth_path: Path to timestamps.json with ground truth
        """
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
        
        # Load ground truth
        print(f"\nLoading ground truth from: {ground_truth_path}")
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
        
        num_files = len(self.ground_truth['audios'])
        num_with_birds = sum(1 for v in self.ground_truth['audios'].values() 
                            if len(v['timestamps']) > 0)
        print(f"  Total files: {num_files}")
        print(f"  Files with bee-eater: {num_with_birds}")
        
        # Load Perch model
        print("\nLoading Perch v2 model...")
        self.perch_model = model_configs.load_model_by_name('perch_v2')
        print("Perch model loaded successfully!")
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Peak normalize to 0.25
            if np.max(np.abs(audio)) > 0:
                audio = audio * (0.25 / np.max(np.abs(audio)))
            
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def extract_embeddings_sliding_window(self, audio, hop_length):
        """Extract Perch embeddings using sliding window."""
        embeddings = []
        timestamps = []
        
        for start in range(0, len(audio) - self.segment_samples + 1, hop_length):
            segment = audio[start:start + self.segment_samples]
            
            if len(segment) == self.segment_samples:
                outputs = self.perch_model.embed(segment)
                embedding = np.array(outputs.embeddings).flatten()
                embeddings.append(embedding)
                timestamps.append(start / self.sample_rate)
        
        # Handle last segment
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
        embeddings_scaled = self.scaler.transform(embeddings)
        probabilities = self.classifier.predict_proba(embeddings_scaled)[:, 1]
        return probabilities
    
    def probabilities_to_timestamps(self, probabilities, timestamps, threshold, min_duration, merge_gap):
        """Convert probability sequence to timestamp intervals."""
        detections = probabilities >= threshold
        
        if not np.any(detections):
            return []
        
        # Find continuous detection regions
        intervals = []
        in_detection = False
        start_idx = 0
        
        for i, detected in enumerate(detections):
            if detected and not in_detection:
                start_idx = i
                in_detection = True
            elif not detected and in_detection:
                end_idx = i - 1
                intervals.append((start_idx, end_idx))
                in_detection = False
        
        if in_detection:
            intervals.append((start_idx, len(detections) - 1))
        
        # Convert to timestamps
        timestamp_intervals = []
        for start_idx, end_idx in intervals:
            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx] + self.segment_length
            
            duration = end_time - start_time
            if duration >= min_duration:
                timestamp_intervals.append([start_time, end_time])
        
        # Merge nearby detections
        timestamp_intervals = self.merge_close_intervals(timestamp_intervals, merge_gap)
        
        return timestamp_intervals
    
    def merge_close_intervals(self, intervals, merge_gap):
        """Merge intervals that are close together."""
        if not intervals:
            return []
        
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            previous = merged[-1]
            
            if current[0] - previous[1] <= merge_gap:
                merged[-1] = [previous[0], max(previous[1], current[1])]
            else:
                merged.append(current)
        
        return merged
    
    def predict_file(self, audio_path, threshold, min_duration, merge_gap, hop_length):
        """Predict bee-eater presence in an audio file."""
        audio = self.load_audio(audio_path)
        if audio is None:
            return []
        
        embeddings, timestamps = self.extract_embeddings_sliding_window(audio, hop_length)
        
        if len(embeddings) == 0:
            return []
        
        probabilities = self.predict_probabilities(embeddings)
        detection_intervals = self.probabilities_to_timestamps(
            probabilities, timestamps, threshold, min_duration, merge_gap
        )
        
        return detection_intervals
    
    def calculate_iou(self, pred_interval, gt_interval):
        """Calculate Intersection over Union for two intervals."""
        pred_start, pred_end = pred_interval
        gt_start, gt_end = gt_interval
        
        # Calculate intersection
        intersection_start = max(pred_start, gt_start)
        intersection_end = min(pred_end, gt_end)
        intersection = max(0, intersection_end - intersection_start)
        
        # Calculate union
        union_start = min(pred_start, gt_start)
        union_end = max(pred_end, gt_end)
        union = union_end - union_start
        
        return intersection / union if union > 0 else 0
    
    def evaluate_predictions(self, predictions, ground_truth, iou_threshold=0.5):
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of predicted intervals [[start, end], ...]
            ground_truth: List of ground truth intervals [[start, end], ...]
            iou_threshold: IoU threshold for matching (default: 0.5)
            
        Returns:
            dict with TP, FP, FN counts and matched/unmatched intervals
        """
        predictions = sorted(predictions, key=lambda x: x[0])
        ground_truth = sorted(ground_truth, key=lambda x: x[0])
        
        matched_gt = set()
        matched_pred = set()
        
        # For each prediction, find best matching ground truth
        for i, pred in enumerate(predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt in enumerate(ground_truth):
                if j in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            if best_iou >= iou_threshold:
                matched_gt.add(best_gt_idx)
                matched_pred.add(i)
        
        tp = len(matched_gt)
        fp = len(predictions) - len(matched_pred)
        fn = len(ground_truth) - len(matched_gt)
        
        return {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'matched_predictions': matched_pred,
            'matched_ground_truth': matched_gt
        }
    
    def calculate_metrics(self, tp, fp, fn, beta=1.0):
        """
        Calculate precision, recall, and F-beta score.
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            beta: Beta value for F-beta score (1=F1, 2=F2, 4=F4)
        """
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)
        else:
            f_beta = 0
        
        return {
            'precision': precision,
            'recall': recall,
            f'f{int(beta)}': f_beta
        }
    
    def evaluate_dataset(self, audio_dir, threshold, min_duration, merge_gap, hop_length, 
                        iou_threshold=0.5, verbose=False, subset_size=None, random_seed=42):
        """
        Evaluate on entire dataset or a random subset.
        
        Args:
            audio_dir: Directory containing audio files
            threshold: Detection threshold
            min_duration: Minimum detection duration
            merge_gap: Gap for merging detections
            hop_length: Hop size in samples
            iou_threshold: IoU threshold for matching
            verbose: Print per-file results
            subset_size: Number of files to evaluate (None = all files)
            random_seed: Random seed for reproducible subset selection
            
        Returns:
            Dictionary with detailed metrics
        """
        audio_dir = Path(audio_dir)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        file_results = []
        
        # Get list of audio IDs
        audio_ids = list(self.ground_truth['audios'].keys())
        
        # Sample subset if requested
        if subset_size is not None and subset_size < len(audio_ids):
            np.random.seed(random_seed)
            audio_ids = np.random.choice(audio_ids, size=subset_size, replace=False).tolist()
            if verbose:
                print(f"Evaluating on subset: {subset_size}/{len(self.ground_truth['audios'])} files")
        
        for audio_id in tqdm(audio_ids, desc="Evaluating", disable=not verbose):
            gt_data = self.ground_truth['audios'][audio_id]
            
            # Find audio file
            audio_file = audio_dir / f"audio_{audio_id}.ogg"
            
            if not audio_file.exists():
                continue
            
            # Get predictions
            predictions = self.predict_file(
                audio_file, threshold, min_duration, merge_gap, hop_length
            )
            
            ground_truth = gt_data['timestamps']
            
            # Evaluate
            eval_result = self.evaluate_predictions(predictions, ground_truth, iou_threshold)
            
            total_tp += eval_result['tp']
            total_fp += eval_result['fp']
            total_fn += eval_result['fn']
            
            file_results.append({
                'audio_id': audio_id,
                'num_predictions': len(predictions),
                'num_ground_truth': len(ground_truth),
                'tp': eval_result['tp'],
                'fp': eval_result['fp'],
                'fn': eval_result['fn']
            })
        
        # Calculate overall metrics
        f1_metrics = self.calculate_metrics(total_tp, total_fp, total_fn, beta=1.0)
        f2_metrics = self.calculate_metrics(total_tp, total_fp, total_fn, beta=2.0)
        f4_metrics = self.calculate_metrics(total_tp, total_fp, total_fn, beta=4.0)
        
        results = {
            'parameters': {
                'threshold': threshold,
                'min_duration': min_duration,
                'merge_gap': merge_gap,
                'hop_length': hop_length / self.sample_rate,
                'iou_threshold': iou_threshold
            },
            'confusion_matrix': {
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn
            },
            'metrics': {
                'precision': f1_metrics['precision'],
                'recall': f1_metrics['recall'],
                'f1': f1_metrics['f1'],
                'f2': f2_metrics['f2'],
                'f4': f4_metrics['f4']
            },
            'file_results': file_results
        }
        
        return results
    
    def grid_search(self, audio_dir, param_grid, iou_threshold=0.5, subset_size=None):
        """
        Perform grid search over hyperparameters.
        
        Args:
            audio_dir: Directory containing audio files
            param_grid: Dictionary of parameter lists to search
            iou_threshold: IoU threshold for matching
            subset_size: Number of files to evaluate per combination (None = all)
            
        Returns:
            List of results sorted by F4 score
        """
        results = []
        
        total_combinations = (
            len(param_grid['threshold']) * 
            len(param_grid['min_duration']) * 
            len(param_grid['merge_gap']) * 
            len(param_grid['hop_size'])
        )
        
        print(f"\nGrid Search: {total_combinations} parameter combinations")
        if subset_size:
            print(f"Using subset: {subset_size} files per evaluation")
        print("="*60)
        
        with tqdm(total=total_combinations, desc="Grid Search Progress") as pbar:
            for threshold in param_grid['threshold']:
                for min_duration in param_grid['min_duration']:
                    for merge_gap in param_grid['merge_gap']:
                        for hop_size in param_grid['hop_size']:
                            hop_length = int(hop_size * self.sample_rate)
                            
                            result = self.evaluate_dataset(
                                audio_dir, threshold, min_duration, 
                                merge_gap, hop_length, iou_threshold, 
                                verbose=False, subset_size=subset_size
                            )
                            
                            results.append(result)
                            pbar.update(1)
                            
                            # Update progress bar with best F4 so far
                            best_f4 = max(r['metrics']['f4'] for r in results)
                            pbar.set_postfix({'Best F4': f"{best_f4:.4f}"})
        
        # Sort by F4 score
        results.sort(key=lambda x: x['metrics']['f4'], reverse=True)
        
        return results
    
    def optuna_optimize(self, audio_dir, n_trials=50, iou_threshold=0.5, subset_size=None, 
                       timeout=None, optimization_metric='f4'):
        """
        Use Optuna to find optimal hyperparameters.
        
        Args:
            audio_dir: Directory containing audio files
            n_trials: Number of optimization trials
            iou_threshold: IoU threshold for matching
            subset_size: Number of files to evaluate per trial (None = all)
            timeout: Time limit in seconds (None = no limit)
            optimization_metric: Metric to optimize ('f1', 'f2', 'f4', 'precision', 'recall')
            
        Returns:
            Dictionary with best parameters and study results
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        print(f"\nOptuna Optimization: {n_trials} trials")
        print(f"Optimizing: {optimization_metric}")
        if subset_size:
            print(f"Using subset: {subset_size} files per trial")
        if timeout:
            print(f"Timeout: {timeout}s")
        print("="*60)
        
        # Store audio_dir for objective function
        self._optuna_audio_dir = audio_dir
        self._optuna_iou_threshold = iou_threshold
        self._optuna_subset_size = subset_size
        self._optuna_metric = optimization_metric
        
        def objective(trial):
            # Suggest hyperparameters
            threshold = trial.suggest_float('threshold', 0.2, 0.8)
            min_duration = trial.suggest_float('min_duration', 0.2, 1.5)
            merge_gap = trial.suggest_float('merge_gap', 0.5, 4.0)
            hop_size = trial.suggest_float('hop_size', 0.25, 2.0)
            
            hop_length = int(hop_size * self.sample_rate)
            
            # Evaluate
            result = self.evaluate_dataset(
                self._optuna_audio_dir, threshold, min_duration,
                merge_gap, hop_length, self._optuna_iou_threshold,
                verbose=False, subset_size=self._optuna_subset_size
            )
            
            # Return metric to maximize
            return result['metrics'][self._optuna_metric]
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        print("\n" + "="*60)
        print("Optuna Optimization Complete!")
        print("="*60)
        print(f"\nBest {optimization_metric}: {study.best_value:.4f}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value:.4f}")
        
        # Evaluate best params on full dataset or larger subset
        print("\nEvaluating best parameters on full dataset...")
        best_hop_length = int(study.best_params['hop_size'] * self.sample_rate)
        final_result = self.evaluate_dataset(
            audio_dir,
            threshold=study.best_params['threshold'],
            min_duration=study.best_params['min_duration'],
            merge_gap=study.best_params['merge_gap'],
            hop_length=best_hop_length,
            iou_threshold=iou_threshold,
            verbose=True,
            subset_size=None  # Use full dataset for final evaluation
        )
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'final_evaluation': final_result,
            'study': study  # For further analysis
        }


def print_results(results, top_n=10):
    """Pretty print evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    if isinstance(results, dict):
        # Single result
        results = [results]
        top_n = 1
    
    print(f"\nTop {min(top_n, len(results))} Parameter Combinations (sorted by F4 score):")
    print("-"*80)
    
    for i, result in enumerate(results[:top_n], 1):
        params = result['parameters']
        metrics = result['metrics']
        cm = result['confusion_matrix']
        
        print(f"\n#{i} - F4: {metrics['f4']:.4f}")
        print(f"  Parameters:")
        print(f"    Threshold:     {params['threshold']:.3f}")
        print(f"    Min Duration:  {params['min_duration']:.2f}s")
        print(f"    Merge Gap:     {params['merge_gap']:.2f}s")
        print(f"    Hop Size:      {params['hop_length']:.2f}s")
        print(f"  Metrics:")
        print(f"    Precision:     {metrics['precision']:.4f}")
        print(f"    Recall:        {metrics['recall']:.4f}")
        print(f"    F1 Score:      {metrics['f1']:.4f}")
        print(f"    F2 Score:      {metrics['f2']:.4f}")
        print(f"    F4 Score:      {metrics['f4']:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    TP: {cm['tp']:3d}  FP: {cm['fp']:3d}  FN: {cm['fn']:3d}")


def main():
    """Main evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description='Evaluate European Bee-eater detection model'
    )
    
    # File paths
    parser.add_argument(
        '--audio-dir',
        type=str,
        default='../dataset/parc_audios/data',
        help='Directory containing audio files'
    )
    parser.add_argument(
        '--ground-truth',
        type=str,
        default='../dataset/parc_audios/timestamps.json',
        help='Path to ground truth timestamps JSON'
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
        default='evaluation_results.json',
        help='Output path for results JSON'
    )
    
    # Evaluation mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'grid', 'custom', 'optuna'],
        default='single',
        help='Evaluation mode: single params, grid search, custom ranges, or optuna optimization'
    )
    
    # Subset sampling
    parser.add_argument(
        '--subset-size',
        type=int,
        default=None,
        help='Number of files to evaluate (default: None = all files). Use for faster testing.'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducible subset selection'
    )
    
    # Single evaluation parameters
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--min-duration', type=float, default=0.5)
    parser.add_argument('--merge-gap', type=float, default=2.0)
    parser.add_argument('--hop', type=float, default=1.0)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    
    # Grid search parameters
    parser.add_argument('--grid-thresholds', type=str, default='0.3,0.4,0.5,0.6,0.7')
    parser.add_argument('--grid-min-durations', type=str, default='0.3,0.5,1.0')
    parser.add_argument('--grid-merge-gaps', type=str, default='1.0,2.0,3.0')
    parser.add_argument('--grid-hops', type=str, default='0.5,1.0')
    
    # Optuna parameters
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds for Optuna')
    parser.add_argument('--optimize-metric', type=str, default='f4', 
                       choices=['f1', 'f2', 'f4', 'precision', 'recall'],
                       help='Metric to optimize with Optuna')
    
    args = parser.parse_args()
    
    # Convert paths
    script_dir = Path(__file__).parent
    audio_dir = script_dir / args.audio_dir
    ground_truth_path = script_dir / args.ground_truth
    model_path = script_dir / args.model
    output_path = script_dir / args.output
    
    print("="*80)
    print("EUROPEAN BEE-EATER DETECTION - MODEL EVALUATION")
    print("="*80)
    print(f"\nAudio directory: {audio_dir}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Model: {model_path}")
    print(f"Mode: {args.mode}")
    
    # Initialize evaluator
    evaluator = BeeEaterEvaluator(model_path, ground_truth_path)
    
    if args.mode == 'single':
        # Single parameter evaluation
        print("\n" + "="*80)
        print("SINGLE PARAMETER EVALUATION")
        print("="*80)
        
        hop_length = int(args.hop * evaluator.sample_rate)
        
        results = evaluator.evaluate_dataset(
            audio_dir,
            threshold=args.threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
            hop_length=hop_length,
            iou_threshold=args.iou_threshold,
            verbose=True,
            subset_size=args.subset_size,
            random_seed=args.random_seed
        )
        
        print_results(results)
        
    elif args.mode == 'grid':
        # Grid search
        param_grid = {
            'threshold': [float(x) for x in args.grid_thresholds.split(',')],
            'min_duration': [float(x) for x in args.grid_min_durations.split(',')],
            'merge_gap': [float(x) for x in args.grid_merge_gaps.split(',')],
            'hop_size': [float(x) for x in args.grid_hops.split(',')]
        }
        
        results = evaluator.grid_search(
            audio_dir, param_grid, args.iou_threshold, subset_size=args.subset_size
        )
        print_results(results, top_n=10)
        
    elif args.mode == 'custom':
        # Custom parameter ranges
        print("\nCustom mode: Define your parameter ranges in the code")
        print("Edit the param_grid dictionary in the script")
        
        # Example custom grid
        param_grid = {
            'threshold': np.linspace(0.2, 0.8, 7),
            'min_duration': [0.3, 0.5, 0.7, 1.0],
            'merge_gap': [1.0, 1.5, 2.0, 2.5, 3.0],
            'hop_size': [0.5, 0.75, 1.0]
        }
        
        results = evaluator.grid_search(
            audio_dir, param_grid, args.iou_threshold, subset_size=args.subset_size
        )
        print_results(results, top_n=15)
        
    elif args.mode == 'optuna':
        # Optuna optimization
        print("\n" + "="*80)
        print("OPTUNA HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        
        optuna_results = evaluator.optuna_optimize(
            audio_dir,
            n_trials=args.n_trials,
            iou_threshold=args.iou_threshold,
            subset_size=args.subset_size,
            timeout=args.timeout,
            optimization_metric=args.optimize_metric
        )
        
        # Print final results
        print_results(optuna_results['final_evaluation'])
        
        # Save to separate results dict
        results = {
            'mode': 'optuna',
            'best_parameters': optuna_results['best_params'],
            'best_metric_value': optuna_results['best_value'],
            'optimization_metric': args.optimize_metric,
            'n_trials': optuna_results['n_trials'],
            'final_evaluation': optuna_results['final_evaluation']
        }
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results if isinstance(results, list) else [results], f, indent=2)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == '__main__':
    main()
