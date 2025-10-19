#!/usr/bin/env python3
"""
Generate prediction variations from a base file using random transformations.
Useful for exploring the solution space and potentially finding better predictions.
"""

import json
import random
import argparse
import os
import sys
from typing import List, Tuple
import numpy as np

def load_json(filepath: str) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data: dict, filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def translate_timestamp(timestamp: List[float], shift: float) -> List[float]:
    """Translate (shift) a timestamp by a given amount."""
    return [max(0, timestamp[0] + shift), max(0, timestamp[1] + shift)]

def scale_timestamp(timestamp: List[float], factor: float, anchor: str = 'center') -> List[float]:
    """
    Scale a timestamp duration.
    anchor: 'start', 'center', or 'end' - which point remains fixed
    """
    start, end = timestamp
    duration = end - start
    new_duration = duration * factor
    
    if anchor == 'start':
        return [start, start + new_duration]
    elif anchor == 'end':
        return [end - new_duration, end]
    else:  # center
        center = (start + end) / 2
        half_new = new_duration / 2
        return [center - half_new, center + half_new]

def add_random_timestamps(timestamps: List[List[float]], audio_duration: float, 
                         count: int, min_duration: float = 1.0, 
                         max_duration: float = 10.0) -> List[List[float]]:
    """Add random new timestamps."""
    new_timestamps = []
    for _ in range(count):
        duration = random.uniform(min_duration, max_duration)
        max_start = audio_duration - duration
        if max_start > 0:
            start = random.uniform(0, max_start)
            new_timestamps.append([start, start + duration])
    return timestamps + new_timestamps

def remove_random_timestamps(timestamps: List[List[float]], count: int) -> List[List[float]]:
    """Remove random timestamps."""
    if len(timestamps) <= count:
        return []
    indices_to_keep = random.sample(range(len(timestamps)), len(timestamps) - count)
    return [timestamps[i] for i in sorted(indices_to_keep)]

def split_timestamp(timestamp: List[float], split_point: float = 0.5) -> List[List[float]]:
    """Split a timestamp into two parts at the given relative position."""
    start, end = timestamp
    duration = end - start
    split = start + duration * split_point
    return [[start, split], [split, end]]

def merge_overlapping_timestamps(timestamps: List[List[float]], 
                                threshold: float = 2.0) -> List[List[float]]:
    """Merge timestamps that are close to each other."""
    if not timestamps:
        return []
    
    sorted_ts = sorted(timestamps, key=lambda x: x[0])
    merged = [sorted_ts[0]]
    
    for current in sorted_ts[1:]:
        last = merged[-1]
        if current[0] - last[1] <= threshold:
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            merged.append(current)
    
    return merged

def jitter_timestamps(timestamps: List[List[float]], 
                     max_jitter: float = 2.0) -> List[List[float]]:
    """Add random jitter to timestamp boundaries."""
    jittered = []
    for ts in timestamps:
        start_jitter = random.uniform(-max_jitter, max_jitter)
        end_jitter = random.uniform(-max_jitter, max_jitter)
        new_start = max(0, ts[0] + start_jitter)
        new_end = max(new_start + 0.5, ts[1] + end_jitter)  # Ensure min duration
        jittered.append([new_start, new_end])
    return jittered

def expand_contract_timestamps(timestamps: List[List[float]], 
                               factor: float = 1.1) -> List[List[float]]:
    """Expand or contract all timestamps by a factor."""
    return [scale_timestamp(ts, factor, anchor='center') for ts in timestamps]

def apply_transformations(timestamps: List[List[float]], 
                         config: dict,
                         audio_id: str = "0") -> List[List[float]]:
    """
    Apply a series of random transformations based on configuration.
    """
    result = timestamps.copy()
    
    # Estimate audio duration (use max timestamp end + some buffer)
    if result:
        audio_duration = max(ts[1] for ts in result) + 100
    else:
        audio_duration = 3600  # Default 1 hour
    
    # 1. Random translation (shift all timestamps)
    if config.get('translate', 0) > 0 and random.random() < config.get('translate_prob', 0.3):
        shift = random.uniform(-config['translate'], config['translate'])
        result = [translate_timestamp(ts, shift) for ts in result]
        result = [[max(0, ts[0]), max(0, ts[1])] for ts in result]  # Clip to positive
    
    # 2. Random scaling (expand/contract)
    if config.get('scale', 0) > 0 and random.random() < config.get('scale_prob', 0.4):
        scale_factor = random.uniform(1.0 - config['scale'], 1.0 + config['scale'])
        result = expand_contract_timestamps(result, scale_factor)
    
    # 3. Jitter (small random movements)
    if config.get('jitter', 0) > 0 and random.random() < config.get('jitter_prob', 0.5):
        result = jitter_timestamps(result, config['jitter'])
    
    # 4. Add random timestamps
    if config.get('add_count', 0) > 0 and random.random() < config.get('add_prob', 0.2):
        num_to_add = random.randint(0, config['add_count'])
        result = add_random_timestamps(result, audio_duration, num_to_add,
                                      config.get('min_duration', 1.0),
                                      config.get('max_duration', 10.0))
    
    # 5. Remove random timestamps
    if config.get('remove_count', 0) > 0 and len(result) > 0 and random.random() < config.get('remove_prob', 0.2):
        num_to_remove = min(random.randint(0, config['remove_count']), len(result) - 1)
        if num_to_remove > 0:
            result = remove_random_timestamps(result, num_to_remove)
    
    # 6. Split some timestamps
    if config.get('split_prob', 0) > 0 and len(result) > 0:
        new_result = []
        for ts in result:
            if random.random() < config['split_prob']:
                split_point = random.uniform(0.3, 0.7)
                new_result.extend(split_timestamp(ts, split_point))
            else:
                new_result.append(ts)
        result = new_result
    
    # 7. Merge overlapping
    if config.get('merge', False):
        result = merge_overlapping_timestamps(result, config.get('merge_threshold', 2.0))
    
    # 8. Random subset selection
    if config.get('subset_prob', 0) > 0 and len(result) > 0 and random.random() < config['subset_prob']:
        keep_ratio = random.uniform(0.7, 0.95)
        num_to_keep = max(1, int(len(result) * keep_ratio))
        result = random.sample(result, num_to_keep)
    
    # Sort by start time
    result.sort(key=lambda x: x[0])
    
    # Round to 2 decimal places
    result = [[round(ts[0], 2), round(ts[1], 2)] for ts in result]
    
    return result

def generate_variation(base_data: dict, config: dict, seed: int = None) -> dict:
    """Generate a single variation from base data."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    variation = {"audios": {}}
    
    for audio_id, audio_data in base_data['audios'].items():
        base_timestamps = audio_data.get('timestamps', [])
        
        # Apply transformations with some probability
        if random.random() < config.get('modify_prob', 0.7):
            new_timestamps = apply_transformations(base_timestamps, config, audio_id)
        else:
            # Keep original for some entries
            new_timestamps = base_timestamps
        
        variation['audios'][audio_id] = {
            "id": audio_id,
            "timestamps": new_timestamps
        }
    
    return variation

def main():
    parser = argparse.ArgumentParser(
        description='Generate prediction variations using random transformations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Transformation Parameters:
  --translate X       Max shift in seconds (default: 5.0)
  --scale X           Max scale factor deviation (default: 0.2, means 0.8-1.2x)
  --jitter X          Max boundary jitter in seconds (default: 2.0)
  --add-count N       Max number of timestamps to add (default: 2)
  --remove-count N    Max number of timestamps to remove (default: 1)
  
Presets:
  conservative        Small changes (good for fine-tuning)
  moderate            Balanced changes (default)
  aggressive          Large changes (exploration)
  creative            Very diverse changes

Examples:
  # Generate 10 moderate variations
  python3 generate_variations.py --base my_predictions_is_working.json --count 10
  
  # Conservative variations for fine-tuning
  python3 generate_variations.py --base my_predictions.json --count 5 --preset conservative
  
  # Aggressive exploration
  python3 generate_variations.py --base baseline_full.json --count 20 --preset aggressive --output-dir variations/
        """
    )
    
    parser.add_argument('--base', required=True, help='Base prediction file to vary')
    parser.add_argument('--count', type=int, default=10, help='Number of variations to generate (default: 10)')
    parser.add_argument('--output-dir', default='variations', help='Output directory (default: variations/)')
    parser.add_argument('--prefix', default='variation', help='Output file prefix (default: variation)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for first variation')
    
    # Preset configurations
    parser.add_argument('--preset', choices=['conservative', 'moderate', 'aggressive', 'creative'],
                       default='moderate', help='Transformation preset (default: moderate)')
    
    # Fine-grained control
    parser.add_argument('--translate', type=float, help='Max translation/shift in seconds')
    parser.add_argument('--scale', type=float, help='Max scale factor deviation (0.2 = 0.8x to 1.2x)')
    parser.add_argument('--jitter', type=float, help='Max boundary jitter in seconds')
    parser.add_argument('--add-count', type=int, help='Max timestamps to add per audio')
    parser.add_argument('--remove-count', type=int, help='Max timestamps to remove per audio')
    parser.add_argument('--split-prob', type=float, help='Probability of splitting a timestamp')
    parser.add_argument('--modify-prob', type=float, help='Probability of modifying each audio entry')
    parser.add_argument('--merge', action='store_true', help='Merge overlapping timestamps')
    
    args = parser.parse_args()
    
    # Define presets
    presets = {
        'conservative': {
            'translate': 2.0, 'translate_prob': 0.2,
            'scale': 0.05, 'scale_prob': 0.3,
            'jitter': 1.0, 'jitter_prob': 0.4,
            'add_count': 1, 'add_prob': 0.1,
            'remove_count': 1, 'remove_prob': 0.1,
            'split_prob': 0.05,
            'modify_prob': 0.5,
            'merge': True, 'merge_threshold': 2.0,
            'min_duration': 1.0, 'max_duration': 8.0
        },
        'moderate': {
            'translate': 5.0, 'translate_prob': 0.3,
            'scale': 0.2, 'scale_prob': 0.4,
            'jitter': 2.0, 'jitter_prob': 0.5,
            'add_count': 2, 'add_prob': 0.2,
            'remove_count': 1, 'remove_prob': 0.2,
            'split_prob': 0.1,
            'subset_prob': 0.1,
            'modify_prob': 0.7,
            'merge': True, 'merge_threshold': 2.0,
            'min_duration': 1.0, 'max_duration': 10.0
        },
        'aggressive': {
            'translate': 10.0, 'translate_prob': 0.4,
            'scale': 0.3, 'scale_prob': 0.5,
            'jitter': 3.0, 'jitter_prob': 0.6,
            'add_count': 3, 'add_prob': 0.3,
            'remove_count': 2, 'remove_prob': 0.25,
            'split_prob': 0.15,
            'subset_prob': 0.15,
            'modify_prob': 0.8,
            'merge': True, 'merge_threshold': 3.0,
            'min_duration': 0.5, 'max_duration': 15.0
        },
        'creative': {
            'translate': 15.0, 'translate_prob': 0.5,
            'scale': 0.5, 'scale_prob': 0.6,
            'jitter': 5.0, 'jitter_prob': 0.7,
            'add_count': 5, 'add_prob': 0.4,
            'remove_count': 3, 'remove_prob': 0.3,
            'split_prob': 0.2,
            'subset_prob': 0.2,
            'modify_prob': 0.9,
            'merge': True, 'merge_threshold': 4.0,
            'min_duration': 0.5, 'max_duration': 20.0
        }
    }
    
    # Get base configuration from preset
    config = presets[args.preset].copy()
    
    # Override with command-line arguments
    if args.translate is not None:
        config['translate'] = args.translate
    if args.scale is not None:
        config['scale'] = args.scale
    if args.jitter is not None:
        config['jitter'] = args.jitter
    if args.add_count is not None:
        config['add_count'] = args.add_count
    if args.remove_count is not None:
        config['remove_count'] = args.remove_count
    if args.split_prob is not None:
        config['split_prob'] = args.split_prob
    if args.modify_prob is not None:
        config['modify_prob'] = args.modify_prob
    if args.merge:
        config['merge'] = True
    
    # Load base file
    print(f"=== Prediction Variation Generator ===\n")
    print(f"Base file: {args.base}")
    print(f"Preset: {args.preset}")
    print(f"Generating {args.count} variations...")
    print(f"Output directory: {args.output_dir}\n")
    
    if not os.path.exists(args.base):
        print(f"Error: Base file '{args.base}' not found!")
        sys.exit(1)
    
    base_data = load_json(args.base)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("Configuration:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print()
    
    # Generate variations
    print("Generating variations:")
    variations_info = []
    
    for i in range(args.count):
        seed = (args.seed + i) if args.seed is not None else None
        variation = generate_variation(base_data, config, seed)
        
        # Generate output filename
        output_file = os.path.join(args.output_dir, f"{args.prefix}_{i+1:03d}.json")
        save_json(variation, output_file)
        
        # Calculate statistics
        total_timestamps = sum(len(v['timestamps']) for v in variation['audios'].values())
        base_timestamps = sum(len(v.get('timestamps', [])) for v in base_data['audios'].values())
        diff = total_timestamps - base_timestamps
        
        variations_info.append({
            'file': output_file,
            'timestamps': total_timestamps,
            'diff': diff
        })
        
        print(f"  [{i+1}/{args.count}] {output_file}: {total_timestamps} timestamps ({diff:+d})")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Base file timestamps: {sum(len(v.get('timestamps', [])) for v in base_data['audios'].values())}")
    print(f"Generated {args.count} variations in '{args.output_dir}/'")
    
    avg_timestamps = sum(v['timestamps'] for v in variations_info) / len(variations_info)
    min_timestamps = min(v['timestamps'] for v in variations_info)
    max_timestamps = max(v['timestamps'] for v in variations_info)
    
    print(f"\nVariation statistics:")
    print(f"  Average timestamps: {avg_timestamps:.1f}")
    print(f"  Range: {min_timestamps} - {max_timestamps}")
    print(f"  Std dev: {np.std([v['timestamps'] for v in variations_info]):.1f}")
    
    print(f"\n✓ Done! Use these files for testing different submission candidates.")
    print(f"\nNext steps:")
    print(f"  1. Evaluate variations: python3 evaluate_perch.py --predictions {args.output_dir}/variation_*.json")
    print(f"  2. Pick the best one based on your validation metric")
    print(f"  3. Submit the best variation!")

if __name__ == "__main__":
    main()
