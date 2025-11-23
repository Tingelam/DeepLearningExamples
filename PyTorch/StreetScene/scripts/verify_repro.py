#!/usr/bin/env python3
"""
Verify reproducibility of a saved run.

Re-evaluates a checkpoint and validates that metrics match within tolerance.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipelines.pipeline import StreetScenePipeline
from src.evaluation.reporting import load_run_metrics
from src.common.utils import setup_logging, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Verify reproducibility of a run')
    
    parser.add_argument('--run-id', type=str, required=True,
                       help='Path to run directory to verify')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data (if different from original)')
    parser.add_argument('--data-yaml', type=str, default=None,
                       help='Path to YOLO dataset YAML file')
    parser.add_argument('--tolerance', type=float, default=0.01,
                       help='Tolerance for metric comparison (default: 0.01)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save verification results (default: run_id/verification)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def compare_metrics(
    original: Dict[str, Any],
    reproduced: Dict[str, Any],
    tolerance: float
) -> tuple:
    """
    Compare original and reproduced metrics.
    
    Returns:
        (all_match, differences)
    """
    differences = {}
    all_match = True
    
    # Extract metrics dictionaries
    orig_metrics = original.get('metrics', {})
    repro_metrics = reproduced.get('metrics', {}) if isinstance(reproduced, dict) else reproduced
    
    # Compare each metric
    for key in orig_metrics:
        orig_value = orig_metrics[key]
        repro_value = repro_metrics.get(key)
        
        # Skip non-numeric values
        if not isinstance(orig_value, (int, float)):
            continue
        
        if repro_value is None:
            differences[key] = {
                'original': orig_value,
                'reproduced': 'MISSING',
                'diff': 'N/A',
                'match': False
            }
            all_match = False
            continue
        
        if not isinstance(repro_value, (int, float)):
            continue
        
        # Compute difference
        diff = abs(orig_value - repro_value)
        match = diff <= tolerance
        
        differences[key] = {
            'original': orig_value,
            'reproduced': repro_value,
            'diff': diff,
            'match': match
        }
        
        if not match:
            all_match = False
    
    return all_match, differences


def print_verification_results(all_match: bool, differences: Dict[str, Dict[str, float]], tolerance: float):
    """Print verification results to console."""
    print("\n" + "=" * 80)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 80 + "\n")
    
    print(f"Tolerance: ±{tolerance}\n")
    
    if all_match:
        print("✓ ALL METRICS MATCH WITHIN TOLERANCE\n")
    else:
        print("✗ SOME METRICS DO NOT MATCH\n")
    
    print("Metric Comparison:")
    print("-" * 80)
    
    for metric, values in sorted(differences.items()):
        orig = values['original']
        repro = values['reproduced']
        diff = values['diff']
        match = values['match']
        
        status = "✓" if match else "✗"
        
        if isinstance(orig, float):
            print(f"{status} {metric:30s} | Original: {orig:10.4f} | Reproduced: {repro:10s} | Diff: {diff}")
        else:
            print(f"{status} {metric:30s} | Original: {orig:10d} | Reproduced: {repro:10s} | Diff: {diff}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info(f"Verifying reproducibility for run: {args.run_id}")
    
    # Load original run info
    run_dir = Path(args.run_id)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return 1
    
    # Load run information
    run_info_path = run_dir / 'run_info.json'
    if not run_info_path.exists():
        logger.warning("No run_info.json found, attempting to load from metrics report")
        run_info = None
    else:
        with open(run_info_path, 'r') as f:
            run_info = json.load(f)
    
    # Load original metrics
    try:
        original_metrics = load_run_metrics(str(run_dir))
        logger.info("Loaded original metrics")
    except Exception as e:
        logger.error(f"Failed to load original metrics: {e}")
        return 1
    
    # Determine paths
    if run_info:
        config_path = run_info.get('config')
        task_type = run_info.get('task_type')
        detection_task = run_info.get('detection_task')
        test_data_path = args.test_data or run_info.get('test_data') or run_info.get('val_data')
        data_yaml = args.data_yaml or run_info.get('data_yaml')
    else:
        # Try to infer from directory structure
        logger.warning("Attempting to infer run configuration from directory structure")
        config_path = None
        task_type = original_metrics.get('task_type', 'detection')
        detection_task = original_metrics.get('task_name')
        test_data_path = args.test_data
        data_yaml = args.data_yaml
    
    if not test_data_path and not data_yaml:
        logger.error("No test data or data YAML specified. Cannot verify.")
        return 1
    
    # Find checkpoint
    checkpoint_candidates = [
        run_dir / 'training' / 'best_model.pth',
        run_dir / 'best_model.pth',
        run_dir / 'evaluation' / 'best_model.pth'
    ]
    
    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if candidate.exists():
            checkpoint_path = str(candidate)
            break
    
    if not checkpoint_path:
        logger.error("No checkpoint found in run directory")
        return 1
    
    logger.info(f"Using checkpoint: {checkpoint_path}")
    
    # Determine config path
    if not config_path:
        # Try to find config in the original metrics
        if 'config' in original_metrics:
            logger.info("Using config from original metrics")
            # Save temporarily
            temp_config_path = run_dir / 'temp_config.yaml'
            import yaml
            with open(temp_config_path, 'w') as f:
                yaml.dump(original_metrics['config'], f)
            config_path = str(temp_config_path)
        else:
            logger.error("No config available for verification")
            return 1
    
    # Initialize pipeline
    logger.info("Initializing pipeline for verification...")
    try:
        pipeline = StreetScenePipeline(
            config_path=config_path,
            task_type=task_type,
            log_level=args.log_level,
            detection_task=detection_task
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return 1
    
    # Run evaluation
    logger.info("Re-evaluating checkpoint...")
    
    output_dir = args.output_dir or str(run_dir / 'verification')
    
    try:
        reproduced_metrics = pipeline.evaluate(
            test_data_path=test_data_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            data_yaml=data_yaml
        )
        
        logger.info("Re-evaluation completed")
        
    except Exception as e:
        logger.error(f"Re-evaluation failed: {e}")
        return 1
    
    # Compare metrics
    logger.info("Comparing metrics...")
    all_match, differences = compare_metrics(original_metrics, reproduced_metrics, args.tolerance)
    
    # Print results
    print_verification_results(all_match, differences, args.tolerance)
    
    # Save verification report
    verification_report = {
        'run_id': str(run_dir),
        'checkpoint': checkpoint_path,
        'tolerance': args.tolerance,
        'all_match': all_match,
        'differences': differences,
        'original_metrics': original_metrics.get('metrics', {}),
        'reproduced_metrics': reproduced_metrics
    }
    
    report_path = Path(output_dir) / 'verification_report.json'
    with open(report_path, 'w') as f:
        json.dump(verification_report, f, indent=2)
    
    logger.info(f"Verification report saved to: {report_path}")
    
    # Return exit code based on verification result
    if all_match:
        logger.info("✓ Reproducibility verified successfully")
        return 0
    else:
        logger.error("✗ Reproducibility verification failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
