#!/usr/bin/env python3
"""
Full lifecycle workflow script for Street Scene tasks.

Executes: data prep → training → evaluation → report archiving
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pipelines.pipeline import StreetScenePipeline
from src.common.utils import setup_logging, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Run full workflow for Street Scene tasks')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--task-type', type=str, required=True,
                       choices=['detection', 'vehicle_classification', 'human_attributes', 'tracking'],
                       help='Type of task to run')
    parser.add_argument('--detection-task', type=str, default=None,
                       help='Specific detection task name (e.g., vehicle_detection, pedestrian_detection)')
    
    parser.add_argument('--train-data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val-data', type=str, default=None,
                       help='Path to validation data')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data')
    parser.add_argument('--data-yaml', type=str, default=None,
                       help='Path to YOLO dataset YAML file (for YOLO models)')
    parser.add_argument('--annotation-file', type=str, default=None,
                       help='Path to annotation file')
    
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Directory to save outputs')
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this run (default: auto-generated)')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training (only run evaluation)')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip evaluation (only run training)')
    
    parser.add_argument('--compare-with', type=str, nargs='+', default=None,
                       help='Previous run directories to compare with')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting workflow...")
    
    # Create run directory
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        task_name = args.detection_task or args.task_type
        run_name = f"{task_name}_{timestamp}"
    
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")
    
    # Initialize pipeline
    logger.info(f"Initializing pipeline for task: {args.task_type}")
    pipeline = StreetScenePipeline(
        config_path=args.config,
        task_type=args.task_type,
        log_level=args.log_level,
        detection_task=args.detection_task
    )
    
    # Training phase
    if not args.skip_training:
        logger.info("=" * 80)
        logger.info("TRAINING PHASE")
        logger.info("=" * 80)
        
        train_output_dir = os.path.join(run_dir, 'training')
        
        try:
            train_results = pipeline.train(
                train_data_path=args.train_data,
                val_data_path=args.val_data,
                annotation_file=args.annotation_file,
                output_dir=train_output_dir,
                resume_checkpoint=args.resume,
                data_yaml=args.data_yaml
            )
            
            logger.info("Training completed successfully")
            logger.info(f"Results saved to: {train_output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            if not args.skip_evaluation:
                logger.error("Aborting workflow")
                return 1
    
    # Evaluation phase
    if not args.skip_evaluation:
        logger.info("=" * 80)
        logger.info("EVALUATION PHASE")
        logger.info("=" * 80)
        
        # Determine checkpoint path
        if args.resume:
            checkpoint_path = args.resume
        elif not args.skip_training:
            train_output_dir = os.path.join(run_dir, 'training')
            checkpoint_path = os.path.join(train_output_dir, 'best_model.pth')
        else:
            logger.error("No checkpoint specified for evaluation")
            return 1
        
        # Use test data if provided, otherwise use validation data
        eval_data_path = args.test_data or args.val_data
        if not eval_data_path:
            logger.warning("No evaluation data provided, skipping evaluation")
        else:
            eval_output_dir = os.path.join(run_dir, 'evaluation')
            
            try:
                eval_results = pipeline.evaluate(
                    test_data_path=eval_data_path,
                    checkpoint_path=checkpoint_path,
                    annotation_file=args.annotation_file,
                    output_dir=eval_output_dir,
                    data_yaml=args.data_yaml
                )
                
                logger.info("Evaluation completed successfully")
                logger.info(f"Results saved to: {eval_output_dir}")
                
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")
                return 1
    
    # Archive run information
    logger.info("=" * 80)
    logger.info("ARCHIVING RUN INFORMATION")
    logger.info("=" * 80)
    
    run_info = {
        'run_name': run_name,
        'task_type': args.task_type,
        'detection_task': args.detection_task,
        'config': args.config,
        'train_data': args.train_data,
        'val_data': args.val_data,
        'test_data': args.test_data,
        'data_yaml': args.data_yaml,
        'timestamp': datetime.now().isoformat(),
        'output_dir': run_dir
    }
    
    import json
    run_info_path = os.path.join(run_dir, 'run_info.json')
    with open(run_info_path, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    logger.info(f"Run information saved to: {run_info_path}")
    
    # Generate comparison if requested
    if args.compare_with:
        logger.info("=" * 80)
        logger.info("GENERATING COMPARISON")
        logger.info("=" * 80)
        
        from src.evaluation.reporting import compare_runs
        
        try:
            comparison_output = os.path.join(run_dir, 'comparison.json')
            compare_runs([run_dir] + args.compare_with, comparison_output)
            logger.info(f"Comparison saved to: {comparison_output}")
        except Exception as e:
            logger.warning(f"Failed to generate comparison: {e}")
    
    logger.info("=" * 80)
    logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"All outputs saved to: {run_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
