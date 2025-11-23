#!/usr/bin/env python3
"""
Training script for Street Scene models.
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipelines.pipeline import StreetScenePipeline


def main():
    parser = argparse.ArgumentParser(description="Train Street Scene models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--task", type=str, required=True, 
                       choices=["detection", "classification"],
                       help="Task type")
    parser.add_argument("--detection-task", type=str, 
                       help="Specific detection task (e.g., vehicle_detection, pedestrian_detection)")
    parser.add_argument("--classification-task", type=str,
                       help="Classification task name from the configuration catalog")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-data", type=str, help="Path to validation data")
    parser.add_argument("--annotations", type=str, help="Path to annotation file")
    parser.add_argument("--data-yaml", type=str, help="Path to YOLO dataset YAML file")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    if args.task == 'classification' and not args.classification_task:
        parser.error("--classification-task is required when --task classification")
    
    # Create output directory with task-specific subdirectory
    output_dir = args.output_dir
    if args.task == 'detection' and args.detection_task:
        output_dir = os.path.join(args.output_dir, args.detection_task)
    elif args.task == 'classification' and args.classification_task:
        output_dir = os.path.join(args.output_dir, args.classification_task)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pipeline
    pipeline = StreetScenePipeline(
        args.config,
        args.task,
        args.log_level,
        detection_task=args.detection_task,
        classification_task=args.classification_task,
    )
    
    # Train model
    train_kwargs = {
        'train_data_path': args.train_data,
        'val_data_path': args.val_data,
        'annotation_file': args.annotations,
        'output_dir': output_dir,
    }
    
    # For YOLO models, add data_yaml if provided
    if hasattr(pipeline.trainer, 'train') and args.data_yaml:
        train_kwargs['data_yaml'] = args.data_yaml
    
    # Handle resume
    if args.resume:
        train_kwargs['resume_checkpoint'] = args.resume
    
    results = pipeline.train(**train_kwargs)
    
    print(f"Training completed. Results saved to {output_dir}")
    if isinstance(results, dict):
        if 'best_val_metric' in results:
            print(f"Best validation metric: {results['best_val_metric']}")
        if 'results' in results:
            print(f"YOLO training results: {results['results']}")


if __name__ == "__main__":
    main()