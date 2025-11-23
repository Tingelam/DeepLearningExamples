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
                       choices=["detection", "vehicle_classification", "human_attributes"],
                       help="Task type")
    parser.add_argument("--detection-task", type=str, 
                       help="Specific detection task (e.g., vehicle_detection, pedestrian_detection)")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-data", type=str, help="Path to validation data")
    parser.add_argument("--annotations", type=str, help="Path to annotation file")
    parser.add_argument("--data-yaml", type=str, help="Path to YOLO dataset YAML file")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--run-name", type=str, help="Name for this experimental run")
    parser.add_argument("--config-override", type=str, action="append", dest="overrides",
                       help="Config override in format 'key=value' (can be used multiple times)")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create pipeline with config overrides and run name
    pipeline = StreetScenePipeline(
        args.config,
        args.task,
        log_level=args.log_level,
        detection_task=args.detection_task,
        overrides=args.overrides,
        run_name=args.run_name,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    # Train model
    train_kwargs = {
        'train_data_path': args.train_data,
        'val_data_path': args.val_data,
        'annotation_file': args.annotations,
    }
    
    # For YOLO models, add data_yaml if provided
    if hasattr(pipeline.trainer, 'train') and args.data_yaml:
        train_kwargs['data_yaml'] = args.data_yaml
    
    # Handle resume
    if args.resume:
        train_kwargs['resume_checkpoint'] = args.resume
    
    results = pipeline.train(**train_kwargs)
    
    output_dir = results.get('output_dir', pipeline.run_dir)
    print(f"Training completed. Results saved to {output_dir}")
    print(f"Run ID: {pipeline.run_id}")
    if isinstance(results, dict):
        if 'best_val_metric' in results:
            print(f"Best validation metric: {results['best_val_metric']}")
        if 'results' in results:
            print(f"YOLO training results: {results['results']}")


if __name__ == "__main__":
    main()