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
    
    # Dataset options: either use catalog or direct paths
    parser.add_argument("--dataset", type=str, help="Dataset name from catalog")
    parser.add_argument("--version", type=str, help="Dataset version from catalog")
    parser.add_argument("--catalog", type=str, default="data/datasets.yaml", help="Path to dataset catalog")
    
    # Legacy direct path options
    parser.add_argument("--train-data", type=str, help="Path to training data (legacy)")
    parser.add_argument("--val-data", type=str, help="Path to validation data (legacy)")
    parser.add_argument("--annotations", type=str, help="Path to annotation file (legacy)")
    parser.add_argument("--data-yaml", type=str, help="Path to YOLO dataset YAML file (legacy)")
    
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dataset and not args.train_data:
        parser.error("Either --dataset or --train-data must be provided")
    
    # Create output directory with task-specific subdirectory
    output_dir = args.output_dir
    if args.task == 'detection' and args.detection_task:
        output_dir = os.path.join(args.output_dir, args.detection_task)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pipeline
    pipeline = StreetScenePipeline(
        args.config,
        args.task,
        args.log_level,
        detection_task=args.detection_task
    )
    
    # Train model
    train_kwargs = {
        'train_data_path': args.train_data,
        'val_data_path': args.val_data,
        'annotation_file': args.annotations,
        'output_dir': output_dir,
    }
    
    if args.dataset:
        train_kwargs['dataset_name'] = args.dataset
        train_kwargs['dataset_version'] = args.version
        train_kwargs['catalog_path'] = args.catalog
    
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