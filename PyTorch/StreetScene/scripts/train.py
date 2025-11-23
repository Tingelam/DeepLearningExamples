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
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-data", type=str, help="Path to validation data")
    parser.add_argument("--annotations", type=str, help="Path to annotation file")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = StreetScenePipeline(args.config, args.task, args.log_level)
    
    # Train model
    results = pipeline.train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        annotation_file=args.annotations,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume
    )
    
    print(f"Training completed. Results saved to {args.output_dir}")
    print(f"Best validation metric: {results['best_val_metric']}")


if __name__ == "__main__":
    main()