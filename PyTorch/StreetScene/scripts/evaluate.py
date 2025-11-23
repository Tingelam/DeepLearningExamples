#!/usr/bin/env python3
"""
Evaluation script for Street Scene models.
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipelines.pipeline import StreetScenePipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate Street Scene models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--task", type=str, required=True,
                        choices=["detection", "vehicle_classification", "human_attributes"],
                        help="Task type")
    parser.add_argument("--detection-task", type=str,
                        help="Specific detection task (e.g., vehicle_detection, pedestrian_detection)")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--annotations", type=str, help="Path to annotation file")
    parser.add_argument("--data-yaml", type=str, help="Path to YOLO dataset YAML file")
    parser.add_argument("--output-dir", type=str, help="Output directory (defaults to run dir)")
    parser.add_argument("--run-name", type=str, help="Name for this experimental run")
    parser.add_argument("--config-override", type=str, action="append", dest="overrides",
                       help="Config override in format 'key=value' (can be used multiple times)")
    parser.add_argument("--verify-config", action="store_true", default=True,
                       help="Verify config hash from checkpoint")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Create pipeline
    pipeline = StreetScenePipeline(
        args.config,
        args.task,
        log_level=args.log_level,
        detection_task=args.detection_task,
        overrides=args.overrides,
        run_name=args.run_name,
        output_dir=args.output_dir or "./outputs",
        seed=args.seed
    )

    # Evaluate model
    eval_kwargs = {
        'test_data_path': args.test_data,
        'checkpoint_path': args.checkpoint,
        'annotation_file': args.annotations,
        'verify_config': args.verify_config
    }

    # For YOLO models, add data_yaml if provided
    if hasattr(pipeline.trainer, 'evaluate') and args.data_yaml:
        eval_kwargs['data_yaml'] = args.data_yaml

    metrics = pipeline.evaluate(**eval_kwargs)

    output_dir = pipeline.run_dir
    print(f"Evaluation completed. Results saved to {output_dir}")
    print(f"Run ID: {pipeline.run_id}")
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()