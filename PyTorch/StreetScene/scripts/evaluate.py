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
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

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

    # Evaluate model
    eval_kwargs = {
        'test_data_path': args.test_data,
        'checkpoint_path': args.checkpoint,
        'annotation_file': args.annotations,
        'output_dir': output_dir
    }

    # For YOLO models, add data_yaml if provided
    if hasattr(pipeline.trainer, 'evaluate') and args.data_yaml:
        eval_kwargs['data_yaml'] = args.data_yaml

    metrics = pipeline.evaluate(**eval_kwargs)

    print(f"Evaluation completed. Results saved to {output_dir}")
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    main()