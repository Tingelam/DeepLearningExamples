#!/usr/bin/env python3
"""
Deployment script stub for Street Scene models.

Packages the best checkpoint + config for downstream deployment.
"""

import os
import sys
import argparse
import logging
import json
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.reporting import load_run_metrics
from src.common.utils import setup_logging, load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Package model for deployment')
    
    parser.add_argument('--run-id', type=str, required=True,
                       help='Path to run directory to deploy')
    parser.add_argument('--output-dir', type=str, default='./deployment',
                       help='Directory to save deployment package')
    parser.add_argument('--deployment-name', type=str, default=None,
                       help='Name for deployment package (default: auto-generated)')
    parser.add_argument('--include-metrics', action='store_true',
                       help='Include metrics report in deployment package')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info(f"Packaging model for deployment from run: {args.run_id}")
    
    # Load run information
    run_dir = Path(args.run_id)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return 1
    
    # Load run info
    run_info_path = run_dir / 'run_info.json'
    if run_info_path.exists():
        with open(run_info_path, 'r') as f:
            run_info = json.load(f)
    else:
        logger.warning("No run_info.json found")
        run_info = {}
    
    # Find checkpoint
    checkpoint_candidates = [
        run_dir / 'training' / 'best_model.pth',
        run_dir / 'best_model.pth',
        run_dir / 'weights' / 'best.pt'  # YOLO format
    ]
    
    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if candidate.exists():
            checkpoint_path = candidate
            break
    
    if not checkpoint_path:
        logger.error("No checkpoint found in run directory")
        return 1
    
    logger.info(f"Found checkpoint: {checkpoint_path}")
    
    # Create deployment name
    if args.deployment_name:
        deployment_name = args.deployment_name
    else:
        task_name = run_info.get('task_name', 'model')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        deployment_name = f"{task_name}_deployment_{timestamp}"
    
    # Create deployment directory
    deployment_dir = Path(args.output_dir) / deployment_name
    deployment_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating deployment package: {deployment_dir}")
    
    # Copy checkpoint
    checkpoint_dest = deployment_dir / checkpoint_path.name
    shutil.copy2(checkpoint_path, checkpoint_dest)
    logger.info(f"Copied checkpoint to: {checkpoint_dest}")
    
    # Copy config if available
    config_path = run_info.get('config')
    if config_path and Path(config_path).exists():
        config_dest = deployment_dir / 'config.yaml'
        shutil.copy2(config_path, config_dest)
        logger.info(f"Copied config to: {config_dest}")
    
    # Copy metrics report if requested
    if args.include_metrics:
        metrics_report_path = run_dir / 'evaluation' / 'metrics_report.json'
        if not metrics_report_path.exists():
            metrics_report_path = run_dir / 'training' / 'metrics_report.json'
        
        if metrics_report_path.exists():
            metrics_dest = deployment_dir / 'metrics_report.json'
            shutil.copy2(metrics_report_path, metrics_dest)
            logger.info(f"Copied metrics report to: {metrics_dest}")
    
    # Create deployment manifest
    manifest = {
        'deployment_name': deployment_name,
        'source_run': str(run_dir),
        'task_type': run_info.get('task_type', 'unknown'),
        'detection_task': run_info.get('detection_task'),
        'checkpoint': checkpoint_path.name,
        'created_at': datetime.now().isoformat(),
        'run_info': run_info
    }
    
    # Add metrics summary if available
    try:
        metrics = load_run_metrics(str(run_dir))
        manifest['metrics_summary'] = metrics.get('metrics', {})
    except:
        pass
    
    manifest_path = deployment_dir / 'deployment_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Created deployment manifest: {manifest_path}")
    
    # Create README
    readme_content = f"""# Deployment Package: {deployment_name}

## Overview
This package contains a trained model ready for deployment.

## Contents
- `{checkpoint_path.name}`: Model checkpoint
- `config.yaml`: Model configuration (if available)
- `deployment_manifest.json`: Deployment metadata
- `metrics_report.json`: Training/evaluation metrics (if included)

## Source
- Run Directory: {run_dir}
- Task Type: {run_info.get('task_type', 'unknown')}
- Created: {datetime.now().isoformat()}

## Usage
To load and use this model:

```python
from src.pipelines.pipeline import StreetScenePipeline

# Initialize pipeline
pipeline = StreetScenePipeline(
    config_path='config.yaml',
    task_type='{run_info.get('task_type', 'detection')}'
)

# Load checkpoint
pipeline.model.load_state_dict(torch.load('{checkpoint_path.name}'))

# Run inference
results = pipeline.inference(...)
```

## Notes
This is a deployment package generated automatically. For reproducibility,
refer to the original run directory and use the verify_repro.py script.
"""
    
    readme_path = deployment_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Created README: {readme_path}")
    
    logger.info("=" * 80)
    logger.info("DEPLOYMENT PACKAGE CREATED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Package location: {deployment_dir}")
    logger.info(f"Package name: {deployment_name}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
