# Evaluation and Reporting Guide

This guide explains how to use the evaluation and reporting system in the Street Scene framework. The system provides comprehensive metrics computation, report generation, and reproducibility verification for all supported tasks.

## Table of Contents

1. [Overview](#overview)
2. [Metrics Computation](#metrics-computation)
3. [Report Generation](#report-generation)
4. [Workflow Automation](#workflow-automation)
5. [Comparing Runs](#comparing-runs)
6. [Verifying Reproducibility](#verifying-reproducibility)
7. [Deployment](#deployment)

## Overview

The evaluation and reporting system provides:

- **Task-aware metrics**: Detection (mAP, PR curves), Tracking (IDF1, MOTA), Classification (accuracy, F1, AUROC)
- **Automated reporting**: JSON, Markdown, and HTML reports with metrics, config metadata, and dataset info
- **Comparison tools**: Compare multiple runs side-by-side
- **Reproducibility verification**: Validate that saved checkpoints reproduce stored metrics
- **Deployment packaging**: Package models for downstream use

## Metrics Computation

### Detection Metrics

For object detection tasks, the system computes:

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.5 to 0.95
- **Precision & Recall**: Per-class and overall metrics
- **PR Curves**: Precision-Recall curves (when available)

Example:
```python
from src.evaluation.metrics import compute_detection_metrics

# Compute detection metrics
metrics = compute_detection_metrics(
    predictions=yolo_results,
    class_names=['car', 'truck', 'bus']
)

print(f"mAP@0.5: {metrics.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.map50_95:.4f}")
```

### Tracking Metrics

For object tracking tasks, the system computes:

- **MOTA**: Multiple Object Tracking Accuracy
- **MOTP**: Multiple Object Tracking Precision
- **IDF1**: ID F1 Score
- **ID Switches**: Number of identity switches
- **Fragmentations**: Number of track fragmentations

Example:
```python
from src.evaluation.metrics import compute_tracking_metrics

# Compute tracking metrics
metrics = compute_tracking_metrics(
    predictions=tracking_results,
    ground_truth=gt_tracks
)

print(f"MOTA: {metrics.mota:.4f}")
print(f"IDF1: {metrics.idf1:.4f}")
```

### Classification Metrics

For classification tasks (vehicle classification, human attributes), the system computes:

- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1**: Per-class and macro-averaged
- **AUROC**: Area Under ROC Curve (when probabilities available)
- **Confusion Matrix**: Full confusion matrix

Example:
```python
from src.evaluation.metrics import compute_classification_metrics

# Compute classification metrics
metrics = compute_classification_metrics(
    predictions=pred_labels,
    ground_truth=true_labels,
    class_names=['sedan', 'suv', 'truck'],
    probabilities=pred_probs
)

print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"F1 Score: {metrics.f1_score:.4f}")
print(f"AUROC: {metrics.auroc:.4f}")
```

## Report Generation

Training and evaluation runs automatically generate comprehensive reports including:

- Metrics summary
- Configuration metadata
- Dataset information
- Training history (loss curves, etc.)
- Visualization plots
- Reproduction checklist

Reports are saved in multiple formats:
- `metrics_report.json`: Machine-readable format
- `metrics_report.md`: Markdown format for GitHub/documentation
- `metrics_report.html`: Standalone HTML report
- `plots/`: Directory with visualization plots

### Manual Report Generation

```python
from src.evaluation.reporting import generate_report

# Generate report
report_paths = generate_report(
    output_dir='./outputs/my_run',
    metrics=computed_metrics,
    config=config_dict,
    task_type='detection',
    task_name='vehicle_detection',
    training_history=history,
    checkpoint_path='./outputs/my_run/best_model.pth'
)

print(f"JSON report: {report_paths['json']}")
print(f"Markdown report: {report_paths['markdown']}")
```

## Workflow Automation

### Full Lifecycle Workflow

The `run_workflow.py` script executes the complete pipeline: data prep → training → evaluation → report archiving.

Basic usage:
```bash
python scripts/run_workflow.py \
    --config configs/detection_config.yaml \
    --task-type detection \
    --detection-task vehicle_detection \
    --train-data /path/to/train \
    --val-data /path/to/val \
    --test-data /path/to/test \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --output-dir ./outputs \
    --run-name my_vehicle_detection_run
```

Options:
- `--skip-training`: Skip training, only run evaluation
- `--skip-evaluation`: Skip evaluation, only run training
- `--resume`: Resume from checkpoint
- `--compare-with`: Compare with previous runs

Example workflow:
```bash
# Full workflow with comparison
python scripts/run_workflow.py \
    --config configs/detection_config.yaml \
    --task-type detection \
    --detection-task vehicle_detection \
    --train-data ./data/train \
    --val-data ./data/val \
    --test-data ./data/test \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --output-dir ./outputs \
    --compare-with ./outputs/previous_run1 ./outputs/previous_run2
```

The script will:
1. Initialize the pipeline
2. Train the model (if not skipped)
3. Evaluate on test data (if not skipped)
4. Generate comprehensive reports
5. Compare with previous runs (if requested)
6. Archive all metadata for reproducibility

## Comparing Runs

Use `compare_runs.py` to compare multiple training/evaluation runs:

```bash
python scripts/compare_runs.py \
    --run-ids ./outputs/run1 ./outputs/run2 ./outputs/run3 \
    --output ./comparison_report \
    --format both \
    --metric mAP@0.5
```

Options:
- `--run-ids`: List of run directories to compare
- `--output`: Output directory for comparison report
- `--format`: Output format (`json`, `markdown`, or `both`)
- `--metric`: Specific metric to highlight

The comparison report includes:
- Summary table of all runs
- Side-by-side metrics comparison
- Best performing run for each metric
- Trend analysis (if applicable)

Example output:
```
Run Comparison Report
=====================

| Run | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-----|---------|--------------|-----------|--------|
| run1 | 0.8523 | 0.7234 | 0.8612 | 0.8234 |
| run2 | 0.8745 | 0.7456 | 0.8823 | 0.8456 |
| run3 | 0.8612 | 0.7345 | 0.8734 | 0.8345 |

Best Performance (mAP@0.5): run2 (0.8745)
```

## Verifying Reproducibility

Use `verify_repro.py` to validate that a saved checkpoint reproduces stored metrics:

```bash
python scripts/verify_repro.py \
    --run-id ./outputs/my_run \
    --test-data /path/to/test \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --tolerance 0.01
```

Options:
- `--run-id`: Path to run directory to verify
- `--test-data`: Path to test data (uses original if not specified)
- `--data-yaml`: Path to dataset YAML
- `--tolerance`: Tolerance for metric comparison (default: 0.01)

The script will:
1. Load the original metrics
2. Load the checkpoint and configuration
3. Re-evaluate on test data
4. Compare metrics within tolerance
5. Report discrepancies (if any)
6. Return exit code 0 if verified, 1 if failed

Example output:
```
REPRODUCIBILITY VERIFICATION
=============================

Tolerance: ±0.01

✓ ALL METRICS MATCH WITHIN TOLERANCE

Metric Comparison:
------------------
✓ mAP@0.5          | Original:   0.8523 | Reproduced:   0.8521 | Diff: 0.0002
✓ mAP@0.5:0.95     | Original:   0.7234 | Reproduced:   0.7233 | Diff: 0.0001
✓ Precision        | Original:   0.8612 | Reproduced:   0.8614 | Diff: 0.0002
✓ Recall           | Original:   0.8234 | Reproduced:   0.8236 | Diff: 0.0002
```

## Deployment

Use `deploy.py` to package a trained model for deployment:

```bash
python scripts/deploy.py \
    --run-id ./outputs/my_best_run \
    --output-dir ./deployment \
    --deployment-name vehicle_detector_v1 \
    --include-metrics
```

Options:
- `--run-id`: Path to run directory to deploy
- `--output-dir`: Directory to save deployment package
- `--deployment-name`: Name for deployment package
- `--include-metrics`: Include metrics report in package

The deployment package includes:
- Model checkpoint
- Configuration file
- Deployment manifest (metadata)
- README with usage instructions
- Metrics report (if included)

## Best Practices

### 1. Always Use run_workflow.py for Production Runs

This ensures all metadata is captured and reports are generated automatically.

### 2. Compare Against Baselines

Use `--compare-with` to track improvements over baseline models.

### 3. Verify Reproducibility Regularly

Run `verify_repro.py` on important checkpoints to ensure they can be reliably reproduced.

### 4. Archive Run Directories

Keep all run directories for future comparison and analysis. They contain:
- Complete configuration
- Training history
- Evaluation metrics
- Checkpoint files
- Generated reports

### 5. Use Meaningful Run Names

Use descriptive names that include:
- Task name
- Model variant
- Key hyperparameters
- Date/version

Example: `vehicle_detection_yolov8m_lr0.0001_20231215`

## Integration with CI/CD

The evaluation and reporting system can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Train and Evaluate
  run: |
    python scripts/run_workflow.py \
      --config configs/detection_config.yaml \
      --task-type detection \
      --detection-task vehicle_detection \
      --train-data ./data/train \
      --val-data ./data/val \
      --test-data ./data/test \
      --data-yaml configs/datasets/vehicle_detection.yaml \
      --output-dir ./outputs/ci_run

- name: Verify Reproducibility
  run: |
    python scripts/verify_repro.py \
      --run-id ./outputs/ci_run \
      --test-data ./data/test \
      --data-yaml configs/datasets/vehicle_detection.yaml \
      --tolerance 0.01

- name: Compare with Baseline
  run: |
    python scripts/compare_runs.py \
      --run-ids ./outputs/ci_run ./outputs/baseline \
      --output ./comparison \
      --format markdown
```

## Troubleshooting

### Missing Metrics

If metrics are not being computed:
- Ensure scikit-learn is installed for classification metrics
- Ensure motmetrics is installed for tracking metrics
- Check that predictions and ground truth are in the correct format

### Report Generation Fails

If report generation fails:
- Check that output directory is writable
- Ensure matplotlib and seaborn are installed
- Verify that metrics data is valid

### Verification Fails

If reproducibility verification fails:
- Check that the same test data is used
- Verify that the configuration matches
- Ensure random seeds are set consistently
- Check for hardware differences (GPU vs CPU)

## API Reference

For detailed API documentation, see [api_reference.md](api_reference.md).

## Examples

For more examples, see [examples.md](examples.md).
