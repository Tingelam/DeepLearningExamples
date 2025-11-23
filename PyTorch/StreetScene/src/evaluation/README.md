# Evaluation Module

This module provides comprehensive evaluation metrics and reporting capabilities for Street Scene tasks.

## Overview

The evaluation module supports:
- **Task-aware metrics** for detection, tracking, and classification
- **Automated report generation** in multiple formats (JSON, Markdown, HTML)
- **Comparison tools** for analyzing multiple runs
- **Reproducibility verification** to ensure saved models can be reliably restored

## Module Structure

```
src/evaluation/
├── __init__.py       # Module exports
├── metrics.py        # Metrics computation
└── reporting.py      # Report generation and comparison
```

## Metrics Module (`metrics.py`)

### Detection Metrics

```python
from src.evaluation.metrics import compute_detection_metrics, DetectionMetrics

# Compute from YOLO results
metrics = compute_detection_metrics(
    predictions=yolo_results,
    class_names=['car', 'truck', 'bus']
)

print(f"mAP@0.5: {metrics.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.map50_95:.4f}")
print(f"Precision: {metrics.precision:.4f}")
print(f"Recall: {metrics.recall:.4f}")
```

**Metrics Computed:**
- mAP@0.5: Mean Average Precision at IoU 0.5
- mAP@0.5:0.95: Mean Average Precision averaged over IoU 0.5-0.95
- Precision: Overall precision
- Recall: Overall recall
- Per-class mAP (when class names provided)

### Tracking Metrics

```python
from src.evaluation.metrics import compute_tracking_metrics, TrackingMetrics

# Compute tracking metrics
metrics = compute_tracking_metrics(
    predictions=tracking_results,
    ground_truth=gt_tracks
)

print(f"MOTA: {metrics.mota:.4f}")
print(f"MOTP: {metrics.motp:.4f}")
print(f"IDF1: {metrics.idf1:.4f}")
print(f"ID Switches: {metrics.num_switches}")
```

**Metrics Computed:**
- MOTA: Multiple Object Tracking Accuracy
- MOTP: Multiple Object Tracking Precision
- IDF1: ID F1 Score
- ID Switches: Number of identity switches
- Fragmentations: Number of track fragmentations
- False Positives: Number of false positive detections
- Misses: Number of missed detections

**Requirements:** `motmetrics>=1.2.0`

### Classification Metrics

```python
from src.evaluation.metrics import compute_classification_metrics, ClassificationMetrics

# Compute classification metrics
metrics = compute_classification_metrics(
    predictions=pred_labels,
    ground_truth=true_labels,
    class_names=['sedan', 'suv', 'truck'],
    task_type='multiclass',
    probabilities=pred_probs
)

print(f"Accuracy: {metrics.accuracy:.4f}")
print(f"F1 Score: {metrics.f1_score:.4f}")
print(f"AUROC: {metrics.auroc:.4f}")
```

**Metrics Computed:**
- Accuracy: Overall accuracy
- Precision: Precision (macro-averaged for multiclass)
- Recall: Recall (macro-averaged for multiclass)
- F1 Score: F1 score
- AUROC: Area Under ROC Curve (when probabilities provided)
- Confusion Matrix: Full confusion matrix
- Per-class metrics: Precision, recall, F1 per class

**Requirements:** `scikit-learn>=0.24.0`

### Multi-Task Classification

For multi-task models (e.g., human attributes with gender, age, clothing):

```python
from src.evaluation.metrics import compute_multi_task_classification_metrics

# Define task configurations
task_configs = {
    'gender': {'class_names': ['male', 'female'], 'task_type': 'binary'},
    'age': {'class_names': ['child', 'adult', 'senior'], 'task_type': 'multiclass'},
    'clothing': {'class_names': ['casual', 'formal'], 'task_type': 'binary'}
}

# Compute metrics for all tasks
all_metrics = compute_multi_task_classification_metrics(
    predictions={'gender': gender_preds, 'age': age_preds, 'clothing': clothing_preds},
    ground_truth={'gender': gender_gt, 'age': age_gt, 'clothing': clothing_gt},
    task_configs=task_configs
)

for task_name, task_metrics in all_metrics.items():
    print(f"{task_name}: accuracy={task_metrics.accuracy:.4f}")
```

## Reporting Module (`reporting.py`)

### MetricsReporter Class

```python
from src.evaluation.reporting import MetricsReporter

# Create reporter
reporter = MetricsReporter(
    output_dir='./outputs/my_run',
    task_type='detection',
    task_name='vehicle_detection'
)

# Generate comprehensive report
report_paths = reporter.generate_report(
    metrics=computed_metrics,
    config=config_dict,
    dataset_info={'data_yaml': 'path/to/data.yaml'},
    training_history=history,
    checkpoint_path='./outputs/best_model.pth',
    compare_with=['./outputs/baseline_run']
)

print(f"JSON report: {report_paths['json']}")
print(f"Markdown report: {report_paths['markdown']}")
print(f"HTML report: {report_paths['html']}")
```

### Report Contents

Each report includes:
- **Metrics Summary**: All computed metrics in table format
- **Configuration**: Complete YAML configuration
- **Dataset Information**: Dataset paths, class names, etc.
- **Training History**: Loss curves, epoch metrics
- **Checkpoint Info**: Path to saved model
- **Reproduction Checklist**: Steps to reproduce results
- **Comparison**: Side-by-side comparison with previous runs (if requested)
- **Plots**: Visualizations in `plots/` directory

### Report Formats

1. **JSON** (`metrics_report.json`): Machine-readable format
   ```json
   {
     "task_name": "vehicle_detection",
     "timestamp": "2023-12-15T10:30:00",
     "metrics": {
       "mAP@0.5": 0.8523,
       "precision": 0.8612
     },
     "config": {...},
     "dataset_info": {...}
   }
   ```

2. **Markdown** (`metrics_report.md`): Documentation-friendly format
   ```markdown
   # vehicle_detection - Metrics Report
   
   ## Metrics
   | Metric | Value |
   |--------|-------|
   | mAP@0.5 | 0.8523 |
   ```

3. **HTML** (`metrics_report.html`): Standalone web page with styled tables

4. **Plots** (`plots/`): PNG images of training curves, confusion matrices, etc.

### Utility Functions

```python
from src.evaluation.reporting import generate_report, compare_runs, load_run_metrics

# Quick report generation
generate_report(
    output_dir='./outputs',
    metrics=metrics,
    config=config,
    task_type='detection',
    task_name='vehicle_detection'
)

# Compare multiple runs
comparison = compare_runs(
    run_dirs=['./outputs/run1', './outputs/run2', './outputs/run3'],
    output_path='./comparison.json'
)

# Load metrics from a run
metrics = load_run_metrics('./outputs/my_run')
```

## Integration with Pipeline

The evaluation module is automatically integrated with the training/evaluation pipeline:

```python
from src.pipelines.pipeline import StreetScenePipeline

# Initialize pipeline
pipeline = StreetScenePipeline(
    config_path='configs/detection_config.yaml',
    task_type='detection',
    detection_task='vehicle_detection'
)

# Training automatically generates report
results = pipeline.train(
    train_data_path='/path/to/train',
    val_data_path='/path/to/val',
    output_dir='./outputs'
)
# Report saved to: ./outputs/metrics_report.{json,md,html}

# Evaluation automatically generates report
metrics = pipeline.evaluate(
    test_data_path='/path/to/test',
    checkpoint_path='./outputs/best_model.pth',
    output_dir='./outputs/eval'
)
# Report saved to: ./outputs/eval/metrics_report.{json,md,html}
```

## Dependencies

### Required
- `numpy>=1.19.0`
- `pyyaml>=5.4.0`

### Optional (with graceful degradation)
- `scikit-learn>=0.24.0` - For classification metrics
- `motmetrics>=1.2.0` - For tracking metrics
- `matplotlib>=3.3.0` - For plot generation
- `seaborn>=0.11.0` - For enhanced visualizations

If optional dependencies are missing, warnings are issued but the module continues to function with reduced capabilities.

## Error Handling

The module uses defensive programming:
- Try-except blocks wrap report generation to avoid breaking training
- Missing dependencies issue warnings but don't crash
- Invalid inputs return empty metrics with warnings
- File I/O errors are caught and logged

## Examples

### Complete Workflow

```python
from src.pipelines.pipeline import StreetScenePipeline
from src.evaluation.reporting import compare_runs
from src.evaluation.metrics import compute_detection_metrics

# 1. Train model
pipeline = StreetScenePipeline('configs/detection_config.yaml', 'detection', detection_task='vehicle_detection')
results = pipeline.train(train_data_path='/data/train', output_dir='./outputs/run1')

# 2. Evaluate model
metrics = pipeline.evaluate(test_data_path='/data/test', checkpoint_path='./outputs/run1/best_model.pth', output_dir='./outputs/run1/eval')

# 3. Compare with baseline
comparison = compare_runs(['./outputs/run1', './outputs/baseline'], output_path='./comparison.json')

# 4. Print comparison
for run in comparison:
    print(f"Run: {run['run_name']}")
    print(f"  mAP@0.5: {run['metrics']['metrics']['mAP@0.5']:.4f}")
```

## See Also

- [Evaluation and Reporting Guide](../../docs/evaluation_and_reporting.md) - Comprehensive documentation
- [API Reference](../../docs/api_reference.md) - Complete API documentation
- Workflow scripts: `scripts/run_workflow.py`, `scripts/compare_runs.py`, `scripts/verify_repro.py`
