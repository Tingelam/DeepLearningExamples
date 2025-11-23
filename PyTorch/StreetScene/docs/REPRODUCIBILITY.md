# Reproducibility Guide for Street Scene Models

This guide explains how to use the reproducibility features of the Street Scene optimization framework to ensure consistent and verifiable experiment results.

## Overview

The reproducibility system consists of the following components:

1. **ConfigManager**: Manages configuration loading, CLI overrides, and experiment IDs
2. **Model Registry**: Tracks trained models and their metadata
3. **Checkpoint Metadata**: Embeds configuration hashes and metadata in checkpoints
4. **Deterministic Seeding**: Locks all random number generators

## Key Features

### 1. Configuration Management

The `ConfigManager` utility provides centralized configuration management with:

- **Base Configuration**: Load from YAML files
- **CLI Overrides**: Apply runtime overrides without modifying files
- **Run IDs**: Automatically generate unique experiment identifiers
- **Config Hashing**: Track configuration changes for reproducibility
- **Metadata Capture**: Record framework versions and git commits

### 2. Experiment Run Directories

Each training run creates a structured output directory containing:

```
outputs/
├── <TIMESTAMP>_<RUN_NAME>/
│   ├── config.yaml              # Resolved configuration for this run
│   ├── metadata.yaml            # Framework versions and system info
│   ├── best_model.pth           # Best model checkpoint
│   ├── checkpoint_epoch_*.pth   # Epoch checkpoints
│   └── other outputs...
```

### 3. Model Registry

A lightweight JSON registry (`registry/model_registry.json`) tracks all trained models:

- Run ID and task type
- Configuration hash (for detecting changes)
- Checkpoint paths
- Training metrics
- Framework versions used

### 4. Checkpoint Metadata

Model checkpoints now embed:

- Configuration hash (for integrity verification)
- Framework metadata (PyTorch, ultralytics versions)
- Training metrics
- Full resolved configuration

## Usage

### Basic Training with Run Name

```bash
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --train-data /path/to/train \
    --val-data /path/to/val \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --run-name my_vehicle_detector_v1
```

Output:
```
Training completed. Results saved to outputs/20240115_143022_my_vehicle_detector_v1
Run ID: 20240115_143022_my_vehicle_detector_v1
```

### Training with Configuration Overrides

Override any configuration parameter without modifying the YAML file:

```bash
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --train-data /path/to/train \
    --val-data /path/to/val \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --run-name exp_high_lr \
    --config-override detection.training.learning_rate=0.001 \
    --config-override detection.training.batch_size=64
```

Nested keys are accessed with dot notation. Supported value types:
- Numbers: `learning_rate=0.001`
- Booleans: `enabled=true`
- Lists: `image_size=[512,512]` (JSON format)

### Reproducing Results with Saved Config

To reproduce a training run with the exact same configuration:

```bash
# Extract the config from a previous run
RUN_ID="20240115_143022_my_vehicle_detector_v1"

python scripts/train.py \
    --config outputs/$RUN_ID/config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --train-data /path/to/train \
    --val-data /path/to/val \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --seed 42
```

### Evaluation with Config Verification

When evaluating a model checkpoint, the system verifies that the configuration matches:

```bash
python scripts/evaluate.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --test-data /path/to/test \
    --checkpoint outputs/20240115_143022_my_vehicle_detector_v1/best_model.pth \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --verify-config
```

If the config hash doesn't match, you'll see a warning:
```
Config hash mismatch! Checkpoint config hash: abc123de, Current config hash: xyz789ab. This may affect reproducibility.
```

### Using the Model Registry

Python API for accessing registered models:

```python
from src.common.model_registry import ModelRegistry

registry = ModelRegistry("registry/model_registry.json")

# Find best model for a task
best_entry = registry.find_best_entry_by_metric(
    task='vehicle_detection',
    metric_name='best_val_metric',
    mode='max'
)
print(f"Best model: {best_entry['run_id']}")
print(f"Metrics: {best_entry['metrics']}")
print(f"Checkpoint: {best_entry['checkpoint_path']}")

# List all vehicle detection runs
vehicle_entries = registry.find_entries_by_task('vehicle_detection')
for entry in vehicle_entries:
    print(f"{entry['run_id']}: {entry['metrics']}")

# Export registry to CSV
registry.export_to_csv('model_registry.csv')
```

## Reproducibility Workflow

### Step 1: Lock Seeds

Always specify a seed for deterministic behavior:

```bash
--seed 42  # Default in scripts
```

All scripts use the same seed specification for consistent results:
- Random number generator seeds (Python, NumPy)
- PyTorch seeds (CPU and CUDA)
- CUDA operations (deterministic mode, disabled benchmarking)

### Step 2: Save Configuration

Configs are automatically saved during training:

```
outputs/
└── 20240115_143022_my_exp/
    ├── config.yaml              # Full resolved config
    ├── metadata.yaml            # Versions, git commit, etc.
    └── best_model.pth           # Checkpoint with embedded config hash
```

### Step 3: Reproduce from Config + Weights

To reproduce results from a previous experiment:

```bash
# 1. Use the saved config from the run directory
# 2. Use the saved checkpoint
# 3. Match the dataset

python scripts/evaluate.py \
    --config outputs/20240115_143022_my_exp/config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --test-data /path/to/test \
    --checkpoint outputs/20240115_143022_my_exp/best_model.pth \
    --data-yaml configs/datasets/vehicle_detection.yaml
```

### Step 4: Register and Track

All training runs are automatically registered in `registry/model_registry.json`:

```json
[
  {
    "run_id": "20240115_143022_my_vehicle_detector_v1",
    "task": "vehicle_detection",
    "config_hash": "abc123de",
    "checkpoint_path": "/full/path/to/best_model.pth",
    "metrics": {"best_val_metric": 0.92},
    "timestamp": "2024-01-15T14:30:22.123456",
    "metadata": {
      "pytorch_version": "2.0.1",
      "ultralytics_version": "8.0.0",
      "git_commit": "abc123def456"
    }
  }
]
```

## Advanced Features

### Comparing Configurations

Track configuration changes across experiments:

```python
from src.common.config_manager import ConfigManager

cm1 = ConfigManager("outputs/run1/config.yaml")
cm2 = ConfigManager("outputs/run2/config.yaml")

# Compare config hashes
if cm1.get_config_hash() == cm2.get_config_hash():
    print("Same configuration")
else:
    print("Different configurations")

# View specific configs
print("Run 1 learning rate:", cm1.get_config()['detection']['training']['learning_rate'])
print("Run 2 learning rate:", cm2.get_config()['detection']['training']['learning_rate'])
```

### Finding Models by Configuration

```python
from src.common.model_registry import ModelRegistry

registry = ModelRegistry()

# Find all models trained with a specific config
config_hash = "abc123de"
entries = registry.find_entries_by_config_hash(config_hash)
for entry in entries:
    print(f"Run {entry['run_id']}: {entry['metrics']}")
```

### Batch Evaluation

```python
from src.common.model_registry import ModelRegistry
from pipelines.pipeline import StreetScenePipeline

registry = ModelRegistry()

# Evaluate all vehicle detection models
for entry in registry.find_entries_by_task('vehicle_detection'):
    checkpoint = entry['checkpoint_path']
    config_path = f"{checkpoint.rsplit('/', 1)[0]}/config.yaml"
    
    pipeline = StreetScenePipeline(
        config_path,
        'detection',
        detection_task='vehicle_detection'
    )
    
    metrics = pipeline.evaluate(
        test_data_path='/path/to/test',
        checkpoint_path=checkpoint
    )
    print(f"{entry['run_id']}: {metrics}")
```

## Environment Variables

Customize paths and behavior:

```bash
# Custom output directory
export STREET_SCENE_OUTPUT_DIR=/data/experiments

# Custom registry location
export STREET_SCENE_REGISTRY=/data/model_registry.json

# Force specific seed
export STREET_SCENE_SEED=123
```

## Troubleshooting

### Config Hash Mismatch

**Problem**: Evaluation shows config hash mismatch warning.

**Solution**: 
- Ensure you're using the same config file: `outputs/<run_id>/config.yaml`
- Or apply the same overrides as the original training run
- Check for environment differences (different frameworks)

### Missing Metadata

**Problem**: Checkpoint doesn't contain metadata.

**Solution**:
- Checkpoints created before metadata feature was added won't have it
- Re-train the model with current version to generate new checkpoints
- Use legacy `load_checkpoint` method without metadata verification

### Registry File Corruption

**Problem**: Registry JSON file is corrupted.

**Solution**:
```bash
# Backup the corrupted file
cp registry/model_registry.json registry/model_registry.json.bak

# Restore from empty state (existing entries will be lost)
echo "[]" > registry/model_registry.json

# Re-train key models to rebuild registry
```

## Best Practices

1. **Always use `--run-name`**: Makes it easier to identify experiments later
2. **Document overrides**: If using `--config-override`, document your changes
3. **Archive configs**: Keep copies of `outputs/<run_id>/config.yaml` for important experiments
4. **Regular registry backups**: Periodically backup `registry/model_registry.json`
5. **Version your datasets**: Track dataset versions in metadata
6. **Use consistent seeds**: Stick with seed 42 unless you have a reason to change it

## Integration with External Tools

### Wandb Integration

```python
import wandb
from pathlib import Path

def log_to_wandb(run_dir):
    """Log run metadata and config to Wandb"""
    metadata_path = Path(run_dir) / 'metadata.yaml'
    config_path = Path(run_dir) / 'config.yaml'
    
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    wandb.log(metadata)
    wandb.config.update(config)
```

### MLflow Integration

```python
import mlflow
from pathlib import Path
import yaml

def log_to_mlflow(run_dir, run_id):
    """Log run to MLflow"""
    metadata_path = Path(run_dir) / 'metadata.yaml'
    config_path = Path(run_dir) / 'config.yaml'
    
    with open(metadata_path) as f:
        metadata = yaml.safe_load(f)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    mlflow.start_run(run_name=run_id)
    mlflow.log_params(mlflow.utils.validation.bad_path_message(config))
    mlflow.log_dict(metadata, 'metadata.json')
    mlflow.end_run()
```

## References

- [ConfigManager](../src/common/config_manager.py): Configuration management
- [ModelRegistry](../src/common/model_registry.py): Model tracking
- [StreetScenePipeline](../src/pipelines/pipeline.py): Main pipeline with reproducibility
