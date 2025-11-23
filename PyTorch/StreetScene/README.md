# StreetScene - Deep Learning for Urban Scene Understanding

StreetScene is a comprehensive deep learning project for computer vision tasks on urban street scenes, featuring a hierarchical configuration management system for reproducible experiments.

## Features

- **Hierarchical Configuration System**: Compose configurations from multiple YAML files (defaults, task-specific, and experiment-level)
- **Multi-task Support**: Detection and classification tasks with pre-configured templates
- **CLI Integration**: Full CLI support with environment variable and command-line overrides
- **Config Validation**: Built-in validation of required configuration fields
- **Reproducibility**: Automatic saving of resolved configurations alongside each run

## Project Structure

```
PyTorch/StreetScene/
├── configs/                          # Configuration files
│   ├── defaults.yaml                 # Default settings for all tasks
│   ├── tasks/                        # Task-specific configurations
│   │   ├── detection/
│   │   │   ├── pedestrian.yaml       # Pedestrian detection config
│   │   │   └── vehicle.yaml          # Vehicle detection config
│   │   └── classification/
│   │       └── vehicle.yaml          # Vehicle classification config
│   └── experiments/                  # Experiment-level configurations
│       ├── baseline.yaml             # Baseline experiment
│       └── large_scale.yaml          # Large-scale distributed training
├── src/
│   └── common/
│       └── config.py                 # Configuration management module
└── scripts/
    └── run_experiment.py             # CLI entry point
```

## Installation

```bash
cd PyTorch/StreetScene
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run a detection task with default settings:

```bash
python scripts/run_experiment.py --task-type detection --task-name pedestrian
```

### With Experiment Config

Apply an experiment configuration:

```bash
python scripts/run_experiment.py \
  --task-type detection \
  --task-name pedestrian \
  --experiment baseline
```

### CLI Overrides

Override configuration values from the command line:

```bash
python scripts/run_experiment.py \
  --task-type detection \
  --task-name pedestrian \
  --experiment baseline \
  training.learning_rate=0.0001 \
  training.batch_size=64 \
  training.epochs=200
```

### Environment Variable Overrides

Use environment variables (prefixed with `APP_`) to override configurations.
Use double underscores (`__`) for nested access; single underscores are preserved:

```bash
export APP_TRAINING__LEARNING_RATE=0.0001
export APP_TRAINING__BATCH_SIZE=64
export APP_HARDWARE__NUM_GPUS=4
python scripts/run_experiment.py --task-type detection --task-name pedestrian
```

The environment variables are converted to dot-notation for config access:
- `APP_TRAINING__LEARNING_RATE` → `training.learning_rate`
- `APP_TRAINING__BATCH_SIZE` → `training.batch_size`
- `APP_HARDWARE__NUM_GPUS` → `hardware.num_gpus`

### Dry Run (Validate Config Only)

Validate and save the configuration without running the experiment:

```bash
python scripts/run_experiment.py \
  --task-type detection \
  --task-name pedestrian \
  --dry-run \
  --print-config
```

## Configuration Hierarchy

Configurations are merged in the following order (later overrides earlier):

1. **Defaults** (`configs/defaults.yaml`) - Base settings for all tasks
2. **Task Config** (`configs/tasks/{type}/{name}.yaml`) - Task-specific overrides
3. **Experiment Config** (`configs/experiments/{name}.yaml`) - Experiment-level settings (optional)
4. **Environment Variables** - Variables prefixed with `APP_` override config values
5. **CLI Overrides** - Command-line arguments have the highest priority

## Configuration Structure

### Default Settings

The `defaults.yaml` file contains base settings for:
- Dataset configuration (paths, types, number of classes)
- Model architecture (backbone, number of layers)
- Training hyperparameters (learning rate, batch size, optimizer)
- Data loading options (workers, prefetching)
- Logging and checkpointing
- Hardware configuration
- Reproducibility settings

### Task-Specific Configs

Task configs override defaults with task-specific settings:
- **Pedestrian Detection**: Single-class detection task
- **Vehicle Detection**: Multi-class object detection
- **Vehicle Classification**: Multi-class image classification

### Experiment Configs

Experiment configs define training scenarios:
- **Baseline**: Minimal overrides for testing
- **Large Scale**: Multi-GPU distributed training configuration

## Advanced Features

### Config Validation

The system validates that all required fields are present. Required fields include:
- Dataset configuration
- Model configuration
- Training parameters
- Hardware settings

### Resolved Config Artifacts

Each run automatically saves the resolved configuration to `outputs/` directory for reproducibility:

```
outputs/
└── {task_type}_{task_name}_{experiment}.yaml
```

This file contains the complete, validated configuration used for that run.

### Custom Overrides

You can use OmegaConf dot-notation for complex overrides:

```bash
python scripts/run_experiment.py \
  --task-type detection \
  --task-name pedestrian \
  dataset.num_classes=2 \
  model.box_score_threshold=0.6 \
  training.scheduler_milestones=[50,100,150]
```

## Configuration Examples

### Pedestrian Detection

```bash
python scripts/run_experiment.py \
  --task-type detection \
  --task-name pedestrian \
  --experiment large_scale \
  --output-dir ./runs/pedestrian_det
```

### Vehicle Classification

```bash
python scripts/run_experiment.py \
  --task-type classification \
  --task-name vehicle \
  --experiment baseline \
  training.epochs=300 \
  hardware.num_gpus=4 \
  hardware.distributed=true
```

## Extending the Configuration System

### Adding New Tasks

1. Create a new YAML file in `configs/tasks/{task_type}/{task_name}.yaml`
2. Define task-specific overrides
3. Use the CLI to run experiments with the new task

### Adding New Experiments

1. Create a new YAML file in `configs/experiments/{experiment_name}.yaml`
2. Define experiment-level settings
3. Reference it via `--experiment {experiment_name}`

### Custom Configuration Fields

You can extend the default configuration with custom fields. They will be validated and accessible in the resolved config:

```yaml
# In your task config
custom_field:
  parameter1: value1
  parameter2: value2
```

Then access via CLI override:
```bash
python scripts/run_experiment.py \
  --task-type detection \
  --task-name pedestrian \
  custom_field.parameter1=new_value
```

## Best Practices

1. **Use Descriptive Names**: Choose clear, self-documenting config field names
2. **Maintain Hierarchy**: Keep related settings organized under parent keys
3. **Document Configs**: Add comments to YAML files explaining non-obvious settings
4. **Version Experiments**: Include version numbers in experiment configs
5. **Save Outputs**: Always preserve resolved configs for reproducibility
6. **Test Overrides**: Use `--dry-run --print-config` to validate configurations before long runs

## References

- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Hydra Documentation](https://hydra.cc/)
