# API Reference

This document provides detailed API documentation for the Street Scene Optimization framework.

## Core Components

### StreetScenePipeline

The main pipeline class for end-to-end training and evaluation.

```python
class StreetScenePipeline:
    def __init__(
        self,
        config_path: str,
        task_type: str,
        log_level: str = "INFO",
        detection_task: Optional[str] = None,
        classification_task: Optional[str] = None,
    )
    def train(self, train_data_path: str, val_data_path: Optional[str] = None, 
              annotation_file: Optional[str] = None, output_dir: str = "./outputs",
              resume_checkpoint: Optional[str] = None) -> Dict[str, Any]
    def evaluate(self, test_data_path: str, checkpoint_path: str,
                 annotation_file: Optional[str] = None, 
                 output_dir: str = "./outputs") -> Dict[str, Any]
```

**Parameters:**
- `config_path`: Path to YAML configuration file
- `task_type`: Either "detection" or "classification"
- `detection_task`: Name of the detection task from `detection_config.yaml` (optional)
- `classification_task`: Name of the classification task from `classification_config.yaml` (required when `task_type="classification"`)
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

**Example:**
```python
from src.pipelines.pipeline import StreetScenePipeline

pipeline = StreetScenePipeline(
    config_path="configs/detection_config.yaml",
    task_type="detection"
)

results = pipeline.train(
    train_data_path="/data/train",
    val_data_path="/data/val",
    output_dir="./outputs"
)
```

### Base Classes

#### BaseDetectionModel

```python
class BaseDetectionModel(nn.Module):
    def __init__(self, num_classes: int, **kwargs)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]
```

#### TimmClassificationModel

```python
class TimmClassificationModel(nn.Module):
    def __init__(self, model_cfg: Dict[str, Any], heads_cfg: Dict[str, Dict[str, Any]])
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]
```

- Builds a ResNet-family backbone via `timm.create_model(..., num_classes=0)`
- Adds one linear head per entry in `heads_cfg`
- Returns a dictionary of logits keyed by head name (e.g., `{"vehicle_type": logits}`)

#### Trainer

```python
class Trainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any], 
                 device: Optional[str] = None)
    def train_epoch(self, train_loader) -> Dict[str, float]
    def validate_epoch(self, val_loader) -> Dict[str, float]
    def save_checkpoint(self, filepath: str, metrics: Dict[str, float])
    def load_checkpoint(self, filepath: str)
```

### Data Components

#### StreetSceneDataset

```python
class StreetSceneDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[transforms.Compose] = None,
                 annotation_file: Optional[str] = None)
    def __len__(self) -> int
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]
```

#### Data Utilities

```python
def get_train_transforms(image_size: Tuple[int, int]) -> transforms.Compose
def get_val_transforms(image_size: Tuple[int, int]) -> transforms.Compose
def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                      num_workers: int = 4, pin_memory: bool = True) -> DataLoader
```

### Model Factories

#### Detection Models

```python
def create_detection_model(config: Dict[str, Any]) -> BaseDetectionModel
```

Supported models:
- `ssd300`: Single Shot Detector with ResNet-50 backbone

#### Classification Models

```python
def create_classification_model(
    config: Dict[str, Any],
    task_name: str,
    task_config: Optional[Dict[str, Any]] = None,
) -> nn.Module
```

`task_name` must correspond to an entry under `classification.tasks` in `configs/classification_config.yaml`. Each task resolves the backbone configuration and the set of heads to attach to the shared features. Multi-head outputs are returned as a dictionary keyed by head name.

### Common Utilities

```python
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None
def load_config(config_path: str) -> Dict[str, Any]
def save_config(config: Dict[str, Any], save_path: str) -> None
def get_device(device: Optional[str] = None) -> str
def seed_everything(seed: int = 42) -> None
```

## Configuration Schema

### Detection Configuration

```yaml
detection:
  model:
    name: str              # Model name (e.g., "ssd300")
    backbone: str          # Backbone network (e.g., "resnet50")
    num_classes: int       # Number of classes including background
  
  training:
    batch_size: int        # Training batch size
    learning_rate: float   # Initial learning rate
    epochs: int           # Number of training epochs
    optimizer: str        # Optimizer type ("sgd", "adam")
    momentum: float       # SGD momentum (if using SGD)
    weight_decay: float   # L2 regularization
    
  data:
    image_size: [int, int] # Input image size [height, width]
    mean: [float, float, float]    # RGB normalization mean
    std: [float, float, float]     # RGB normalization std
    
  augmentation:
    horizontal_flip: float # Probability of horizontal flip
    brightness: float      # Brightness adjustment range
    contrast: float        # Contrast adjustment range
    saturation: float      # Saturation adjustment range

mixed_precision:
  enabled: bool          # Enable mixed precision training
  loss_scale: str        # Loss scaling ("dynamic" or numeric value)
```

### Classification Configuration

```yaml
classification:
  defaults:
    model:
      backbone: str        # timm backbone name (e.g., "resnet50")
      pretrained: bool     # Whether to load ImageNet weights
      global_pool: str     # Pooling mode passed to timm ("avg", "max", etc.)
      freeze_backbone: bool # Freeze backbone parameters
    training:
      batch_size: int
      learning_rate: float
      epochs: int
      optimizer: str
      weight_decay: float
    data:
      image_size: [int, int]
      mean: [float, float, float]
      std: [float, float, float]
  tasks:
    <task_name>:
      description: str
      model: {...}      # Optional overrides (e.g., change backbone)
      training: {...}   # Optional overrides (batch size, lr, etc.)
      data: {...}       # Optional overrides (image size, normalization)
      heads:
        <head_name>:
          num_classes: int
          loss: str           # "cross_entropy", "bce_with_logits", etc.
          metrics: [str, ...] # Metrics to log for this head (e.g., ["accuracy"])
          class_weights: [float, ...]   # Optional PyTorch class weights
          pos_weight: [float, ...]      # Optional BCE positive-class weights
```

- `defaults` supply shared settings; each task inherits and may override them.
- `heads` defines one classifier per attribute. The trainer sums the per-head losses (optionally weighted) and logs the requested metrics for every head each epoch.
- Select a task at runtime via `--classification-task <task_name>` when using the CLI, or by passing `classification_task="<task_name>"` to `StreetScenePipeline`.

## Error Handling

The framework uses standard Python exceptions with custom error messages:

- `ValueError`: Invalid configuration parameters or task types
- `FileNotFoundError`: Missing data files or checkpoints
- `RuntimeError`: Training/inference errors
- `torch.cuda.OutOfMemoryError`: GPU memory errors

## Logging

The framework provides comprehensive logging:

```python
import logging
from src.common.utils import setup_logging

# Setup logging
setup_logging(log_level="INFO", log_file="training.log")

# Use in your code
logger = logging.getLogger(__name__)
logger.info("Training started")
logger.error("Error occurred during training")
```

Log levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General information messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors