# API Reference

This document provides detailed API documentation for the Street Scene Optimization framework.

## Core Components

### StreetScenePipeline

The main pipeline class for end-to-end training and evaluation.

```python
class StreetScenePipeline:
    def __init__(self, config_path: str, task_type: str, log_level: str = "INFO")
    def train(self, train_data_path: str, val_data_path: Optional[str] = None, 
              annotation_file: Optional[str] = None, output_dir: str = "./outputs",
              resume_checkpoint: Optional[str] = None) -> Dict[str, Any]
    def evaluate(self, test_data_path: str, checkpoint_path: str,
                 annotation_file: Optional[str] = None, 
                 output_dir: str = "./outputs") -> Dict[str, Any]
```

**Parameters:**
- `config_path`: Path to YAML configuration file
- `task_type`: One of "detection", "vehicle_classification", "human_attributes"
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

#### BaseClassificationModel

```python
class BaseClassificationModel(nn.Module):
    def __init__(self, num_classes: int, **kwargs)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    def predict(self, x: torch.Tensor) -> torch.Tensor
```

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
def create_classification_model(config: Dict[str, Any], task_type: str) -> BaseClassificationModel
```

Supported task types:
- `vehicle`: Vehicle type classification
- `human_attributes`: Multi-task human attribute classification

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
  vehicle:
    model:
      name: str           # Model name (e.g., "resnet50")
      num_classes: int    # Number of vehicle classes
      pretrained: bool    # Use pretrained weights
    
    training:
      batch_size: int     # Training batch size
      learning_rate: float # Initial learning rate
      epochs: int         # Number of training epochs
      optimizer: str      # Optimizer type
      momentum: float     # SGD momentum
      weight_decay: float # L2 regularization
      
    data:
      image_size: [int, int] # Input image size
      mean: [float, float, float]    # RGB normalization mean
      std: [float, float, float]     # RGB normalization std
      
    augmentation:
      horizontal_flip: float # Probability of horizontal flip
      rotation: int          # Max rotation degrees
      brightness: float      # Brightness adjustment range
      contrast: float        # Contrast adjustment range
      
  human_attributes:
    model:
      name: str           # Model name (e.g., "resnet34")
      num_classes: int    # Total number of attribute combinations
      pretrained: bool    # Use pretrained weights
    
    training:
      batch_size: int     # Training batch size
      learning_rate: float # Initial learning rate
      epochs: int         # Number of training epochs
      optimizer: str      # Optimizer type
      weight_decay: float # L2 regularization
      
    data:
      image_size: [int, int] # Input image size
      mean: [float, float, float]    # RGB normalization mean
      std: [float, float, float]     # RGB normalization std
```

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