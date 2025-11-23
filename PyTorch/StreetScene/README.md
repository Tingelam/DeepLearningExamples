# Street Scene Optimization for PyTorch

This repository provides a comprehensive framework for street scene optimization, supporting multiple computer vision tasks including pedestrian/vehicle detection, vehicle classification, and human attribute analysis. The framework is designed for modularity, scalability, and performance optimization on NVIDIA Tensor Core GPUs.

## Table of Contents

- [Overview](#overview)
- [Supported Tasks](#supported-tasks)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [End-to-End Workflow](#end-to-end-workflow)
- [Adding New Tasks](#adding-new-tasks)
- [Performance Optimization](#performance-optimization)
- [Configuration](#configuration)
- [Scripts and Tools](#scripts-and-tools)

## Overview

The Street Scene Optimization framework provides:

- **Modular Architecture**: Clean separation between data, models, training, and evaluation components
- **Multiple Task Support**: Detection, vehicle classification, and human attribute analysis
- **Performance Optimized**: Mixed precision training, multi-GPU support, and Tensor Core utilization
- **Extensible Design**: Easy to add new models, datasets, and tasks
- **Production Ready**: Comprehensive logging, checkpointing, and monitoring

## Supported Tasks

### 1. Object Detection
- **Pedestrian Detection**: Detect pedestrians in street scenes
- **Vehicle Detection**: Detect various types of vehicles (cars, trucks, buses, motorcycles)
- **Multi-class Detection**: Support for COCO-style object detection
- **Tracking Support**: Foundation for object tracking applications

### 2. Vehicle Classification
- **Vehicle Type Classification**: Classify vehicles into categories (sedan, SUV, truck, bus, motorcycle)
- **Make/Model Classification**: Fine-grained vehicle recognition (extensible)
- **Color Classification**: Vehicle color classification
- **Multi-task Learning**: Combine multiple classification tasks

### 3. Human Attribute Analysis
- **Gender Classification**: Binary gender classification
- **Age Group Estimation**: Categorical age group prediction
- **Clothing Type Recognition**: Clothing category classification
- **Multi-task Learning**: Joint learning of multiple attributes

## Architecture

The framework follows a modular architecture with clear separation of concerns:

```
StreetScene/
├── configs/           # Configuration files for different tasks
├── src/
│   ├── common/        # Shared utilities and base classes
│   ├── data/          # Data loading and preprocessing
│   ├── detection/     # Detection models and utilities
│   ├── classification/# Classification models and utilities
│   └── pipelines/     # End-to-end training/evaluation pipelines
├── scripts/           # Training, evaluation, and inference scripts
└── docs/             # Documentation and examples
```

### Key Design Principles

1. **Modularity**: Each component can be used independently or combined as needed
2. **Extensibility**: Easy to add new models, datasets, and tasks
3. **Reusability**: Common functionality shared across tasks
4. **Performance**: Optimized for NVIDIA Tensor Core GPUs
5. **Maintainability**: Clean code structure and comprehensive documentation

## Directory Structure

```
StreetScene/
├── configs/
│   ├── detection_config.yaml      # Detection task configuration
│   └── classification_config.yaml # Classification task configuration
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── utils.py               # Common utilities (logging, device management)
│   │   └── trainer.py             # Base trainer class
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py             # Dataset classes and data utilities
│   ├── detection/
│   │   ├── __init__.py
│   │   └── models.py              # Detection models (SSD, YOLO, etc.)
│   ├── classification/
│   │   ├── __init__.py
│   │   └── models.py              # Classification models (ResNet, EfficientNet, etc.)
│   └── pipelines/
│       ├── __init__.py
│       └── pipeline.py            # End-to-end training/evaluation pipelines
├── scripts/
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── inference.py               # Inference script
├── docs/
│   ├── api_reference.md           # API documentation
│   ├── examples/                  # Usage examples
│   └── tutorials/                 # Step-by-step tutorials
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker container setup
└── README.md                      # This file
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+ with CUDA support
- NVIDIA GPU with Tensor Core support (recommended)
- CUDA 11.0+ (for GPU training)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/StreetScene

# Install Python dependencies
pip install -r requirements.txt
```

### Docker Installation

```bash
# Build Docker image
docker build -t street-scene-pytorch .

# Run container
docker run --gpus all -it --rm -v $(pwd):/workspace street-scene-pytorch bash
```

## Quick Start

### 1. Train a Detection Model

```bash
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --train-data /path/to/train/data \
    --val-data /path/to/val/data \
    --output-dir ./outputs/detection
```

### 2. Train a Vehicle Classification Model

```bash
python scripts/train.py \
    --config configs/classification_config.yaml \
    --task vehicle_classification \
    --train-data /path/to/vehicle/data \
    --val-data /path/to/val/data \
    --output-dir ./outputs/vehicle_classification
```

### 3. Run Inference

```bash
python scripts/inference.py \
    --config configs/detection_config.yaml \
    --task detection \
    --checkpoint ./outputs/detection/best_model.pth \
    --input /path/to/test/image.jpg \
    --output results.json
```

## End-to-End Workflow

The framework supports a complete machine learning workflow:

### 1. Data Preparation
```
Raw Data → Preprocessing → Augmentation → Dataset Creation
```

### 2. Model Training
```
Dataset → Data Loader → Model → Optimizer → Training Loop → Checkpoints
```

### 3. Evaluation
```
Trained Model → Test Dataset → Metrics → Performance Analysis
```

### 4. Deployment
```
Best Model → Optimization → TensorRT (optional) → Inference Service
```

### Data Flow Example

```python
from src.pipelines.pipeline import StreetScenePipeline

# Initialize pipeline for detection
pipeline = StreetScenePipeline(
    config_path="configs/detection_config.yaml",
    task_type="detection"
)

# Train model
results = pipeline.train(
    train_data_path="/data/train",
    val_data_path="/data/val",
    output_dir="./outputs"
)

# Evaluate model
metrics = pipeline.evaluate(
    test_data_path="/data/test",
    checkpoint_path="./outputs/best_model.pth"
)
```

## Adding New Tasks

The framework is designed to make adding new tasks straightforward. Here's how to add a new detection model:

### 1. Extend the Model Factory

```python
# src/detection/models.py
class NewDetectionModel(BaseDetectionModel):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, **kwargs)
        # Model implementation
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Forward pass implementation

def create_detection_model(config: Dict[str, Any]) -> BaseDetectionModel:
    model_config = config['detection']['model']
    
    if model_config['name'] == 'new_model':
        return NewDetectionModel(
            num_classes=model_config['num_classes'],
            **model_config
        )
    # ... existing models
```

### 2. Create Configuration

```yaml
# configs/new_detection_config.yaml
detection:
  model:
    name: "new_model"
    num_classes: 80
    # ... model-specific parameters
```

### 3. Add Training Logic (if needed)

If the new model requires specialized training logic, extend the trainer:

```python
# src/common/trainer.py
class NewDetectionTrainer(Trainer):
    def _create_criterion(self) -> nn.Module:
        # Custom loss function
    
    def _compute_loss(self, outputs, targets) -> torch.Tensor:
        # Custom loss computation
```

### 4. Update Pipeline Factory

```python
# src/pipelines/pipeline.py
def _create_trainer(self) -> Trainer:
    if self.task_type == 'detection':
        model_config = self.config['detection']['model']
        if model_config['name'] == 'new_model':
            return NewDetectionTrainer(self.model, self.config)
    # ... existing trainers
```

## Performance Optimization

### Mixed Precision Training

The framework automatically supports mixed precision training for improved performance:

```yaml
mixed_precision:
  enabled: true
  loss_scale: "dynamic"
```

### Multi-GPU Training

Multi-GPU training is supported through PyTorch's DistributedDataParallel:

```bash
# Single node, multiple GPUs
python -m torch.distributed.launch --nproc_per_node=4 scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --train-data /path/to/data
```

### Memory Optimization

- Gradient checkpointing for large models
- Efficient data loading with multiple workers
- Automatic batch size scaling based on GPU memory

## Configuration

The framework uses YAML configuration files for easy customization:

### Detection Configuration

```yaml
detection:
  model:
    name: "ssd300"
    backbone: "resnet50"
    num_classes: 80
  
  training:
    batch_size: 32
    learning_rate: 0.001
    epochs: 120
    optimizer: "sgd"
    momentum: 0.9
    weight_decay: 0.0005
    
  data:
    image_size: [300, 300]
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    
  augmentation:
    horizontal_flip: 0.5
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    
mixed_precision:
  enabled: true
  loss_scale: "dynamic"
```

### Classification Configuration

```yaml
classification:
  vehicle:
    model:
      name: "resnet50"
      num_classes: 10
      pretrained: true
    
    training:
      batch_size: 64
      learning_rate: 0.01
      epochs: 90
      optimizer: "sgd"
      momentum: 0.9
      weight_decay: 0.0001
```

## Scripts and Tools

### Training Script (`scripts/train.py`)

Comprehensive training script with support for:
- Resume from checkpoints
- Multi-GPU training
- Custom logging and monitoring
- Automatic model saving

```bash
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --train-data /path/to/train \
    --val-data /path/to/val \
    --output-dir ./outputs \
    --resume ./outputs/checkpoint_epoch_50.pth
```

### Evaluation Script (`scripts/evaluate.py`)

Model evaluation with comprehensive metrics:
- Accuracy, precision, recall, F1-score
- mAP for detection tasks
- Per-class performance analysis

```bash
python scripts/evaluate.py \
    --config configs/detection_config.yaml \
    --task detection \
    --test-data /path/to/test \
    --checkpoint ./outputs/best_model.pth
```

### Inference Script (`scripts/inference.py`)

Production-ready inference with:
- Batch processing support
- Confidence thresholding
- JSON output format
- Visualization options

```bash
python scripts/inference.py \
    --config configs/detection_config.yaml \
    --task detection \
    --checkpoint ./outputs/best_model.pth \
    --input /path/to/images \
    --output results.json \
    --confidence 0.7
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone in development mode
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/StreetScene

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if available

# Run tests
python -m pytest tests/
```

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support:
- GitHub Issues: Report bugs and request features
- NVIDIA Forums: Community support and discussions
- Documentation: Comprehensive API reference and tutorials

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{nvidia_street_scene_2024,
  title={NVIDIA Street Scene Optimization Framework},
  author={NVIDIA},
  year={2024},
  url={https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/StreetScene}
}
```