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
- [Evaluation and Reporting](#evaluation-and-reporting)
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

# Install Python dependencies (includes ultralytics for YOLO)
pip install -r requirements.txt
```

**Important Installation Notes for YOLO:**

The framework uses `ultralytics>=8.0.0` for YOLOv8/YOLO11 detection tasks. On some systems, you may need additional dependencies:

```bash
# For systems with display servers (X11)
pip install ultralytics opencv-python

# For headless systems (Docker, remote servers)
# The Dockerfile automatically includes libsm6 and libxext6 which are required
# If installing manually, ensure these system packages are available:
sudo apt-get install libsm6 libxext6 libxrender-dev libgl1-mesa-glx libglib2.0-0
```

### Docker Installation

```bash
# Build Docker image
docker build -t street-scene-pytorch .

# Run container (with GPU support)
docker run --gpus all -it --rm -v $(pwd):/workspace street-scene-pytorch bash

# Run container (CPU only)
docker run -it --rm -v $(pwd):/workspace street-scene-pytorch bash
```

## Quick Start

### 1. Train a YOLO Detection Model

The framework uses YOLOv8/YOLO11 models for modern detection. Available detection tasks are defined in the config:

```bash
# Train vehicle detection using YOLO
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --train-data /path/to/train/data \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --output-dir ./outputs

# Train pedestrian detection using YOLO
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task pedestrian_detection \
    --train-data /path/to/train/data \
    --data-yaml configs/datasets/pedestrian_detection.yaml \
    --output-dir ./outputs

# Train vehicle tracking model
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_tracking \
    --train-data /path/to/train/data \
    --data-yaml configs/datasets/vehicle_tracking.yaml \
    --output-dir ./outputs
```

### 2. Run YOLO Detection Inference

```bash
# Run object detection on images
python scripts/inference.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --checkpoint ./outputs/vehicle_detection/best.pt \
    --input /path/to/test/image.jpg \
    --output-dir ./outputs/predictions

# Run on directory of images
python scripts/inference.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --checkpoint ./outputs/vehicle_detection/best.pt \
    --input /path/to/test/images/ \
    --output-dir ./outputs/predictions
```

### 3. Run YOLO Tracking

```bash
# Track objects in video
python scripts/inference.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_tracking \
    --mode track \
    --checkpoint ./outputs/vehicle_tracking/best.pt \
    --input /path/to/video.mp4 \
    --output-dir ./outputs/tracks
```

### 4. Train a Vehicle Classification Model

```bash
python scripts/train.py \
    --config configs/classification_config.yaml \
    --task vehicle_classification \
    --train-data /path/to/vehicle/data \
    --val-data /path/to/val/data \
    --output-dir ./outputs/vehicle_classification
```

### 5. Evaluate Models

```bash
# Evaluate detection model
python scripts/evaluate.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --test-data /path/to/test/data \
    --checkpoint ./outputs/vehicle_detection/best.pt \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --output-dir ./outputs/eval_results
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

## Adding New Detection Tasks

The framework makes it easy to add new detection tasks without modifying code. Detection tasks are defined in the task catalog in `configs/detection_config.yaml`.

### Adding a New YOLO Detection Task

To add a new detection task, simply add an entry to the `tasks` section in `detection_config.yaml`:

```yaml
# configs/detection_config.yaml
tasks:
  your_new_task:
    description: "Description of your detection task"
    model:
      variant: "yolov8m"  # or yolov8n, yolov8s, yolov8l, yolov8x, yolo11m, etc.
      num_classes: 3
    dataset:
      yaml: "configs/datasets/your_dataset.yaml"
      classes: ["class1", "class2", "class3"]
    training:
      epochs: 100
      batch_size: 32
      learning_rate: 0.0001
      optimizer: "adam"
    augmentation:
      horizontal_flip: 0.5
      brightness: 0.1
      contrast: 0.1
    tracking_enabled: false  # Set to true if task requires tracking
```

Then use the task immediately:

```bash
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task your_new_task \
    --train-data /path/to/data \
    --data-yaml configs/datasets/your_dataset.yaml \
    --output-dir ./outputs
```

### YOLO Model Variants

The framework supports all YOLOv8 and YOLO11 variants:

- **YOLOv8**: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- **YOLO11**: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`

Choose based on your accuracy/speed tradeoff:
- `n` (nano): Smallest, fastest, least accurate
- `s` (small): Small model
- `m` (medium): Balanced model (recommended)
- `l` (large): Large model, higher accuracy
- `x` (extra-large): Largest model, highest accuracy

### Creating a YOLO Dataset YAML

For YOLO training, you need a dataset YAML file pointing to your data:

```yaml
# configs/datasets/your_dataset.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 3  # number of classes
names: ['class1', 'class2', 'class3']  # class names
```

The dataset directory structure should be:
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image should have a corresponding `.txt` label file in YOLO format (one detection per line):
```
<class_id> <x_center> <y_center> <width> <height>
```

Where coordinates are normalized to [0, 1].

### Legacy Model Extension

If you need to add a custom PyTorch detection model (not YOLO), you can extend the framework:

```python
# src/detection/models.py
class CustomDetectionModel(BaseDetectionModel):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, **kwargs)
        # Model implementation
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Forward pass implementation
```

Then update the model factory to support your type:

```python
def create_detection_model(config: Dict[str, Any], task_config: Optional[Dict[str, Any]] = None) -> BaseDetectionModel:
    model_config = task_config.get('model') if task_config else config.get('detection', {}).get('model', {})
    model_type = model_config.get('type', 'yolo').lower()
    
    if model_type == 'custom':
        return CustomDetectionModel(
            num_classes=model_config.get('num_classes', 80),
            **model_config
        )
    # ... existing models
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

### YOLO Detection Configuration with Task Catalog

```yaml
detection:
  # Default model configuration (used if no task is specified)
  model:
    type: "yolo"
    variant: "yolov8n"
    num_classes: 80
  
  # Global training defaults
  training:
    batch_size: 32
    learning_rate: 0.001
    epochs: 120
    optimizer: "sgd"
    momentum: 0.9
    weight_decay: 0.0005
    
  # Global data configuration
  data:
    image_size: [640, 640]
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    
  # Global augmentation configuration
  augmentation:
    horizontal_flip: 0.5
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2

  # Task catalog - each task inherits defaults and can override them
  tasks:
    vehicle_detection:
      description: "Vehicle detection in street scenes"
      model:
        variant: "yolov8m"
        num_classes: 5
      dataset:
        yaml: "configs/datasets/vehicle_detection.yaml"
        classes: ["car", "truck", "bus", "motorcycle", "bicycle"]
      training:
        epochs: 120
        batch_size: 32
        learning_rate: 0.0001
        optimizer: "adam"
      tracking_enabled: false

    pedestrian_detection:
      description: "Pedestrian detection in street scenes"
      model:
        variant: "yolov8m"
        num_classes: 1
      dataset:
        yaml: "configs/datasets/pedestrian_detection.yaml"
        classes: ["person"]
      training:
        epochs: 100
        batch_size: 32
      tracking_enabled: false

    vehicle_tracking:
      description: "Vehicle tracking and re-identification"
      model:
        variant: "yolov8m"
        num_classes: 1
      dataset:
        yaml: "configs/datasets/vehicle_tracking.yaml"
        classes: ["vehicle"]
      training:
        epochs: 150
        batch_size: 16
      tracking:
        enabled: true
        tracker: "botsort.yaml"
        max_age: 30
        min_hits: 3
      
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

## Evaluation and Reporting

The framework includes a comprehensive evaluation and reporting system that automatically generates metrics, reports, and supports reproducibility verification.

### Automated Metrics Computation

Training and evaluation runs automatically compute task-aware metrics:

- **Detection**: mAP@0.5, mAP@0.5:0.95, precision, recall, PR curves
- **Tracking**: MOTA, MOTP, IDF1, ID switches, fragmentations
- **Classification**: Accuracy, precision, recall, F1, AUROC, confusion matrices

### Report Generation

Every run automatically generates comprehensive reports in multiple formats:

- **JSON**: Machine-readable metrics (`metrics_report.json`)
- **Markdown**: Documentation-friendly format (`metrics_report.md`)
- **HTML**: Standalone report (`metrics_report.html`)
- **Plots**: Visualization of metrics and training history (`plots/`)

Reports include:
- All computed metrics
- Configuration metadata
- Dataset information
- Training history
- Reproduction checklist

### Workflow Automation

#### Full Lifecycle Workflow

Run the complete pipeline (data prep → training → evaluation → reporting):

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
    --run-name vehicle_detection_v1
```

#### Compare Multiple Runs

Compare metrics across multiple training runs:

```bash
python scripts/compare_runs.py \
    --run-ids ./outputs/run1 ./outputs/run2 ./outputs/run3 \
    --output ./comparison_report \
    --format both \
    --metric mAP@0.5
```

This generates a comparison table showing:
- Side-by-side metrics for all runs
- Best performing run for each metric
- Detailed comparison report

#### Verify Reproducibility

Validate that a saved checkpoint reproduces stored metrics:

```bash
python scripts/verify_repro.py \
    --run-id ./outputs/my_run \
    --test-data /path/to/test \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --tolerance 0.01
```

The script:
1. Loads original metrics
2. Re-evaluates the checkpoint
3. Compares metrics within tolerance
4. Reports discrepancies (if any)
5. Returns exit code 0 if verified, 1 if failed

#### Deploy Models

Package a trained model for deployment:

```bash
python scripts/deploy.py \
    --run-id ./outputs/my_best_run \
    --output-dir ./deployment \
    --deployment-name vehicle_detector_v1 \
    --include-metrics
```

The deployment package includes:
- Model checkpoint
- Configuration file
- Deployment manifest
- README with usage instructions
- Metrics report (optional)

### Best Practices

1. **Use `run_workflow.py` for production runs** to ensure all metadata is captured
2. **Compare against baselines** using `--compare-with` to track improvements
3. **Verify reproducibility** on important checkpoints with `verify_repro.py`
4. **Archive run directories** - they contain complete configuration and history
5. **Use meaningful run names** that include task, model variant, and key parameters

For detailed documentation on evaluation and reporting, see [docs/evaluation_and_reporting.md](docs/evaluation_and_reporting.md).

## Scripts and Tools

### Training Script (`scripts/train.py`)

Comprehensive training script with support for:
- Task-specific configuration via `--detection-task`
- YOLO dataset YAML configuration via `--data-yaml`
- Resume from checkpoints
- Custom logging and monitoring
- Automatic model saving

```bash
# Train a specific YOLO detection task
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --train-data /path/to/train \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --output-dir ./outputs

# Resume training
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --train-data /path/to/train \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --output-dir ./outputs \
    --resume ./outputs/vehicle_detection/last.pt
```

### Evaluation Script (`scripts/evaluate.py`)

Model evaluation with comprehensive metrics:
- Accuracy, precision, recall, F1-score
- mAP for detection tasks
- Per-class performance analysis
- YOLO-compatible evaluation

```bash
# Evaluate detection model
python scripts/evaluate.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --test-data /path/to/test \
    --checkpoint ./outputs/vehicle_detection/best.pt \
    --data-yaml configs/datasets/vehicle_detection.yaml \
    --output-dir ./outputs/eval_results
```

### Inference Script (`scripts/inference.py`)

Production-ready inference with:
- Batch processing support
- Confidence thresholding
- JSON output format
- **Object tracking support** via YOLO tracker
- Video and image processing

```bash
# Detection on images
python scripts/inference.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_detection \
    --checkpoint ./outputs/vehicle_detection/best.pt \
    --input /path/to/images \
    --output-dir ./outputs/predictions \
    --confidence 0.5

# Object tracking on video
python scripts/inference.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_tracking \
    --mode track \
    --checkpoint ./outputs/vehicle_tracking/best.pt \
    --input /path/to/video.mp4 \
    --output-dir ./outputs/tracks \
    --tracker botsort.yaml

# Custom tracker configuration
python scripts/inference.py \
    --config configs/detection_config.yaml \
    --task detection \
    --detection-task vehicle_tracking \
    --mode track \
    --checkpoint ./outputs/vehicle_tracking/best.pt \
    --input /path/to/video.mp4 \
    --output-dir ./outputs/tracks \
    --tracker bytetrack.yaml
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