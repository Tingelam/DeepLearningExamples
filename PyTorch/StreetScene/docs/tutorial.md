# Getting Started Tutorial

This tutorial will walk you through the basics of using the Street Scene Optimization framework.

## Prerequisites

- Python 3.8+
- PyTorch 1.8+ with CUDA support
- NVIDIA GPU (recommended for training)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NVIDIA/DeepLearningExamples.git
cd DeepLearningExamples/PyTorch/StreetScene
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Tutorial 1: Your First Detection Model

### Step 1: Prepare Your Data

Organize your data in the following structure:

```
dataset/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── annotations.json
├── val/
│   ├── images/
│   └── annotations.json
└── test/
    ├── images/
    └── annotations.json
```

The annotation file should be in COCO format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image_001.jpg",
      "height": 1080,
      "width": 1920
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 200, 300],
      "area": 60000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "bicycle"}
  ]
}
```

### Step 2: Configure Your Model

Edit `configs/detection_config.yaml`:

```yaml
detection:
  model:
    name: "ssd300"
    backbone: "resnet50"
    num_classes: 4  # 3 classes + background
  
  training:
    batch_size: 16
    learning_rate: 0.001
    epochs: 50
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

### Step 3: Train Your Model

```bash
python scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --train-data dataset/train \
    --val-data dataset/val \
    --annotations dataset/train/annotations.json \
    --output-dir ./outputs/my_detection_model
```

### Step 4: Evaluate Your Model

```bash
python scripts/evaluate.py \
    --config configs/detection_config.yaml \
    --task detection \
    --test-data dataset/test \
    --checkpoint ./outputs/my_detection_model/best_model.pth \
    --annotations dataset/test/annotations.json \
    --output-dir ./outputs/my_detection_model/test_results
```

### Step 5: Run Inference

```bash
python scripts/inference.py \
    --config configs/detection_config.yaml \
    --task detection \
    --checkpoint ./outputs/my_detection_model/best_model.pth \
    --input dataset/test/images \
    --output results.json \
    --confidence 0.5
```

## Tutorial 2: Vehicle Classification

### Step 1: Prepare Vehicle Data

Organize your vehicle data:

```
vehicle_dataset/
├── train/
│   ├── sedan/
│   │   ├── car_001.jpg
│   │   └── ...
│   ├── suv/
│   │   ├── suv_001.jpg
│   │   └── ...
│   └── truck/
│       ├── truck_001.jpg
│       └── ...
├── val/
│   ├── sedan/
│   ├── suv/
│   └── truck/
└── test/
    ├── sedan/
    ├── suv/
    └── truck/
```

### Step 2: Configure Vehicle Classification

Edit `configs/classification_config.yaml` and add or modify entries under `classification.tasks`:

```yaml
classification:
  defaults:
    model:
      backbone: "resnet50"
      pretrained: true
      global_pool: "avg"
    training:
      batch_size: 64
      learning_rate: 0.01
      epochs: 60
      optimizer: "sgd"
    data:
      image_size: [224, 224]
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  tasks:
    vehicle_types:
      description: "Sedan/SUV/truck classification"
      heads:
        vehicle_type:
          num_classes: 3
          loss: "cross_entropy"
          metrics: ["accuracy"]
    human_attributes:
      description: "Multi-head attribute recognition"
      model:
        backbone: "resnet34"
        freeze_backbone: true
      heads:
        gender:
          num_classes: 2
          loss: "cross_entropy"
          metrics: ["accuracy"]
        clothing_style:
          num_classes: 8
          loss: "cross_entropy"
          metrics: ["accuracy"]
```

Each task inherits the defaults and can override the timm backbone or training schedule. The `heads` dictionary defines one classifier per attribute—add as many as you need, choosing the appropriate loss (`cross_entropy`, `bce_with_logits`, etc.) and the metrics you want logged per head.

### Step 3: Train Vehicle Classifier

```bash
python scripts/train.py \
    --config configs/classification_config.yaml \
    --task classification \
    --classification-task vehicle_types \
    --train-data vehicle_dataset/train \
    --val-data vehicle_dataset/val \
    --output-dir ./outputs/vehicle_classifier
```

## Tutorial 3: Using the Python API

### Basic Training Pipeline

```python
import sys
import os
sys.path.append('src')

from pipelines.pipeline import StreetScenePipeline

def train_detection_model():
    """Train a detection model using the Python API."""
    
    # Initialize pipeline
    pipeline = StreetScenePipeline(
        config_path="configs/detection_config.yaml",
        task_type="detection",
        log_level="INFO"
    )
    
    # Train the model
    results = pipeline.train(
        train_data_path="dataset/train",
        val_data_path="dataset/val",
        annotation_file="dataset/train/annotations.json",
        output_dir="./outputs/api_detection"
    )
    
    print(f"Training completed. Best validation metric: {results['best_val_metric']}")
    
    # Evaluate on test set
    test_metrics = pipeline.evaluate(
        test_data_path="dataset/test",
        checkpoint_path="./outputs/api_detection/best_model.pth",
        annotation_file="dataset/test/annotations.json"
    )
    
    print(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    train_detection_model()
```

### Custom Model Training

```python
import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
from detection.models import BaseDetectionModel
from common.trainer import Trainer
from common.utils import load_config

class CustomDetectionModel(BaseDetectionModel):
    """Custom detection model example."""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, **kwargs)
        
        # Simple backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Detection head
        self.detection_head = nn.Linear(256, num_classes * 4)
        self.confidence_head = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        
        detections = self.detection_head(features)
        confidences = self.confidence_head(features)
        
        return {
            'detections': detections,
            'confidences': confidences
        }

def train_custom_model():
    """Train a custom model."""
    
    # Create custom model
    model = CustomDetectionModel(num_classes=4)
    
    # Load config
    config = load_config("configs/detection_config.yaml")
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Your training loop here
    print("Custom model training setup complete!")

if __name__ == "__main__":
    train_custom_model()
```

## Tutorial 4: Advanced Features

### Mixed Precision Training

The framework automatically supports mixed precision training:

```python
# Enable in config
mixed_precision:
  enabled: true
  loss_scale: "dynamic"
```

### Multi-GPU Training

```bash
# Launch multi-GPU training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/detection_config.yaml \
    --task detection \
    --train-data dataset/train \
    --val-data dataset/val
```

### Custom Data Augmentation

```python
import torchvision.transforms as transforms
from data.dataset import StreetSceneDataset

# Custom augmentation
custom_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Create dataset with custom transform
dataset = StreetSceneDataset(
    data_path="dataset/train",
    transform=custom_transform,
    annotation_file="dataset/train/annotations.json"
)
```

## Common Issues and Solutions

### 1. CUDA Out of Memory

Reduce batch size in your config:

```yaml
training:
  batch_size: 8  # Reduce from 32 to 8
```

### 2. Slow Training

Enable mixed precision:

```yaml
mixed_precision:
  enabled: true
```

### 3. Poor Accuracy

- Increase training epochs
- Adjust learning rate
- Add more data augmentation
- Use pretrained backbones

### 4. Data Loading Issues

Check your data format and paths:

```python
# Verify dataset loading
from data.dataset import StreetSceneDataset
dataset = StreetSceneDataset(
    data_path="dataset/train",
    annotation_file="dataset/train/annotations.json"
)
print(f"Dataset size: {len(dataset)}")
print(f"Sample: {dataset[0]}")
```

## Next Steps

1. **Explore Different Models**: Try different backbones and architectures
2. **Hyperparameter Tuning**: Use the configuration system to experiment
3. **Production Deployment**: Learn about model optimization and deployment
4. **Advanced Topics**: Explore multi-task learning and custom losses

For more examples, see the [Examples Documentation](examples.md).