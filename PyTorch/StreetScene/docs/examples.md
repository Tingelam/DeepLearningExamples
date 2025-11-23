# Usage Examples

This document provides practical examples of using the Street Scene Optimization framework.

## Example 1: Training a Pedestrian Detection Model

```python
import sys
import os
sys.path.append('src')

from pipelines.pipeline import StreetScenePipeline

def main():
    # Initialize pipeline for pedestrian detection
    pipeline = StreetScenePipeline(
        config_path="configs/detection_config.yaml",
        task_type="detection",
        log_level="INFO"
    )
    
    # Train the model
    results = pipeline.train(
        train_data_path="/data/pedestrian/train",
        val_data_path="/data/pedestrian/val",
        annotation_file="/data/pedestrian/annotations.json",
        output_dir="./outputs/pedestrian_detection"
    )
    
    print(f"Training completed. Best validation metric: {results['best_val_metric']}")

if __name__ == "__main__":
    main()
```

## Example 2: Vehicle Classification with Custom Data

```python
import sys
import os
sys.path.append('src')

from pipelines.pipeline import StreetScenePipeline
from data.dataset import StreetSceneDataset, get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

def main():
    # Initialize pipeline for vehicle classification
    pipeline = StreetScenePipeline(
        config_path="configs/classification_config.yaml",
        task_type="vehicle_classification"
    )
    
    # Custom training with specific data loaders
    train_transform = get_train_transforms((224, 224))
    val_transform = get_val_transforms((224, 224))
    
    train_dataset = StreetSceneDataset(
        data_path="/data/vehicles/train",
        transform=train_transform,
        annotation_file="/data/vehicles/labels.json"
    )
    
    val_dataset = StreetSceneDataset(
        data_path="/data/vehicles/val",
        transform=val_transform,
        annotation_file="/data/vehicles/labels.json"
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Train using the pipeline's internal methods
    results = pipeline.train(
        train_data_path="/data/vehicles/train",
        val_data_path="/data/vehicles/val",
        output_dir="./outputs/vehicle_classification"
    )
    
    print("Vehicle classification training completed!")

if __name__ == "__main__":
    main()
```

## Example 3: Human Attribute Analysis with Multi-task Learning

```python
import sys
import os
sys.path.append('src')

from pipelines.pipeline import StreetScenePipeline

def main():
    # Initialize pipeline for human attribute analysis
    pipeline = StreetScenePipeline(
        config_path="configs/classification_config.yaml",
        task_type="human_attributes",
        log_level="DEBUG"
    )
    
    # Train with multi-task learning
    results = pipeline.train(
        train_data_path="/data/humans/train",
        val_data_path="/data/humans/val",
        annotation_file="/data/humans/attributes.json",
        output_dir="./outputs/human_attributes"
    )
    
    # Evaluate on test set
    test_metrics = pipeline.evaluate(
        test_data_path="/data/humans/test",
        checkpoint_path="./outputs/human_attributes/best_model.pth",
        annotation_file="/data/humans/attributes.json",
        output_dir="./outputs/human_attributes/test_results"
    )
    
    print(f"Test metrics: {test_metrics}")

if __name__ == "__main__":
    main()
```

## Example 4: Custom Model Implementation

```python
import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
from detection.models import BaseDetectionModel

class CustomDetectionModel(BaseDetectionModel):
    """Custom detection model implementation."""
    
    def __init__(self, num_classes: int, backbone: str = "resnet50", **kwargs):
        super().__init__(num_classes, **kwargs)
        
        # Custom backbone
        self.backbone = self._build_backbone(backbone)
        
        # Custom detection heads
        self.detection_head = nn.Conv2d(2048, num_classes * 4, 3, padding=1)
        self.confidence_head = nn.Conv2d(2048, num_classes, 3, padding=1)
        
        self._init_weights()
    
    def _build_backbone(self, backbone: str):
        """Build custom backbone."""
        if backbone == "resnet50":
            from torchvision.models import resnet50
            model = resnet50(pretrained=True)
            return nn.Sequential(*list(model.children())[:-2])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def _init_weights(self):
        """Initialize custom weights."""
        for m in [self.detection_head, self.confidence_head]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass."""
        features = self.backbone(x)
        
        detections = self.detection_head(features)
        confidences = self.confidence_head(features)
        
        return {
            'detections': detections,
            'confidences': confidences
        }

def main():
    # Create custom model
    model = CustomDetectionModel(num_classes=80, backbone="resnet50")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 300, 300)
    output = model(dummy_input)
    
    print(f"Output shapes: {k}: {v.shape} for k, v in output.items()}")

if __name__ == "__main__":
    main()
```

## Example 5: Batch Inference on Directory

```python
import sys
import os
sys.path.append('src')

import torch
from pathlib import Path
from detection.models import create_detection_model
from common.utils import load_config, get_device
from data.dataset import get_val_transforms
import json

def batch_inference(config_path: str, checkpoint_path: str, 
                   input_dir: str, output_file: str):
    """Run batch inference on all images in a directory."""
    
    # Load model and config
    config = load_config(config_path)
    device = get_device()
    
    model = create_detection_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Setup transforms
    image_size = tuple(config['detection']['data']['image_size'])
    transform = get_val_transforms(image_size)
    
    # Process all images
    input_path = Path(input_dir)
    results = {}
    
    for image_file in input_path.glob("*.jpg"):
        print(f"Processing {image_file.name}...")
        
        # Load and preprocess image
        import cv2
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # Store results
        results[image_file.name] = {
            'detections': outputs['detections'].cpu().numpy().tolist(),
            'confidences': outputs['confidences'].cpu().numpy().tolist()
        }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} images. Results saved to {output_file}")

def main():
    batch_inference(
        config_path="configs/detection_config.yaml",
        checkpoint_path="./outputs/best_model.pth",
        input_dir="/data/test_images",
        output_file="batch_results.json"
    )

if __name__ == "__main__":
    main()
```

## Example 6: Multi-GPU Training

```python
import sys
import os
sys.path.append('src')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pipelines.pipeline import StreetScenePipeline

def train_worker(rank: int, world_size: int, config_path: str, task_type: str):
    """Training function for each GPU process."""
    
    # Initialize distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    
    # Initialize pipeline
    pipeline = StreetScenePipeline(config_path, task_type)
    pipeline.model = torch.nn.parallel.DistributedDataParallel(
        pipeline.model, device_ids=[rank]
    )
    
    # Create distributed sampler and data loader
    # (This would require modifying the pipeline to support distributed sampling)
    
    # Train
    results = pipeline.train(
        train_data_path="/data/train",
        val_data_path="/data/val",
        output_dir=f"./outputs/gpu_{rank}"
    )
    
    # Clean up
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()
    config_path = "configs/detection_config.yaml"
    task_type = "detection"
    
    print(f"Starting training on {world_size} GPUs")
    mp.spawn(
        train_worker,
        args=(world_size, config_path, task_type),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
```

## Example 7: Custom Data Augmentation

```python
import sys
import os
sys.path.append('src')

import torchvision.transforms as transforms
import random
from PIL import Image, ImageEnhance

class StreetSceneAugmentation:
    """Custom augmentation for street scene images."""
    
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            ], p=0.8),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        return self.transform(image)

class WeatherAugmentation:
    """Simulate weather conditions for robust training."""
    
    def __init__(self, rain_prob=0.2, fog_prob=0.1):
        self.rain_prob = rain_prob
        self.fog_prob = fog_prob
    
    def add_rain(self, image):
        """Add rain effect to image."""
        # Simple rain simulation
        image_array = np.array(image)
        h, w = image_array.shape[:2]
        
        # Create rain streaks
        num_drops = random.randint(50, 200)
        for _ in range(num_drops):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            length = random.randint(10, 30)
            
            for i in range(length):
                if y + i < h:
                    image_array[y+i, x] = [200, 200, 200]  # White rain drops
        
        return Image.fromarray(image_array)
    
    def add_fog(self, image):
        """Add fog effect to image."""
        image_array = np.array(image, dtype=float)
        fog_layer = np.ones_like(image_array) * 200  # Light gray fog
        
        # Blend with original image
        alpha = 0.3  # Fog intensity
        foggy_image = alpha * fog_layer + (1 - alpha) * image_array
        
        return Image.fromarray(foggy_image.astype(np.uint8))
    
    def __call__(self, image):
        """Apply weather augmentation."""
        if random.random() < self.rain_prob:
            image = self.add_rain(image)
        
        if random.random() < self.fog_prob:
            image = self.add_fog(image)
        
        return image

def main():
    # Create custom augmentation pipeline
    weather_aug = WeatherAugmentation(rain_prob=0.3, fog_prob=0.15)
    street_aug = StreetSceneAugmentation()
    
    # Combined transform
    transform = transforms.Compose([
        weather_aug,
        street_aug
    ])
    
    # Test augmentation
    from PIL import Image
    test_image = Image.open("/data/test_image.jpg")
    augmented = transform(test_image)
    
    print("Custom augmentation pipeline created successfully!")

if __name__ == "__main__":
    main()
```

## Example 8: Performance Monitoring

```python
import sys
import os
sys.path.append('src')

import time
import psutil
import torch
from torch.profiler import profile, record_function, ProfilerActivity

class PerformanceMonitor:
    """Monitor training performance and resource usage."""
    
    def __init__(self):
        self.start_time = None
        self.batch_times = []
        self.memory_usage = []
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
    
    def log_batch(self, batch_size: int):
        """Log batch processing time."""
        current_time = time.time()
        if self.start_time:
            batch_time = current_time - self.start_time
            self.batch_times.append((batch_time, batch_size))
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            else:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            self.memory_usage.append(memory_mb)
        
        self.start_time = current_time
    
    def get_stats(self):
        """Get performance statistics."""
        if not self.batch_times:
            return {}
        
        total_time = sum(t[0] for t in self.batch_times)
        total_samples = sum(t[1] for t in self.batch_times)
        
        return {
            'total_time': total_time,
            'total_samples': total_samples,
            'avg_batch_time': total_time / len(self.batch_times),
            'samples_per_second': total_samples / total_time,
            'avg_memory_mb': sum(self.memory_usage) / len(self.memory_usage),
            'max_memory_mb': max(self.memory_usage)
        }

def profile_model(model, input_tensor, num_runs=100):
    """Profile model performance with PyTorch profiler."""
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        with record_function("model_inference"):
            for _ in range(num_runs):
                _ = model(input_tensor)
    
    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # Export to Chrome trace
    prof.export_chrome_trace("model_profile.json")

def main():
    # Example usage
    from detection.models import create_detection_model
    from common.utils import load_config
    
    # Load model
    config = load_config("configs/detection_config.yaml")
    model = create_detection_model(config)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 300, 300)
    
    # Profile model
    profile_model(model, dummy_input, num_runs=50)
    
    # Monitor training performance
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    for i in range(10):
        # Simulate batch processing
        time.sleep(0.1)  # Simulate computation time
        monitor.log_batch(batch_size=32)
    
    stats = monitor.get_stats()
    print(f"Performance stats: {stats}")

if __name__ == "__main__":
    main()
```