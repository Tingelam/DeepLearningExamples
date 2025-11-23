# Data Workflow Guide

This guide explains the complete data lifecycle in the Street Scene framework, from acquiring raw data to training models.

## Table of Contents

- [Overview](#overview)
- [Data Lifecycle](#data-lifecycle)
- [Dataset Catalog](#dataset-catalog)
- [Data Preparation](#data-preparation)
- [Using Datasets](#using-datasets)
- [Adding New Datasets](#adding-new-datasets)
- [Supported Data Formats](#supported-data-formats)
- [Best Practices](#best-practices)

## Overview

The Street Scene framework includes a comprehensive data management system with:

- **Dataset Catalog**: Centralized registry of datasets with version control
- **Data Preparation Tools**: Scripts to convert raw data to standard formats
- **Flexible Loaders**: Support for YOLO, classification, and custom formats
- **Augmentation Pipelines**: Task-specific preprocessing and augmentation
- **Provenance Tracking**: Hash-based reproducibility for experiments

## Data Lifecycle

The data workflow follows four main stages:

### 1. Acquire
Obtain raw data from various sources (cameras, public datasets, annotations tools, etc.)

### 2. Preprocess
Convert raw data to standardized formats (YOLO for detection, image folders or CSV for classification)

### 3. Register
Add dataset to the catalog with version tags, label schemas, and metadata

### 4. Use
Load datasets in training/evaluation pipelines via catalog lookups

```
┌──────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────┐
│ Acquire  │ --> │  Preprocess  │ --> │ Register │ --> │   Use    │
│ Raw Data │     │   & Convert  │     │   in     │     │  in      │
│          │     │   Format     │     │ Catalog  │     │ Pipeline │
└──────────┘     └──────────────┘     └──────────┘     └──────────┘
```

## Dataset Catalog

The dataset catalog (`data/datasets.yaml`) maintains a registry of all datasets with complete metadata.

### Catalog Entry Structure

```yaml
vehicle_detection:
  v1:
    name: vehicle_detection
    version: v1
    task_type: detection
    data_format: yolo
    splits:
      train:
        path: /path/to/data/vehicle_detection/v1/images/train
        num_samples: 10000
      val:
        path: /path/to/data/vehicle_detection/v1/images/val
        num_samples: 2000
    label_schema:
      classes: [car, truck, bus, motorcycle, bicycle]
      num_classes: 5
      format: yolo
    preprocessing_config:
      source_format: coco
      output_format: yolo
      splits: [train, val, test]
    metadata:
      description: "Vehicle detection dataset v1"
      source: /data/raw/vehicles
      output: /data/processed/vehicle_detection/v1
    provenance:
      hash: "a1b2c3d4e5f6g7h8"
      source_hash: "h8g7f6e5d4c3b2a1"
      registration_date: "2024-01-15T10:30:00"
      command: "python scripts/data_prepare.py --dataset vehicle_detection ..."
```

### Using the Catalog API

```python
from src.data.catalog import get_catalog

# Load catalog
catalog = get_catalog("data/datasets.yaml")

# Get dataset information
dataset = catalog.get_dataset("vehicle_detection", version="v1")

# Get specific split path
train_path = catalog.get_dataset_path("vehicle_detection", "v1", "train")

# Get label schema
labels = catalog.get_label_schema("vehicle_detection", "v1")

# List all datasets
datasets = catalog.list_datasets(task_type="detection")

# Export manifest for experiment tracking
catalog.export_dataset_manifest("vehicle_detection", "v1", "outputs/manifest.yaml")
```

## Data Preparation

Use the `data_prepare.py` script to convert raw data and register it in the catalog.

### Basic Usage

```bash
python scripts/data_prepare.py \
  --dataset vehicle_detection \
  --version v1 \
  --raw /data/raw/vehicles \
  --output data/processed \
  --task-type detection \
  --data-format coco \
  --splits train val test \
  --classes car truck bus motorcycle bicycle \
  --catalog data/datasets.yaml \
  --description "Vehicle detection dataset v1"
```

### Parameters

- `--dataset`: Dataset name (identifier)
- `--version`: Version tag (e.g., v1, v1.0.0, 2024-01)
- `--raw`: Path to raw data directory
- `--output`: Output directory for processed data
- `--task-type`: Task type (detection, classification, tracking)
- `--data-format`: Source data format (yolo, coco, pascal_voc, image_folder, csv_attribute)
- `--splits`: Dataset splits to process (default: train val test)
- `--classes`: List of class names
- `--catalog`: Path to catalog file (default: data/datasets.yaml)
- `--description`: Human-readable description

### Detection Dataset Example

```bash
# Prepare YOLO detection dataset
python scripts/data_prepare.py \
  --dataset pedestrian_detection \
  --version v2 \
  --raw /data/raw/pedestrians \
  --task-type detection \
  --data-format yolo \
  --splits train val \
  --classes person
```

### Classification Dataset Example

```bash
# Prepare image folder classification dataset
python scripts/data_prepare.py \
  --dataset vehicle_classification \
  --version v1 \
  --raw /data/raw/vehicle_types \
  --task-type classification \
  --data-format image_folder \
  --splits train val test \
  --classes sedan suv truck van motorcycle
```

### What Happens During Preparation

1. **Data Conversion**: Converts from source format to target format (e.g., COCO → YOLO)
2. **Directory Structure**: Creates standardized directory layout
3. **Manifest Generation**: Creates YAML files with dataset configuration
4. **Hash Computation**: Calculates provenance hashes for reproducibility
5. **Catalog Registration**: Adds entry to dataset catalog with all metadata

## Using Datasets

### In Training Scripts

#### Using Catalog (Recommended)

```python
# Train with dataset from catalog
python scripts/train.py \
  --config configs/detection_config.yaml \
  --task detection \
  --detection-task vehicle_detection \
  --dataset vehicle_detection \
  --version v1 \
  --output-dir outputs/vehicle_det_v1
```

#### Using Direct Paths (Legacy)

```python
# Train with direct paths
python scripts/train.py \
  --config configs/detection_config.yaml \
  --task detection \
  --detection-task vehicle_detection \
  --train-data data/processed/vehicle_detection/v1/images/train \
  --val-data data/processed/vehicle_detection/v1/images/val \
  --data-yaml data/processed/vehicle_detection/v1/data.yaml
```

### In Python Code

```python
from src.pipelines.pipeline import StreetScenePipeline

# Initialize pipeline
pipeline = StreetScenePipeline(
    config_path="configs/detection_config.yaml",
    task_type="detection",
    detection_task="vehicle_detection"
)

# Prepare data from catalog
train_loader, val_loader = pipeline.prepare_data(
    dataset_name="vehicle_detection",
    dataset_version="v1",
    catalog_path="data/datasets.yaml"
)

# Train model
results = pipeline.train(
    dataset_name="vehicle_detection",
    dataset_version="v1",
    output_dir="outputs/vehicle_det_v1"
)
```

### Dataset Loaders

#### YOLO Dataset Loader

```python
from src.data.yolo_dataset import create_yolo_dataloader

# Create YOLO dataloader
loader = create_yolo_dataloader(
    data_yaml="data/processed/vehicle_detection/v1/data.yaml",
    split="train",
    batch_size=32,
    shuffle=True,
    load_tracks=True  # For tracking tasks
)

# Iterate over batches
for images, targets in loader:
    # images: torch.Tensor of shape (B, 3, H, W)
    # targets: List of dicts with 'bboxes', 'track_ids', etc.
    pass
```

#### Classification Dataset Loader

```python
from src.data.classification_dataset import create_classification_dataloader

# Image folder classification
loader = create_classification_dataloader(
    data_path="data/processed/vehicle_classification/v1/train",
    dataset_type="image_folder",
    batch_size=64,
    shuffle=True
)

# CSV attributes classification
loader = create_classification_dataloader(
    data_path="data/processed/human_attributes/v1/train.csv",
    dataset_type="csv_attribute",
    batch_size=32,
    image_dir="data/processed/human_attributes/v1/images"
)
```

## Adding New Datasets

### Step 1: Organize Raw Data

Organize your raw data according to the source format:

**YOLO Format:**
```
raw_data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── classes.txt
```

**Image Folder Format:**
```
raw_data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── class3/
├── val/
│   └── ...
└── test/
    └── ...
```

### Step 2: Run Data Preparation

```bash
python scripts/data_prepare.py \
  --dataset my_new_dataset \
  --version v1 \
  --raw /path/to/raw_data \
  --task-type detection \
  --data-format yolo \
  --splits train val test \
  --classes class1 class2 class3
```

### Step 3: Verify Registration

Check that the dataset is in the catalog:

```python
from src.data.catalog import get_catalog

catalog = get_catalog("data/datasets.yaml")
dataset = catalog.get_dataset("my_new_dataset", "v1")
print(dataset)
```

### Step 4: Use in Training

```bash
python scripts/train.py \
  --config configs/detection_config.yaml \
  --task detection \
  --detection-task my_new_dataset \
  --dataset my_new_dataset \
  --version v1
```

## Supported Data Formats

### Detection Formats

#### YOLO Format
- Annotations: One `.txt` file per image
- Format: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)
- Optional tracking: `<class_id> <x_center> <y_center> <width> <height> <track_id>`

#### COCO Format (input only)
- JSON annotations with bounding boxes
- Converted to YOLO format during preparation

#### Pascal VOC Format (input only)
- XML annotations per image
- Converted to YOLO format during preparation

### Classification Formats

#### Image Folder
- Directory structure defines classes
- Each subdirectory is a class
- Supports train/val/test splits

#### CSV Attributes
- CSV file with image paths and attribute columns
- Supports multi-attribute classification
- Format: `image_path,attr1,attr2,attr3,...`

## Best Practices

### Versioning

1. **Use Semantic Versioning**: `v1.0.0` for major datasets
2. **Date-based Versions**: `2024-01-15` for time-series data
3. **Incremental Versions**: `v1`, `v2`, `v3` for quick iterations

### Data Organization

1. **Keep Raw Data Separate**: Never modify raw data
2. **Version Processed Data**: Store each version separately
3. **Document Changes**: Use descriptions and metadata
4. **Track Provenance**: Hashes ensure reproducibility

### Catalog Management

1. **Commit Catalog**: Version control `data/datasets.yaml`
2. **Export Manifests**: Save with each experiment
3. **Review Regularly**: Clean up unused datasets
4. **Backup**: Keep catalog backups

### Performance

1. **Cache Small Datasets**: Use `cache_images=True` for datasets < 1GB
2. **Optimize Workers**: Set `num_workers` based on CPU cores
3. **Batch Size**: Adjust based on GPU memory
4. **Preprocessing**: Do heavy preprocessing once during preparation

### Augmentation

Configure augmentation in config files:

```yaml
augmentation:
  horizontal_flip: 0.5
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  # YOLO-specific
  hsv_augmentation: true
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
```

## Example Workflows

### Complete Detection Workflow

```bash
# 1. Prepare dataset
python scripts/data_prepare.py \
  --dataset vehicle_detection \
  --version v1 \
  --raw /data/raw/vehicles \
  --task-type detection \
  --data-format coco \
  --splits train val \
  --classes car truck bus motorcycle bicycle

# 2. Train model
python scripts/train.py \
  --config configs/detection_config.yaml \
  --task detection \
  --detection-task vehicle_detection \
  --dataset vehicle_detection \
  --version v1 \
  --output-dir outputs/vehicle_det_v1

# 3. Evaluate model
python scripts/evaluate.py \
  --config configs/detection_config.yaml \
  --task detection \
  --detection-task vehicle_detection \
  --dataset vehicle_detection \
  --version v1 \
  --checkpoint outputs/vehicle_det_v1/best.pt
```

### Complete Classification Workflow

```bash
# 1. Prepare dataset
python scripts/data_prepare.py \
  --dataset human_attributes \
  --version v1 \
  --raw /data/raw/humans \
  --task-type classification \
  --data-format csv_attribute \
  --splits train val test \
  --classes male female young adult elderly

# 2. Train model
python scripts/train.py \
  --config configs/classification_config.yaml \
  --task human_attributes \
  --dataset human_attributes \
  --version v1 \
  --output-dir outputs/human_attr_v1
```

## Troubleshooting

### Dataset Not Found

**Error**: `Dataset 'xxx' not found in catalog`

**Solution**: Run `data_prepare.py` to register the dataset

### Wrong Format

**Error**: `Split 'train' not found in YAML`

**Solution**: Check that your data.yaml includes all required splits

### Path Issues

**Error**: `Image directory not found`

**Solution**: Use absolute paths or ensure relative paths are correct from working directory

### Memory Issues

**Error**: `Out of memory during data loading`

**Solutions**:
- Reduce batch size
- Disable image caching
- Reduce number of workers
- Use smaller images

## Next Steps

- See [Examples](examples.md) for complete code examples
- Check [API Reference](api_reference.md) for detailed API docs
- Read [Training Guide](training.md) for training workflows
