#!/usr/bin/env python3
"""
Data preparation script for Street Scene datasets.

Converts raw data to YOLO/classification formats, generates manifests,
and registers datasets in the catalog.
"""

import argparse
import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import hashlib
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.catalog import DatasetCatalog


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:16]


def compute_dataset_hash(data_dir: str) -> str:
    """Compute hash for entire dataset directory."""
    file_hashes = []
    
    for root, dirs, files in sorted(os.walk(data_dir)):
        for filename in sorted(files):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.txt', '.csv')):
                file_path = os.path.join(root, filename)
                file_hashes.append(compute_file_hash(file_path))
    
    # Combine all file hashes
    combined = ''.join(file_hashes)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def convert_to_yolo_format(
    raw_data_dir: str,
    output_dir: str,
    splits: List[str],
    annotation_format: str = 'coco',
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Convert raw data to YOLO format.
    
    Args:
        raw_data_dir: Path to raw data directory
        output_dir: Output directory for YOLO format data
        splits: List of splits to process (e.g., ['train', 'val', 'test'])
        annotation_format: Format of source annotations ('coco', 'pascal_voc', 'yolo')
        logger: Logger instance
        
    Returns:
        Dictionary with conversion statistics
    """
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Converting {annotation_format} format to YOLO format")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {}
    
    for split in splits:
        split_input = os.path.join(raw_data_dir, split)
        if not os.path.exists(split_input):
            logger.warning(f"Split directory not found: {split_input}")
            continue
        
        # Create output directories
        images_dir = os.path.join(output_dir, 'images', split)
        labels_dir = os.path.join(output_dir, 'labels', split)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Handle different annotation formats
        if annotation_format == 'yolo':
            # Already in YOLO format, just copy
            num_samples = copy_yolo_format(split_input, images_dir, labels_dir, logger)
        elif annotation_format == 'coco':
            num_samples = convert_coco_to_yolo(split_input, images_dir, labels_dir, logger)
        elif annotation_format == 'pascal_voc':
            num_samples = convert_pascal_voc_to_yolo(split_input, images_dir, labels_dir, logger)
        else:
            raise ValueError(f"Unsupported annotation format: {annotation_format}")
        
        stats[split] = {
            'num_samples': num_samples,
            'images_dir': images_dir,
            'labels_dir': labels_dir
        }
        
        logger.info(f"Processed {num_samples} samples for split '{split}'")
    
    return stats


def copy_yolo_format(
    source_dir: str,
    images_dir: str,
    labels_dir: str,
    logger: logging.Logger
) -> int:
    """Copy data already in YOLO format."""
    num_samples = 0
    
    # Look for images and labels subdirectories
    source_images = os.path.join(source_dir, 'images')
    source_labels = os.path.join(source_dir, 'labels')
    
    # If images/labels subdirectories don't exist, assume flat structure
    if not os.path.exists(source_images):
        source_images = source_dir
        source_labels = source_dir
    
    # Copy images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for ext in image_extensions:
        for image_path in Path(source_images).glob(f'*{ext}'):
            shutil.copy2(image_path, images_dir)
            num_samples += 1
            
            # Copy corresponding label if it exists
            label_path = Path(source_labels) / (image_path.stem + '.txt')
            if label_path.exists():
                shutil.copy2(label_path, labels_dir)
    
    return num_samples


def convert_coco_to_yolo(
    source_dir: str,
    images_dir: str,
    labels_dir: str,
    logger: logging.Logger
) -> int:
    """Convert COCO format to YOLO format."""
    # Placeholder implementation
    # In practice, this would parse COCO JSON and convert bbox coordinates
    logger.warning("COCO conversion not fully implemented. Using placeholder.")
    return 0


def convert_pascal_voc_to_yolo(
    source_dir: str,
    images_dir: str,
    labels_dir: str,
    logger: logging.Logger
) -> int:
    """Convert Pascal VOC format to YOLO format."""
    # Placeholder implementation
    # In practice, this would parse XML annotations
    logger.warning("Pascal VOC conversion not fully implemented. Using placeholder.")
    return 0


def convert_to_classification_format(
    raw_data_dir: str,
    output_dir: str,
    splits: List[str],
    dataset_type: str = 'image_folder',
    logger: logging.Logger = None
) -> Dict[str, Any]:
    """
    Convert raw data to classification format.
    
    Args:
        raw_data_dir: Path to raw data directory
        output_dir: Output directory
        splits: List of splits to process
        dataset_type: Type of dataset ('image_folder', 'csv_attribute')
        logger: Logger instance
        
    Returns:
        Dictionary with conversion statistics
    """
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Preparing classification data in {dataset_type} format")
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {}
    
    for split in splits:
        split_input = os.path.join(raw_data_dir, split)
        if not os.path.exists(split_input):
            logger.warning(f"Split directory not found: {split_input}")
            continue
        
        split_output = os.path.join(output_dir, split)
        os.makedirs(split_output, exist_ok=True)
        
        # Copy classification data
        if dataset_type == 'image_folder':
            num_samples = copy_image_folder_structure(split_input, split_output, logger)
        elif dataset_type == 'csv_attribute':
            num_samples = copy_csv_attributes(split_input, split_output, logger)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        stats[split] = {
            'num_samples': num_samples,
            'path': split_output
        }
        
        logger.info(f"Processed {num_samples} samples for split '{split}'")
    
    return stats


def copy_image_folder_structure(
    source_dir: str,
    output_dir: str,
    logger: logging.Logger
) -> int:
    """Copy image folder structure for classification."""
    num_samples = 0
    
    # Get all class directories
    for class_dir in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        # Create output class directory
        output_class_dir = os.path.join(output_dir, class_dir)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Copy images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for ext in image_extensions:
            for image_path in Path(class_path).glob(f'*{ext}'):
                shutil.copy2(image_path, output_class_dir)
                num_samples += 1
    
    return num_samples


def copy_csv_attributes(
    source_dir: str,
    output_dir: str,
    logger: logging.Logger
) -> int:
    """Copy CSV attributes dataset."""
    # Copy CSV file and images
    csv_files = list(Path(source_dir).glob('*.csv'))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {source_dir}")
        return 0
    
    # Copy CSV file
    csv_file = csv_files[0]
    shutil.copy2(csv_file, output_dir)
    
    # Copy images directory if it exists
    images_dir = os.path.join(source_dir, 'images')
    if os.path.exists(images_dir):
        output_images_dir = os.path.join(output_dir, 'images')
        shutil.copytree(images_dir, output_images_dir, dirs_exist_ok=True)
    
    # Count samples from CSV
    import pandas as pd
    df = pd.read_csv(os.path.join(output_dir, csv_file.name))
    return len(df)


def generate_yolo_yaml(
    dataset_name: str,
    output_dir: str,
    class_names: List[str],
    splits: Dict[str, str]
) -> str:
    """
    Generate YOLO dataset YAML file.
    
    Args:
        dataset_name: Name of dataset
        output_dir: Base output directory
        class_names: List of class names
        splits: Dictionary mapping split names to paths
        
    Returns:
        Path to generated YAML file
    """
    yaml_path = os.path.join(output_dir, 'data.yaml')
    
    yaml_content = {
        'path': output_dir,
        'train': splits.get('train', ''),
        'val': splits.get('val', ''),
        'test': splits.get('test', ''),
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Street Scene tasks")
    parser.add_argument("--dataset", type=str, required=True, 
                       help="Dataset name (e.g., vehicle_detection, pedestrian_detection)")
    parser.add_argument("--version", type=str, required=True,
                       help="Dataset version (e.g., v1, v1.0.0)")
    parser.add_argument("--raw", type=str, required=True,
                       help="Path to raw data directory")
    parser.add_argument("--output", type=str, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--task-type", type=str, required=True,
                       choices=["detection", "classification", "tracking"],
                       help="Task type")
    parser.add_argument("--data-format", type=str, default="yolo",
                       choices=["yolo", "coco", "pascal_voc", "image_folder", "csv_attribute"],
                       help="Source data format")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                       help="Dataset splits to process")
    parser.add_argument("--classes", type=str, nargs="+", required=True,
                       help="Class names")
    parser.add_argument("--catalog", type=str, default="data/datasets.yaml",
                       help="Path to dataset catalog file")
    parser.add_argument("--description", type=str, default="",
                       help="Dataset description")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info(f"Preparing dataset: {args.dataset} version {args.version}")
    logger.info(f"Task type: {args.task_type}, Data format: {args.data_format}")
    
    # Create output directory
    dataset_output = os.path.join(args.output, args.dataset, args.version)
    os.makedirs(dataset_output, exist_ok=True)
    
    # Convert data based on task type
    if args.task_type in ['detection', 'tracking']:
        # Convert to YOLO format
        conversion_stats = convert_to_yolo_format(
            args.raw,
            dataset_output,
            args.splits,
            args.data_format,
            logger
        )
        
        # Generate YOLO YAML
        split_paths = {
            split: os.path.join('images', split)
            for split in args.splits if split in conversion_stats
        }
        
        yaml_path = generate_yolo_yaml(
            args.dataset,
            dataset_output,
            args.classes,
            split_paths
        )
        
        logger.info(f"Generated YOLO dataset YAML: {yaml_path}")
        
        # Prepare split information for catalog
        splits_info = {
            split: {
                'path': os.path.abspath(os.path.join(dataset_output, 'images', split)),
                'num_samples': stats['num_samples']
            }
            for split, stats in conversion_stats.items()
        }
        
        # Label schema
        label_schema = {
            'classes': args.classes,
            'num_classes': len(args.classes),
            'format': 'yolo'
        }
        
        output_format = 'yolo'
    
    elif args.task_type == 'classification':
        # Convert to classification format
        dataset_type = 'csv_attribute' if args.data_format == 'csv_attribute' else 'image_folder'
        
        conversion_stats = convert_to_classification_format(
            args.raw,
            dataset_output,
            args.splits,
            dataset_type,
            logger
        )
        
        # Prepare split information for catalog
        splits_info = {
            split: {
                'path': os.path.abspath(stats['path']),
                'num_samples': stats['num_samples']
            }
            for split, stats in conversion_stats.items()
        }
        
        # Label schema
        label_schema = {
            'classes': args.classes,
            'num_classes': len(args.classes),
            'format': dataset_type
        }
        
        output_format = dataset_type
    
    else:
        raise ValueError(f"Unsupported task type: {args.task_type}")
    
    # Compute provenance hash
    source_hash = compute_dataset_hash(args.raw)
    
    # Prepare metadata
    metadata = {
        'description': args.description,
        'source': os.path.abspath(args.raw),
        'output': os.path.abspath(dataset_output)
    }
    
    provenance = {
        'source_hash': source_hash,
        'creation_date': None,  # Will be set by catalog
        'command': ' '.join(sys.argv)
    }
    
    preprocessing_config = {
        'source_format': args.data_format,
        'output_format': output_format,
        'splits': args.splits
    }
    
    # Register in catalog
    logger.info(f"Registering dataset in catalog: {args.catalog}")
    
    catalog = DatasetCatalog(args.catalog)
    catalog.register_dataset(
        name=args.dataset,
        version=args.version,
        task_type=args.task_type,
        data_format=output_format,
        splits=splits_info,
        label_schema=label_schema,
        preprocessing_config=preprocessing_config,
        metadata=metadata,
        provenance=provenance
    )
    
    logger.info("Dataset preparation complete!")
    logger.info(f"Dataset registered: {args.dataset} version {args.version}")
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET PREPARATION SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Version: {args.version}")
    print(f"Task Type: {args.task_type}")
    print(f"Output Format: {output_format}")
    print(f"Output Directory: {dataset_output}")
    print(f"\nSplits:")
    for split, info in splits_info.items():
        print(f"  {split}: {info['num_samples']} samples")
    print(f"\nProvenance Hash: {provenance['source_hash']}")
    print(f"Catalog: {args.catalog}")
    print("="*60)


if __name__ == "__main__":
    main()
