"""
YOLO-format dataset loader with support for detection and tracking.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import yaml
import logging
import random


class YOLODataset(Dataset):
    """
    Dataset for YOLO-format annotations (detection and tracking).
    
    Expected directory structure:
        dataset/
            images/
                train/
                    img001.jpg
                    img002.jpg
                val/
                    img003.jpg
            labels/
                train/
                    img001.txt
                    img002.txt
                val/
                    img003.txt
            data.yaml
    
    YOLO annotation format (per line in .txt file):
        <class_id> <x_center> <y_center> <width> <height> [<track_id>]
        
    All coordinates are normalized (0-1).
    """
    
    def __init__(
        self,
        data_yaml: Optional[str] = None,
        image_dir: Optional[str] = None,
        label_dir: Optional[str] = None,
        split: str = 'train',
        transform: Optional[Callable] = None,
        preprocessing_hooks: Optional[List[Callable]] = None,
        load_tracks: bool = False,
        cache_images: bool = False
    ):
        """
        Initialize YOLO dataset.
        
        Args:
            data_yaml: Path to YOLO dataset YAML file
            image_dir: Direct path to images directory (alternative to data_yaml)
            label_dir: Direct path to labels directory (alternative to data_yaml)
            split: Dataset split ('train', 'val', 'test')
            transform: Transform function to apply to images
            preprocessing_hooks: List of preprocessing functions to apply
            load_tracks: Whether to load tracking IDs from annotations
            cache_images: Whether to cache images in memory (for small datasets)
        """
        self.split = split
        self.transform = transform
        self.preprocessing_hooks = preprocessing_hooks or []
        self.load_tracks = load_tracks
        self.cache_images = cache_images
        self.logger = logging.getLogger(__name__)
        
        # Load dataset configuration
        if data_yaml:
            self._load_from_yaml(data_yaml)
        elif image_dir and label_dir:
            self.image_dir = image_dir
            self.label_dir = label_dir
            self.class_names = []
            self.num_classes = 0
        else:
            raise ValueError("Either data_yaml or both image_dir and label_dir must be provided")
        
        # Load image and label paths
        self.samples = self._load_samples()
        
        # Image cache
        self.image_cache = {} if cache_images else None
        
        self.logger.info(f"Loaded {len(self.samples)} samples for split '{split}'")
    
    def _load_from_yaml(self, yaml_path: str) -> None:
        """Load dataset configuration from YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get base path (relative to YAML file location)
        base_path = os.path.dirname(yaml_path)
        
        # Get split paths
        split_key = self.split
        if split_key not in config:
            raise ValueError(f"Split '{split_key}' not found in YAML. Available: {list(config.keys())}")
        
        # Handle both absolute and relative paths
        split_path = config[split_key]
        if not os.path.isabs(split_path):
            split_path = os.path.join(base_path, split_path)
        
        # Determine image and label directories
        # Convention: images in 'images/<split>', labels in 'labels/<split>'
        if os.path.isdir(split_path):
            self.image_dir = split_path
            # Try to find corresponding labels directory
            parent_dir = os.path.dirname(os.path.dirname(split_path))
            self.label_dir = os.path.join(parent_dir, 'labels', self.split)
            if not os.path.exists(self.label_dir):
                # Try alternative structure
                self.label_dir = split_path.replace('images', 'labels')
        else:
            raise ValueError(f"Split path not found: {split_path}")
        
        # Load class names
        if 'names' in config:
            if isinstance(config['names'], dict):
                # Dictionary format: {0: 'class1', 1: 'class2'}
                self.class_names = [config['names'][i] for i in sorted(config['names'].keys())]
            else:
                # List format
                self.class_names = config['names']
            self.num_classes = len(self.class_names)
        else:
            self.class_names = []
            self.num_classes = config.get('nc', 0)
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load image and label file paths."""
        samples = []
        
        if not os.path.exists(self.image_dir):
            self.logger.warning(f"Image directory not found: {self.image_dir}")
            return samples
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(self.image_dir).glob(f'*{ext}'))
            image_files.extend(Path(self.image_dir).glob(f'*{ext.upper()}'))
        
        for image_path in sorted(image_files):
            # Get corresponding label file
            label_filename = image_path.stem + '.txt'
            label_path = os.path.join(self.label_dir, label_filename)
            
            # Check if label exists (for test set, labels might not exist)
            has_label = os.path.exists(label_path)
            
            samples.append({
                'image_path': str(image_path),
                'label_path': label_path if has_label else None,
                'image_id': image_path.stem
            })
        
        return samples
    
    def _load_annotations(self, label_path: Optional[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load YOLO format annotations.
        
        Args:
            label_path: Path to label file
            
        Returns:
            Tuple of (bboxes, track_ids)
            - bboxes: Array of shape (N, 5) with [class_id, x_center, y_center, width, height]
            - track_ids: Array of shape (N,) with tracking IDs (None if not available)
        """
        if label_path is None or not os.path.exists(label_path):
            return np.zeros((0, 5)), None
        
        annotations = []
        track_ids = [] if self.load_tracks else None
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # Parse: class_id x_center y_center width height [track_id]
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                annotations.append([class_id, x_center, y_center, width, height])
                
                # Load track ID if available and requested
                if self.load_tracks and len(parts) >= 6:
                    track_ids.append(int(parts[5]))
                elif self.load_tracks:
                    track_ids.append(-1)  # No track ID
        
        bboxes = np.array(annotations, dtype=np.float32) if annotations else np.zeros((0, 5), dtype=np.float32)
        tracks = np.array(track_ids, dtype=np.int32) if track_ids else None
        
        return bboxes, tracks
    
    def _apply_preprocessing_hooks(self, image: np.ndarray, annotations: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply preprocessing hooks to image and annotations."""
        for hook in self.preprocessing_hooks:
            image, annotations = hook(image, annotations)
        return image, annotations
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get item by index.
        
        Returns:
            Tuple of (image, target) where target contains:
            - bboxes: Bounding boxes (N, 5) [class_id, x_center, y_center, width, height]
            - track_ids: Tracking IDs (N,) [optional]
            - image_path: Path to image
            - image_id: Image identifier
        """
        sample = self.samples[idx]
        
        # Load image
        if self.cache_images and sample['image_path'] in self.image_cache:
            image = self.image_cache[sample['image_path']].copy()
        else:
            image = cv2.imread(sample['image_path'])
            if image is None:
                raise ValueError(f"Failed to load image: {sample['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.cache_images:
                self.image_cache[sample['image_path']] = image.copy()
        
        # Load annotations
        bboxes, track_ids = self._load_annotations(sample['label_path'])
        
        # Create target dictionary
        target = {
            'bboxes': bboxes,
            'image_path': sample['image_path'],
            'image_id': sample['image_id']
        }
        
        if track_ids is not None:
            target['track_ids'] = track_ids
        
        # Apply preprocessing hooks
        image, target = self._apply_preprocessing_hooks(image, target)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, target
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return self.class_names
    
    def get_num_classes(self) -> int:
        """Get number of classes."""
        return self.num_classes


def create_yolo_dataloader(
    data_yaml: str,
    split: str = 'train',
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    preprocessing_hooks: Optional[List[Callable]] = None,
    load_tracks: bool = False,
    cache_images: bool = False,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create YOLO dataloader.
    
    Args:
        data_yaml: Path to YOLO dataset YAML file
        split: Dataset split ('train', 'val', 'test')
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        transform: Transform function
        preprocessing_hooks: Preprocessing hooks
        load_tracks: Whether to load tracking IDs
        cache_images: Whether to cache images
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader instance
    """
    dataset = YOLODataset(
        data_yaml=data_yaml,
        split=split,
        transform=transform,
        preprocessing_hooks=preprocessing_hooks,
        load_tracks=load_tracks,
        cache_images=cache_images
    )
    
    # Custom collate function to handle variable-length annotations
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        targets = [item[1] for item in batch]
        return images, targets
    
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        **kwargs
    )
