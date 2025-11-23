"""
Classification dataset loaders for image folders and CSV-based annotations.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import logging


class ImageFolderClassificationDataset(Dataset):
    """
    Classification dataset from image folders.
    
    Expected directory structure:
        dataset/
            train/
                class1/
                    img001.jpg
                    img002.jpg
                class2/
                    img003.jpg
            val/
                class1/
                    img004.jpg
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        preprocessing_hooks: Optional[List[Callable]] = None,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize image folder classification dataset.
        
        Args:
            data_dir: Path to data directory
            transform: Transform function to apply to images
            preprocessing_hooks: List of preprocessing functions
            class_names: Explicit list of class names (optional, will auto-detect if None)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.preprocessing_hooks = preprocessing_hooks or []
        self.logger = logging.getLogger(__name__)
        
        # Load class names and samples
        if class_names:
            self.class_names = class_names
        else:
            self.class_names = self._discover_classes()
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        self.samples = self._load_samples()
        
        self.logger.info(f"Loaded {len(self.samples)} samples from {len(self.class_names)} classes")
    
    def _discover_classes(self) -> List[str]:
        """Discover class names from directory structure."""
        classes = []
        for item in sorted(os.listdir(self.data_dir)):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path):
                classes.append(item)
        return classes
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load image paths and labels."""
        samples = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            # Get all images in class directory
            for ext in image_extensions:
                image_files = list(Path(class_dir).glob(f'*{ext}'))
                image_files.extend(Path(class_dir).glob(f'*{ext.upper()}'))
                
                for image_path in image_files:
                    samples.append({
                        'image_path': str(image_path),
                        'label': class_idx,
                        'class_name': class_name
                    })
        
        return samples
    
    def _apply_preprocessing_hooks(self, image: np.ndarray, label: int) -> Tuple[np.ndarray, int]:
        """Apply preprocessing hooks."""
        for hook in self.preprocessing_hooks:
            image, label = hook(image, label)
        return image, label
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index."""
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label = sample['label']
        
        # Apply preprocessing hooks
        image, label = self._apply_preprocessing_hooks(image, label)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return self.class_names


class CSVAttributeDataset(Dataset):
    """
    Multi-attribute classification dataset from CSV annotations.
    
    CSV format:
        image_path,attribute1,attribute2,attribute3,...
        /path/to/img001.jpg,0,2,1,...
        /path/to/img002.jpg,1,0,3,...
    
    Supports multi-task classification where each attribute is a separate classification task.
    """
    
    def __init__(
        self,
        csv_path: str,
        image_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        preprocessing_hooks: Optional[List[Callable]] = None,
        attribute_names: Optional[List[str]] = None,
        attribute_schemas: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize CSV attribute dataset.
        
        Args:
            csv_path: Path to CSV file with annotations
            image_dir: Base directory for images (prepended to image paths if relative)
            transform: Transform function to apply to images
            preprocessing_hooks: List of preprocessing functions
            attribute_names: List of attribute column names in CSV (optional)
            attribute_schemas: Dictionary mapping attribute names to their possible values
                              e.g., {'gender': ['male', 'female'], 'age': ['young', 'adult', 'elderly']}
        """
        self.csv_path = csv_path
        self.image_dir = image_dir or ''
        self.transform = transform
        self.preprocessing_hooks = preprocessing_hooks or []
        self.logger = logging.getLogger(__name__)
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Determine attribute columns
        if attribute_names:
            self.attribute_names = attribute_names
        else:
            # Use all columns except 'image_path'
            self.attribute_names = [col for col in self.df.columns if col != 'image_path']
        
        self.attribute_schemas = attribute_schemas or {}
        self.num_attributes = len(self.attribute_names)
        
        self.logger.info(f"Loaded {len(self.df)} samples with {self.num_attributes} attributes")
    
    def _apply_preprocessing_hooks(
        self,
        image: np.ndarray,
        attributes: Dict[str, int]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Apply preprocessing hooks."""
        for hook in self.preprocessing_hooks:
            image, attributes = hook(image, attributes)
        return image, attributes
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Get item by index."""
        row = self.df.iloc[idx]
        
        # Get image path
        image_path = row['image_path']
        if not os.path.isabs(image_path) and self.image_dir:
            image_path = os.path.join(self.image_dir, image_path)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get attributes
        attributes = {attr: int(row[attr]) for attr in self.attribute_names}
        
        # Apply preprocessing hooks
        image, attributes = self._apply_preprocessing_hooks(image, attributes)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Convert to tensor if no transform provided
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, attributes
    
    def get_attribute_names(self) -> List[str]:
        """Get attribute names."""
        return self.attribute_names
    
    def get_attribute_schema(self, attribute_name: str) -> Optional[List[str]]:
        """Get possible values for an attribute."""
        return self.attribute_schemas.get(attribute_name)


def create_classification_dataloader(
    data_path: str,
    dataset_type: str = 'image_folder',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    preprocessing_hooks: Optional[List[Callable]] = None,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create classification dataloader.
    
    Args:
        data_path: Path to data directory or CSV file
        dataset_type: Type of dataset ('image_folder' or 'csv_attribute')
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        transform: Transform function
        preprocessing_hooks: Preprocessing hooks
        **kwargs: Additional dataset and DataLoader arguments
        
    Returns:
        DataLoader instance
    """
    if dataset_type == 'image_folder':
        dataset = ImageFolderClassificationDataset(
            data_dir=data_path,
            transform=transform,
            preprocessing_hooks=preprocessing_hooks,
            class_names=kwargs.get('class_names')
        )
        
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
    
    elif dataset_type == 'csv_attribute':
        dataset = CSVAttributeDataset(
            csv_path=data_path,
            image_dir=kwargs.get('image_dir'),
            transform=transform,
            preprocessing_hooks=preprocessing_hooks,
            attribute_names=kwargs.get('attribute_names'),
            attribute_schemas=kwargs.get('attribute_schemas')
        )
        
        # Custom collate function for multi-attribute labels
        def collate_fn(batch):
            images = torch.stack([item[0] for item in batch])
            
            # Collect attributes
            attribute_names = dataset.get_attribute_names()
            attributes = {
                attr: torch.tensor([item[1][attr] for item in batch], dtype=torch.long)
                for attr in attribute_names
            }
            
            return images, attributes
        
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
