"""
Shared preprocessing and augmentation pipelines for detection and classification tasks.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple, List, Dict, Any, Optional, Callable
import random
import logging


class YOLOMosaicAugmentation:
    """YOLO-style mosaic augmentation combining 4 images."""
    
    def __init__(self, output_size: Tuple[int, int] = (640, 640), prob: float = 1.0):
        """
        Initialize mosaic augmentation.
        
        Args:
            output_size: Output image size (height, width)
            prob: Probability of applying mosaic
        """
        self.output_size = output_size
        self.prob = prob
    
    def __call__(self, images: List[np.ndarray], targets: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply mosaic augmentation to 4 images.
        
        Args:
            images: List of 4 images
            targets: List of 4 target dictionaries with 'bboxes' key
            
        Returns:
            Tuple of (mosaic_image, mosaic_target)
        """
        if random.random() > self.prob or len(images) != 4:
            # Return first image if not applying mosaic
            return images[0], targets[0]
        
        h, w = self.output_size
        
        # Create empty mosaic canvas
        mosaic = np.full((h, w, 3), 114, dtype=np.uint8)
        
        # Center point for mosaic
        cx = w // 2
        cy = h // 2
        
        all_bboxes = []
        
        # Place 4 images in quadrants
        positions = [
            (0, 0, cx, cy),           # Top-left
            (cx, 0, w, cy),           # Top-right
            (0, cy, cx, h),           # Bottom-left
            (cx, cy, w, h)            # Bottom-right
        ]
        
        for i, (img, target) in enumerate(zip(images, targets)):
            x1, y1, x2, y2 = positions[i]
            quad_w = x2 - x1
            quad_h = y2 - y1
            
            # Resize image to fit quadrant
            img_resized = cv2.resize(img, (quad_w, quad_h))
            
            # Place in mosaic
            mosaic[y1:y2, x1:x2] = img_resized
            
            # Adjust bounding boxes
            if 'bboxes' in target and len(target['bboxes']) > 0:
                bboxes = target['bboxes'].copy()
                
                # Scale and translate bboxes
                # YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                bboxes[:, 1] = (bboxes[:, 1] * quad_w + x1) / w  # x_center
                bboxes[:, 2] = (bboxes[:, 2] * quad_h + y1) / h  # y_center
                bboxes[:, 3] = bboxes[:, 3] * quad_w / w         # width
                bboxes[:, 4] = bboxes[:, 4] * quad_h / h         # height
                
                all_bboxes.append(bboxes)
        
        # Combine all bboxes
        if all_bboxes:
            combined_bboxes = np.concatenate(all_bboxes, axis=0)
        else:
            combined_bboxes = np.zeros((0, 5), dtype=np.float32)
        
        mosaic_target = {
            'bboxes': combined_bboxes,
            'image_path': 'mosaic',
            'image_id': 'mosaic'
        }
        
        return mosaic, mosaic_target


class HSVAugmentation:
    """HSV color space augmentation (YOLO-style)."""
    
    def __init__(
        self,
        hue_gain: float = 0.015,
        saturation_gain: float = 0.7,
        value_gain: float = 0.4
    ):
        """
        Initialize HSV augmentation.
        
        Args:
            hue_gain: Hue augmentation factor
            saturation_gain: Saturation augmentation factor
            value_gain: Value augmentation factor
        """
        self.hue_gain = hue_gain
        self.saturation_gain = saturation_gain
        self.value_gain = value_gain
    
    def __call__(self, image: np.ndarray, target: Any = None) -> Tuple[np.ndarray, Any]:
        """
        Apply HSV augmentation.
        
        Args:
            image: Input image (RGB)
            target: Target annotations (passed through unchanged)
            
        Returns:
            Tuple of (augmented_image, target)
        """
        # Random gains
        r = np.random.uniform(-1, 1, 3) * [self.hue_gain, self.saturation_gain, self.value_gain] + 1
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Apply gains
        hsv[..., 0] = (hsv[..., 0] * r[0]) % 180  # Hue (wrap around)
        hsv[..., 1] = np.clip(hsv[..., 1] * r[1], 0, 255)  # Saturation
        hsv[..., 2] = np.clip(hsv[..., 2] * r[2], 0, 255)  # Value
        
        # Convert back to RGB
        image_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return image_aug, target


class RandomHorizontalFlipWithBBoxes:
    """Random horizontal flip with bbox adjustment."""
    
    def __init__(self, prob: float = 0.5):
        """
        Initialize random horizontal flip.
        
        Args:
            prob: Probability of applying flip
        """
        self.prob = prob
    
    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply random horizontal flip.
        
        Args:
            image: Input image
            target: Target dictionary with 'bboxes' key
            
        Returns:
            Tuple of (flipped_image, flipped_target)
        """
        if random.random() < self.prob:
            # Flip image
            image = np.fliplr(image).copy()
            
            # Flip bboxes
            if 'bboxes' in target and len(target['bboxes']) > 0:
                target = target.copy()
                bboxes = target['bboxes'].copy()
                # YOLO format: flip x_center
                bboxes[:, 1] = 1.0 - bboxes[:, 1]
                target['bboxes'] = bboxes
        
        return image, target


class ClassificationAugmentationPolicy:
    """Classification augmentation policy with common transforms."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        horizontal_flip: float = 0.5,
        rotation: int = 15,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        normalize: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ):
        """
        Initialize classification augmentation policy.
        
        Args:
            image_size: Target image size
            horizontal_flip: Probability of horizontal flip
            rotation: Maximum rotation angle in degrees
            brightness: Brightness jitter factor
            contrast: Contrast jitter factor
            saturation: Saturation jitter factor
            normalize: Whether to normalize with ImageNet stats
            mean: Normalization mean
            std: Normalization std
        """
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
        ]
        
        if horizontal_flip > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip))
        
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))
        
        if brightness > 0 or contrast > 0 or saturation > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation
                )
            )
        
        transform_list.append(transforms.ToTensor())
        
        if normalize:
            transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        self.transform = transforms.Compose(transform_list)
    
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Apply augmentation policy."""
        return self.transform(image)


class DetectionAugmentationPipeline:
    """
    Comprehensive augmentation pipeline for detection tasks.
    Combines YOLO-specific augmentations.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (640, 640),
        horizontal_flip: float = 0.5,
        hsv_augmentation: bool = True,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        normalize: bool = False
    ):
        """
        Initialize detection augmentation pipeline.
        
        Args:
            image_size: Target image size
            horizontal_flip: Probability of horizontal flip
            hsv_augmentation: Whether to apply HSV augmentation
            hsv_h: Hue gain
            hsv_s: Saturation gain
            hsv_v: Value gain
            normalize: Whether to normalize images
        """
        self.image_size = image_size
        self.augmentations = []
        
        if horizontal_flip > 0:
            self.augmentations.append(RandomHorizontalFlipWithBBoxes(prob=horizontal_flip))
        
        if hsv_augmentation:
            self.augmentations.append(HSVAugmentation(hue_gain=hsv_h, saturation_gain=hsv_s, value_gain=hsv_v))
        
        self.normalize = normalize
    
    def __call__(self, image: np.ndarray, target: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply augmentation pipeline.
        
        Args:
            image: Input image
            target: Target dictionary
            
        Returns:
            Tuple of (augmented_image_tensor, target)
        """
        # Apply augmentations
        for aug in self.augmentations:
            image, target = aug(image, target)
        
        # Resize
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Normalize
        if self.normalize:
            image_tensor = image_tensor / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
        else:
            image_tensor = image_tensor / 255.0
        
        return image_tensor, target


def get_train_transforms(
    image_size: Tuple[int, int],
    task_type: str = 'classification',
    config: Optional[Dict[str, Any]] = None
) -> Callable:
    """
    Get training transforms based on task type and config.
    
    Args:
        image_size: Target image size
        task_type: Task type ('detection', 'classification', 'tracking')
        config: Configuration dictionary with augmentation settings
        
    Returns:
        Transform function
    """
    config = config or {}
    
    if task_type == 'detection' or task_type == 'tracking':
        return DetectionAugmentationPipeline(
            image_size=image_size,
            horizontal_flip=config.get('horizontal_flip', 0.5),
            hsv_augmentation=config.get('hsv_augmentation', True),
            hsv_h=config.get('hsv_h', 0.015),
            hsv_s=config.get('hsv_s', 0.7),
            hsv_v=config.get('hsv_v', 0.4),
            normalize=config.get('normalize', False)
        )
    
    elif task_type == 'classification':
        return ClassificationAugmentationPolicy(
            image_size=image_size,
            horizontal_flip=config.get('horizontal_flip', 0.5),
            rotation=config.get('rotation', 15),
            brightness=config.get('brightness', 0.2),
            contrast=config.get('contrast', 0.2),
            saturation=config.get('saturation', 0.2),
            normalize=config.get('normalize', True),
            mean=config.get('mean', [0.485, 0.456, 0.406]),
            std=config.get('std', [0.229, 0.224, 0.225])
        )
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def get_val_transforms(
    image_size: Tuple[int, int],
    task_type: str = 'classification',
    normalize: bool = True
) -> Callable:
    """
    Get validation transforms (no augmentation).
    
    Args:
        image_size: Target image size
        task_type: Task type ('detection', 'classification', 'tracking')
        normalize: Whether to normalize
        
    Returns:
        Transform function
    """
    if task_type == 'classification':
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)
    
    elif task_type == 'detection' or task_type == 'tracking':
        # For detection, create a simple pipeline
        return DetectionAugmentationPipeline(
            image_size=image_size,
            horizontal_flip=0.0,  # No flip for validation
            hsv_augmentation=False,
            normalize=normalize
        )
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def create_preprocessing_hooks(
    task_type: str,
    config: Optional[Dict[str, Any]] = None
) -> List[Callable]:
    """
    Create preprocessing hooks based on task type.
    
    Args:
        task_type: Task type
        config: Configuration dictionary
        
    Returns:
        List of preprocessing hook functions
    """
    hooks = []
    config = config or {}
    
    # Add common preprocessing hooks here based on config
    # For example: bbox filtering, image quality checks, etc.
    
    return hooks
