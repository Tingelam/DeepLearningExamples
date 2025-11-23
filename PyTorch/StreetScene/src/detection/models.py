"""
Detection models for Street Scene optimization.

Supports both legacy SSD models and modern YOLO models via ultralytics.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

from .yolo_adapter import YOLOAdapter


class BaseDetectionModel(nn.Module):
    """Base class for detection models."""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = kwargs.get('model_name', 'base_detection')
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        raise NotImplementedError
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference method."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class YOLODetectionModel(BaseDetectionModel):
    """YOLO detection model wrapper for ultralytics YOLO variants."""
    
    def __init__(
        self,
        num_classes: int,
        model_variant: str = "yolov8n",
        device: Optional[str] = None,
        **kwargs
    ):
        super().__init__(num_classes, **kwargs)
        self.model_variant = model_variant
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create YOLO adapter
        self.yolo = YOLOAdapter(
            model_variant=model_variant,
            task_name="detect",
            num_classes=num_classes,
            device=self.device,
            **kwargs
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through YOLO model.
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            
        Returns:
            Dictionary with detection outputs (for compatibility)
        """
        # YOLO expects PIL images or file paths, not tensors
        # For direct tensor forward pass, we use the underlying ultralytics model
        self.yolo.model.eval()
        with torch.no_grad():
            outputs = self.yolo.model(x)
        return outputs


def create_detection_model(config: Dict[str, Any], task_config: Optional[Dict[str, Any]] = None) -> BaseDetectionModel:
    """
    Factory function to create detection models.
    
    Args:
        config: Main configuration dictionary
        task_config: Task-specific configuration (overrides config['detection'])
        
    Returns:
        Detection model instance
    """
    # Use task_config if provided, otherwise use detection config
    model_config = task_config.get('model') if task_config else config.get('detection', {}).get('model', {})
    
    if not model_config:
        raise ValueError("No model configuration found in config or task_config")
    
    model_type = model_config.get('type', 'yolo').lower()
    
    if model_type == 'yolo':
        return YOLODetectionModel(
            num_classes=model_config.get('num_classes', 80),
            model_variant=model_config.get('variant', 'yolov8n'),
            device=model_config.get('device', None)
        )
    else:
        raise ValueError(f"Unsupported detection model type: {model_type}")