"""
Base detection models for Street Scene optimization.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet34
from typing import Dict, Any


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


class SSD300(BaseDetectionModel):
    """SSD300 model for street scene detection."""
    
    def __init__(self, num_classes: int, backbone: str = "resnet50", **kwargs):
        super().__init__(num_classes, **kwargs)
        self.backbone_name = backbone
        
        # Build backbone
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=True)
            backbone_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Detection heads (simplified)
        self.num_anchors = 6  # Simplified anchor configuration
        self.loc_head = nn.Conv2d(backbone_features, self.num_anchors * 4, 3, padding=1)
        self.conf_head = nn.Conv2d(backbone_features, self.num_anchors * (num_classes + 1), 3, padding=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize detection heads."""
        for m in [self.loc_head, self.conf_head]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        features = self.backbone(x)
        
        # Predictions
        loc_preds = self.loc_head(features)
        conf_preds = self.conf_head(features)
        
        # Reshape for loss computation
        batch_size = x.size(0)
        loc_preds = loc_preds.permute(0, 2, 3, 1).contiguous()
        loc_preds = loc_preds.view(batch_size, -1, 4)
        
        conf_preds = conf_preds.permute(0, 2, 3, 1).contiguous()
        conf_preds = conf_preds.view(batch_size, -1, self.num_classes + 1)
        
        return {
            'loc_preds': loc_preds,
            'conf_preds': conf_preds
        }


def create_detection_model(config: Dict[str, Any]) -> BaseDetectionModel:
    """Factory function to create detection models."""
    model_config = config['detection']['model']
    
    if model_config['name'] == 'ssd300':
        return SSD300(
            num_classes=model_config['num_classes'],
            backbone=model_config['backbone']
        )
    else:
        raise ValueError(f"Unsupported detection model: {model_config['name']}")