"""
Base classification models for Street Scene optimization.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet34
from typing import Dict, Any


class BaseClassificationModel(nn.Module):
    """Base class for classification models."""
    
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = kwargs.get('model_name', 'base_classification')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        raise NotImplementedError
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference method."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


class VehicleClassifier(BaseClassificationModel):
    """Vehicle classification model."""
    
    def __init__(self, num_classes: int, backbone: str = "resnet50", pretrained: bool = True, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.backbone_name = backbone
        
        # Build backbone
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=pretrained)
            backbone_features = 2048
        elif backbone == "resnet34":
            self.backbone = resnet34(pretrained=pretrained)
            backbone_features = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classification head
        self.backbone.fc = nn.Linear(backbone_features, num_classes)
        
        # Initialize new classification layer
        nn.init.xavier_uniform_(self.backbone.fc.weight)
        nn.init.constant_(self.backbone.fc.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)


class HumanAttributeClassifier(BaseClassificationModel):
    """Human attribute classification model."""
    
    def __init__(self, num_classes: int, backbone: str = "resnet34", pretrained: bool = True, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.backbone_name = backbone
        
        # Build backbone
        if backbone == "resnet34":
            self.backbone = resnet34(pretrained=pretrained)
            backbone_features = 512
        else:
            raise ValueError(f"Unsupported backbone for attribute classification: {backbone}")
        
        # Multi-task heads for different attributes
        self.backbone.fc = nn.Identity()
        
        # Attribute heads (example: gender, age group, clothing type, etc.)
        self.gender_head = nn.Linear(backbone_features, 2)  # male/female
        self.age_head = nn.Linear(backbone_features, 4)     # age groups
        self.clothing_head = nn.Linear(backbone_features, 8) # clothing types
        
        # Initialize heads
        for head in [self.gender_head, self.age_head, self.clothing_head]:
            nn.init.xavier_uniform_(head.weight)
            nn.init.constant_(head.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        features = self.backbone(x)
        
        return {
            'gender': self.gender_head(features),
            'age': self.age_head(features),
            'clothing': self.clothing_head(features)
        }


def create_classification_model(config: Dict[str, Any], task_type: str) -> BaseClassificationModel:
    """Factory function to create classification models."""
    if task_type == "vehicle":
        model_config = config['classification']['vehicle']
        return VehicleClassifier(
            num_classes=model_config['num_classes'],
            backbone=model_config['name'],
            pretrained=model_config['pretrained']
        )
    elif task_type == "human_attributes":
        model_config = config['classification']['human_attributes']
        return HumanAttributeClassifier(
            num_classes=model_config['num_classes'],
            backbone=model_config['name'],
            pretrained=model_config['pretrained']
        )
    else:
        raise ValueError(f"Unsupported classification task: {task_type}")