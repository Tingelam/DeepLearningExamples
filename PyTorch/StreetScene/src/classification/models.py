"""
Classification models built on top of timm backbones.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional

import timm
import torch
import torch.nn as nn


def _merge_dict(base: Optional[Dict[str, Any]], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating the inputs."""
    base = copy.deepcopy(base) if base else {}
    if not override:
        return base
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge_dict(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def resolve_classification_task_config(config: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """Resolve classification defaults and overrides for a given task."""
    classification_cfg = config.get('classification', {})
    tasks = classification_cfg.get('tasks', {})
    if task_name not in tasks:
        raise ValueError(
            f"Unknown classification task '{task_name}'. Available tasks: {list(tasks.keys())}"
        )

    defaults = classification_cfg.get('defaults', {})
    task_cfg = copy.deepcopy(tasks[task_name])

    merged_model = _merge_dict(defaults.get('model'), task_cfg.get('model'))
    merged_training = _merge_dict(defaults.get('training'), task_cfg.get('training'))
    merged_data = _merge_dict(defaults.get('data'), task_cfg.get('data'))
    heads_cfg = task_cfg.get('heads', {})

    if not heads_cfg:
        raise ValueError(
            f"Classification task '{task_name}' must define at least one prediction head in 'heads'."
        )

    return {
        'name': task_name,
        'description': task_cfg.get('description', ''),
        'model': merged_model,
        'training': merged_training,
        'data': merged_data,
        'heads': copy.deepcopy(heads_cfg),
    }


def _build_activation(name: str) -> Optional[nn.Module]:
    activation = name.lower()
    if activation in ('', 'identity', 'none'):
        return None
    if activation == 'relu':
        return nn.ReLU(inplace=True)
    if activation == 'gelu':
        return nn.GELU()
    if activation in ('silu', 'swish'):
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation '{name}' for classification head")


class ClassificationHead(nn.Module):
    """Configurable classification head made of optional hidden layers."""

    def __init__(self, in_features: int, head_cfg: Dict[str, Any]):
        super().__init__()
        if 'num_classes' not in head_cfg:
            raise ValueError("Each head configuration must include 'num_classes'.")

        num_classes = head_cfg['num_classes']
        hidden_dims: Iterable[int]
        hidden_dims = head_cfg.get('hidden_dims') or []
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        dropout = float(head_cfg.get('dropout', 0.0))
        activation = _build_activation(head_cfg.get('activation', 'relu'))

        layers: List[nn.Module] = []
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation is not None:
                layers.append(copy.deepcopy(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class TimmClassificationModel(nn.Module):
    """Classification model that wraps a timm backbone with configurable heads."""

    def __init__(self, model_cfg: Dict[str, Any], heads_cfg: Dict[str, Dict[str, Any]]):
        super().__init__()
        if not heads_cfg:
            raise ValueError("At least one prediction head must be provided.")

        backbone_name = model_cfg.get('backbone', 'resnet50')
        pretrained = model_cfg.get('pretrained', True)
        in_chans = model_cfg.get('in_channels', 3)
        global_pool = model_cfg.get('global_pool', 'avg')
        drop_rate = model_cfg.get('dropout', 0.0)
        drop_path_rate = model_cfg.get('drop_path_rate', 0.0)
        checkpoint_path = model_cfg.get('checkpoint_path')
        backbone_kwargs = copy.deepcopy(model_cfg.get('backbone_kwargs') or {})

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool=global_pool,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            checkpoint_path=checkpoint_path,
            **backbone_kwargs,
        )

        self.feature_dim = getattr(self.backbone, 'num_features', None)
        if self.feature_dim is None:
            raise ValueError(
                f"Backbone '{backbone_name}' does not expose 'num_features' needed for heads."
            )

        freeze_backbone = model_cfg.get('freeze_backbone', False)
        feature_extract = model_cfg.get('feature_extract', False)
        if freeze_backbone or feature_extract:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.heads = nn.ModuleDict({
            head_name: ClassificationHead(self.feature_dim, head_cfg)
            for head_name, head_cfg in heads_cfg.items()
        })
        self.head_names = list(self.heads.keys())
        self.return_features = model_cfg.get('return_features', False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.dim() > 2:
            features = torch.flatten(features, 1)

        outputs = {name: head(features) for name, head in self.heads.items()}
        if self.return_features:
            outputs['features'] = features
        return outputs


def create_classification_model(
    config: Dict[str, Any],
    task_name: str,
    task_config: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Factory that builds a classification model for the requested task."""
    resolved_config = task_config or resolve_classification_task_config(config, task_name)
    model_cfg = resolved_config['model']
    heads_cfg = resolved_config['heads']
    return TimmClassificationModel(model_cfg, heads_cfg)
