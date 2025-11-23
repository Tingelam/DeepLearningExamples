"""
Training utilities for Street Scene optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm

from ..common.utils import get_device


class Trainer:
    """Base trainer class."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.device = get_device(device)
        self.model.to(self.device)
        
        # Setup mixed precision
        self.use_amp = config.get('mixed_precision', {}).get('enabled', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        training_config = self.config.get('training', {})
        optimizer_name = training_config.get('optimizer', 'sgd').lower()
        lr = training_config.get('learning_rate', 0.001)
        weight_decay = training_config.get('weight_decay', 0.0001)
        
        if optimizer_name == 'sgd':
            momentum = training_config.get('momentum', 0.9)
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on task type."""
        # This should be overridden by subclasses
        return nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(self.device)
            
            # Move targets to device (handle different target formats)
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(data)
                    loss = self._compute_loss(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                loss = self._compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch_idx, (data, targets) in enumerate(pbar):
                data = data.to(self.device)
                
                # Move targets to device
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                else:
                    targets = targets.to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(data)
                        loss = self._compute_loss(outputs, targets)
                else:
                    outputs = self.model(data)
                    loss = self._compute_loss(outputs, targets)
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'val_loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def _compute_loss(self, outputs, targets) -> torch.Tensor:
        """Compute loss. Should be overridden by subclasses."""
        return self.criterion(outputs, targets)
    
    def save_checkpoint(
        self,
        filepath: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        config_hash: Optional[str] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            metrics: Dictionary of metrics to save
            metadata: Optional metadata to embed in checkpoint
            config_hash: Optional config hash for reproducibility
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        if config_hash is not None:
            checkpoint['config_hash'] = config_hash
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        return checkpoint['metrics']