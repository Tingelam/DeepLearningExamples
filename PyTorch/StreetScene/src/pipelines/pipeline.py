"""
End-to-end pipelines for Street Scene optimization.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from ..common.utils import setup_logging, load_config, seed_everything
from ..common.trainer import Trainer
from ..data.dataset import StreetSceneDataset, get_train_transforms, get_val_transforms, create_dataloader
from ..detection.models import create_detection_model
from ..classification.models import create_classification_model


class StreetScenePipeline:
    """End-to-end pipeline for street scene tasks."""
    
    def __init__(self, config_path: str, task_type: str, log_level: str = "INFO"):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
            task_type: Type of task ('detection', 'vehicle_classification', 'human_attributes')
            log_level: Logging level
        """
        self.config = load_config(config_path)
        self.task_type = task_type
        self.logger = setup_logging(log_level)
        
        # Set random seeds
        seed_everything()
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize trainer
        self.trainer = self._create_trainer()
        
        self.logger.info(f"Initialized {task_type} pipeline")
    
    def _create_model(self) -> nn.Module:
        """Create model based on task type."""
        if self.task_type == 'detection':
            return create_detection_model(self.config)
        elif self.task_type in ['vehicle_classification', 'human_attributes']:
            classification_type = self.task_type.replace('_classification', '')
            return create_classification_model(self.config, classification_type)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def _create_trainer(self) -> Trainer:
        """Create trainer for the model."""
        # Create task-specific trainer
        if self.task_type == 'detection':
            return DetectionTrainer(self.model, self.config)
        elif self.task_type in ['vehicle_classification', 'human_attributes']:
            return ClassificationTrainer(self.model, self.config, self.task_type)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def prepare_data(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        annotation_file: Optional[str] = None
    ) -> tuple:
        """Prepare training and validation data loaders."""
        # Get image size from config
        if self.task_type == 'detection':
            image_size = tuple(self.config['detection']['data']['image_size'])
        else:
            config_key = 'vehicle' if self.task_type == 'vehicle_classification' else 'human_attributes'
            image_size = tuple(self.config['classification'][config_key]['data']['image_size'])
        
        # Create transforms
        train_transform = get_train_transforms(image_size)
        val_transform = get_val_transforms(image_size)
        
        # Create datasets
        train_dataset = StreetSceneDataset(
            data_path=train_data_path,
            transform=train_transform,
            annotation_file=annotation_file
        )
        
        val_dataset = None
        if val_data_path:
            val_dataset = StreetSceneDataset(
                data_path=val_data_path,
                transform=val_transform,
                annotation_file=annotation_file
            )
        
        # Create data loaders
        batch_size = self._get_batch_size()
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = create_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        return train_loader, val_loader
    
    def _get_batch_size(self) -> int:
        """Get batch size from config."""
        if self.task_type == 'detection':
            return self.config['detection']['training']['batch_size']
        else:
            config_key = 'vehicle' if self.task_type == 'vehicle_classification' else 'human_attributes'
            return self.config['classification'][config_key]['training']['batch_size']
    
    def train(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        annotation_file: Optional[str] = None,
        output_dir: str = "./outputs",
        resume_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            annotation_file: Path to annotation file
            output_dir: Directory to save outputs
            resume_checkpoint: Path to checkpoint to resume from
        
        Returns:
            Training metrics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            train_data_path, val_data_path, annotation_file
        )
        
        # Resume from checkpoint if specified
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            metrics = self.trainer.load_checkpoint(resume_checkpoint)
            self.logger.info(f"Resumed training from {resume_checkpoint}")
        
        # Get number of epochs
        epochs = self._get_num_epochs()
        
        # Training loop
        best_val_metric = 0.0
        train_metrics = []
        
        for epoch in range(self.trainer.current_epoch, epochs):
            self.trainer.current_epoch = epoch
            
            # Train epoch
            train_metrics_epoch = self.trainer.train_epoch(train_loader)
            
            # Validate epoch
            val_metrics_epoch = {}
            if val_loader:
                val_metrics_epoch = self.trainer.validate_epoch(val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics_epoch, **val_metrics_epoch}
            train_metrics.append(epoch_metrics)
            
            # Log metrics
            self.logger.info(f"Epoch {epoch}: {epoch_metrics}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pth")
            self.trainer.save_checkpoint(checkpoint_path, epoch_metrics)
            
            # Save best model
            if val_metrics_epoch:
                current_metric = list(val_metrics_epoch.values())[0]  # Use first validation metric
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    best_model_path = os.path.join(output_dir, "best_model.pth")
                    self.trainer.save_checkpoint(best_model_path, epoch_metrics)
        
        return {
            'train_metrics': train_metrics,
            'best_val_metric': best_val_metric,
            'output_dir': output_dir
        }
    
    def _get_num_epochs(self) -> int:
        """Get number of epochs from config."""
        if self.task_type == 'detection':
            return self.config['detection']['training']['epochs']
        else:
            config_key = 'vehicle' if self.task_type == 'vehicle_classification' else 'human_attributes'
            return self.config['classification'][config_key]['training']['epochs']
    
    def evaluate(
        self,
        test_data_path: str,
        checkpoint_path: str,
        annotation_file: Optional[str] = None,
        output_dir: str = "./outputs"
    ) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            test_data_path: Path to test data
            checkpoint_path: Path to model checkpoint
            annotation_file: Path to annotation file
            output_dir: Directory to save outputs
        
        Returns:
            Evaluation metrics
        """
        # Load checkpoint
        metrics = self.trainer.load_checkpoint(checkpoint_path)
        
        # Prepare test data
        image_size = self._get_image_size()
        val_transform = get_val_transforms(image_size)
        
        test_dataset = StreetSceneDataset(
            data_path=test_data_path,
            transform=val_transform,
            annotation_file=annotation_file
        )
        
        batch_size = self._get_batch_size()
        test_loader = create_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Evaluate
        test_metrics = self.trainer.validate_epoch(test_loader)
        
        self.logger.info(f"Test metrics: {test_metrics}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_path = os.path.join(output_dir, "test_results.pth")
        torch.save({
            'metrics': test_metrics,
            'config': self.config,
            'task_type': self.task_type
        }, results_path)
        
        return test_metrics
    
    def _get_image_size(self) -> tuple:
        """Get image size from config."""
        if self.task_type == 'detection':
            return tuple(self.config['detection']['data']['image_size'])
        else:
            config_key = 'vehicle' if self.task_type == 'vehicle_classification' else 'human_attributes'
            return tuple(self.config['classification'][config_key]['data']['image_size'])


class DetectionTrainer(Trainer):
    """Trainer for detection models."""
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function for detection."""
        # Simplified detection loss - in practice, this would be more complex
        return nn.MultiLabelSoftMarginLoss()


class ClassificationTrainer(Trainer):
    """Trainer for classification models."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], task_type: str, **kwargs):
        super().__init__(model, config, **kwargs)
        self.task_type = task_type
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function for classification."""
        if self.task_type == 'human_attributes':
            # Multi-task loss for human attributes
            return {
                'gender': nn.CrossEntropyLoss(),
                'age': nn.CrossEntropyLoss(),
                'clothing': nn.CrossEntropyLoss()
            }
        else:
            # Standard classification loss
            return nn.CrossEntropyLoss()
    
    def _compute_loss(self, outputs, targets) -> torch.Tensor:
        """Compute loss for classification tasks."""
        if isinstance(self.criterion, dict):
            # Multi-task loss
            total_loss = 0
            for task_name, criterion in self.criterion.items():
                task_loss = criterion(outputs[task_name], targets[task_name])
                total_loss += task_loss
            return total_loss
        else:
            # Single task loss
            return self.criterion(outputs, targets)