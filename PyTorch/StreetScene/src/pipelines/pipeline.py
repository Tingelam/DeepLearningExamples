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
from ..data.dataset import StreetSceneDataset, create_dataloader
from ..data.transforms import get_train_transforms, get_val_transforms, create_preprocessing_hooks
from ..data.yolo_dataset import create_yolo_dataloader
from ..data.classification_dataset import create_classification_dataloader
from ..data.catalog import get_catalog
from ..detection.models import create_detection_model
from ..detection.yolo_adapter import YOLOAdapter
from ..classification.models import create_classification_model


class StreetScenePipeline:
    """End-to-end pipeline for street scene tasks."""
    
    def __init__(self, config_path: str, task_type: str, log_level: str = "INFO", detection_task: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
            task_type: Type of task ('detection', 'vehicle_classification', 'human_attributes')
            log_level: Logging level
            detection_task: Specific detection task name (e.g., 'vehicle_detection', 'pedestrian_detection')
        """
        self.config = load_config(config_path)
        self.task_type = task_type
        self.detection_task = detection_task
        self.task_config = None
        self.logger = setup_logging(log_level)
        
        catalog_cfg = self.config.get('data_catalog', {})
        self.catalog_path = catalog_cfg.get('path', 'data/datasets.yaml')
        
        # Set random seeds
        seed_everything()
        
        # For detection tasks, load task-specific configuration
        if self.task_type == 'detection' and detection_task:
            self._load_detection_task_config(detection_task)
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize trainer
        self.trainer = self._create_trainer()
        
        self.logger.info(f"Initialized {task_type}" + (f" ({detection_task})" if detection_task else "") + " pipeline")
    
    def _load_detection_task_config(self, task_name: str) -> None:
        """Load task-specific detection configuration."""
        if 'tasks' not in self.config.get('detection', {}):
            self.logger.warning(f"No tasks catalog found for detection task: {task_name}")
            return
        
        tasks = self.config['detection']['tasks']
        if task_name not in tasks:
            raise ValueError(f"Unknown detection task: {task_name}. Available tasks: {list(tasks.keys())}")
        
        self.task_config = tasks[task_name]
        self.logger.info(f"Loaded task configuration for: {task_name}")
    
    def _create_model(self) -> nn.Module:
        """Create model based on task type."""
        if self.task_type == 'detection':
            return create_detection_model(self.config, self.task_config)
        elif self.task_type in ['vehicle_classification', 'human_attributes']:
            classification_type = self.task_type.replace('_classification', '')
            return create_classification_model(self.config, classification_type)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def _create_trainer(self) -> Trainer:
        """Create trainer for the model."""
        # Create task-specific trainer
        if self.task_type == 'detection':
            # Check if this is a YOLO model
            if hasattr(self.model, 'yolo'):
                return YOLODetectionTrainer(self.model, self.config, self.task_config)
            else:
                return DetectionTrainer(self.model, self.config)
        elif self.task_type in ['vehicle_classification', 'human_attributes']:
            return ClassificationTrainer(self.model, self.config, self.task_type)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def prepare_data(
        self,
        train_data_path: Optional[str] = None,
        val_data_path: Optional[str] = None,
        annotation_file: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_version: Optional[str] = None,
        catalog_path: str = "data/datasets.yaml"
    ) -> tuple:
        """
        Prepare training and validation data loaders.
        
        Can load data either from:
        1. Dataset catalog (preferred) - using dataset_name parameter
        2. Direct paths - using train_data_path and val_data_path parameters
        
        Args:
            train_data_path: Direct path to training data (legacy)
            val_data_path: Direct path to validation data (legacy)
            annotation_file: Path to annotation file (legacy)
            dataset_name: Name of dataset in catalog
            dataset_version: Version of dataset in catalog
            catalog_path: Path to catalog file
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Resolve dataset from catalog if dataset_name is provided
        if dataset_name:
            catalog = get_catalog(catalog_path)
            dataset_info = catalog.get_dataset(dataset_name, dataset_version)
            
            self.logger.info(f"Loading dataset from catalog: {dataset_name} version {dataset_info['version']}")
            
            # Get paths from catalog
            train_data_path = catalog.get_dataset_path(dataset_name, dataset_version, 'train')
            val_data_path = catalog.get_dataset_path(dataset_name, dataset_version, 'val') \
                if 'val' in dataset_info['splits'] else None
            
            # Get label schema
            label_schema = catalog.get_label_schema(dataset_name, dataset_version)
            
            # Determine data format
            data_format = dataset_info['data_format']
            task_type = dataset_info['task_type']
            
        else:
            # Legacy path: use direct paths
            data_format = 'legacy'
            task_type = self.task_type
        
        # Get image size from config
        if self.task_type == 'detection' or self.task_type == 'tracking':
            image_size = tuple(self.config['detection']['data']['image_size'])
            augmentation_config = self.config.get('detection', {}).get('augmentation', {})
        else:
            config_key = 'vehicle' if self.task_type == 'vehicle_classification' else 'human_attributes'
            image_size = tuple(self.config['classification'][config_key]['data']['image_size'])
            augmentation_config = self.config['classification'][config_key].get('augmentation', {})
        
        # Create preprocessing hooks
        preprocessing_hooks = create_preprocessing_hooks(self.task_type, augmentation_config)
        
        # Create loaders based on data format
        if dataset_name and data_format == 'yolo':
            # Use YOLO dataloader
            train_transform = get_train_transforms(image_size, task_type='detection', config=augmentation_config)
            val_transform = get_val_transforms(image_size, task_type='detection')
            
            # Note: For YOLO training via ultralytics, we typically pass the data.yaml directly
            # These loaders are for custom training loops
            batch_size = self._get_batch_size()
            
            train_loader = create_yolo_dataloader(
                data_yaml=os.path.join(os.path.dirname(train_data_path), '..', 'data.yaml'),
                split='train',
                batch_size=batch_size,
                shuffle=True,
                transform=train_transform,
                preprocessing_hooks=preprocessing_hooks
            )
            
            val_loader = None
            if val_data_path:
                val_loader = create_yolo_dataloader(
                    data_yaml=os.path.join(os.path.dirname(val_data_path), '..', 'data.yaml'),
                    split='val',
                    batch_size=batch_size,
                    shuffle=False,
                    transform=val_transform,
                    preprocessing_hooks=preprocessing_hooks
                )
        
        elif dataset_name and data_format in ['image_folder', 'csv_attribute']:
            # Use classification dataloader
            train_transform = get_train_transforms(image_size, task_type='classification', config=augmentation_config)
            val_transform = get_val_transforms(image_size, task_type='classification')
            
            batch_size = self._get_batch_size()
            
            train_loader = create_classification_dataloader(
                data_path=train_data_path,
                dataset_type=data_format,
                batch_size=batch_size,
                shuffle=True,
                transform=train_transform,
                preprocessing_hooks=preprocessing_hooks
            )
            
            val_loader = None
            if val_data_path:
                val_loader = create_classification_dataloader(
                    data_path=val_data_path,
                    dataset_type=data_format,
                    batch_size=batch_size,
                    shuffle=False,
                    transform=val_transform,
                    preprocessing_hooks=preprocessing_hooks
                )
        
        else:
            # Legacy: use original StreetSceneDataset
            train_transform = get_train_transforms(image_size)
            val_transform = get_val_transforms(image_size)
            
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
    
    def _resolve_dataset_params(
        self,
        dataset_name: Optional[str],
        dataset_version: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolve dataset name/version from config if not provided."""
        resolved_name = dataset_name
        resolved_version = dataset_version
        
        if self.task_type == 'detection':
            dataset_cfg = (self.task_config or {}).get('dataset', {})
            if resolved_name is None:
                resolved_name = dataset_cfg.get('name', self.detection_task)
            if resolved_version is None:
                resolved_version = dataset_cfg.get('version')
        else:
            config_key = 'vehicle' if self.task_type == 'vehicle_classification' else 'human_attributes'
            dataset_cfg = self.config.get('classification', {}).get(config_key, {}).get('dataset', {})
            if resolved_name is None:
                resolved_name = dataset_cfg.get('name')
            if resolved_version is None:
                resolved_version = dataset_cfg.get('version')
        
        return resolved_name, resolved_version
    
    def train(
        self,
        train_data_path: Optional[str] = None,
        val_data_path: Optional[str] = None,
        annotation_file: Optional[str] = None,
        output_dir: str = "./outputs",
        resume_checkpoint: Optional[str] = None,
        data_yaml: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_version: Optional[str] = None,
        catalog_path: str = "data/datasets.yaml",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data
            annotation_file: Path to annotation file
            output_dir: Directory to save outputs
            resume_checkpoint: Path to checkpoint to resume from
            data_yaml: Path to YOLO dataset YAML file (for YOLO models)
            dataset_name: Name of dataset in catalog
            dataset_version: Version of dataset in catalog
            catalog_path: Path to catalog file
            **kwargs: Additional arguments for YOLO trainer
        
        Returns:
            Training metrics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export dataset manifest for experiment tracking if using catalog
        if dataset_name:
            catalog = get_catalog(catalog_path)
            dataset_info = catalog.get_dataset(dataset_name, dataset_version)
            manifest_path = os.path.join(output_dir, f"dataset_manifest_{dataset_name}_{dataset_info['version']}.yaml")
            catalog.export_dataset_manifest(dataset_name, dataset_info['version'], manifest_path)
            self.logger.info(f"Exported dataset manifest to {manifest_path}")
        
        # For YOLO models, use YOLO trainer directly
        if isinstance(self.trainer, YOLODetectionTrainer):
            return self.trainer.train(
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                annotation_file=annotation_file,
                data_yaml=data_yaml,
                output_dir=output_dir,
                resume=bool(resume_checkpoint),
                **kwargs
            )
        
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
        output_dir: str = "./outputs",
        data_yaml: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Args:
            test_data_path: Path to test data
            checkpoint_path: Path to model checkpoint
            annotation_file: Path to annotation file
            output_dir: Directory to save outputs
            data_yaml: Path to YOLO dataset YAML file (for YOLO models)
            **kwargs: Additional arguments for YOLO evaluator
        
        Returns:
            Evaluation metrics
        """
        # For YOLO models, use YOLO evaluator directly
        if isinstance(self.trainer, YOLODetectionTrainer):
            return self.trainer.evaluate(
                test_data_path=test_data_path,
                checkpoint_path=checkpoint_path,
                data_yaml=data_yaml,
                output_dir=output_dir,
                **kwargs
            )
        
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


class YOLODetectionTrainer(Trainer):
    """Trainer for YOLO detection models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        task_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(model, config, **kwargs)
        self.task_config = task_config or {}
    
    def train(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        annotation_file: Optional[str] = None,
        data_yaml: Optional[str] = None,
        output_dir: str = "./outputs",
        resume: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train YOLO model using ultralytics trainer.
        
        Args:
            train_data_path: Path to training data directory
            val_data_path: Path to validation data directory
            annotation_file: Path to annotation file (unused for YOLO)
            data_yaml: Path to YOLO dataset YAML file
            output_dir: Directory to save outputs
            resume: Whether to resume from last checkpoint
            **kwargs: Additional training arguments
        
        Returns:
            Training results
        """
        if not data_yaml:
            self.logger.warning("No data_yaml provided for YOLO training. Using train_data_path as source.")
            data_yaml = train_data_path
        
        # Get training configuration
        training_config = self.task_config.get('training', {}) if self.task_config else {}
        epochs = training_config.get('epochs', self.config.get('detection', {}).get('training', {}).get('epochs', 100))
        batch_size = training_config.get('batch_size', self.config.get('detection', {}).get('training', {}).get('batch_size', 16))
        optimizer = training_config.get('optimizer', self.config.get('detection', {}).get('training', {}).get('optimizer', 'SGD'))
        lr = training_config.get('learning_rate', self.config.get('detection', {}).get('training', {}).get('learning_rate', 0.001))
        
        # Train using YOLO adapter
        results = self.model.yolo.train(
            data_yaml=data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            lr0=lr,
            output_dir=output_dir,
            resume=resume,
            **kwargs
        )
        
        self.logger.info(f"YOLO training completed. Results saved to {output_dir}")
        return results
    
    def evaluate(
        self,
        test_data_path: str,
        checkpoint_path: Optional[str] = None,
        data_yaml: Optional[str] = None,
        output_dir: str = "./outputs",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate YOLO model using ultralytics evaluator.
        
        Args:
            test_data_path: Path to test data directory
            checkpoint_path: Path to model checkpoint (unused for YOLO)
            data_yaml: Path to YOLO dataset YAML file
            output_dir: Directory to save outputs
            **kwargs: Additional evaluation arguments
        
        Returns:
            Evaluation metrics
        """
        if not data_yaml:
            self.logger.warning("No data_yaml provided for YOLO evaluation. Using test_data_path as source.")
            data_yaml = test_data_path
        
        # Evaluate using YOLO adapter
        results = self.model.yolo.eval(
            data_yaml=data_yaml,
            output_dir=output_dir,
            **kwargs
        )
        
        self.logger.info(f"YOLO evaluation completed. Results saved to {output_dir}")
        return results
    
    def _create_criterion(self) -> nn.Module:
        """YOLO models manage their own loss functions."""
        return nn.Identity()


class DetectionTrainer(Trainer):
    """Trainer for legacy detection models."""
    
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