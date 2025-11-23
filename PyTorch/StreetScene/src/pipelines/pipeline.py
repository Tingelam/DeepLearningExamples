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
from ..detection.yolo_adapter import YOLOAdapter
from ..classification.models import create_classification_model
from ..evaluation.metrics import compute_detection_metrics, compute_tracking_metrics, compute_classification_metrics
from ..evaluation.reporting import MetricsReporter, generate_report


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
        resume_checkpoint: Optional[str] = None,
        data_yaml: Optional[str] = None,
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
            **kwargs: Additional arguments for YOLO trainer
        
        Returns:
            Training metrics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # For YOLO models, use YOLO trainer directly
        if isinstance(self.trainer, YOLODetectionTrainer):
            results = self.trainer.train(
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                annotation_file=annotation_file,
                data_yaml=data_yaml,
                output_dir=output_dir,
                resume=bool(resume_checkpoint),
                **kwargs
            )
            
            # Generate report for YOLO training
            self._generate_training_report(results, output_dir, data_yaml=data_yaml)
            
            return results
        
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
        
        results = {
            'train_metrics': train_metrics,
            'best_val_metric': best_val_metric,
            'output_dir': output_dir
        }
        
        # Generate report for non-YOLO training
        self._generate_training_report(results, output_dir, training_history=train_metrics)
        
        return results
    
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
            results = self.trainer.evaluate(
                test_data_path=test_data_path,
                checkpoint_path=checkpoint_path,
                data_yaml=data_yaml,
                output_dir=output_dir,
                **kwargs
            )
            
            # Generate evaluation report
            self._generate_evaluation_report(results, output_dir, checkpoint_path, data_yaml=data_yaml)
            
            return results
        
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
        
        # Generate evaluation report
        self._generate_evaluation_report(test_metrics, output_dir, checkpoint_path)
        
        return test_metrics
    
    def _get_image_size(self) -> tuple:
        """Get image size from config."""
        if self.task_type == 'detection':
            return tuple(self.config['detection']['data']['image_size'])
        else:
            config_key = 'vehicle' if self.task_type == 'vehicle_classification' else 'human_attributes'
            return tuple(self.config['classification'][config_key]['data']['image_size'])
    
    def _generate_training_report(
        self,
        results: Dict[str, Any],
        output_dir: str,
        data_yaml: Optional[str] = None,
        training_history: Optional[List[Dict[str, float]]] = None
    ) -> None:
        """Generate training report with metrics and metadata."""
        try:
            # Extract metrics from results
            if 'metrics' in results:
                metrics = results['metrics']
            elif 'train_metrics' in results:
                metrics = results['train_metrics'][-1] if results['train_metrics'] else {}
            else:
                metrics = {}
            
            # Compute metrics based on task type
            if self.task_type == 'detection':
                computed_metrics = compute_detection_metrics(results)
            else:
                computed_metrics = metrics
            
            # Prepare dataset info
            dataset_info = {}
            if data_yaml:
                dataset_info['data_yaml'] = data_yaml
            
            # Generate report
            reporter = MetricsReporter(output_dir, self.task_type, self.detection_task)
            reporter.generate_report(
                metrics=computed_metrics,
                config=self.config,
                dataset_info=dataset_info,
                training_history=training_history or results.get('train_metrics'),
                checkpoint_path=os.path.join(output_dir, 'best_model.pth')
            )
            
            self.logger.info(f"Training report generated in {output_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to generate training report: {e}")
    
    def _generate_evaluation_report(
        self,
        results: Dict[str, Any],
        output_dir: str,
        checkpoint_path: str,
        data_yaml: Optional[str] = None
    ) -> None:
        """Generate evaluation report with metrics and metadata."""
        try:
            # Compute metrics based on task type
            if self.task_type == 'detection':
                metrics = compute_detection_metrics(results)
            elif self.task_type == 'tracking':
                metrics = results if isinstance(results, dict) else {}
            else:
                metrics = results if isinstance(results, dict) else {}
            
            # Prepare dataset info
            dataset_info = {}
            if data_yaml:
                dataset_info['data_yaml'] = data_yaml
            
            # Generate report
            reporter = MetricsReporter(output_dir, self.task_type, self.detection_task)
            reporter.generate_report(
                metrics=metrics,
                config=self.config,
                dataset_info=dataset_info,
                checkpoint_path=checkpoint_path
            )
            
            self.logger.info(f"Evaluation report generated in {output_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to generate evaluation report: {e}")


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