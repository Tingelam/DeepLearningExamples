"""
End-to-end pipelines for Street Scene optimization.
"""

import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from ..common.utils import setup_logging, load_config, seed_everything
from ..common.trainer import Trainer
from ..data.dataset import StreetSceneDataset, get_train_transforms, get_val_transforms, create_dataloader
from ..detection.models import create_detection_model
from ..detection.yolo_adapter import YOLOAdapter
from ..classification.models import (
    create_classification_model,
    resolve_classification_task_config,
)


class StreetScenePipeline:
    """End-to-end pipeline for street scene tasks."""
    
    def __init__(
        self,
        config_path: str,
        task_type: str,
        log_level: str = "INFO",
        detection_task: Optional[str] = None,
        classification_task: Optional[str] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            config_path: Path to configuration file
            task_type: Type of task ('detection' or 'classification')
            log_level: Logging level
            detection_task: Specific detection task name (e.g., 'vehicle_detection', 'pedestrian_detection')
            classification_task: Specific classification task name from the config catalog
        """
        self.config = load_config(config_path)
        self.task_type = task_type
        self.detection_task = detection_task
        self.classification_task = classification_task
        self.task_config = None
        self.logger = setup_logging(log_level)
        
        # Set random seeds
        seed_everything()
        
        # Load task-specific configuration when available
        if self.task_type == 'detection' and detection_task:
            self.task_config = self._load_detection_task_config(detection_task)
        elif self.task_type == 'classification':
            if not classification_task:
                raise ValueError("classification_task must be provided when task_type='classification'")
            self.task_config = self._load_classification_task_config(classification_task)
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize trainer
        self.trainer = self._create_trainer()
        
        suffix = ""
        if detection_task:
            suffix = f" ({detection_task})"
        elif classification_task:
            suffix = f" ({classification_task})"
        self.logger.info(f"Initialized {task_type}{suffix} pipeline")
    
    def _load_detection_task_config(self, task_name: str) -> None:
        """Load task-specific detection configuration."""
        if 'tasks' not in self.config.get('detection', {}):
            self.logger.warning(f"No tasks catalog found for detection task: {task_name}")
            return
        
        tasks = self.config['detection']['tasks']
        if task_name not in tasks:
            raise ValueError(f"Unknown detection task: {task_name}. Available tasks: {list(tasks.keys())}")
        
        task_config = tasks[task_name]
        self.logger.info(f"Loaded task configuration for: {task_name}")
        return task_config
    
    def _load_classification_task_config(self, task_name: str) -> Dict[str, Any]:
        """Load and resolve classification task configuration."""
        resolved = resolve_classification_task_config(self.config, task_name)
        self.logger.info(f"Loaded classification task configuration for: {task_name}")
        return resolved
    
    def _create_model(self) -> nn.Module:
        """Create model based on task type."""
        if self.task_type == 'detection':
            return create_detection_model(self.config, self.task_config)
        elif self.task_type == 'classification':
            return create_classification_model(
                self.config,
                self.classification_task,
                task_config=self.task_config,
            )
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
        elif self.task_type == 'classification':
            trainer_config = {
                'training': (self.task_config or {}).get('training', {}),
                'mixed_precision': self.config.get('mixed_precision', {}),
            }
            return ClassificationTrainer(
                self.model,
                trainer_config,
                self.task_config,
                self.classification_task,
            )
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
    def _get_classification_section(self, section: str) -> Dict[str, Any]:
        """Helper to fetch a classification section merged with defaults."""
        if self.task_type != 'classification' or not self.task_config:
            raise ValueError("Classification task configuration is not initialized")
        return self.task_config.get(section, {})
    
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
        elif self.task_type == 'classification':
            data_cfg = self._get_classification_section('data')
            image_size = tuple(data_cfg.get('image_size', (224, 224)))
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
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
        elif self.task_type == 'classification':
            training_cfg = self._get_classification_section('training')
            return training_cfg.get('batch_size', 32)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
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
                current_metric = self._select_monitor_metric(val_metrics_epoch)
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
        elif self.task_type == 'classification':
            training_cfg = self._get_classification_section('training')
            return training_cfg.get('epochs', 1)
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    
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
        elif self.task_type == 'classification':
            data_cfg = self._get_classification_section('data')
            return tuple(data_cfg.get('image_size', (224, 224)))
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _select_monitor_metric(self, metrics: Dict[str, float]) -> float:
        """Pick the validation metric to monitor for checkpointing."""
        if not metrics:
            return 0.0
        for key, value in metrics.items():
            if key.startswith('val_') and key != 'val_loss':
                return value
        return metrics.get('val_loss', 0.0)


class YOLODetectionTrainer(Trainer):

    
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
    """Trainer for classification models with configurable multi-head support."""
    
    def __init__(
        self,
        model: nn.Module,
        trainer_config: Dict[str, Any],
        task_config: Dict[str, Any],
        task_name: str,
        **kwargs,
    ):
        self.task_config = task_config or {}
        self.task_name = task_name
        self.head_configs = self.task_config.get('heads', {}) or {}
        self.head_loss_types: Dict[str, str] = {}
        self.head_loss_weights: Dict[str, float] = {}
        self.head_metric_map: Dict[str, List[str]] = {}
        self.metric_keys: List[str] = []
        super().__init__(model, trainer_config, **kwargs)
        self._initialize_metric_plan()
    
    def _initialize_metric_plan(self) -> None:
        """Prepare metric tracking plan per head."""
        metric_keys: List[str] = []
        for head_name, head_cfg in self.head_configs.items():
            metrics = head_cfg.get('metrics')
            if metrics is None:
                metrics = [head_cfg.get('metric', 'accuracy')]
            if isinstance(metrics, str):
                metrics = [metrics]
            normalized = [metric.lower() for metric in metrics if metric]
            if not normalized:
                normalized = ['accuracy']
            self.head_metric_map[head_name] = normalized
            for metric_name in normalized:
                metric_keys.append(f"{head_name}_{metric_name}")
        self.metric_keys = metric_keys
    
    def _create_criterion(self) -> nn.Module:
        """Create loss functions per head based on configuration."""
        if not self.head_configs:
            return nn.CrossEntropyLoss()
        criteria: Dict[str, nn.Module] = {}
        for head_name, head_cfg in self.head_configs.items():
            loss_type = head_cfg.get('loss', 'cross_entropy').lower()
            self.head_loss_types[head_name] = loss_type
            self.head_loss_weights[head_name] = float(head_cfg.get('loss_weight', 1.0))
            if loss_type == 'cross_entropy':
                weight = head_cfg.get('class_weights')
                weight_tensor = None
                if weight is not None:
                    weight_tensor = torch.tensor(weight, dtype=torch.float32, device=self.device)
                criteria[head_name] = nn.CrossEntropyLoss(weight=weight_tensor)
            elif loss_type in ('bce', 'bce_with_logits'):
                pos_weight = head_cfg.get('pos_weight')
                pos_weight_tensor = None
                if pos_weight is not None:
                    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=self.device)
                criteria[head_name] = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                raise ValueError(f"Unsupported loss '{loss_type}' for head '{head_name}'")
        return criteria
    
    def _move_targets_to_device(self, targets):
        if isinstance(targets, dict):
            return {k: v.to(self.device) for k, v in targets.items()}
        return targets.to(self.device)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch with per-head metrics."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        metric_totals = {key: 0.0 for key in self.metric_keys}
        total_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for data, targets in pbar:
            data = data.to(self.device)
            targets = self._move_targets_to_device(targets)
            self.optimizer.zero_grad()
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
            batch_size = data.size(0)
            total_samples += batch_size
            batch_metrics = self._calculate_metrics(outputs, targets)
            for key, value in batch_metrics.items():
                if key in metric_totals:
                    metric_totals[key] += value * batch_size
            pbar.set_postfix({'loss': loss.item()})
        avg_loss = total_loss / max(1, num_batches)
        metrics = {'train_loss': avg_loss}
        if total_samples > 0:
            for key in self.metric_keys:
                metrics[f"train_{key}"] = metric_totals.get(key, 0.0) / total_samples
        return metrics
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """Validate for one epoch with per-head metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        metric_totals = {key: 0.0 for key in self.metric_keys}
        total_samples = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for data, targets in pbar:
                data = data.to(self.device)
                targets = self._move_targets_to_device(targets)
                if self.use_amp:
                    with autocast():
                        outputs = self.model(data)
                        loss = self._compute_loss(outputs, targets)
                else:
                    outputs = self.model(data)
                    loss = self._compute_loss(outputs, targets)
                total_loss += loss.item()
                batch_size = data.size(0)
                total_samples += batch_size
                batch_metrics = self._calculate_metrics(outputs, targets)
                for key, value in batch_metrics.items():
                    if key in metric_totals:
                        metric_totals[key] += value * batch_size
                pbar.set_postfix({'val_loss': loss.item()})
        avg_loss = total_loss / max(1, num_batches)
        metrics = {'val_loss': avg_loss}
        if total_samples > 0:
            for key in self.metric_keys:
                metrics[f"val_{key}"] = metric_totals.get(key, 0.0) / total_samples
        return metrics
    
    def _compute_loss(self, outputs, targets) -> torch.Tensor:
        if not isinstance(self.criterion, dict):
            return self.criterion(outputs, targets)
        total_loss = 0.0
        output_dict = outputs if isinstance(outputs, dict) else {'logits': outputs}
        for head_name, criterion in self.criterion.items():
            if head_name not in output_dict:
                continue
            head_target = self._get_head_target(targets, head_name)
            loss_weight = self.head_loss_weights.get(head_name, 1.0)
            total_loss += criterion(output_dict[head_name], head_target) * loss_weight
        return total_loss
    
    def _get_head_target(self, targets, head_name: str):
        if isinstance(targets, dict):
            if head_name not in targets:
                raise KeyError(f"Missing target for head '{head_name}'")
            target = targets[head_name]
        else:
            target = targets
        loss_type = self.head_loss_types.get(head_name, 'cross_entropy')
        if loss_type in ('bce', 'bce_with_logits'):
            return target.float()
        return target
    
    def _calculate_metrics(self, outputs, targets) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        output_dict = outputs if isinstance(outputs, dict) else {'logits': outputs}
        for head_name, metric_names in self.head_metric_map.items():
            if head_name not in output_dict or head_name == 'features':
                continue
            head_target = self._get_head_target(targets, head_name)
            for metric_name in metric_names:
                metric_value = self._compute_metric(
                    metric_name,
                    output_dict[head_name],
                    head_target,
                    self.head_loss_types.get(head_name, 'cross_entropy'),
                )
                if metric_value is not None:
                    metrics[f"{head_name}_{metric_name}"] = metric_value
        return metrics
    
    def _compute_metric(
        self,
        metric_name: str,
        logits: torch.Tensor,
        target: torch.Tensor,
        loss_type: str,
    ) -> Optional[float]:
        metric = metric_name.lower()
        if metric == 'accuracy':
            if loss_type in ('bce', 'bce_with_logits'):
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct = (preds == target.float()).float().mean(dim=1)
                return correct.mean().item()
            else:
                if target.dim() > 1 and target.size(-1) == logits.size(-1):
                    target = torch.argmax(target, dim=1)
                preds = torch.argmax(logits, dim=1)
                return (preds == target.long()).float().mean().item()
        return None

