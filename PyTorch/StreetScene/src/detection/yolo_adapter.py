"""
YOLO model adapter for ultralytics YOLOv8/YOLO11 variants.

This module provides a unified interface to load, train, evaluate, and perform
inference/tracking with YOLOv8/YOLO11 models.
"""

import os
import torch
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLOAdapter:
    """Wrapper around ultralytics YOLO for Street Scene detection tasks."""

    def __init__(
        self,
        model_variant: str,
        task_name: str = "detect",
        num_classes: int = 80,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize YOLO adapter.

        Args:
            model_variant: YOLO model variant (e.g., 'yolov8n', 'yolov8s', 'yolov11m')
            task_name: Task type ('detect', 'track')
            num_classes: Number of classes for detection
            device: Device to use ('cuda', 'cpu', or specific GPU)
            **kwargs: Additional arguments for model configuration
        """
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Install with: pip install ultralytics")

        self.model_variant = model_variant
        self.task_name = task_name
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.extra_config = kwargs

        # Load YOLO model
        model_name = f"{model_variant}.pt"
        self.model = YOLO(model_name)
        self.model.to(self.device)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded YOLO variant: {model_variant} on device: {self.device}")

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch_size: int = 16,
        patience: int = 20,
        device: Optional[str] = None,
        optimizer: str = "SGD",
        lr0: float = 0.01,
        lrf: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        output_dir: str = "./runs",
        resume: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train YOLO model.

        Args:
            data_yaml: Path to YOLO dataset YAML file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Training batch size
            patience: Early stopping patience
            device: Device for training
            optimizer: Optimizer name ('SGD', 'Adam', etc.)
            lr0: Initial learning rate
            lrf: Final learning rate as fraction of initial
            momentum: Optimizer momentum
            weight_decay: Weight decay regularization
            output_dir: Directory for saving outputs
            resume: Whether to resume from last checkpoint
            **kwargs: Additional training arguments

        Returns:
            Dictionary with training results
        """
        device = device or self.device

        # Prepare training arguments
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'patience': patience,
            'device': device,
            'optimizer': optimizer,
            'lr0': lr0,
            'lrf': lrf,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'project': output_dir,
            'name': self.task_name,
            'exist_ok': True,
            'resume': resume,
            'save': True,
            'plots': True,
        }

        # Add any extra configuration
        train_args.update(self.extra_config)
        train_args.update(kwargs)

        self.logger.info(f"Starting training with config: {train_args}")

        # Train model
        results = self.model.train(**train_args)

        return {
            'results': results,
            'output_dir': output_dir,
            'model_variant': self.model_variant,
            'task_name': self.task_name
        }

    def eval(
        self,
        data_yaml: str,
        imgsz: int = 640,
        batch_size: int = 16,
        device: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate YOLO model.

        Args:
            data_yaml: Path to YOLO dataset YAML file
            imgsz: Image size for evaluation
            batch_size: Evaluation batch size
            device: Device for evaluation
            **kwargs: Additional evaluation arguments

        Returns:
            Dictionary with evaluation metrics
        """
        device = device or self.device

        eval_args = {
            'data': data_yaml,
            'imgsz': imgsz,
            'batch': batch_size,
            'device': device,
            'plots': True,
        }

        eval_args.update(self.extra_config)
        eval_args.update(kwargs)

        self.logger.info(f"Starting evaluation with config: {eval_args}")

        # Evaluate model
        results = self.model.val(**eval_args)

        return {
            'metrics': results.results_dict if hasattr(results, 'results_dict') else {},
            'model_variant': self.model_variant,
            'task_name': self.task_name
        }

    def predict(
        self,
        source,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None,
        save: bool = False,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run inference on images.

        Args:
            source: Input image(s) path or list of paths
            imgsz: Image size for inference
            conf: Confidence threshold
            iou: IOU threshold for NMS
            device: Device for inference
            save: Whether to save predictions
            save_dir: Directory to save predictions
            **kwargs: Additional prediction arguments

        Returns:
            List of prediction results
        """
        device = device or self.device

        predict_args = {
            'conf': conf,
            'iou': iou,
            'imgsz': imgsz,
            'device': device,
            'save': save,
            'save_dir': save_dir,
            'verbose': False,
        }

        predict_args.update(self.extra_config)
        predict_args.update(kwargs)

        self.logger.info(f"Running inference on {source}")

        # Run predictions
        results = self.model.predict(source, **predict_args)

        # Convert results to list of dicts
        predictions = []
        for result in results:
            pred_dict = {
                'image': result.path if hasattr(result, 'path') else '',
                'boxes': result.boxes.xyxy.cpu().numpy().tolist() if result.boxes else [],
                'confidences': result.boxes.conf.cpu().numpy().tolist() if result.boxes else [],
                'classes': result.boxes.cls.cpu().numpy().tolist() if result.boxes else [],
                'class_names': [result.names[int(cls)] for cls in result.boxes.cls] if result.boxes else []
            }
            predictions.append(pred_dict)

        return predictions

    def track(
        self,
        source,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None,
        tracker: str = "botsort.yaml",
        save: bool = False,
        save_dir: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run object tracking.

        Args:
            source: Input video or image(s) path
            imgsz: Image size for tracking
            conf: Confidence threshold
            iou: IOU threshold for NMS
            device: Device for tracking
            tracker: Tracker configuration (default: botsort.yaml)
            save: Whether to save tracking results
            save_dir: Directory to save results
            **kwargs: Additional tracking arguments

        Returns:
            List of tracking results
        """
        device = device or self.device

        track_args = {
            'conf': conf,
            'iou': iou,
            'imgsz': imgsz,
            'device': device,
            'tracker': tracker,
            'save': save,
            'save_dir': save_dir,
            'verbose': False,
            'persist': True,
        }

        track_args.update(self.extra_config)
        track_args.update(kwargs)

        self.logger.info(f"Running tracking on {source}")

        # Run tracking
        results = self.model.track(source, **track_args)

        # Convert results to list of dicts
        tracks = []
        for result in results:
            track_dict = {
                'frame': result.frame if hasattr(result, 'frame') else 0,
                'image': result.path if hasattr(result, 'path') else '',
                'boxes': result.boxes.xyxy.cpu().numpy().tolist() if result.boxes else [],
                'confidences': result.boxes.conf.cpu().numpy().tolist() if result.boxes else [],
                'classes': result.boxes.cls.cpu().numpy().tolist() if result.boxes else [],
                'track_ids': result.boxes.id.cpu().numpy().tolist() if (result.boxes and result.boxes.id is not None) else [],
                'class_names': [result.names[int(cls)] for cls in result.boxes.cls] if result.boxes else []
            }
            tracks.append(track_dict)

        return tracks

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.model = YOLO(checkpoint_path)
        self.model.to(self.device)

    def save_checkpoint(self, checkpoint_dir: str, name: str = "best") -> str:
        """
        Save model checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint
            name: Name for the checkpoint (default: 'best')

        Returns:
            Path to saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pt")
        
        # YOLO models are typically saved during training
        # This saves a reference to the best model
        if hasattr(self.model, 'trainer') and self.model.trainer.best:
            checkpoint_path = self.model.trainer.best
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'variant': self.model_variant,
            'task': self.task_name,
            'num_classes': self.num_classes,
            'device': self.device,
            'model_type': type(self.model).__name__
        }
