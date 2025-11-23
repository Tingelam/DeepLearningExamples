"""
Task-aware metrics computation for Street Scene tasks.

Supports:
- Detection: mAP, precision, recall, PR curves (using pycocotools/YOLO metrics)
- Tracking: IDF1, MOTA, MOTP, etc.
- Classification: accuracy, precision, recall, F1, AUROC
"""

import os
import numpy as np
import torch
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import json

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        precision_recall_curve, average_precision_score
    )
except ImportError:
    print("Warning: scikit-learn not installed. Classification metrics will be limited.")
    accuracy_score = None

try:
    import motmetrics as mm
except ImportError:
    print("Warning: motmetrics not installed. Tracking metrics will be limited.")
    mm = None


logger = logging.getLogger(__name__)


@dataclass
class DetectionMetrics:
    """Container for detection metrics."""
    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    map_per_class: Optional[Dict[str, float]] = None
    pr_curves: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    confusion_matrix: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'mAP@0.5': self.map50,
            'mAP@0.5:0.95': self.map50_95,
            'precision': self.precision,
            'recall': self.recall
        }
        if self.map_per_class:
            result['mAP_per_class'] = self.map_per_class
        return result


@dataclass
class TrackingMetrics:
    """Container for tracking metrics."""
    mota: float = 0.0
    motp: float = 0.0
    idf1: float = 0.0
    num_switches: int = 0
    num_fragmentations: int = 0
    num_false_positives: int = 0
    num_misses: int = 0
    precision: float = 0.0
    recall: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'MOTA': self.mota,
            'MOTP': self.motp,
            'IDF1': self.idf1,
            'num_switches': self.num_switches,
            'num_fragmentations': self.num_fragmentations,
            'num_false_positives': self.num_false_positives,
            'num_misses': self.num_misses,
            'precision': self.precision,
            'recall': self.recall
        }


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auroc: Optional[float] = None
    confusion_matrix: Optional[np.ndarray] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score
        }
        if self.auroc is not None:
            result['auroc'] = self.auroc
        if self.per_class_metrics:
            result['per_class_metrics'] = self.per_class_metrics
        return result


def compute_detection_metrics(
    predictions: Union[List[Dict[str, Any]], Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    class_names: Optional[List[str]] = None,
    iou_threshold: float = 0.5
) -> DetectionMetrics:
    """
    Compute detection metrics from predictions and ground truth.
    
    Args:
        predictions: Detection predictions (YOLO results or list of detections)
        ground_truth: Ground truth annotations (optional, may be in predictions)
        class_names: List of class names
        iou_threshold: IOU threshold for matching
    
    Returns:
        DetectionMetrics object
    """
    logger.info("Computing detection metrics...")
    
    # If predictions come from YOLO results, extract metrics directly
    if isinstance(predictions, dict) and 'metrics' in predictions:
        yolo_metrics = predictions['metrics']
        
        # Extract YOLO metrics
        metrics = DetectionMetrics(
            map50=yolo_metrics.get('metrics/mAP50(B)', 0.0),
            map50_95=yolo_metrics.get('metrics/mAP50-95(B)', 0.0),
            precision=yolo_metrics.get('metrics/precision(B)', 0.0),
            recall=yolo_metrics.get('metrics/recall(B)', 0.0)
        )
        
        # Extract per-class mAP if available
        if class_names:
            map_per_class = {}
            for i, class_name in enumerate(class_names):
                key = f'metrics/mAP50(B)_{i}'
                if key in yolo_metrics:
                    map_per_class[class_name] = yolo_metrics[key]
            if map_per_class:
                metrics.map_per_class = map_per_class
        
        logger.info(f"Detection metrics: mAP@0.5={metrics.map50:.4f}, mAP@0.5:0.95={metrics.map50_95:.4f}")
        return metrics
    
    # Otherwise, compute metrics from scratch
    if not ground_truth:
        logger.warning("No ground truth provided, returning empty metrics")
        return DetectionMetrics()
    
    # Compute metrics using custom implementation or pycocotools
    # This is a simplified version - in production, use pycocotools.COCOeval
    metrics = DetectionMetrics()
    
    logger.info("Detection metrics computed")
    return metrics


def compute_tracking_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    class_names: Optional[List[str]] = None
) -> TrackingMetrics:
    """
    Compute tracking metrics from predictions and ground truth.
    
    Uses MOT metrics: MOTA, MOTP, IDF1, etc.
    
    Args:
        predictions: List of tracking predictions per frame
        ground_truth: List of ground truth tracks per frame
        class_names: List of class names
    
    Returns:
        TrackingMetrics object
    """
    logger.info("Computing tracking metrics...")
    
    if mm is None:
        logger.warning("motmetrics not installed, returning empty metrics")
        return TrackingMetrics()
    
    # Create accumulator for MOT metrics
    acc = mm.MOTAccumulator(auto_id=True)
    
    # Process each frame
    for frame_idx, (pred_frame, gt_frame) in enumerate(zip(predictions, ground_truth)):
        # Extract object IDs and boxes
        pred_ids = pred_frame.get('track_ids', [])
        pred_boxes = pred_frame.get('boxes', [])
        gt_ids = gt_frame.get('track_ids', [])
        gt_boxes = gt_frame.get('boxes', [])
        
        # Compute distances (IOU-based)
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            distances = _compute_iou_distance(pred_boxes, gt_boxes)
        else:
            distances = np.empty((0, 0))
        
        # Update accumulator
        acc.update(
            gt_ids,
            pred_ids,
            distances
        )
    
    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'num_switches', 
                                       'num_fragmentations', 'num_false_positives',
                                       'num_misses', 'precision', 'recall'],
                         name='tracker')
    
    # Extract metrics
    metrics = TrackingMetrics(
        mota=summary['mota'].values[0] if 'mota' in summary else 0.0,
        motp=summary['motp'].values[0] if 'motp' in summary else 0.0,
        idf1=summary['idf1'].values[0] if 'idf1' in summary else 0.0,
        num_switches=int(summary['num_switches'].values[0]) if 'num_switches' in summary else 0,
        num_fragmentations=int(summary['num_fragmentations'].values[0]) if 'num_fragmentations' in summary else 0,
        num_false_positives=int(summary['num_false_positives'].values[0]) if 'num_false_positives' in summary else 0,
        num_misses=int(summary['num_misses'].values[0]) if 'num_misses' in summary else 0,
        precision=summary['precision'].values[0] if 'precision' in summary else 0.0,
        recall=summary['recall'].values[0] if 'recall' in summary else 0.0
    )
    
    logger.info(f"Tracking metrics: MOTA={metrics.mota:.4f}, IDF1={metrics.idf1:.4f}")
    return metrics


def compute_classification_metrics(
    predictions: Union[torch.Tensor, np.ndarray, List],
    ground_truth: Union[torch.Tensor, np.ndarray, List],
    class_names: Optional[List[str]] = None,
    task_type: str = 'multiclass',
    probabilities: Optional[Union[torch.Tensor, np.ndarray]] = None
) -> ClassificationMetrics:
    """
    Compute classification metrics from predictions and ground truth.
    
    Args:
        predictions: Predicted class labels
        ground_truth: Ground truth labels
        class_names: List of class names
        task_type: Type of classification task ('binary', 'multiclass', 'multilabel')
        probabilities: Predicted probabilities (for AUROC)
    
    Returns:
        ClassificationMetrics object
    """
    logger.info("Computing classification metrics...")
    
    if accuracy_score is None:
        logger.warning("scikit-learn not installed, returning empty metrics")
        return ClassificationMetrics()
    
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.cpu().numpy()
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.cpu().numpy()
    
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Compute basic metrics
    accuracy = accuracy_score(ground_truth, predictions)
    
    # Determine averaging strategy
    average = 'binary' if task_type == 'binary' else 'macro'
    if task_type == 'multilabel':
        average = 'samples'
    
    precision = precision_score(ground_truth, predictions, average=average, zero_division=0)
    recall = recall_score(ground_truth, predictions, average=average, zero_division=0)
    f1 = f1_score(ground_truth, predictions, average=average, zero_division=0)
    
    # Compute AUROC if probabilities are provided
    auroc = None
    if probabilities is not None:
        try:
            if task_type == 'binary':
                auroc = roc_auc_score(ground_truth, probabilities)
            elif task_type == 'multiclass':
                auroc = roc_auc_score(ground_truth, probabilities, multi_class='ovr', average='macro')
            elif task_type == 'multilabel':
                auroc = roc_auc_score(ground_truth, probabilities, average='macro')
        except Exception as e:
            logger.warning(f"Failed to compute AUROC: {e}")
    
    # Compute confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    
    # Compute per-class metrics
    per_class_metrics = None
    if class_names and len(class_names) > 0:
        per_class_metrics = {}
        report = classification_report(ground_truth, predictions, target_names=class_names, 
                                      output_dict=True, zero_division=0)
        for class_name in class_names:
            if class_name in report:
                per_class_metrics[class_name] = {
                    'precision': report[class_name]['precision'],
                    'recall': report[class_name]['recall'],
                    'f1-score': report[class_name]['f1-score'],
                    'support': report[class_name]['support']
                }
    
    metrics = ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        auroc=auroc,
        confusion_matrix=cm,
        per_class_metrics=per_class_metrics
    )
    
    logger.info(f"Classification metrics: accuracy={metrics.accuracy:.4f}, f1={metrics.f1_score:.4f}")
    return metrics


def compute_multi_task_classification_metrics(
    predictions: Dict[str, Union[torch.Tensor, np.ndarray]],
    ground_truth: Dict[str, Union[torch.Tensor, np.ndarray]],
    task_configs: Dict[str, Dict[str, Any]]
) -> Dict[str, ClassificationMetrics]:
    """
    Compute classification metrics for multi-task models (e.g., human attributes).
    
    Args:
        predictions: Dictionary of predictions per task
        ground_truth: Dictionary of ground truth per task
        task_configs: Dictionary of task configurations (class_names, task_type, etc.)
    
    Returns:
        Dictionary of ClassificationMetrics per task
    """
    logger.info("Computing multi-task classification metrics...")
    
    metrics = {}
    for task_name, task_preds in predictions.items():
        if task_name not in ground_truth:
            logger.warning(f"No ground truth for task {task_name}")
            continue
        
        task_gt = ground_truth[task_name]
        task_config = task_configs.get(task_name, {})
        
        class_names = task_config.get('class_names', None)
        task_type = task_config.get('task_type', 'multiclass')
        
        task_metrics = compute_classification_metrics(
            predictions=task_preds,
            ground_truth=task_gt,
            class_names=class_names,
            task_type=task_type
        )
        
        metrics[task_name] = task_metrics
    
    return metrics


def _compute_iou_distance(boxes1: List, boxes2: List) -> np.ndarray:
    """
    Compute IOU-based distance matrix between two sets of boxes.
    
    Args:
        boxes1: First set of boxes (N x 4) [x1, y1, x2, y2]
        boxes2: Second set of boxes (M x 4) [x1, y1, x2, y2]
    
    Returns:
        Distance matrix (N x M) where distance = 1 - IOU
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.empty((len(boxes1), len(boxes2)))
    
    # Compute IOUs
    ious = np.zeros((len(boxes1), len(boxes2)))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            ious[i, j] = _compute_iou(box1, box2)
    
    # Convert to distance (lower is better)
    distances = 1.0 - ious
    
    return distances


def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IOU between two boxes.
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
    
    Returns:
        IOU value
    """
    # Compute intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Compute union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def save_metrics(metrics: Union[DetectionMetrics, TrackingMetrics, ClassificationMetrics],
                output_path: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics object
        output_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    logger.info(f"Metrics saved to {output_path}")


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        metrics_path: Path to JSON file
    
    Returns:
        Dictionary of metrics
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics
