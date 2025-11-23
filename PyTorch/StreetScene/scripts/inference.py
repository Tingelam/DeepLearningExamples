#!/usr/bin/env python3
"""
Inference script for Street Scene models.
"""

import argparse
import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection.models import create_detection_model
from classification.models import create_classification_model, resolve_classification_task_config
from common.utils import load_config, get_device
from data.dataset import get_val_transforms


def load_model(
    config_path: str,
    task_type: str,
    checkpoint_path: str,
    device: str,
    detection_task: Optional[str] = None,
    classification_task: Optional[str] = None,
):
    """Load trained model and return associated configuration."""
    config = load_config(config_path)
    task_config: Optional[Dict[str, Any]] = None
    
    if task_type == 'detection':
        detection_cfg = config.get('detection', {})
        if detection_task:
            tasks = detection_cfg.get('tasks', {})
            if detection_task not in tasks:
                raise ValueError(f"Unknown detection task: {detection_task}")
            task_config = tasks[detection_task]
        model = create_detection_model(config, task_config)
    elif task_type == 'classification':
        if not classification_task:
            raise ValueError("classification_task must be provided for classification inference")
        task_config = resolve_classification_task_config(config, classification_task)
        model = create_classification_model(
            config,
            classification_task,
            task_config=task_config,
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config, task_config


def preprocess_image(image_path: str, image_size: tuple, device: str):
    """Preprocess single image for inference."""
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transform = get_val_transforms(image_size)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    return image_tensor, image


def run_detection_inference(model, image_tensor, config, confidence_threshold=0.5):
    """Run detection inference."""
    with torch.no_grad():
        outputs = model(image_tensor)
    
    # Process detection outputs (simplified)
    loc_preds = outputs['loc_preds'].cpu().numpy()
    conf_preds = outputs['conf_preds'].cpu().numpy()
    
    # Convert to detections (this is simplified - real implementation would need NMS, etc.)
    detections = []
    batch_size = loc_preds.shape[0]
    
    for i in range(batch_size):
        # Get confidence scores
        max_conf = np.max(conf_preds[i, :, 1:], axis=1)  # Skip background class
        valid_indices = max_conf > confidence_threshold
        
        if np.any(valid_indices):
            # Get boxes and scores for valid detections
            boxes = loc_preds[i][valid_indices]
            scores = max_conf[valid_indices]
            classes = np.argmax(conf_preds[i][valid_indices, 1:], axis=1) + 1
            
            for box, score, cls in zip(boxes, scores, classes):
                detections.append({
                    'box': box.tolist(),
                    'score': float(score),
                    'class': int(cls)
                })
    
    return detections


def run_classification_inference(
    model,
    image_tensor,
    head_configs: Dict[str, Any],
):
    """Run classification inference and return per-head predictions."""
    with torch.no_grad():
        outputs = model(image_tensor)
    
    output_dict = outputs if isinstance(outputs, dict) else {'logits': outputs}
    results = {}
    for head_name, logits in output_dict.items():
        if head_name == 'features':
            continue
        head_cfg = head_configs.get(head_name, {}) if head_configs else {}
        loss_type = head_cfg.get('loss', 'cross_entropy').lower()
        if loss_type in ('bce', 'bce_with_logits'):
            probs = torch.sigmoid(logits).cpu().numpy()[0].tolist()
            preds = [int(score > 0.5) for score in probs]
        else:
            probs_tensor = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs = probs_tensor.tolist()
            preds = int(np.argmax(probs_tensor))
        results[head_name] = {
            'predictions': probs,
            'predicted_class': preds
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with Street Scene models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--task", type=str, required=True,
                       choices=["detection", "classification"],
                       help="Task type")
    parser.add_argument("--detection-task", type=str,
                       help="Specific detection task (e.g., vehicle_detection, pedestrian_detection)")
    parser.add_argument("--classification-task", type=str,
                       help="Classification task name from the configuration catalog")
    parser.add_argument("--mode", type=str, default="predict",
                       choices=["predict", "track"],
                       help="Inference mode (predict or track)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input image, video or directory")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--output-dir", type=str, help="Output directory for predictions/tracks")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detection")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker configuration for tracking mode")
    
    args = parser.parse_args()
    
    if args.task == 'classification' and not args.classification_task:
        parser.error("--classification-task is required when --task classification")
    
    # Setup
    device = get_device()
    
    # Create output directory if specified
    output_dir = args.output_dir
    if args.task == 'detection' and args.detection_task:
        if not output_dir:
            output_dir = f"./outputs/{args.detection_task}"
        else:
            output_dir = os.path.join(output_dir, args.detection_task)
    elif args.task == 'classification' and args.classification_task:
        if not output_dir:
            output_dir = f"./outputs/{args.classification_task}"
        else:
            output_dir = os.path.join(output_dir, args.classification_task)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, config, task_config = load_model(
        args.config,
        args.task,
        args.checkpoint,
        device,
        detection_task=args.detection_task,
        classification_task=args.classification_task,
    )
    
    # Get image size from config
    if args.task == 'detection':
        image_size = tuple(config['detection']['data']['image_size'])
        is_yolo = hasattr(model, 'yolo')
    else:
        data_cfg = (task_config or {}).get('data', {})
        image_size = tuple(data_cfg.get('image_size', (224, 224)))
        is_yolo = False
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.mp4"))
    else:
        raise ValueError(f"Invalid input path: {args.input}")
    
    # Run inference
    all_results = {}
    
    # For YOLO models with tracking
    if is_yolo and args.task == 'detection' and args.mode == 'track':
        print(f"Running YOLO tracking on {args.input}...")
        save_dir = os.path.join(output_dir, 'tracks') if output_dir else None
        track_results = model.yolo.track(
            source=args.input,
            conf=args.confidence,
            tracker=args.tracker,
            save_dir=save_dir,
            save=output_dir is not None
        )
        
        all_results['tracking'] = [
            {
                'frame': r.get('frame', 0),
                'image': r.get('image', ''),
                'detections': {
                    'boxes': r.get('boxes', []),
                    'confidences': r.get('confidences', []),
                    'classes': r.get('classes', []),
                    'track_ids': r.get('track_ids', []),
                    'class_names': r.get('class_names', [])
                }
            }
            for r in track_results
        ]
    # For YOLO models with prediction
    elif is_yolo and args.task == 'detection' and args.mode == 'predict':
        print(f"Running YOLO prediction on {args.input}...")
        save_dir = os.path.join(output_dir, 'predictions') if output_dir else None
        pred_results = model.yolo.predict(
            source=args.input,
            conf=args.confidence,
            save_dir=save_dir,
            save=output_dir is not None
        )
        
        all_results['predictions'] = [
            {
                'image': r.get('image', ''),
                'detections': {
                    'boxes': r.get('boxes', []),
                    'confidences': r.get('confidences', []),
                    'classes': r.get('classes', []),
                    'class_names': r.get('class_names', [])
                }
            }
            for r in pred_results
        ]
    # For legacy detection models
    elif args.task == 'detection':
        for image_file in image_files:
            print(f"Processing {image_file}...")
            image_tensor, original_image = preprocess_image(str(image_file), image_size, device)
            results = run_detection_inference(model, image_tensor, config, args.confidence)
            all_results[str(image_file)] = results
    # For classification models
    else:
        head_configs = (task_config or {}).get('heads', {})
        for image_file in image_files:
            print(f"Processing {image_file}...")
            image_tensor, _ = preprocess_image(str(image_file), image_size, device)
            results = run_classification_inference(model, image_tensor, head_configs)
            all_results[str(image_file)] = results
    
    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")
    elif output_dir:
        import json
        result_file = os.path.join(output_dir, f"results_{args.mode}.json")
        with open(result_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {result_file}")
    else:
        print("Results:")
        for image_path, results in all_results.items():
            print(f"{image_path}: {results}")


if __name__ == "__main__":
    main()