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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detection.models import create_detection_model
from classification.models import create_classification_model
from common.utils import load_config, get_device
from data.dataset import get_val_transforms


def load_model(config_path: str, task_type: str, checkpoint_path: str, device: str):
    """Load trained model."""
    config = load_config(config_path)
    
    if task_type == 'detection':
        model = create_detection_model(config)
    elif task_type in ['vehicle_classification', 'human_attributes']:
        classification_type = task_type.replace('_classification', '')
        model = create_classification_model(config, classification_type)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


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


def run_classification_inference(model, image_tensor, task_type):
    """Run classification inference."""
    with torch.no_grad():
        outputs = model(image_tensor)
    
    if task_type == 'human_attributes':
        # Multi-task outputs
        results = {}
        for task_name, output in outputs.items():
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            results[task_name] = {
                'predictions': probs.tolist(),
                'predicted_class': int(np.argmax(probs))
            }
        return results
    else:
        # Single task output
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        return {
            'predictions': probs.tolist(),
            'predicted_class': int(np.argmax(probs))
        }


def main():
    parser = argparse.ArgumentParser(description="Run inference with Street Scene models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--task", type=str, required=True,
                       choices=["detection", "vehicle_classification", "human_attributes"],
                       help="Task type")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detection")
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    
    # Load model
    model, config = load_model(args.config, args.task, args.checkpoint, device)
    
    # Get image size from config
    if args.task == 'detection':
        image_size = tuple(config['detection']['data']['image_size'])
    else:
        config_key = 'vehicle' if args.task == 'vehicle_classification' else 'human_attributes'
        image_size = tuple(config['classification'][config_key]['data']['image_size'])
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    else:
        raise ValueError(f"Invalid input path: {args.input}")
    
    # Run inference
    all_results = {}
    
    for image_file in image_files:
        print(f"Processing {image_file}...")
        
        # Preprocess
        image_tensor, original_image = preprocess_image(str(image_file), image_size, device)
        
        # Run inference
        if args.task == 'detection':
            results = run_detection_inference(model, image_tensor, config, args.confidence)
        else:
            results = run_classification_inference(model, image_tensor, args.task)
        
        all_results[str(image_file)] = results
    
    # Save results
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print("Results:")
        for image_path, results in all_results.items():
            print(f"{image_path}: {results}")


if __name__ == "__main__":
    main()