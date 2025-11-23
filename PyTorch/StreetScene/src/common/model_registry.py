"""
Model registry for tracking trained models and their metadata.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class ModelRegistry:
    """Registry for tracking trained models and experiments."""
    
    def __init__(self, registry_path: str = "./registry/model_registry.json"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to the registry JSON file
        """
        self.registry_path = registry_path
        self.logger = logging.getLogger(__name__)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        # Load or initialize registry
        self.entries = self._load_registry()
    
    def _load_registry(self) -> List[Dict[str, Any]]:
        """Load existing registry or initialize empty list."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not load registry: {e}. Starting with empty registry.")
                return []
        else:
            return []
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.entries, f, indent=2, default=str)
        self.logger.info(f"Saved registry with {len(self.entries)} entries")
    
    def register_model(
        self,
        run_id: str,
        task: str,
        config_hash: str,
        checkpoint_path: str,
        metrics: Dict[str, Any],
        dataset_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        **extra_fields
    ) -> Dict[str, Any]:
        """
        Register a trained model in the registry.
        
        Args:
            run_id: Unique run identifier
            task: Task type (e.g., 'detection', 'vehicle_classification')
            config_hash: Hash of the configuration
            checkpoint_path: Path to the model checkpoint file
            metrics: Dictionary of training metrics
            dataset_version: Version identifier for the dataset
            metadata: Framework and system metadata
            config: Full configuration dictionary
            **extra_fields: Additional fields to store
            
        Returns:
            The registered entry dictionary
        """
        entry = {
            'run_id': run_id,
            'task': task,
            'config_hash': config_hash,
            'checkpoint_path': os.path.abspath(checkpoint_path),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'dataset_version': dataset_version or 'unknown',
        }
        
        # Add optional fields
        if metadata:
            entry['metadata'] = metadata
        if config:
            entry['config'] = config
        
        # Add extra fields
        entry.update(extra_fields)
        
        # Check for duplicate run_id
        existing = self.find_entry_by_run_id(run_id)
        if existing:
            self.logger.info(f"Updating existing entry for run_id: {run_id}")
            # Update existing entry
            for i, e in enumerate(self.entries):
                if e.get('run_id') == run_id:
                    self.entries[i] = entry
                    break
        else:
            self.entries.append(entry)
        
        self._save_registry()
        self.logger.info(f"Registered model: {run_id}")
        
        return entry
    
    def find_entry_by_run_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Find registry entry by run ID."""
        for entry in self.entries:
            if entry.get('run_id') == run_id:
                return entry
        return None
    
    def find_entries_by_task(self, task: str) -> List[Dict[str, Any]]:
        """Find all entries for a specific task."""
        return [e for e in self.entries if e.get('task') == task]
    
    def find_entries_by_config_hash(self, config_hash: str) -> List[Dict[str, Any]]:
        """Find all entries with a specific config hash."""
        return [e for e in self.entries if e.get('config_hash') == config_hash]
    
    def find_best_entry_by_metric(
        self,
        task: str,
        metric_name: str,
        mode: str = 'max'
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best entry for a task based on a specific metric.
        
        Args:
            task: Task type
            metric_name: Name of the metric to evaluate
            mode: 'max' to maximize metric, 'min' to minimize
            
        Returns:
            Entry with best metric value, or None if no entries found
        """
        entries = self.find_entries_by_task(task)
        if not entries:
            return None
        
        def get_metric(entry):
            metrics = entry.get('metrics', {})
            if isinstance(metrics, dict):
                return metrics.get(metric_name, float('-inf') if mode == 'max' else float('inf'))
            return float('-inf') if mode == 'max' else float('inf')
        
        if mode == 'max':
            return max(entries, key=get_metric)
        else:
            return min(entries, key=get_metric)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the registry."""
        if not self.entries:
            return {'total_entries': 0, 'tasks': {}}
        
        summary = {
            'total_entries': len(self.entries),
            'tasks': {}
        }
        
        # Group by task
        for entry in self.entries:
            task = entry.get('task', 'unknown')
            if task not in summary['tasks']:
                summary['tasks'][task] = {'count': 0, 'entries': []}
            summary['tasks'][task]['count'] += 1
            summary['tasks'][task]['entries'].append(entry.get('run_id'))
        
        return summary
    
    def list_all_entries(self) -> List[Dict[str, Any]]:
        """Get all registry entries."""
        return self.entries
    
    def delete_entry(self, run_id: str) -> bool:
        """Delete an entry from the registry."""
        initial_count = len(self.entries)
        self.entries = [e for e in self.entries if e.get('run_id') != run_id]
        
        if len(self.entries) < initial_count:
            self._save_registry()
            self.logger.info(f"Deleted entry: {run_id}")
            return True
        return False
    
    def export_to_csv(self, output_path: str) -> None:
        """
        Export registry to CSV file.
        
        Args:
            output_path: Path to save CSV file
        """
        import csv
        
        if not self.entries:
            self.logger.warning("Registry is empty, nothing to export")
            return
        
        # Flatten entries for CSV export
        rows = []
        for entry in self.entries:
            row = {
                'run_id': entry.get('run_id'),
                'task': entry.get('task'),
                'config_hash': entry.get('config_hash'),
                'checkpoint_path': entry.get('checkpoint_path'),
                'timestamp': entry.get('timestamp'),
                'dataset_version': entry.get('dataset_version'),
            }
            
            # Flatten metrics
            metrics = entry.get('metrics', {})
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    row[f'metric_{metric_name}'] = metric_value
            
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            
            self.logger.info(f"Exported registry to {output_path}")
