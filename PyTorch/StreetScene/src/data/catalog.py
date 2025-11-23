"""
Dataset catalog for managing versioned datasets with metadata.
"""

import os
import yaml
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging


class DatasetCatalog:
    """
    Manages dataset registration, versioning, and metadata.
    
    The catalog stores dataset information including:
    - Version tags and provenance hashes
    - Label schemas (class names, attribute definitions)
    - Split manifests (train/val/test paths and statistics)
    - Preprocessing configurations
    """
    
    def __init__(self, catalog_path: str = "data/datasets.yaml"):
        """
        Initialize dataset catalog.
        
        Args:
            catalog_path: Path to catalog YAML file
        """
        self.catalog_path = catalog_path
        self.logger = logging.getLogger(__name__)
        self.datasets = self._load_catalog()
    
    def _load_catalog(self) -> Dict[str, Any]:
        """Load catalog from file."""
        if not os.path.exists(self.catalog_path):
            self.logger.info(f"Catalog not found at {self.catalog_path}. Creating new catalog.")
            return {}
        
        try:
            with open(self.catalog_path, 'r') as f:
                data = yaml.safe_load(f)
                return data or {}
        except Exception as e:
            self.logger.error(f"Error loading catalog: {e}")
            return {}
    
    def _save_catalog(self) -> None:
        """Save catalog to file."""
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(self.catalog_path) or '.', exist_ok=True)
        
        try:
            with open(self.catalog_path, 'w') as f:
                yaml.dump(self.datasets, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"Catalog saved to {self.catalog_path}")
        except Exception as e:
            self.logger.error(f"Error saving catalog: {e}")
            raise
    
    def register_dataset(
        self,
        name: str,
        version: str,
        task_type: str,
        data_format: str,
        splits: Dict[str, Dict[str, Any]],
        label_schema: Dict[str, Any],
        preprocessing_config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a dataset in the catalog.
        
        Args:
            name: Dataset name (e.g., 'vehicle_detection')
            version: Version tag (e.g., 'v1', 'v1.0.0')
            task_type: Task type ('detection', 'classification', 'tracking')
            data_format: Data format ('yolo', 'coco', 'image_folder', 'csv')
            splits: Dictionary with split information
                   e.g., {'train': {'path': '...', 'num_samples': 1000}, 'val': {...}}
            label_schema: Label information
                         e.g., {'classes': ['car', 'truck'], 'num_classes': 2}
                         or {'attributes': {'gender': ['male', 'female']}}
            preprocessing_config: Preprocessing configuration used
            metadata: Additional metadata (description, source, etc.)
            provenance: Provenance information (source_hash, creation_date, etc.)
        """
        # Generate unique dataset key
        dataset_key = f"{name}_{version}"
        
        # Calculate provenance hash
        if provenance is None:
            provenance = {}
        
        provenance_hash = self._compute_provenance_hash(splits, label_schema, preprocessing_config)
        provenance['hash'] = provenance_hash
        provenance['registration_date'] = datetime.now().isoformat()
        
        # Create dataset entry
        dataset_entry = {
            'name': name,
            'version': version,
            'task_type': task_type,
            'data_format': data_format,
            'splits': splits,
            'label_schema': label_schema,
            'preprocessing_config': preprocessing_config or {},
            'metadata': metadata or {},
            'provenance': provenance
        }
        
        # Add to catalog
        if name not in self.datasets:
            self.datasets[name] = {}
        
        self.datasets[name][version] = dataset_entry
        
        # Save catalog
        self._save_catalog()
        
        self.logger.info(f"Registered dataset: {name} version {version}")
    
    def _compute_provenance_hash(
        self,
        splits: Dict[str, Any],
        label_schema: Dict[str, Any],
        preprocessing_config: Optional[Dict[str, Any]]
    ) -> str:
        """
        Compute provenance hash for reproducibility.
        
        Args:
            splits: Split information
            label_schema: Label schema
            preprocessing_config: Preprocessing configuration
            
        Returns:
            SHA256 hash string
        """
        # Create deterministic representation
        data = {
            'splits': splits,
            'label_schema': label_schema,
            'preprocessing_config': preprocessing_config or {}
        }
        
        # Sort keys for deterministic ordering
        json_str = json.dumps(data, sort_keys=True)
        
        # Compute hash
        hash_obj = hashlib.sha256(json_str.encode())
        return hash_obj.hexdigest()[:16]  # Use first 16 characters
    
    def get_dataset(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get dataset information from catalog.
        
        Args:
            name: Dataset name
            version: Version tag (if None, returns latest version)
            
        Returns:
            Dataset information dictionary
            
        Raises:
            ValueError: If dataset not found
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found in catalog. Available datasets: {list(self.datasets.keys())}")
        
        versions = self.datasets[name]
        
        if version is None:
            # Get latest version (assumes semantic versioning or chronological ordering)
            version = max(versions.keys())
            self.logger.info(f"No version specified. Using latest version: {version}")
        
        if version not in versions:
            raise ValueError(f"Version '{version}' not found for dataset '{name}'. Available versions: {list(versions.keys())}")
        
        return versions[version]
    
    def list_datasets(self, task_type: Optional[str] = None) -> List[Dict[str, str]]:
        """
        List all datasets in the catalog.
        
        Args:
            task_type: Filter by task type (optional)
            
        Returns:
            List of dataset information dictionaries
        """
        results = []
        
        for name, versions in self.datasets.items():
            for version, info in versions.items():
                if task_type is None or info.get('task_type') == task_type:
                    results.append({
                        'name': name,
                        'version': version,
                        'task_type': info.get('task_type'),
                        'data_format': info.get('data_format'),
                        'num_classes': info.get('label_schema', {}).get('num_classes'),
                        'provenance_hash': info.get('provenance', {}).get('hash')
                    })
        
        return results
    
    def get_dataset_path(self, name: str, version: Optional[str] = None, split: str = 'train') -> str:
        """
        Get path to dataset split.
        
        Args:
            name: Dataset name
            version: Version tag
            split: Split name ('train', 'val', 'test')
            
        Returns:
            Path to dataset split
        """
        dataset = self.get_dataset(name, version)
        
        if split not in dataset['splits']:
            raise ValueError(f"Split '{split}' not found for dataset '{name}'. Available splits: {list(dataset['splits'].keys())}")
        
        return dataset['splits'][split]['path']
    
    def get_label_schema(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get label schema for a dataset.
        
        Args:
            name: Dataset name
            version: Version tag
            
        Returns:
            Label schema dictionary
        """
        dataset = self.get_dataset(name, version)
        return dataset['label_schema']
    
    def update_dataset_metadata(
        self,
        name: str,
        version: str,
        metadata_updates: Dict[str, Any]
    ) -> None:
        """
        Update metadata for an existing dataset.
        
        Args:
            name: Dataset name
            version: Version tag
            metadata_updates: Dictionary of metadata updates
        """
        if name not in self.datasets or version not in self.datasets[name]:
            raise ValueError(f"Dataset '{name}' version '{version}' not found in catalog")
        
        # Update metadata
        self.datasets[name][version]['metadata'].update(metadata_updates)
        self.datasets[name][version]['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Save catalog
        self._save_catalog()
        
        self.logger.info(f"Updated metadata for dataset: {name} version {version}")
    
    def export_dataset_manifest(self, name: str, version: str, output_path: str) -> None:
        """
        Export dataset manifest to a file for experiment tracking.
        
        Args:
            name: Dataset name
            version: Version tag
            output_path: Output file path
        """
        dataset = self.get_dataset(name, version)
        
        # Create manifest with essential information
        manifest = {
            'dataset_name': name,
            'version': version,
            'task_type': dataset['task_type'],
            'data_format': dataset['data_format'],
            'label_schema': dataset['label_schema'],
            'splits': dataset['splits'],
            'provenance': dataset['provenance'],
            'exported_at': datetime.now().isoformat()
        }
        
        # Save manifest
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
        
        self.logger.info(f"Exported dataset manifest to {output_path}")


def get_catalog(catalog_path: str = "data/datasets.yaml") -> DatasetCatalog:
    """
    Get or create dataset catalog instance.
    
    Args:
        catalog_path: Path to catalog file
        
    Returns:
        DatasetCatalog instance
    """
    return DatasetCatalog(catalog_path)
