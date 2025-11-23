"""
Configuration management utilities for reproducible experiments.
Handles loading configs, applying overrides, managing run directories, and seeding.
"""

import os
import yaml
import hashlib
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class ConfigManager:
    """Manager for experiment configuration and reproducibility."""
    
    def __init__(
        self,
        config_path: str,
        overrides: Optional[List[str]] = None,
        run_name: Optional[str] = None,
        output_dir: str = "./outputs",
        seed: int = 42
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to base configuration YAML file
            overrides: List of config overrides in format 'key=value' or 'key.subkey=value'
            run_name: Name for this experimental run (used in run_id)
            output_dir: Base output directory for runs
            seed: Random seed for reproducibility
        """
        self.config_path = config_path
        self.base_config = self._load_config(config_path)
        self.overrides = overrides or []
        self.run_name = run_name
        self.output_dir = output_dir
        self.seed = seed
        self.logger = logging.getLogger(__name__)
        
        # Apply overrides
        self.resolved_config = self._apply_overrides(self.base_config, self.overrides)
        
        # Generate run ID and directory
        self.run_id = self._generate_run_id()
        self.run_dir = os.path.join(output_dir, self.run_id)
        
        # Calculate config hash
        self.config_hash = self._compute_config_hash(self.resolved_config)
        
        # Capture metadata
        self.metadata = self._capture_metadata()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _apply_overrides(self, config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
        """
        Apply CLI overrides to configuration.
        
        Supports nested key access via dot notation (e.g., 'detection.training.epochs=100')
        """
        import copy
        config = copy.deepcopy(config)
        
        for override in overrides:
            if '=' not in override:
                self.logger.warning(f"Invalid override format: {override}. Expected 'key=value'")
                continue
            
            key_path, value = override.split('=', 1)
            keys = key_path.split('.')
            
            # Navigate to the target dictionary and set value
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Try to parse value as JSON, otherwise treat as string
            try:
                current[keys[-1]] = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                current[keys[-1]] = value
            
            self.logger.info(f"Applied override: {key_path}={value}")
        
        return config
    
    def _generate_run_id(self) -> str:
        """
        Generate unique run ID combining timestamp and run name.
        Format: <timestamp>_<run_name> or <timestamp> if no run_name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.run_name:
            return f"{timestamp}_{self.run_name}"
        else:
            return timestamp
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Compute SHA256 hash of configuration for reproducibility tracking.
        """
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _capture_metadata(self) -> Dict[str, Any]:
        """
        Capture framework versions and system metadata.
        """
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_id,
            'config_hash': self.config_hash,
            'seed': self.seed,
            'config_file': os.path.abspath(self.config_path),
            'overrides': self.overrides,
        }
        
        # Capture framework versions
        try:
            import torch
            metadata['pytorch_version'] = torch.__version__
        except ImportError:
            pass
        
        try:
            import ultralytics
            metadata['ultralytics_version'] = ultralytics.__version__
        except ImportError:
            pass
        
        try:
            import timm
            metadata['timm_version'] = timm.__version__
        except ImportError:
            pass
        
        # Capture git commit information
        try:
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                cwd=os.path.dirname(self.config_path),
                stderr=subprocess.DEVNULL
            ).decode().strip()
            metadata['git_commit'] = commit_hash
        except (subprocess.CalledProcessError, FileNotFoundError):
            metadata['git_commit'] = 'unknown'
        
        return metadata
    
    def get_config(self) -> Dict[str, Any]:
        """Get the resolved configuration."""
        return self.resolved_config
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get captured metadata."""
        return self.metadata
    
    def get_run_dir(self) -> str:
        """Get the run directory path."""
        return self.run_dir
    
    def get_run_id(self) -> str:
        """Get the run ID."""
        return self.run_id
    
    def get_config_hash(self) -> str:
        """Get the config hash."""
        return self.config_hash
    
    def save_config(self) -> str:
        """
        Save the resolved configuration to the run directory.
        
        Returns:
            Path to saved config file
        """
        os.makedirs(self.run_dir, exist_ok=True)
        config_path = os.path.join(self.run_dir, 'config.yaml')
        
        with open(config_path, 'w') as f:
            yaml.dump(self.resolved_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Saved resolved config to {config_path}")
        return config_path
    
    def save_metadata(self) -> str:
        """
        Save metadata to the run directory.
        
        Returns:
            Path to saved metadata file
        """
        os.makedirs(self.run_dir, exist_ok=True)
        metadata_path = os.path.join(self.run_dir, 'metadata.yaml')
        
        with open(metadata_path, 'w') as f:
            yaml.dump(self.metadata, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Saved metadata to {metadata_path}")
        return metadata_path
    
    def save_all(self) -> Dict[str, str]:
        """
        Save both config and metadata to run directory.
        
        Returns:
            Dictionary with 'config' and 'metadata' keys pointing to saved files
        """
        return {
            'config': self.save_config(),
            'metadata': self.save_metadata(),
            'run_dir': self.run_dir
        }
    
    @staticmethod
    def load_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """
        Load configuration saved with a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Configuration dictionary embedded in checkpoint
        """
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            return checkpoint['config']
        else:
            raise KeyError(f"No 'config' found in checkpoint: {checkpoint_path}")
    
    @staticmethod
    def load_metadata_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """
        Load metadata saved with a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Metadata dictionary embedded in checkpoint
        """
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'metadata' in checkpoint:
            return checkpoint['metadata']
        else:
            return {}
    
    @staticmethod
    def verify_config_hash(checkpoint_path: str, expected_hash: Optional[str] = None) -> bool:
        """
        Verify that config hash in checkpoint matches expected value.
        
        Args:
            checkpoint_path: Path to checkpoint file
            expected_hash: Expected config hash (if None, returns hash from checkpoint)
            
        Returns:
            True if hashes match or no expected_hash provided
        """
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        saved_hash = checkpoint.get('config_hash')
        
        if expected_hash is None:
            return saved_hash
        
        return saved_hash == expected_hash
