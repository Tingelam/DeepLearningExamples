"""
Hierarchical configuration management system using OmegaConf.

This module provides utilities for loading, merging, and validating configurations
from YAML files with support for CLI and environment variable overrides.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime

from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigAttributeError


logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages hierarchical configuration loading and merging.
    
    Supports loading from:
    - Shared defaults (configs/defaults.yaml)
    - Task-level configs (configs/tasks/{task_type}/{task_name}.yaml)
    - Experiment configs (configs/experiments/{experiment_name}.yaml)
    - CLI overrides
    - Environment variable overrides
    """

    def __init__(self, config_dir: Path):
        """
        Initialize the ConfigManager.
        
        Args:
            config_dir: Path to the configs directory
        """
        self.config_dir = Path(config_dir)
        self.defaults_file = self.config_dir / "defaults.yaml"
        self.tasks_dir = self.config_dir / "tasks"
        self.experiments_dir = self.config_dir / "experiments"
        
    def load_defaults(self) -> DictConfig:
        """
        Load the default configuration.
        
        Returns:
            DictConfig with default settings
            
        Raises:
            FileNotFoundError: If defaults.yaml doesn't exist
        """
        if not self.defaults_file.exists():
            raise FileNotFoundError(f"Default config not found at {self.defaults_file}")
        
        logger.info(f"Loading defaults from {self.defaults_file}")
        return OmegaConf.load(self.defaults_file)
    
    def load_task_config(self, task_type: str, task_name: str) -> DictConfig:
        """
        Load a task-level configuration.
        
        Args:
            task_type: Type of task (e.g., 'detection', 'classification')
            task_name: Name of the specific task (e.g., 'pedestrian', 'vehicle')
            
        Returns:
            DictConfig with task-specific settings
            
        Raises:
            FileNotFoundError: If task config doesn't exist
        """
        task_file = self.tasks_dir / task_type / f"{task_name}.yaml"
        
        if not task_file.exists():
            raise FileNotFoundError(f"Task config not found at {task_file}")
        
        logger.info(f"Loading task config from {task_file}")
        return OmegaConf.load(task_file)
    
    def load_experiment_config(self, experiment_name: str) -> DictConfig:
        """
        Load an experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            DictConfig with experiment settings
            
        Raises:
            FileNotFoundError: If experiment config doesn't exist
        """
        exp_file = self.experiments_dir / f"{experiment_name}.yaml"
        
        if not exp_file.exists():
            raise FileNotFoundError(f"Experiment config not found at {exp_file}")
        
        logger.info(f"Loading experiment config from {exp_file}")
        return OmegaConf.load(exp_file)
    
    def load_configs(
        self,
        task_type: str,
        task_name: str,
        experiment_name: Optional[str] = None,
        cli_overrides: Optional[List[str]] = None,
        env_prefix: str = "APP_"
    ) -> DictConfig:
        """
        Load and merge configurations from multiple sources.
        
        Merging order (later overrides earlier):
        1. Defaults
        2. Task config
        3. Experiment config (if provided)
        4. Environment variables (with prefix, using double underscores for nesting)
        5. CLI overrides
        
        Args:
            task_type: Type of task (e.g., 'detection', 'classification')
            task_name: Name of the specific task (e.g., 'pedestrian', 'vehicle')
            experiment_name: Optional experiment name to include in config composition
            cli_overrides: Optional list of CLI overrides in format 'key=value'
            env_prefix: Prefix for environment variables (e.g., 'APP_').
                Environment variables use double underscores for nested keys,
                e.g., APP_TRAINING__BATCH_SIZE maps to training.batch_size
            
        Returns:
            Merged DictConfig
        """
        # Load defaults
        cfg = self.load_defaults()
        
        # Merge task config
        task_cfg = self.load_task_config(task_type, task_name)
        cfg = OmegaConf.merge(cfg, task_cfg)
        
        # Merge experiment config if provided
        if experiment_name:
            exp_cfg = self.load_experiment_config(experiment_name)
            cfg = OmegaConf.merge(cfg, exp_cfg)
        
        # Apply environment variable overrides
        cfg = self._apply_env_overrides(cfg, env_prefix)
        
        # Apply CLI overrides
        if cli_overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(cli_overrides))
        
        return cfg
    
    def _apply_env_overrides(self, cfg: DictConfig, prefix: str = "APP_") -> DictConfig:
        """
        Apply environment variable overrides to config.
        
        Environment variables should follow the pattern:
        {prefix}{KEY}={VALUE} where KEY uses double underscores for nesting, e.g.
        APP_DATASET__NAME=my_dataset (which maps to dataset.name)
        Single underscores are preserved in the key name.
        
        Args:
            cfg: Configuration to update
            prefix: Prefix for environment variables
            
        Returns:
            Updated DictConfig
        """
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Remove prefix and convert to config key
                # Replace double underscores with dots for nested access
                config_key = env_key[len(prefix):].lower().replace("__", ".")
                
                # Check if the key exists in the config before applying
                if OmegaConf.select(cfg, config_key) is not None:
                    logger.debug(f"Applying environment override: {env_key} -> {config_key}={env_value}")
                    try:
                        OmegaConf.update(cfg, config_key, env_value, merge=False)
                    except (ConfigAttributeError, KeyError) as e:
                        logger.warning(f"Failed to apply environment override {env_key}: {e}")
                else:
                    logger.debug(f"Skipping environment override {env_key}: key {config_key} not found in config")
        
        return cfg
    
    def validate_config(
        self,
        cfg: DictConfig,
        required_fields: Optional[List[str]] = None
    ) -> bool:
        """
        Validate that all required fields are present in the config.
        
        Args:
            cfg: Configuration to validate
            required_fields: List of required field paths (dot notation)
            
        Returns:
            True if valid, False otherwise
        """
        if not required_fields:
            return True
        
        missing_fields = []
        for field in required_fields:
            try:
                OmegaConf.select(cfg, field)
            except (ConfigAttributeError, KeyError):
                missing_fields.append(field)
        
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return False
        
        return True
    
    def save_resolved_config(
        self,
        cfg: DictConfig,
        output_dir: Path,
        run_name: Optional[str] = None
    ) -> Path:
        """
        Save the resolved configuration to a file for reproducibility.
        
        Args:
            cfg: Configuration to save
            output_dir: Directory to save the config to
            run_name: Optional run name for the config file
            
        Returns:
            Path to the saved config file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"resolved_config_{timestamp}"
        
        config_file = output_dir / f"{run_name}.yaml"
        
        logger.info(f"Saving resolved config to {config_file}")
        OmegaConf.save(cfg, config_file)
        
        return config_file
    
    def to_dict(self, cfg: DictConfig) -> Dict[str, Any]:
        """
        Convert DictConfig to a regular Python dict.
        
        Args:
            cfg: DictConfig to convert
            
        Returns:
            Regular Python dict
        """
        return OmegaConf.to_container(cfg, resolve=True)


def setup_config(
    config_dir: Path,
    task_type: str,
    task_name: str,
    experiment_name: Optional[str] = None,
    cli_overrides: Optional[List[str]] = None,
    env_prefix: str = "APP_",
    required_fields: Optional[List[str]] = None
) -> DictConfig:
    """
    Setup and load configuration with sensible defaults.
    
    Args:
        config_dir: Path to the configs directory
        task_type: Type of task (e.g., 'detection', 'classification')
        task_name: Name of the specific task
        experiment_name: Optional experiment name
        cli_overrides: Optional list of CLI overrides
        env_prefix: Prefix for environment variable overrides
        required_fields: Optional list of required fields to validate
        
    Returns:
        Resolved DictConfig
        
    Raises:
        FileNotFoundError: If required config files don't exist
        ValueError: If validation fails
    """
    manager = ConfigManager(config_dir)
    cfg = manager.load_configs(
        task_type=task_type,
        task_name=task_name,
        experiment_name=experiment_name,
        cli_overrides=cli_overrides,
        env_prefix=env_prefix
    )
    
    if required_fields and not manager.validate_config(cfg, required_fields):
        raise ValueError("Config validation failed: missing required fields")
    
    return cfg
