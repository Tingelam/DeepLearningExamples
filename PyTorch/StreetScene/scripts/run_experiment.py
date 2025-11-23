#!/usr/bin/env python3
"""
CLI entry point for running StreetScene experiments.

This script loads hierarchical configurations, applies overrides, validates
required fields, and saves the resolved config for reproducibility.

Usage:
    python run_experiment.py --task-type detection --task-name pedestrian
    python run_experiment.py --task-type detection --task-name pedestrian --experiment baseline
    python run_experiment.py --task-type detection --task-name pedestrian --experiment baseline lr=0.0001 batch_size=64
    python run_experiment.py --help
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from common.config import ConfigManager


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run StreetScene experiments with hierarchical config management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pedestrian detection task with default settings
  python run_experiment.py --task-type detection --task-name pedestrian

  # Run with specific experiment config
  python run_experiment.py --task-type detection --task-name pedestrian --experiment baseline

  # Override settings via CLI
  python run_experiment.py --task-type detection --task-name pedestrian \\
    training.learning_rate=0.0001 training.batch_size=64

  # Use environment variables for overrides
  # Note: Use double underscores for nesting, single underscores are preserved
  export APP_TRAINING__LEARNING_RATE=0.0001
  python run_experiment.py --task-type detection --task-name pedestrian

  # Save config and exit (dry-run)
  python run_experiment.py --task-type detection --task-name pedestrian --dry-run
        """
    )
    
    parser.add_argument(
        "--task-type",
        required=True,
        help="Type of task (e.g., detection, classification)"
    )
    parser.add_argument(
        "--task-name",
        required=True,
        help="Name of the specific task (e.g., pedestrian, vehicle)"
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Optional experiment config to include in composition"
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Path to configs directory (auto-detected if not provided)"
    )
    parser.add_argument(
        "--output-dir",
        default="./outputs",
        help="Directory to save resolved config and outputs"
    )
    parser.add_argument(
        "--env-prefix",
        default="APP_",
        help="Prefix for environment variable overrides"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and save it without running the experiment"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved config and exit"
    )
    
    # Allow CLI overrides as positional arguments
    parser.add_argument(
        "overrides",
        nargs="*",
        help="CLI overrides in format key=value (e.g., training.lr=0.001)"
    )
    
    return parser.parse_args()


def validate_required_fields(config) -> bool:
    """
    Validate that all required fields are present in the config.
    
    Args:
        config: The configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        "dataset.name",
        "dataset.root_dir",
        "model.name",
        "model.backbone",
        "training.epochs",
        "training.batch_size",
        "training.learning_rate",
        "hardware.device",
    ]
    
    logger = logging.getLogger(__name__)
    missing_fields = []
    
    for field in required_fields:
        try:
            OmegaConf.select(config, field)
        except (KeyError, AttributeError):
            missing_fields.append(field)
    
    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
        return False
    
    return True


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Determine config directory
        if args.config_dir:
            config_dir = Path(args.config_dir)
        else:
            config_dir = PROJECT_ROOT / "configs"
        
        if not config_dir.exists():
            logger.error(f"Config directory not found: {config_dir}")
            return 1
        
        logger.info(f"Using config directory: {config_dir}")
        
        # Initialize config manager
        manager = ConfigManager(config_dir)
        
        # Load and merge configurations
        logger.info(f"Loading config for task_type={args.task_type}, task_name={args.task_name}")
        
        cli_overrides = args.overrides if args.overrides else None
        
        config = manager.load_configs(
            task_type=args.task_type,
            task_name=args.task_name,
            experiment_name=args.experiment,
            cli_overrides=cli_overrides,
            env_prefix=args.env_prefix
        )
        
        # Validate required fields
        if not validate_required_fields(config):
            logger.error("Config validation failed")
            return 1
        
        logger.info("Config validation passed")
        
        # Print config if requested
        if args.print_config:
            logger.info("=" * 80)
            logger.info("RESOLVED CONFIGURATION")
            logger.info("=" * 80)
            print(OmegaConf.to_yaml(config))
            logger.info("=" * 80)
        
        # Save resolved config
        output_dir = Path(args.output_dir)
        config_file = manager.save_resolved_config(
            config,
            output_dir,
            run_name=f"{args.task_type}_{args.task_name}_{args.experiment or 'default'}"
        )
        
        logger.info(f"Resolved config saved to: {config_file}")
        
        # Exit early if dry-run
        if args.dry_run:
            logger.info("Dry-run mode: exiting after config validation and saving")
            return 0
        
        # At this point, we would normally pass the config to the actual training/eval code
        logger.info("Config setup complete. Ready for experiment execution.")
        logger.info(f"Dataset: {config.dataset.name}")
        logger.info(f"Model: {config.model.name}")
        logger.info(f"Training epochs: {config.training.epochs}")
        logger.info(f"Batch size: {config.training.batch_size}")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
