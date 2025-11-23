"""
Report generation and comparison utilities for Street Scene tasks.

Generates JSON and Markdown/HTML reports with metrics, config metadata,
dataset info, and comparisons against previous runs.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None

from .metrics import (
    DetectionMetrics, TrackingMetrics, ClassificationMetrics,
    load_metrics
)

logger = logging.getLogger(__name__)


class MetricsReporter:
    """Generate reports for training/evaluation runs."""
    
    def __init__(self, output_dir: str, task_type: str, task_name: Optional[str] = None):
        """
        Initialize reporter.
        
        Args:
            output_dir: Directory to save reports
            task_type: Type of task ('detection', 'tracking', 'classification')
            task_name: Specific task name (e.g., 'vehicle_detection')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.task_type = task_type
        self.task_name = task_name or task_type
        self.logger = logging.getLogger(__name__)
    
    def generate_report(
        self,
        metrics: Union[DetectionMetrics, TrackingMetrics, ClassificationMetrics, Dict[str, Any]],
        config: Dict[str, Any],
        dataset_info: Optional[Dict[str, Any]] = None,
        training_history: Optional[List[Dict[str, float]]] = None,
        checkpoint_path: Optional[str] = None,
        compare_with: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate comprehensive report.
        
        Args:
            metrics: Metrics object or dictionary
            config: Configuration dictionary
            dataset_info: Dataset information
            training_history: Training history (loss per epoch, etc.)
            checkpoint_path: Path to model checkpoint
            compare_with: List of run directories to compare with
        
        Returns:
            Dictionary with paths to generated reports
        """
        self.logger.info(f"Generating report for {self.task_name}...")
        
        # Convert metrics to dict if needed
        if hasattr(metrics, 'to_dict'):
            metrics_dict = metrics.to_dict()
        else:
            metrics_dict = metrics
        
        # Create report data
        report_data = {
            'task_name': self.task_name,
            'task_type': self.task_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics_dict,
            'config': config,
            'dataset_info': dataset_info or {},
            'checkpoint_path': checkpoint_path
        }
        
        # Add training history if available
        if training_history:
            report_data['training_history'] = training_history
        
        # Save JSON report
        json_path = self.output_dir / 'metrics_report.json'
        self._save_json_report(report_data, json_path)
        
        # Generate Markdown report
        md_path = self.output_dir / 'metrics_report.md'
        self._save_markdown_report(report_data, md_path)
        
        # Generate HTML report
        html_path = self.output_dir / 'metrics_report.html'
        self._save_html_report(report_data, html_path)
        
        # Generate plots
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        self._generate_plots(metrics, training_history, plots_dir)
        
        # Generate comparison if requested
        comparison_path = None
        if compare_with:
            comparison_path = self.output_dir / 'comparison.md'
            self._generate_comparison(report_data, compare_with, comparison_path)
        
        self.logger.info(f"Report generated in {self.output_dir}")
        
        return {
            'json': str(json_path),
            'markdown': str(md_path),
            'html': str(html_path),
            'plots_dir': str(plots_dir),
            'comparison': str(comparison_path) if comparison_path else None
        }
    
    def _save_json_report(self, data: Dict[str, Any], output_path: Path) -> None:
        """Save JSON report."""
        # Convert numpy arrays to lists for JSON serialization
        data_copy = self._convert_for_json(data)
        
        with open(output_path, 'w') as f:
            json.dump(data_copy, f, indent=2)
        
        self.logger.info(f"JSON report saved to {output_path}")
    
    def _save_markdown_report(self, data: Dict[str, Any], output_path: Path) -> None:
        """Save Markdown report."""
        lines = []
        
        # Title
        lines.append(f"# {data['task_name']} - Metrics Report\n")
        lines.append(f"**Generated:** {data['timestamp']}\n")
        lines.append(f"**Task Type:** {data['task_type']}\n\n")
        
        # Metrics section
        lines.append("## Metrics\n")
        lines.append(self._format_metrics_table(data['metrics']))
        lines.append("\n")
        
        # Dataset info
        if data.get('dataset_info'):
            lines.append("## Dataset Information\n")
            lines.append(self._format_dict_as_table(data['dataset_info']))
            lines.append("\n")
        
        # Configuration
        lines.append("## Configuration\n")
        lines.append("```yaml\n")
        lines.append(yaml.dump(data['config'], default_flow_style=False))
        lines.append("```\n\n")
        
        # Training history summary
        if data.get('training_history'):
            lines.append("## Training Summary\n")
            lines.append(self._format_training_summary(data['training_history']))
            lines.append("\n")
        
        # Checkpoint info
        if data.get('checkpoint_path'):
            lines.append("## Model Checkpoint\n")
            lines.append(f"**Path:** `{data['checkpoint_path']}`\n\n")
        
        # Reproduction checklist
        lines.append("## Reproduction Checklist\n")
        lines.append(self._generate_reproduction_checklist(data))
        lines.append("\n")
        
        with open(output_path, 'w') as f:
            f.writelines(lines)
        
        self.logger.info(f"Markdown report saved to {output_path}")
    
    def _save_html_report(self, data: Dict[str, Any], output_path: Path) -> None:
        """Save HTML report."""
        html = []
        
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<meta charset='utf-8'>")
        html.append(f"<title>{data['task_name']} - Metrics Report</title>")
        html.append("<style>")
        html.append(self._get_html_style())
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        html.append(f"<h1>{data['task_name']} - Metrics Report</h1>")
        html.append(f"<p><strong>Generated:</strong> {data['timestamp']}</p>")
        html.append(f"<p><strong>Task Type:</strong> {data['task_type']}</p>")
        
        html.append("<h2>Metrics</h2>")
        html.append(self._format_metrics_html_table(data['metrics']))
        
        if data.get('dataset_info'):
            html.append("<h2>Dataset Information</h2>")
            html.append(self._format_dict_html_table(data['dataset_info']))
        
        if data.get('training_history'):
            html.append("<h2>Training History</h2>")
            html.append("<p>See plots directory for visualization</p>")
        
        html.append("</body>")
        html.append("</html>")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(html))
        
        self.logger.info(f"HTML report saved to {output_path}")
    
    def _generate_plots(
        self,
        metrics: Union[DetectionMetrics, TrackingMetrics, ClassificationMetrics, Dict],
        training_history: Optional[List[Dict[str, float]]],
        plots_dir: Path
    ) -> None:
        """Generate visualization plots."""
        if plt is None:
            self.logger.warning("matplotlib not available, skipping plots")
            return
        
        # Plot training history
        if training_history:
            self._plot_training_history(training_history, plots_dir)
        
        # Plot confusion matrix if available
        if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
            self._plot_confusion_matrix(metrics.confusion_matrix, plots_dir)
        elif isinstance(metrics, dict) and 'confusion_matrix' in metrics:
            self._plot_confusion_matrix(metrics['confusion_matrix'], plots_dir)
    
    def _plot_training_history(self, history: List[Dict[str, float]], plots_dir: Path) -> None:
        """Plot training history."""
        if not history:
            return
        
        # Extract metrics
        epochs = list(range(len(history)))
        metrics_keys = set()
        for h in history:
            metrics_keys.update(h.keys())
        
        # Plot each metric
        for key in metrics_keys:
            values = [h.get(key, 0) for h in history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, values, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel(key)
            plt.title(f'{key} over Epochs')
            plt.grid(True)
            
            plot_path = plots_dir / f'{key}_history.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"Saved plot: {plot_path}")
    
    def _plot_confusion_matrix(self, cm: Union[np.ndarray, List], plots_dir: Path) -> None:
        """Plot confusion matrix."""
        if sns is None:
            return
        
        cm = np.array(cm)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        plot_path = plots_dir / 'confusion_matrix.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.debug(f"Saved confusion matrix: {plot_path}")
    
    def _generate_comparison(
        self,
        current_data: Dict[str, Any],
        previous_runs: List[str],
        output_path: Path
    ) -> None:
        """Generate comparison against previous runs."""
        lines = []
        
        lines.append(f"# Comparison: {current_data['task_name']}\n\n")
        lines.append(f"**Current Run:** {current_data['timestamp']}\n\n")
        
        # Load previous runs
        comparison_data = [('Current', current_data)]
        
        for run_dir in previous_runs:
            try:
                prev_metrics = load_run_metrics(run_dir)
                run_name = Path(run_dir).name
                comparison_data.append((run_name, prev_metrics))
            except Exception as e:
                self.logger.warning(f"Failed to load run from {run_dir}: {e}")
        
        # Create comparison table
        lines.append("## Metrics Comparison\n\n")
        lines.append(self._format_comparison_table(comparison_data))
        
        with open(output_path, 'w') as f:
            f.writelines(lines)
        
        self.logger.info(f"Comparison saved to {output_path}")
    
    def _format_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as Markdown table."""
        lines = ["| Metric | Value |", "|--------|-------|"]
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                continue
            if isinstance(value, float):
                value = f"{value:.4f}"
            lines.append(f"| {key} | {value} |")
        
        return '\n'.join(lines)
    
    def _format_dict_as_table(self, data: Dict[str, Any]) -> str:
        """Format dictionary as Markdown table."""
        lines = ["| Key | Value |", "|-----|-------|"]
        
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            lines.append(f"| {key} | {value} |")
        
        return '\n'.join(lines)
    
    def _format_training_summary(self, history: List[Dict[str, float]]) -> str:
        """Format training history summary."""
        if not history:
            return "No training history available."
        
        lines = []
        lines.append(f"- **Total Epochs:** {len(history)}")
        
        # Get final metrics
        final_metrics = history[-1]
        lines.append(f"- **Final Metrics:**")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                lines.append(f"  - {key}: {value:.4f}")
        
        return '\n'.join(lines)
    
    def _format_metrics_html_table(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as HTML table."""
        html = ["<table>", "<tr><th>Metric</th><th>Value</th></tr>"]
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                continue
            if isinstance(value, float):
                value = f"{value:.4f}"
            html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
        
        html.append("</table>")
        return '\n'.join(html)
    
    def _format_dict_html_table(self, data: Dict[str, Any]) -> str:
        """Format dictionary as HTML table."""
        html = ["<table>", "<tr><th>Key</th><th>Value</th></tr>"]
        
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            html.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
        
        html.append("</table>")
        return '\n'.join(html)
    
    def _format_comparison_table(self, comparison_data: List[tuple]) -> str:
        """Format comparison table."""
        if not comparison_data:
            return "No comparison data available."
        
        # Get all metric keys
        all_keys = set()
        for _, data in comparison_data:
            if 'metrics' in data:
                all_keys.update(data['metrics'].keys())
        
        # Create header
        lines = ["| Metric | " + " | ".join([name for name, _ in comparison_data]) + " |"]
        lines.append("|--------|" + "|".join(["-------" for _ in comparison_data]) + "|")
        
        # Create rows
        for key in sorted(all_keys):
            row = [key]
            for _, data in comparison_data:
                value = data.get('metrics', {}).get(key, 'N/A')
                if isinstance(value, float):
                    value = f"{value:.4f}"
                row.append(str(value))
            lines.append("| " + " | ".join(row) + " |")
        
        return '\n'.join(lines)
    
    def _generate_reproduction_checklist(self, data: Dict[str, Any]) -> str:
        """Generate reproduction checklist."""
        lines = []
        lines.append("- [x] Configuration saved")
        lines.append("- [x] Metrics computed and saved")
        
        if data.get('checkpoint_path'):
            lines.append("- [x] Model checkpoint saved")
        else:
            lines.append("- [ ] Model checkpoint saved")
        
        if data.get('dataset_info'):
            lines.append("- [x] Dataset information recorded")
        else:
            lines.append("- [ ] Dataset information recorded")
        
        lines.append("- [ ] Verification run completed (use `verify_repro.py`)")
        
        return '\n'.join(lines)
    
    def _get_html_style(self) -> str:
        """Get HTML style."""
        return """
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1, h2 {
            color: #333;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        """
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert objects for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def generate_report(
    output_dir: str,
    metrics: Union[DetectionMetrics, TrackingMetrics, ClassificationMetrics, Dict[str, Any]],
    config: Dict[str, Any],
    task_type: str,
    task_name: Optional[str] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Generate report (convenience function).
    
    Args:
        output_dir: Directory to save reports
        metrics: Metrics object or dictionary
        config: Configuration dictionary
        task_type: Type of task
        task_name: Specific task name
        **kwargs: Additional arguments for MetricsReporter.generate_report
    
    Returns:
        Dictionary with paths to generated reports
    """
    reporter = MetricsReporter(output_dir, task_type, task_name)
    return reporter.generate_report(metrics, config, **kwargs)


def compare_runs(run_dirs: List[str], output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare multiple runs.
    
    Args:
        run_dirs: List of run directories
        output_path: Path to save comparison (optional)
    
    Returns:
        Comparison data
    """
    logger.info(f"Comparing {len(run_dirs)} runs...")
    
    comparison_data = []
    
    for run_dir in run_dirs:
        try:
            metrics = load_run_metrics(run_dir)
            comparison_data.append({
                'run_dir': run_dir,
                'run_name': Path(run_dir).name,
                'metrics': metrics
            })
        except Exception as e:
            logger.warning(f"Failed to load run from {run_dir}: {e}")
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        logger.info(f"Comparison saved to {output_path}")
    
    return comparison_data


def load_run_metrics(run_dir: str) -> Dict[str, Any]:
    """
    Load metrics from a run directory.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        Dictionary with metrics and metadata
    """
    run_path = Path(run_dir)
    
    # Try to load JSON report
    json_path = run_path / 'metrics_report.json'
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    
    # Try to load metrics.json
    metrics_path = run_path / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"No metrics found in {run_dir}")
