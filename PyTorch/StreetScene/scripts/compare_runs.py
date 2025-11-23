#!/usr/bin/env python3
"""
Compare multiple training/evaluation runs.

Loads run directories and generates comparative reports.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.reporting import compare_runs, load_run_metrics
from src.common.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description='Compare multiple runs')
    
    parser.add_argument('--run-ids', type=str, nargs='+', required=True,
                       help='List of run directories to compare')
    parser.add_argument('--output', type=str, default='./comparison_report',
                       help='Output directory for comparison report')
    parser.add_argument('--format', type=str, choices=['json', 'markdown', 'both'], default='both',
                       help='Output format for comparison')
    parser.add_argument('--metric', type=str, default=None,
                       help='Specific metric to highlight (optional)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def format_comparison_markdown(comparison_data: List[Dict[str, Any]], 
                               highlight_metric: str = None) -> str:
    """Format comparison data as Markdown."""
    lines = []
    
    lines.append("# Run Comparison Report\n\n")
    lines.append(f"Comparing {len(comparison_data)} runs\n\n")
    
    # Create summary table
    lines.append("## Summary\n\n")
    lines.append("| Run | Task | Timestamp |\n")
    lines.append("|-----|------|----------|\n")
    
    for run in comparison_data:
        run_name = run.get('run_name', 'Unknown')
        task = run.get('metrics', {}).get('task_name', 'N/A')
        timestamp = run.get('metrics', {}).get('timestamp', 'N/A')
        lines.append(f"| {run_name} | {task} | {timestamp} |\n")
    
    lines.append("\n")
    
    # Collect all metrics
    all_metrics = set()
    for run in comparison_data:
        metrics = run.get('metrics', {}).get('metrics', {})
        if isinstance(metrics, dict):
            all_metrics.update(metrics.keys())
    
    # Create metrics comparison table
    if all_metrics:
        lines.append("## Metrics Comparison\n\n")
        
        # Header
        header_cols = ['Metric'] + [run.get('run_name', f"Run {i}") for i, run in enumerate(comparison_data)]
        lines.append("| " + " | ".join(header_cols) + " |\n")
        lines.append("|" + "|".join(["-" * (len(col) + 2) for col in header_cols]) + "|\n")
        
        # Rows
        for metric in sorted(all_metrics):
            if highlight_metric and metric == highlight_metric:
                row = [f"**{metric}**"]
            else:
                row = [metric]
            
            values = []
            for run in comparison_data:
                metrics = run.get('metrics', {}).get('metrics', {})
                value = metrics.get(metric, 'N/A')
                if isinstance(value, float):
                    value = f"{value:.4f}"
                elif isinstance(value, dict):
                    continue
                values.append(str(value))
            
            if values:
                row.extend(values)
                lines.append("| " + " | ".join(row) + " |\n")
        
        lines.append("\n")
    
    # Best performing run
    if highlight_metric and all_metrics:
        lines.append(f"## Best Performance ({highlight_metric})\n\n")
        
        best_run = None
        best_value = -float('inf')
        
        for run in comparison_data:
            metrics = run.get('metrics', {}).get('metrics', {})
            value = metrics.get(highlight_metric)
            if value is not None and isinstance(value, (int, float)):
                if value > best_value:
                    best_value = value
                    best_run = run
        
        if best_run:
            lines.append(f"**Best Run:** {best_run.get('run_name', 'Unknown')}\n")
            lines.append(f"**Value:** {best_value:.4f}\n\n")
    
    return ''.join(lines)


def print_comparison_table(comparison_data: List[Dict[str, Any]]):
    """Print comparison as formatted table to console."""
    print("\n" + "=" * 80)
    print("RUN COMPARISON")
    print("=" * 80 + "\n")
    
    # Print summary
    for i, run in enumerate(comparison_data):
        run_name = run.get('run_name', f'Run {i}')
        print(f"Run {i+1}: {run_name}")
        print(f"  Directory: {run.get('run_dir', 'N/A')}")
        
        metrics = run.get('metrics', {}).get('metrics', {})
        if metrics:
            print(f"  Metrics:")
            for key, value in sorted(metrics.items()):
                if isinstance(value, dict):
                    continue
                if isinstance(value, float):
                    print(f"    - {key}: {value:.4f}")
                else:
                    print(f"    - {key}: {value}")
        print()
    
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info(f"Comparing {len(args.run_ids)} runs...")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and compare runs
    try:
        comparison_data = compare_runs(args.run_ids)
        
        if not comparison_data:
            logger.error("No valid runs found")
            return 1
        
        # Print to console
        print_comparison_table(comparison_data)
        
        # Save JSON format
        if args.format in ['json', 'both']:
            json_path = os.path.join(args.output, 'comparison.json')
            with open(json_path, 'w') as f:
                json.dump(comparison_data, f, indent=2)
            logger.info(f"JSON comparison saved to: {json_path}")
        
        # Save Markdown format
        if args.format in ['markdown', 'both']:
            md_path = os.path.join(args.output, 'comparison.md')
            markdown = format_comparison_markdown(comparison_data, args.metric)
            with open(md_path, 'w') as f:
                f.write(markdown)
            logger.info(f"Markdown comparison saved to: {md_path}")
        
        logger.info("Comparison completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
