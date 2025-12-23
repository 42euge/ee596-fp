#!/usr/bin/env python3
"""
Training Monitoring Dashboard

Provides real-time monitoring of training runs:
- Parse logs from training runs
- Visualize metrics (loss, reward, accuracy)
- Track training progress
- Compare multiple runs
- Export metrics to CSV/JSON
"""

import argparse
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitor and visualize training metrics"""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.metrics_cache = {}

    def parse_log_file(self, log_file: Path) -> Dict[str, Any]:
        """
        Parse training log file and extract metrics

        Args:
            log_file: Path to log file

        Returns:
            Dictionary with parsed metrics
        """
        logger.info(f"Parsing log file: {log_file}")

        metrics = {
            "run_name": log_file.stem,
            "log_file": str(log_file),
            "steps": [],
            "loss": [],
            "reward": [],
            "kl_div": [],
            "learning_rate": [],
            "accuracy": [],
            "timestamps": [],
        }

        # Regex patterns for common log formats
        patterns = {
            "step": re.compile(r"step[:\s]+(\d+)", re.IGNORECASE),
            "loss": re.compile(r"loss[:\s]+([\d\.]+)", re.IGNORECASE),
            "reward": re.compile(r"reward[:\s]+([\d\.]+)", re.IGNORECASE),
            "kl": re.compile(r"kl[_\s]div[:\s]+([\d\.]+)", re.IGNORECASE),
            "lr": re.compile(r"learning[_\s]rate[:\s]+([\d\.e\-]+)", re.IGNORECASE),
            "accuracy": re.compile(r"acc(?:uracy)?[:\s]+([\d\.]+)", re.IGNORECASE),
            "timestamp": re.compile(r"(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})"),
        }

        with open(log_file, 'r') as f:
            for line in f:
                # Try to extract step number
                step_match = patterns["step"].search(line)
                if step_match:
                    step = int(step_match.group(1))

                    # Extract other metrics from the same line
                    loss_match = patterns["loss"].search(line)
                    reward_match = patterns["reward"].search(line)
                    kl_match = patterns["kl"].search(line)
                    lr_match = patterns["lr"].search(line)
                    acc_match = patterns["accuracy"].search(line)
                    ts_match = patterns["timestamp"].search(line)

                    metrics["steps"].append(step)
                    metrics["loss"].append(float(loss_match.group(1)) if loss_match else None)
                    metrics["reward"].append(float(reward_match.group(1)) if reward_match else None)
                    metrics["kl_div"].append(float(kl_match.group(1)) if kl_match else None)
                    metrics["learning_rate"].append(float(lr_match.group(1)) if lr_match else None)
                    metrics["accuracy"].append(float(acc_match.group(1)) if acc_match else None)
                    metrics["timestamps"].append(ts_match.group(1) if ts_match else None)

        logger.info(f"Parsed {len(metrics['steps'])} training steps")

        return metrics

    def parse_wandb_metrics(self, run_name: str, project: str = "reward-model-dev") -> Dict[str, Any]:
        """
        Fetch metrics from Weights & Biases

        Args:
            run_name: W&B run name
            project: W&B project name

        Returns:
            Dictionary with metrics
        """
        try:
            import wandb

            api = wandb.Api()
            run = api.run(f"{project}/{run_name}")

            # Get history
            history = run.history()

            metrics = {
                "run_name": run_name,
                "steps": history["_step"].tolist(),
                "loss": history.get("loss", [None] * len(history)).tolist(),
                "reward": history.get("reward", [None] * len(history)).tolist(),
                "kl_div": history.get("kl_div", [None] * len(history)).tolist(),
                "learning_rate": history.get("learning_rate", [None] * len(history)).tolist(),
                "accuracy": history.get("accuracy", [None] * len(history)).tolist(),
            }

            logger.info(f"Fetched {len(metrics['steps'])} steps from W&B")

            return metrics

        except ImportError:
            logger.error("wandb not installed. Install with: pip install wandb")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch W&B metrics: {e}")
            raise

    def generate_ascii_plot(
        self,
        values: List[float],
        width: int = 80,
        height: int = 20,
        title: str = "Metric"
    ) -> str:
        """
        Generate ASCII art plot of metrics

        Args:
            values: List of metric values
            width: Plot width in characters
            height: Plot height in characters
            title: Plot title

        Returns:
            ASCII art plot as string
        """
        # Filter out None values
        values = [v for v in values if v is not None]

        if not values:
            return f"{title}: No data available"

        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val != min_val else 1

        lines = []
        lines.append(f"\n{title} ({min_val:.4f} to {max_val:.4f})")
        lines.append("─" * width)

        # Create plot grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]

        # Plot values
        for i, val in enumerate(values):
            x = int((i / len(values)) * (width - 1))
            y = height - 1 - int(((val - min_val) / val_range) * (height - 1))
            y = max(0, min(height - 1, y))  # Clamp to grid
            grid[y][x] = '●'

        # Convert grid to string
        for row in grid:
            lines.append(''.join(row))

        lines.append("─" * width)

        return '\n'.join(lines)

    def generate_summary(self, metrics: Dict[str, Any]) -> str:
        """Generate text summary of training run"""

        lines = [
            "=" * 80,
            f"Training Run: {metrics['run_name']}",
            "=" * 80,
            f"Total Steps: {len(metrics['steps'])}",
            "",
        ]

        # Add metric summaries
        for metric_name in ["loss", "reward", "kl_div", "accuracy"]:
            values = [v for v in metrics[metric_name] if v is not None]
            if values:
                lines.append(f"{metric_name.upper()}:")
                lines.append(f"  Final: {values[-1]:.4f}")
                lines.append(f"  Min: {min(values):.4f}")
                lines.append(f"  Max: {max(values):.4f}")
                lines.append(f"  Mean: {sum(values)/len(values):.4f}")
                lines.append("")

        return "\n".join(lines)

    def generate_dashboard(
        self,
        metrics: Dict[str, Any],
        include_plots: bool = True
    ) -> str:
        """Generate full dashboard with summary and plots"""

        dashboard = []

        # Add summary
        dashboard.append(self.generate_summary(metrics))

        # Add plots
        if include_plots:
            dashboard.append("\nMetric Plots:")

            for metric_name in ["loss", "reward", "accuracy"]:
                values = metrics[metric_name]
                if any(v is not None for v in values):
                    plot = self.generate_ascii_plot(
                        values,
                        title=metric_name.upper()
                    )
                    dashboard.append(plot)

        dashboard.append("=" * 80)

        return "\n".join(dashboard)

    def compare_runs(self, run_names: List[str]) -> str:
        """Compare multiple training runs"""

        lines = [
            "=" * 100,
            "Training Run Comparison",
            "=" * 100,
            "",
        ]

        # Header
        header = f"{'Run Name':<30} {'Steps':<10} {'Final Loss':<15} {'Final Reward':<15} {'Final Acc':<15}"
        lines.append(header)
        lines.append("─" * 100)

        # Load metrics for each run
        for run_name in run_names:
            log_file = self.log_dir / f"{run_name}.log"

            if not log_file.exists():
                lines.append(f"{run_name:<30} {'Not found':>10}")
                continue

            metrics = self.parse_log_file(log_file)

            steps = len(metrics["steps"])
            final_loss = metrics["loss"][-1] if metrics["loss"] and metrics["loss"][-1] else "N/A"
            final_reward = metrics["reward"][-1] if metrics["reward"] and metrics["reward"][-1] else "N/A"
            final_acc = metrics["accuracy"][-1] if metrics["accuracy"] and metrics["accuracy"][-1] else "N/A"

            # Format values
            loss_str = f"{final_loss:.4f}" if isinstance(final_loss, float) else final_loss
            reward_str = f"{final_reward:.4f}" if isinstance(final_reward, float) else final_reward
            acc_str = f"{final_acc:.4f}" if isinstance(final_acc, float) else final_acc

            row = f"{run_name:<30} {steps:<10} {loss_str:<15} {reward_str:<15} {acc_str:<15}"
            lines.append(row)

        lines.append("=" * 100)

        return "\n".join(lines)

    def export_metrics(
        self,
        metrics: Dict[str, Any],
        output_file: Path,
        format: str = "json"
    ) -> None:
        """
        Export metrics to file

        Args:
            metrics: Metrics dictionary
            output_file: Output file path
            format: Export format (json, csv)
        """
        logger.info(f"Exporting metrics to {output_file}")

        if format == "json":
            with open(output_file, 'w') as f:
                json.dump(metrics, f, indent=2)

        elif format == "csv":
            import csv

            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)

                # Write header
                header = ["step"]
                for key in ["loss", "reward", "kl_div", "learning_rate", "accuracy"]:
                    if any(v is not None for v in metrics[key]):
                        header.append(key)

                writer.writerow(header)

                # Write data
                for i, step in enumerate(metrics["steps"]):
                    row = [step]
                    for key in header[1:]:
                        row.append(metrics[key][i] if i < len(metrics[key]) else None)
                    writer.writerow(row)

        logger.info(f"Metrics exported to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Training monitoring dashboard")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing log files"
    )
    parser.add_argument(
        "--run-name",
        help="Specific run to monitor"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple runs"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Fetch metrics from W&B instead of log files"
    )
    parser.add_argument(
        "--wandb-project",
        default="reward-model-dev",
        help="W&B project name"
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export metrics to file"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Export format"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable ASCII plots"
    )

    args = parser.parse_args()

    monitor = TrainingMonitor(args.log_dir)

    try:
        if args.compare:
            # Compare multiple runs
            comparison = monitor.compare_runs(args.compare)
            print(comparison)

        elif args.run_name:
            # Monitor single run
            if args.wandb:
                metrics = monitor.parse_wandb_metrics(args.run_name, args.wandb_project)
            else:
                log_file = args.log_dir / f"{args.run_name}.log"
                if not log_file.exists():
                    logger.error(f"Log file not found: {log_file}")
                    sys.exit(1)
                metrics = monitor.parse_log_file(log_file)

            # Generate dashboard
            dashboard = monitor.generate_dashboard(metrics, include_plots=not args.no_plots)
            print(dashboard)

            # Export if requested
            if args.export:
                monitor.export_metrics(metrics, args.export, args.format)

        else:
            # List all available runs
            logger.info(f"Available runs in {args.log_dir}:")
            for log_file in sorted(args.log_dir.glob("*.log")):
                logger.info(f"  - {log_file.stem}")

            logger.info("\nUse --run-name to monitor a specific run")
            logger.info("Use --compare to compare multiple runs")

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise


if __name__ == "__main__":
    main()
