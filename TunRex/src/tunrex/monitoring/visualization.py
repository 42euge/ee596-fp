"""
Visualization Utilities for Reward Monitoring

Creates plots, dashboards, and visualizations for reward signal analysis.
"""

from typing import Dict, List, Optional, Any
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available, visualization will be disabled")

try:
    import numpy as np
except ImportError:
    np = None
    warnings.warn("numpy not available, some visualizations will be disabled")


class RewardVisualizer:
    """
    Creates visualizations for reward signals and monitoring data.
    """

    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        self.style = style

        if not MATPLOTLIB_AVAILABLE:
            warnings.warn("Matplotlib not available, visualization disabled")

    def plot_reward_history(
        self,
        reward_history: Dict[str, List[float]],
        title: str = "Reward Signals Over Time",
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot reward signal history over time.

        Args:
            reward_history: Dictionary mapping reward names to value lists
            title: Plot title
            save_path: Optional path to save figure
            show: Whether to show the plot
        """
        if not MATPLOTLIB_AVAILABLE or not reward_history:
            return

        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=(12, 6))

        for name, values in reward_history.items():
            steps = list(range(len(values)))
            ax.plot(steps, values, label=name, alpha=0.7, linewidth=1.5)

        ax.set_xlabel("Step")
        ax.set_ylabel("Reward Value")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_reward_distributions(
        self,
        reward_history: Dict[str, List[float]],
        title: str = "Reward Distributions",
        save_path: Optional[str] = None,
        show: bool = True,
        bins: int = 50,
    ):
        """
        Plot distributions of reward signals.

        Args:
            reward_history: Dictionary mapping reward names to value lists
            title: Plot title
            save_path: Optional path to save figure
            show: Whether to show the plot
            bins: Number of histogram bins
        """
        if not MATPLOTLIB_AVAILABLE or not reward_history:
            return

        plt.style.use(self.style)
        n_rewards = len(reward_history)
        n_cols = min(3, n_rewards)
        n_rows = (n_rewards + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rewards == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (name, values) in enumerate(reward_history.items()):
            ax = axes[idx]
            ax.hist(values, bins=bins, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Reward Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{name}\nMean: {np.mean(values):.3f}, Std: {np.std(values):.3f}")
            ax.grid(True, alpha=0.3)

        # Hide extra subplots
        for idx in range(n_rewards, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=14, y=1.00)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_reward_statistics(
        self,
        reward_stats: Dict[str, Any],
        title: str = "Reward Statistics",
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot reward statistics (mean, std, percentiles).

        Args:
            reward_stats: Dictionary of reward statistics
            title: Plot title
            save_path: Optional path to save figure
            show: Whether to show the plot
        """
        if not MATPLOTLIB_AVAILABLE or not reward_stats:
            return

        plt.style.use(self.style)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        names = list(reward_stats.keys())
        means = [stats.mean if hasattr(stats, "mean") else stats.get("mean", 0) for stats in reward_stats.values()]
        stds = [stats.std if hasattr(stats, "std") else stats.get("std", 0) for stats in reward_stats.values()]
        mins = [stats.min if hasattr(stats, "min") else stats.get("min", 0) for stats in reward_stats.values()]
        maxs = [stats.max if hasattr(stats, "max") else stats.get("max", 0) for stats in reward_stats.values()]

        # Mean values
        axes[0, 0].barh(names, means, alpha=0.7, color="steelblue")
        axes[0, 0].set_xlabel("Mean Reward")
        axes[0, 0].set_title("Mean Reward Values")
        axes[0, 0].grid(True, alpha=0.3, axis="x")

        # Standard deviation
        axes[0, 1].barh(names, stds, alpha=0.7, color="coral")
        axes[0, 1].set_xlabel("Standard Deviation")
        axes[0, 1].set_title("Reward Variability")
        axes[0, 1].grid(True, alpha=0.3, axis="x")

        # Min/Max ranges
        y_pos = np.arange(len(names))
        axes[1, 0].barh(y_pos, np.array(maxs) - np.array(mins), left=mins, alpha=0.7, color="mediumseagreen")
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(names)
        axes[1, 0].set_xlabel("Reward Value")
        axes[1, 0].set_title("Min-Max Ranges")
        axes[1, 0].grid(True, alpha=0.3, axis="x")

        # Distribution quality (positive/negative/zero fractions)
        pos_fracs = []
        neg_fracs = []
        zero_fracs = []

        for stats in reward_stats.values():
            if hasattr(stats, "positive_fraction"):
                pos_fracs.append(stats.positive_fraction)
                neg_fracs.append(stats.negative_fraction)
                zero_fracs.append(stats.zeros_fraction)
            else:
                pos_fracs.append(stats.get("positive_fraction", 0))
                neg_fracs.append(stats.get("negative_fraction", 0))
                zero_fracs.append(stats.get("zeros_fraction", 0))

        x = np.arange(len(names))
        width = 0.6

        axes[1, 1].bar(x, pos_fracs, width, label="Positive", alpha=0.7, color="green")
        axes[1, 1].bar(x, neg_fracs, width, bottom=pos_fracs, label="Negative", alpha=0.7, color="red")
        axes[1, 1].bar(
            x,
            zero_fracs,
            width,
            bottom=np.array(pos_fracs) + np.array(neg_fracs),
            label="Zero",
            alpha=0.7,
            color="gray",
        )

        axes[1, 1].set_ylabel("Fraction")
        axes[1, 1].set_title("Distribution Quality")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(names, rotation=45, ha="right")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis="y")

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_reward_dashboard(
        self,
        reward_monitor,
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Create a comprehensive dashboard of reward monitoring data.

        Args:
            reward_monitor: RewardMonitor instance
            save_path: Optional path to save figure
            show: Whether to show the plot
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        plt.style.use(self.style)
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Get data
        stats = reward_monitor.get_all_stats()
        history = reward_monitor.reward_history

        # 1. Time series (top row, full width)
        ax1 = fig.add_subplot(gs[0, :])
        for name, values in history.items():
            steps = list(range(len(values)))
            ax1.plot(steps, values, label=name, alpha=0.7, linewidth=1.5)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Reward Value")
        ax1.set_title("Reward Signals Over Time")
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. Mean values (middle left)
        ax2 = fig.add_subplot(gs[1, 0])
        names = list(stats.keys())
        means = [s.mean for s in stats.values()]
        ax2.barh(names, means, alpha=0.7, color="steelblue")
        ax2.set_xlabel("Mean")
        ax2.set_title("Mean Reward Values")
        ax2.grid(True, alpha=0.3, axis="x")

        # 3. EMAs (middle center)
        ax3 = fig.add_subplot(gs[1, 1])
        for name, s in stats.items():
            ax3.plot([s.ema_long, s.ema_short], [name, name], marker="o", linewidth=2)
        ax3.set_xlabel("EMA Value")
        ax3.set_title("EMAs (Long→Short)")
        ax3.grid(True, alpha=0.3, axis="x")
        ax3.legend(["Long EMA", "Short EMA"], loc="best", fontsize=8)

        # 4. Distribution quality (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        pos_fracs = [s.positive_fraction for s in stats.values()]
        neg_fracs = [s.negative_fraction for s in stats.values()]
        zero_fracs = [s.zeros_fraction for s in stats.values()]

        x = np.arange(len(names))
        width = 0.6
        ax4.bar(x, pos_fracs, width, label="Positive", alpha=0.7, color="green")
        ax4.bar(x, neg_fracs, width, bottom=pos_fracs, label="Negative", alpha=0.7, color="red")
        ax4.bar(
            x,
            zero_fracs,
            width,
            bottom=np.array(pos_fracs) + np.array(neg_fracs),
            label="Zero",
            alpha=0.7,
            color="gray",
        )
        ax4.set_ylabel("Fraction")
        ax4.set_title("Distribution Quality")
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3, axis="y")

        # 5-7. Individual histograms (bottom row)
        for idx, (name, values) in enumerate(list(history.items())[:3]):
            ax = fig.add_subplot(gs[2, idx])
            ax.hist(values, bins=30, alpha=0.7, edgecolor="black")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{name} Distribution")
            ax.grid(True, alpha=0.3)

        # Add info text
        info_text = f"Step: {reward_monitor.global_step} | Alerts: {len(reward_monitor.all_alerts)}"
        fig.text(0.99, 0.01, info_text, ha="right", va="bottom", fontsize=10, style="italic")

        fig.suptitle("Reward Monitoring Dashboard", fontsize=16, fontweight="bold")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_ema_comparison(
        self,
        reward_stats: Dict[str, Any],
        title: str = "EMA Comparison (Short vs Long)",
        save_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Plot comparison of short and long EMAs for detecting trends.

        Args:
            reward_stats: Dictionary of reward statistics
            title: Plot title
            save_path: Optional path to save figure
            show: Whether to show the plot
        """
        if not MATPLOTLIB_AVAILABLE or not reward_stats:
            return

        plt.style.use(self.style)
        fig, ax = plt.subplots(figsize=(10, 6))

        names = list(reward_stats.keys())
        short_emas = []
        long_emas = []

        for stats in reward_stats.values():
            if hasattr(stats, "ema_short"):
                short_emas.append(stats.ema_short)
                long_emas.append(stats.ema_long)
            else:
                short_emas.append(stats.get("ema_short", 0))
                long_emas.append(stats.get("ema_long", 0))

        x = np.arange(len(names))
        width = 0.35

        ax.bar(x - width / 2, long_emas, width, label="Long EMA", alpha=0.7, color="steelblue")
        ax.bar(x + width / 2, short_emas, width, label="Short EMA", alpha=0.7, color="coral")

        ax.set_ylabel("EMA Value")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Add divergence indicators
        for i, (short, long) in enumerate(zip(short_emas, long_emas)):
            if abs(short - long) > 0.1 * abs(long):  # >10% divergence
                color = "green" if short > long else "red"
                ax.plot([i], [max(short, long) * 1.05], marker="^", color=color, markersize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()


def create_monitoring_summary_html(
    reward_monitor,
    output_path: str = "monitoring_summary.html",
):
    """
    Create an HTML summary report of monitoring data.

    Args:
        reward_monitor: RewardMonitor instance
        output_path: Path to save HTML file
    """
    stats = reward_monitor.get_all_stats()
    alerts = reward_monitor.all_alerts

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reward Monitoring Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; }}
            table {{ border-collapse: collapse; width: 100%; background-color: white; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .alert {{ background-color: #fff3cd; padding: 10px; margin: 10px 0; border-left: 4px solid #ffc107; }}
            .alert.warning {{ border-left-color: #ff9800; }}
            .alert.error {{ border-left-color: #f44336; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <h1>Reward Monitoring Summary</h1>
        <p><strong>Training Step:</strong> {reward_monitor.global_step}</p>
        <p><strong>Total Alerts:</strong> {len(alerts)}</p>

        <h2>Reward Statistics</h2>
        <table>
            <tr>
                <th>Reward</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Max</th>
                <th>EMA Short</th>
                <th>EMA Long</th>
                <th>Samples</th>
            </tr>
    """

    for name, s in stats.items():
        html += f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td>{s.mean:.4f}</td>
                <td>{s.std:.4f}</td>
                <td>{s.min:.4f}</td>
                <td>{s.max:.4f}</td>
                <td>{s.ema_short:.4f}</td>
                <td>{s.ema_long:.4f}</td>
                <td>{s.count}</td>
            </tr>
        """

    html += """
        </table>

        <h2>Recent Alerts</h2>
    """

    recent_alerts = [a for a in alerts[-20:]]  # Last 20 alerts
    if recent_alerts:
        for alert in reversed(recent_alerts):
            alert_type = alert["message"].split(":")[0]
            alert_class = "warning" if alert_type in ["DROP", "SPIKE"] else "error"
            html += f"""
            <div class="alert {alert_class}">
                <strong>Step {alert['step']}:</strong> {alert['message']}
            </div>
            """
    else:
        html += "<p>No recent alerts.</p>"

    html += """
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html)

    print(f"✓ Monitoring summary saved to {output_path}")
