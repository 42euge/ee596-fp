"""
TensorBoard Configuration and Custom Visualizations

Provides enhanced TensorBoard logging with custom layouts and visualizations.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional


def create_tensorboard_layout_config(log_dir: str):
    """
    Create custom TensorBoard layout configuration.

    This creates a layout configuration that organizes metrics into
    meaningful groups for easier analysis.

    Args:
        log_dir: TensorBoard log directory
    """
    layout = {
        "version": 0,
        "charts": {
            "Reward Overview": {
                "type": "line",
                "series": [
                    {"tag": "rewards/mean"},
                    {"tag": "rewards/max"},
                    {"tag": "rewards/min"},
                ]
            },
            "Reward Functions": {
                "type": "line",
                "series": [
                    {"tag": "rewards/functions/format_exact"},
                    {"tag": "rewards/functions/format_approx"},
                    {"tag": "rewards/functions/check_answer"},
                    {"tag": "rewards/functions/check_numbers"},
                ]
            },
            "Quality Metrics": {
                "type": "line",
                "series": [
                    {"tag": "rewards/signal_to_noise"},
                    {"tag": "rewards/entropy"},
                ]
            },
            "Training Dynamics": {
                "type": "line",
                "series": [
                    {"tag": "training/kl_divergence"},
                    {"tag": "training/policy_entropy"},
                ]
            },
            "Contributions": {
                "type": "bar",
                "series": [
                    {"tag": "contributions/format_exact"},
                    {"tag": "contributions/format_approx"},
                    {"tag": "contributions/check_answer"},
                    {"tag": "contributions/check_numbers"},
                ]
            }
        }
    }

    # Write layout config
    layout_file = Path(log_dir) / "layout.json"
    os.makedirs(log_dir, exist_ok=True)

    with open(layout_file, 'w') as f:
        json.dump(layout, f, indent=2)

    print(f"✓ TensorBoard layout config created: {layout_file}")

    return layout


def create_tensorboard_readme(log_dir: str):
    """
    Create README for TensorBoard logs.

    Args:
        log_dir: TensorBoard log directory
    """
    readme_content = """# GRPO Reward Monitoring TensorBoard Logs

This directory contains TensorBoard logs for GRPO training with enhanced reward monitoring.

## Viewing the Dashboard

Run TensorBoard with:

```bash
tensorboard --logdir={log_dir}
```

Then open http://localhost:6006 in your browser.

## Metric Groups

### Reward Overview
- `rewards/mean`: Mean reward across all functions
- `rewards/std`: Standard deviation of rewards
- `rewards/min`: Minimum reward
- `rewards/max`: Maximum reward
- `rewards/total`: Total cumulative reward

### Reward Functions
Individual reward function values:
- `rewards/functions/format_exact`: Exact format match reward
- `rewards/functions/format_approx`: Approximate format match reward
- `rewards/functions/check_answer`: Answer correctness reward
- `rewards/functions/check_numbers`: Numerical extraction reward

### Quality Metrics
Signal quality indicators:
- `rewards/signal_to_noise`: Signal-to-noise ratio (higher is better)
- `rewards/entropy`: Reward distribution entropy
- `rewards/variance`: Reward variance

### Training Dynamics
GRPO training metrics:
- `training/kl_divergence`: KL divergence from reference policy
- `training/policy_entropy`: Policy distribution entropy
- `training/advantage_mean`: Mean advantage estimate
- `training/advantage_std`: Advantage standard deviation

### Contributions
Percentage contribution of each reward function:
- `contributions/format_exact`: % of total reward
- `contributions/format_approx`: % of total reward
- `contributions/check_answer`: % of total reward
- `contributions/check_numbers`: % of total reward

## Tips

1. Use the **Smoothing** slider to reduce noise in charts
2. Compare multiple runs by selecting them in the left panel
3. Use **Download** buttons to export charts as images or data
4. Check the **Distributions** tab for reward histograms
5. Use **Time** view to see real-time training progress

## Custom Layouts

A custom layout configuration is provided in `layout.json` that organizes
metrics into logical groups for easier analysis.
""".format(log_dir=log_dir)

    readme_file = Path(log_dir) / "README.md"

    with open(readme_file, 'w') as f:
        f.write(readme_content)

    print(f"✓ TensorBoard README created: {readme_file}")


def setup_tensorboard_monitoring(log_dir: str = "/tmp/tensorboard/grpo_rewards"):
    """
    Setup TensorBoard monitoring with custom configuration.

    Args:
        log_dir: TensorBoard log directory

    Returns:
        Dict with setup information
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create layout config
    create_tensorboard_layout_config(log_dir)

    # Create README
    create_tensorboard_readme(log_dir)

    return {
        'log_dir': log_dir,
        'tensorboard_command': f"tensorboard --logdir={log_dir}",
        'url': 'http://localhost:6006',
    }


if __name__ == "__main__":
    # Setup TensorBoard monitoring
    info = setup_tensorboard_monitoring()
    print("\n" + "="*60)
    print("TensorBoard Monitoring Setup Complete")
    print("="*60)
    print(f"\nLog directory: {info['log_dir']}")
    print(f"\nTo view dashboard, run:")
    print(f"  {info['tensorboard_command']}")
    print(f"\nThen open: {info['url']}")
    print("="*60)
