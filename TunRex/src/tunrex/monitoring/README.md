# TunRex Monitoring System

Comprehensive monitoring and observability for reward signals during GRPO training.

## Overview

The TunRex monitoring system provides real-time tracking, analysis, and visualization of reward signals during training. It helps identify issues early, understand reward dynamics, and ensure training progresses correctly.

## Features

### 🎯 Core Monitoring Capabilities

- **Individual Reward Tracking**: Monitor each reward function separately
- **Statistical Analysis**: Mean, std, min, max, percentiles, and moving averages
- **Anomaly Detection**: Automatic detection of outliers, drops, spikes, and flatlines
- **Distribution Analysis**: Histograms and distribution quality metrics
- **Execution Monitoring**: Track reward function call counts, errors, and timing
- **W&B Integration**: Automatic logging to Weights & Biases
- **Visualizations**: Dashboards, plots, and HTML reports

### 📊 Tracked Metrics

For each reward function:
- Mean, median, standard deviation
- Min/max values and ranges
- Percentiles (P25, P75, P90, P99)
- Exponential moving averages (short and long term)
- Distribution breakdown (positive/negative/zero fractions)
- Call counts and error rates
- Execution time statistics

### 🚨 Anomaly Detection

Automatically detects:
- **Outliers**: Values beyond 3 standard deviations
- **Drops**: Sudden decreases in moving averages
- **Spikes**: Sudden increases in moving averages
- **Flatlines**: Reward functions returning zero consistently
- **High Negative Rate**: Excessive negative rewards

## Quick Start

### Basic Usage

```python
from tunrex.datasets.rewards import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)
from tunrex.monitoring import setup_grpo_monitoring

# Create monitoring integration
monitoring = setup_grpo_monitoring(
    reward_functions=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    use_wandb=True,
    enable_anomaly_detection=True,
    output_dir="/tmp/reward_monitoring",
)

# Get wrapped reward functions to pass to GRPO trainer
wrapped_rewards = monitoring.get_wrapped_reward_functions()

# Pass to GRPO trainer
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=wrapped_rewards,  # Use wrapped functions
    algo_config=grpo_config,
)

# During training loop, update monitoring
for step in range(num_steps):
    # Training happens...

    # Update monitoring (automatically extracts reward values)
    monitoring.update_step(step)

# Finalize and create reports
monitoring.finalize()
```

### Integration with Existing Training Script

To integrate monitoring into an existing GRPO training script:

1. **Import monitoring**:
```python
from tunrex.monitoring import setup_grpo_monitoring
```

2. **Wrap reward functions** (before creating GRPOLearner):
```python
# Original reward functions
from tunrex.datasets.rewards import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)

# Setup monitoring
monitoring = setup_grpo_monitoring(
    reward_functions=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    use_wandb=True,  # Enable W&B logging
    log_frequency=10,  # Log every 10 steps
    summary_frequency=100,  # Print summary every 100 steps
    visualization_frequency=500,  # Create plots every 500 steps
)

# Get wrapped functions
wrapped_rewards = monitoring.get_wrapped_reward_functions()
```

3. **Use wrapped functions** in GRPOLearner:
```python
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=wrapped_rewards,  # Use wrapped functions instead
    algo_config=grpo_config,
)
```

4. **Update during training**:
```python
# After each training step
monitoring.update_step(current_step)
```

5. **Finalize when done**:
```python
# At the end of training
monitoring.finalize()
```

## Components

### RewardMonitor

Core monitoring class that tracks reward statistics and detects anomalies.

```python
from tunrex.monitoring import RewardMonitor

monitor = RewardMonitor(
    reward_names=["format", "accuracy", "numbers"],
    window_size=1000,
    enable_anomaly_detection=True,
)

# Update with new values
stats = monitor.update({
    "format": 2.5,
    "accuracy": 3.0,
    "numbers": 1.5,
}, step=100)

# Get summary
print(monitor.get_summary())

# Get metrics for logging
metrics = monitor.get_metrics_dict()
```

### RewardMetricsLogger

Handles logging to W&B and TensorBoard with histogram support.

```python
from tunrex.monitoring import RewardMetricsLogger

logger = RewardMetricsLogger(
    use_wandb=True,
    log_histograms=True,
    histogram_bins=50,
)

# Log batch of rewards
logger.log_reward_batch({
    "format": [2.0, 2.5, 3.0],
    "accuracy": [3.0, 3.0, 1.5],
})

# Log distributions
logger.log_distributions(step=100)
```

### RewardFunctionWrapper

Wraps reward functions to track execution statistics.

```python
from tunrex.monitoring import RewardFunctionWrapper

wrapped_fn = RewardFunctionWrapper(
    reward_fn=check_answer,
    name="answer_checker",
    track_timing=True,
)

# Use as normal function
result = wrapped_fn(model_output, ground_truth)

# Get statistics
stats = wrapped_fn.get_stats()
print(f"Called {stats['calls']} times, {stats['errors']} errors")
print(f"Average time: {stats['avg_time']*1000:.2f}ms")
```

### RewardVisualizer

Creates plots and dashboards for reward analysis.

```python
from tunrex.monitoring import RewardVisualizer

viz = RewardVisualizer()

# Plot reward history
viz.plot_reward_history(
    reward_history={
        "format": [2.0, 2.5, 3.0, 2.8],
        "accuracy": [3.0, 3.0, 1.5, 3.0],
    },
    save_path="reward_history.png",
)

# Plot distributions
viz.plot_reward_distributions(
    reward_history=reward_data,
    save_path="distributions.png",
)

# Create full dashboard
viz.plot_reward_dashboard(
    reward_monitor=monitor,
    save_path="dashboard.png",
)
```

## Configuration Options

### GRPOMonitoringIntegration

```python
monitoring = GRPOMonitoringIntegration(
    reward_functions=reward_fns,
    reward_names=None,  # Auto-detect from function names
    use_wandb=True,  # Log to W&B
    enable_anomaly_detection=True,  # Enable anomaly detection
    log_frequency=10,  # Log metrics every N steps
    summary_frequency=100,  # Print summary every N steps
    visualization_frequency=500,  # Create plots every N steps
    output_dir="/tmp/reward_monitoring",  # Output directory
    verbose=True,  # Print monitoring info
)
```

### RewardMonitor

```python
monitor = RewardMonitor(
    reward_names=["reward1", "reward2"],
    window_size=1000,  # Rolling window size
    ema_alpha_short=0.1,  # Short-term EMA smoothing (~10 samples)
    ema_alpha_long=0.01,  # Long-term EMA smoothing (~100 samples)
    enable_anomaly_detection=True,
    verbose=True,
)
```

### RewardAnomalyDetector

```python
detector = RewardAnomalyDetector(
    window_size=100,
    std_threshold=3.0,  # Z-score threshold for outliers
    drop_threshold=0.5,  # Fraction drop to trigger alert (50%)
    spike_threshold=2.0,  # Multiplier for spike detection (2x)
)
```

## Output Files

The monitoring system generates several output files:

### Regular Outputs (during training)

- `dashboard_step_N.png`: Monitoring dashboard at step N
- `summary_step_N.html`: HTML summary report at step N

### Final Outputs (at end of training)

- `final_dashboard.png`: Final comprehensive dashboard
- `final_summary.html`: Final HTML report with all statistics
- `reward_history.png`: Time series plot of all rewards
- `reward_distributions.png`: Distribution histograms
- `reward_statistics.png`: Statistical summary plots
- `monitoring_data.json`: Complete monitoring data for analysis

## W&B Metrics

When W&B logging is enabled, the following metrics are logged:

### Per-Reward Metrics

For each reward function `<name>`:
- `<name>/mean`: Mean value
- `<name>/std`: Standard deviation
- `<name>/min`: Minimum value
- `<name>/max`: Maximum value
- `<name>/median`: Median value
- `<name>/p25, p75, p90, p99`: Percentiles
- `<name>/ema_short`: Short-term moving average
- `<name>/ema_long`: Long-term moving average
- `<name>/zeros_fraction`: Fraction of zero values
- `<name>/negative_fraction`: Fraction of negative values
- `<name>/positive_fraction`: Fraction of positive values
- `<name>/histogram`: Distribution histogram

### Aggregate Metrics

- `rewards/total_mean`: Mean across all rewards
- `rewards/total_std`: Average standard deviation
- `rewards/sum`: Sum of all mean rewards
- `rewards/min_mean`: Minimum mean reward
- `rewards/max_mean`: Maximum mean reward

### Function Execution Metrics

For each reward function `<name>`:
- `reward_fn/<name>/calls`: Number of calls
- `reward_fn/<name>/errors`: Number of errors
- `reward_fn/<name>/error_rate`: Error rate (0-1)
- `reward_fn/<name>/avg_time_ms`: Average execution time (ms)
- `reward_fn/<name>/max_time_ms`: Maximum execution time (ms)

### Alert Metrics

- `alerts/<TYPE>`: Count of each alert type (OUTLIER, DROP, SPIKE, etc.)
- `alerts/total`: Total number of alerts
- `monitoring/alert_count`: Recent alert count
- `monitoring/total_alerts`: Cumulative alert count

## Advanced Usage

### Custom Reward Monitoring

For fine-grained control:

```python
from tunrex.monitoring import (
    RewardMonitor,
    RewardMetricsLogger,
    RewardFunctionWrapper,
)

# Setup components separately
monitor = RewardMonitor(reward_names=["my_reward"])
logger = RewardMetricsLogger(use_wandb=True)
wrapper = RewardFunctionWrapper(my_reward_fn, name="my_reward")

# Manual monitoring loop
for step in range(num_steps):
    # Get reward value
    reward_value = wrapper(model_output, ground_truth)

    # Update monitor
    stats = monitor.update({"my_reward": reward_value}, step=step)

    # Log to W&B
    logger.log_step(monitor.get_metrics_dict(), step=step)

    # Check for alerts
    if monitor.all_alerts:
        print(f"Alert: {monitor.all_alerts[-1]['message']}")
```

### Creating Custom Visualizations

```python
from tunrex.monitoring import RewardVisualizer

viz = RewardVisualizer()

# Access monitoring data
history = monitor.reward_history
stats = monitor.get_all_stats()

# Create custom plots
viz.plot_ema_comparison(stats, save_path="ema_comparison.png")
viz.plot_reward_statistics(stats, save_path="statistics.png")
```

### Exporting Data for Analysis

```python
import json

# Export monitoring data
data = {
    "stats": {name: s.to_dict() for name, s in monitor.stats.items()},
    "alerts": monitor.all_alerts,
    "history": {name: list(vals) for name, vals in monitor.reward_history.items()},
}

with open("monitoring_export.json", "w") as f:
    json.dump(data, f, indent=2)
```

## Interpreting Results

### Healthy Reward Signals

✅ **Good signs:**
- Steady or increasing mean values
- Low standard deviation (stable rewards)
- High positive fraction
- Short and long EMAs trending together
- Few or no anomaly alerts

### Warning Signs

⚠️ **Potential issues:**
- **Flatlines (high zeros_fraction)**: Reward function not triggering
  - Check reward logic and thresholds
  - Verify model outputs match expected format

- **High negative fraction**: Model performing poorly
  - May need hyperparameter adjustment
  - Check if model is learning

- **Large EMA divergence**: Unstable training
  - Short EMA >> Long EMA: Recent spike (could be good or bad)
  - Short EMA << Long EMA: Recent drop (concerning)

- **High error rates**: Implementation issues
  - Check reward function for bugs
  - Verify input data format

- **Frequent outliers**: Inconsistent rewards
  - May indicate noisy reward signal
  - Consider reward normalization

## Troubleshooting

### No metrics logged to W&B

1. Check W&B is initialized: `wandb.init()` called before monitoring setup
2. Verify WANDB_API_KEY environment variable is set
3. Ensure `use_wandb=True` in monitoring setup
4. Check console for W&B warnings

### Visualization errors

1. Install matplotlib: `pip install matplotlib`
2. Check output directory is writable
3. Verify sufficient disk space

### Memory issues with large windows

- Reduce `window_size` in RewardMonitor (default: 1000)
- Increase `log_frequency` to log less often
- Disable histograms: `log_histograms=False`

### Missing reward values

- Ensure reward functions are wrapped before passing to GRPO
- Check reward functions return numeric values
- Verify `update_step()` is called after reward computation

## Best Practices

1. **Enable monitoring from the start**: Easier to debug early issues
2. **Set appropriate frequencies**: Balance detail vs. performance
   - Log frequently for debugging: `log_frequency=1`
   - Log less often for long runs: `log_frequency=10-50`
3. **Monitor alerts**: Don't ignore anomaly warnings
4. **Save monitoring data**: Keep JSON exports for later analysis
5. **Review visualizations**: Check dashboards periodically
6. **Tune detection thresholds**: Adjust for your specific reward scale
7. **Use W&B tags**: Tag runs with monitoring info for comparison

## Examples

See the training script for a complete integration example:
```bash
python scripts/train_grpo.py --num-steps 1000 --wandb-project my-project
```

## API Reference

### Main Classes

- `GRPOMonitoringIntegration`: All-in-one monitoring setup
- `RewardMonitor`: Core reward tracking and statistics
- `RewardMetricsLogger`: W&B and TensorBoard logging
- `RewardFunctionWrapper`: Wrap individual reward functions
- `RewardFunctionMonitor`: Manage multiple wrapped functions
- `RewardVisualizer`: Create plots and visualizations
- `RewardStats`: Statistics dataclass
- `RewardAnomalyDetector`: Anomaly detection

### Helper Functions

- `setup_grpo_monitoring()`: Quick setup helper
- `create_monitoring_summary_html()`: Generate HTML reports

## Performance Considerations

- Monitoring overhead: ~1-5% (mostly from logging and visualization)
- Memory usage: O(window_size × num_rewards)
- Disk usage: Depends on visualization frequency
- W&B network: Batched uploads every log_frequency steps

To minimize overhead:
- Use larger `log_frequency` (e.g., 50-100)
- Reduce `window_size` (e.g., 500)
- Disable histograms for very frequent logging
- Reduce `visualization_frequency` (e.g., 1000+)

## License

Part of the TunRex dataset and reward framework.
