# Reward Monitoring and Observability

Comprehensive guide to monitoring reward signals during GRPO training.

## Quick Start

Add monitoring to your training script in 3 steps:

### 1. Setup Monitoring

```python
from tunrex.monitoring import setup_grpo_monitoring
from tunrex.datasets.rewards import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)

monitoring = setup_grpo_monitoring(
    reward_functions=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    use_wandb=True,
    output_dir="/tmp/reward_monitoring",
)
```

### 2. Use Wrapped Functions in GRPO

```python
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=monitoring.get_wrapped_reward_functions(),  # Use wrapped functions
    algo_config=grpo_config,
)
```

### 3. Update During Training

```python
for step in range(num_steps):
    # Training happens...
    monitoring.update_step(step)

# Finalize at the end
monitoring.finalize()
```

## What You Get

### Real-time Monitoring
- ✅ Individual reward signal tracking
- ✅ Statistical analysis (mean, std, percentiles)
- ✅ Moving averages for trend detection
- ✅ Automatic anomaly detection
- ✅ Execution time and error tracking

### W&B Integration
- ✅ Automatic metric logging
- ✅ Histogram visualization
- ✅ Alert tracking
- ✅ Distribution analysis

### Visual Reports
- ✅ Comprehensive dashboards
- ✅ Time series plots
- ✅ Distribution histograms
- ✅ HTML summary reports
- ✅ Statistical breakdowns

## Anomaly Detection

The system automatically detects and alerts on:

| Alert Type | Description | Threshold |
|------------|-------------|-----------|
| **OUTLIER** | Value far from mean | >3 std deviations |
| **DROP** | Sudden decrease in rewards | >50% drop in short EMA |
| **SPIKE** | Sudden increase in rewards | >2x increase in short EMA |
| **FLATLINE** | Reward stuck at zero | >95% zeros |
| **NEGATIVE** | Too many negative rewards | >90% negative |

## Metrics Logged to W&B

### Per-Reward Metrics
For each reward function (e.g., `format_exact`):
```
format_exact/mean           - Mean value
format_exact/std            - Standard deviation
format_exact/min            - Minimum value
format_exact/max            - Maximum value
format_exact/median         - Median value
format_exact/p25, p75, p90  - Percentiles
format_exact/ema_short      - Short-term moving average
format_exact/ema_long       - Long-term moving average
format_exact/histogram      - Distribution histogram
```

### Aggregate Metrics
```
rewards/total_mean          - Mean across all rewards
rewards/total_std           - Average standard deviation
rewards/sum                 - Sum of mean rewards
```

### Execution Metrics
```
reward_fn/format_exact/calls      - Number of calls
reward_fn/format_exact/errors     - Number of errors
reward_fn/format_exact/avg_time_ms - Avg execution time
```

### Alert Metrics
```
alerts/OUTLIER              - Count of outlier alerts
alerts/DROP                 - Count of drop alerts
alerts/SPIKE                - Count of spike alerts
alerts/total                - Total alerts
```

## Output Files

Generated during training:
```
output_dir/
├── dashboard_step_100.png        # Dashboard at step 100
├── dashboard_step_500.png        # Dashboard at step 500
├── summary_step_100.html         # HTML summary at step 100
├── final_dashboard.png           # Final comprehensive dashboard
├── final_summary.html            # Final HTML report
├── reward_history.png            # Time series of rewards
├── reward_distributions.png      # Distribution histograms
├── reward_statistics.png         # Statistical summary
└── monitoring_data.json          # Complete data export
```

## Configuration

### Basic Configuration

```python
monitoring = setup_grpo_monitoring(
    reward_functions=my_reward_fns,
    use_wandb=True,              # Enable W&B logging
    enable_anomaly_detection=True, # Enable alerts
    log_frequency=10,            # Log every 10 steps
    summary_frequency=100,       # Print summary every 100 steps
    visualization_frequency=500, # Create plots every 500 steps
    output_dir="/tmp/monitoring", # Where to save files
    verbose=True,                # Print monitoring info
)
```

### Advanced Configuration

```python
from tunrex.monitoring import GRPOMonitoringIntegration

monitoring = GRPOMonitoringIntegration(
    reward_functions=my_reward_fns,
    reward_names=["custom_name_1", "custom_name_2"],  # Custom names
    use_wandb=True,
    enable_anomaly_detection=True,
    log_frequency=10,
    summary_frequency=100,
    visualization_frequency=500,
    output_dir="/tmp/monitoring",
    verbose=True,
)

# Access individual components
monitor = monitoring.reward_monitor
logger = monitoring.metrics_logger
visualizer = monitoring.visualizer
```

## Interpreting Results

### Healthy Training

✅ **Good indicators:**
- Mean rewards steady or increasing
- Low standard deviation (< 1.0)
- EMAs trending together
- Few anomaly alerts
- High positive fraction

### Warning Signs

⚠️ **Potential issues:**

**Flatline (all zeros)**
- Reward function not triggering
- Check reward logic and thresholds
- Verify model output format

**High negative fraction**
- Model performing poorly
- May need hyperparameter tuning
- Consider reward shaping

**Large EMA divergence**
- Training unstable
- Short >> Long: Recent improvement (good)
- Short << Long: Recent degradation (bad)

**Frequent outliers**
- Noisy reward signal
- Consider reward normalization
- Check for bugs in reward functions

**High error rates**
- Implementation issues
- Verify input data format
- Check reward function logic

## Examples

### Example 1: Training with Monitoring

```python
from tunrex.monitoring import setup_grpo_monitoring

# Setup
monitoring = setup_grpo_monitoring(
    reward_functions=[match_format_exactly, check_answer],
    use_wandb=True,
)

# Train
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=monitoring.get_wrapped_reward_functions(),
    algo_config=grpo_config,
)

grpo_trainer.train(train_dataset, val_dataset)

# Finalize
monitoring.finalize()
```

### Example 2: Manual Monitoring

```python
from tunrex.monitoring import RewardMonitor, RewardMetricsLogger

monitor = RewardMonitor(reward_names=["my_reward"])
logger = RewardMetricsLogger(use_wandb=True)

for step in range(num_steps):
    reward_value = compute_reward()

    stats = monitor.update({"my_reward": reward_value}, step=step)
    logger.log_step(monitor.get_metrics_dict(), step=step)

    if monitor.all_alerts:
        print(f"Alert: {monitor.all_alerts[-1]['message']}")
```

### Example 3: Offline Analysis

```python
import json
from tunrex.monitoring import RewardVisualizer

# Load saved monitoring data
with open("monitoring_data.json") as f:
    data = json.load(f)

# Create visualizations
viz = RewardVisualizer()
viz.plot_reward_history(data["history"], save_path="analysis.png")
```

## Troubleshooting

### Issue: No W&B metrics

**Solution:**
1. Ensure `wandb.init()` called before monitoring setup
2. Set `WANDB_API_KEY` environment variable
3. Check `use_wandb=True` in configuration

### Issue: Visualization errors

**Solution:**
1. Install matplotlib: `pip install matplotlib`
2. Check output directory permissions
3. Verify disk space available

### Issue: Out of memory

**Solution:**
1. Reduce window size: `RewardMonitor(window_size=500)`
2. Increase log frequency: `log_frequency=50`
3. Disable histograms: `log_histograms=False`

### Issue: Missing reward values

**Solution:**
1. Ensure functions wrapped before GRPO trainer
2. Verify functions return numeric values
3. Call `update_step()` after reward computation

## Best Practices

1. **Enable early**: Add monitoring from the start of training
2. **Monitor alerts**: Don't ignore anomaly warnings
3. **Tune thresholds**: Adjust detection for your reward scale
4. **Save data**: Keep JSON exports for later analysis
5. **Use W&B tags**: Tag runs for easy comparison
6. **Check dashboards**: Review visualizations periodically

## Performance Impact

Monitoring overhead is minimal:
- CPU: ~1-3% additional compute
- Memory: ~10-50MB (depends on window size)
- Disk: ~10-100MB (depends on visualization frequency)
- Network: Batched W&B uploads

To minimize overhead:
- Use `log_frequency=50` or higher
- Reduce `window_size=500`
- Set `visualization_frequency=1000+`
- Disable histograms if not needed

## Full Documentation

See [TunRex/src/tunrex/monitoring/README.md](../TunRex/src/tunrex/monitoring/README.md) for complete API documentation.

## Example Output

### Console Output
```
================================================================================
Reward Monitoring Summary (Step 100)
================================================================================

format_exact:
  Mean: 2.4500 ± 0.9800
  Range: [0.0000, 3.0000]
  Median: 3.0000
  Percentiles: P25=0.0000, P75=3.0000, P90=3.0000
  EMAs: Short=2.5200, Long=2.4800
  Distribution: 68.0% positive, 0.0% negative, 32.0% zero
  Samples: 100

⚠️  Step 50: OUTLIER: answer value -10.000 is 5.2 std deviations from mean 2.100
⚠️  Step 105: FLATLINE: format_exact is returning zero 95.0% of the time
================================================================================
```

### W&B Dashboard
![W&B Dashboard](https://via.placeholder.com/800x400?text=W%26B+Reward+Monitoring+Dashboard)

### HTML Report
Interactive HTML report with:
- Summary statistics table
- Recent alerts with details
- Trend indicators
- Export links

## Additional Resources

- [Example Integration Script](../examples/monitoring_integration_example.py)
- [Training Script](../scripts/train_grpo.py)
- [API Documentation](../TunRex/src/tunrex/monitoring/README.md)
