# Reward Monitoring System Documentation

## Overview

The GRPO Reward Monitoring System provides comprehensive visibility into reward signal quality across training runs. It helps researchers understand how different reward functions contribute to model training, detect anomalies, and diagnose training issues.

## Features

### 🎯 Core Capabilities

- **Per-Function Tracking**: Monitor each reward function individually
- **Quality Metrics**: Signal-to-noise ratio, entropy, variance analysis
- **Multi-Backend Support**: W&B, TensorBoard, JSON export, Web dashboard
- **Real-time Alerting**: Threshold and trend-based alerts
- **Statistical Analysis**: Comprehensive trend analysis and correlation studies
- **Reward Hacking Detection**: Identify when rewards increase but performance doesn't

### 📊 Visualization Options

1. **Weights & Biases (W&B)**: Cloud-based experiment tracking with custom charts
2. **TensorBoard**: Local visualization with custom layouts
3. **Web Dashboard**: Interactive Streamlit dashboard with Plotly charts
4. **JSON Export**: Raw data for custom analysis

---

## Quick Start

### 1. Installation

Install required dependencies:

```bash
# Core monitoring dependencies
pip install numpy scipy

# W&B (optional)
pip install wandb

# TensorBoard (optional)
pip install tensorboard torch

# Web dashboard (optional)
pip install streamlit plotly pandas
```

### 2. Basic Usage

Run training with monitoring:

```bash
# With W&B and TensorBoard
python scripts/train_grpo_monitored.py \
    --num-steps 100 \
    --model-id google/gemma-3-1b-it \
    --enable-alerts

# Disable specific backends
python scripts/train_grpo_monitored.py \
    --num-steps 100 \
    --no-wandb \
    --no-tensorboard
```

### 3. View Dashboards

**TensorBoard:**
```bash
tensorboard --logdir=/tmp/tensorboard/grpo_rewards
# Open http://localhost:6006
```

**Web Dashboard:**
```bash
streamlit run src/monitoring/web_dashboard.py -- --data-dir ./monitoring_data
# Open http://localhost:8501
```

**W&B:**
- Access your run at: https://wandb.ai/your-username/tunix-grpo-monitored

---

## Architecture

### Components

```
monitoring/
├── __init__.py              # Package exports
├── metrics_tracker.py       # Core metrics tracking
├── dashboard_manager.py     # Multi-backend dashboard management
├── reward_analyzer.py       # Statistical analysis
├── realtime_monitor.py      # Real-time alerting
├── tensorboard_config.py    # TensorBoard setup
└── web_dashboard.py         # Streamlit dashboard
```

### Data Flow

```
Training Loop
    ↓
Reward Functions
    ↓
RewardMetricsTracker ─→ Per-step metrics
    ↓                   ├─ Mean, std, min, max
    ├─────────────────→ ├─ Per-function scores
    │                   ├─ Quality indicators (SNR, entropy)
    │                   └─ Training dynamics (KL, advantages)
    ↓
DashboardManager
    ├─→ W&B (wandb.log)
    ├─→ TensorBoard (SummaryWriter)
    └─→ JSON Export
         ↓
    Web Dashboard (Streamlit)
```

---

## Metrics Reference

### Aggregate Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| `rewards/mean` | Mean reward across all functions | -∞ to +∞ |
| `rewards/std` | Standard deviation of rewards | 0 to +∞ |
| `rewards/min` | Minimum reward in batch | -∞ to +∞ |
| `rewards/max` | Maximum reward in batch | -∞ to +∞ |
| `rewards/total` | Sum of all rewards | -∞ to +∞ |

### Quality Indicators

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `rewards/signal_to_noise` | \|mean\| / std | Higher = better signal quality |
| `rewards/entropy` | Distribution entropy | Higher = more uniform distribution |
| `rewards/variance` | Reward variance | Track for stability |

### Per-Function Metrics

For each reward function (e.g., `format_exact`, `check_answer`):

- `rewards/functions/{name}`: Mean reward value
- `rewards/functions/{name}_std`: Standard deviation
- `contributions/{name}`: Percentage contribution to total reward

### Training Dynamics

| Metric | Description |
|--------|-------------|
| `training/kl_divergence` | KL divergence from reference policy |
| `training/policy_entropy` | Entropy of policy distribution |
| `training/advantage_mean` | Mean advantage estimate |
| `training/advantage_std` | Advantage standard deviation |

---

## Reward Functions

The system tracks four reward functions:

### 1. Format Exact (`format_exact`)
- **Range**: 0 or 3.0
- **Purpose**: Strict format validation
- **Checks**: Proper `<reasoning>` and `<answer>` tags

### 2. Format Approximate (`format_approx`)
- **Range**: -2.5 to +2.5
- **Purpose**: Partial credit for format
- **Checks**: Tag presence, counts, positioning

### 3. Check Answer (`check_answer`)
- **Range**: -1.0 to +3.0
- **Purpose**: Answer correctness
- **Scoring**:
  - Exact match: 3.0
  - Normalized match: 1.5
  - Within 10% ratio: 0.5
  - Within 20% ratio: 0.25
  - Wrong: -1.0

### 4. Check Numbers (`check_numbers`)
- **Range**: 0 or 1.5
- **Purpose**: Numerical extraction
- **Checks**: First number extracted matches expected

---

## Analysis Tools

### RewardAnalyzer

Provides comprehensive quality analysis:

```python
from src.monitoring import RewardAnalyzer

analyzer = RewardAnalyzer()

# Analyze reward quality
report = analyzer.analyze_quality(reward_history, steps)

print(f"Quality Score: {report.quality_score:.1f}/100")
print(f"Issues: {report.issues}")
print(f"Recommendations: {report.recommendations}")

# Analyze function importance
importance = analyzer.analyze_function_importance(reward_history, function_names)

for func, metrics in importance.items():
    print(f"{func}: {metrics['contribution']:.1f}% contribution")

# Detect reward hacking
hacking_report = analyzer.detect_reward_hacking(
    reward_history,
    function_names,
    performance_metrics=accuracy_over_time
)
```

### Quality Dimensions

1. **Consistency** (25%): How stable rewards are over time
2. **Discriminability** (35%): How well rewards separate good/bad outputs
3. **Stability** (25%): How stable the variance is
4. **Coverage** (15%): How much of reward range is used

### Real-time Alerting

```python
from src.monitoring import RealtimeMonitor, AlertLevel

monitor = RealtimeMonitor(print_alerts=True)

# Add custom alerts
monitor.add_threshold_alert(
    metric_name='signal_to_noise',
    threshold=0.5,
    condition='less',
    level=AlertLevel.WARNING
)

monitor.add_trend_alert(
    metric_name='mean_reward',
    min_slope=-0.01,  # Alert if declining
    window_size=10,
    level=AlertLevel.WARNING
)

# Check metrics each step
monitor.check_metrics(step, metrics_dict)

# Get alert summary
summary = monitor.get_alert_summary()
```

---

## Command-Line Options

### Training Script Arguments

```bash
python scripts/train_grpo_monitored.py [OPTIONS]
```

**Monitoring Options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--monitoring-dir` | `./monitoring_data` | Directory for JSON exports |
| `--tensorboard-dir` | `/tmp/tensorboard/grpo_rewards` | TensorBoard log directory |
| `--no-tensorboard` | False | Disable TensorBoard |
| `--no-wandb` | False | Disable W&B |
| `--enable-alerts` | False | Enable real-time alerting |
| `--log-distributions` | False | Log full reward distributions |

**Example:**

```bash
python scripts/train_grpo_monitored.py \
    --num-steps 500 \
    --model-id google/gemma-3-1b-it \
    --use-lora \
    --lora-rank 64 \
    --enable-alerts \
    --log-distributions \
    --monitoring-dir ./my_experiment/monitoring \
    --wandb-project my-grpo-experiments
```

---

## Dashboard Guides

### W&B Dashboard

**Custom Charts Available:**

1. **Reward Breakdown**: Per-function rewards over time
2. **Quality Metrics**: SNR, entropy, variance trends
3. **Training Dynamics**: KL divergence and policy entropy
4. **Anomaly Detection**: Flagged anomalies
5. **Contributions**: Reward function contribution percentages

**Tips:**
- Use grouping to compare multiple runs
- Create custom panels for specific analyses
- Export charts as PNG or SVG
- Use W&B Reports for sharing results

### TensorBoard Dashboard

**Layout Structure:**

```
Reward Signals/
├── Total Rewards (mean, total, max)
├── Per-Function Rewards (all 4 functions)
└── Quality Metrics (SNR, entropy, variance)

Training Dynamics/
├── Policy Metrics (KL, entropy)
└── Advantages (mean, std)

Contributions/
└── Reward Contributions (%) by function
```

**Tips:**
- Adjust smoothing slider for clearer trends
- Use scalar comparison for multi-run analysis
- Check Distributions tab for histograms
- Toggle runs in left sidebar for comparison

### Web Dashboard (Streamlit)

**Features:**

1. **Summary Statistics**: Key metrics at a glance
2. **Reward Overview**: 4-panel overview chart
3. **Function Breakdown**: Per-function analysis
4. **Training Dynamics**: KL, entropy, advantages
5. **Quality Metrics**: SNR, entropy, variance
6. **Raw Data Table**: Full metrics export
7. **JSON Summary**: Detailed analysis

**Running:**

```bash
streamlit run src/monitoring/web_dashboard.py -- --data-dir ./monitoring_data
```

**Customization:**

Edit `src/monitoring/web_dashboard.py` to:
- Add new chart types
- Modify color schemes
- Add custom analyses
- Change layout structure

---

## Programmatic API

### RewardMetricsTracker

```python
from src.monitoring import RewardMetricsTracker

tracker = RewardMetricsTracker(
    reward_function_names=['func1', 'func2'],
    window_size=100,
    track_distributions=True
)

# Track a training step
metrics = tracker.track_step(
    step=42,
    rewards_by_function={
        'func1': np.array([1.0, 2.0, 1.5]),
        'func2': np.array([0.5, 0.7, 0.6]),
    },
    kl_divergence=0.05,
    policy_entropy=2.3,
    batch_size=3,
    num_generations=1
)

# Get statistics
summary = tracker.get_summary()
trends = tracker.get_recent_trends(num_steps=50)
func_stats = tracker.get_function_statistics('func1')
```

### DashboardManager

```python
from src.monitoring import DashboardManager

with DashboardManager(
    project_name='my-project',
    enable_wandb=True,
    enable_tensorboard=True,
    json_export_dir='./data'
) as dashboard:

    # Define custom charts
    dashboard.define_wandb_charts(['func1', 'func2'])

    # Log metrics
    dashboard.log_metrics(step, metrics)

    # Log distributions
    dashboard.log_reward_distributions(step, metrics)

    # Log summary
    dashboard.log_summary(tracker, step)
```

---

## Best Practices

### 1. Choosing Backends

| Use Case | Recommended Backends |
|----------|---------------------|
| Quick experiments | TensorBoard + JSON |
| Team collaboration | W&B |
| Offline analysis | JSON + Web Dashboard |
| Real-time monitoring | W&B + Alerts |
| Publication figures | All (highest quality exports) |

### 2. Monitoring Frequency

- **Real-time training**: Log every step
- **Long training runs**: Log every 5-10 steps
- **Memory constraints**: Disable `--log-distributions`

### 3. Alert Configuration

```python
# Conservative alerts (fewer false positives)
monitor.add_threshold_alert('signal_to_noise', threshold=0.3, condition='less')

# Aggressive alerts (catch issues early)
monitor.add_threshold_alert('signal_to_noise', threshold=0.7, condition='less')
monitor.add_trend_alert('mean_reward', min_slope=0, window_size=20)
```

### 4. Storage Considerations

**Disk Usage:**
- JSON exports: ~10KB per step
- TensorBoard: ~5KB per step
- Full distributions: +50KB per step

**Recommendations:**
- Use `--log-distributions` only when needed
- Clean up old monitoring data: `rm -rf monitoring_data/step_*.json`
- Archive completed runs: `tar -czf run_001.tar.gz monitoring_data/`

---

## Troubleshooting

### Issue: W&B not logging

**Check:**
1. `WANDB_API_KEY` environment variable set
2. `wandb` package installed: `pip install wandb`
3. Run `wandb login` to authenticate
4. Check `--no-wandb` flag not set

### Issue: TensorBoard empty

**Check:**
1. TensorBoard directory exists and has write permissions
2. `torch` installed: `pip install torch`
3. Run with correct `--logdir` path
4. Logs may take 10-30 seconds to appear

### Issue: Web dashboard not showing data

**Check:**
1. Monitoring directory path correct
2. JSON files exist: `ls monitoring_data/step_*.json`
3. Streamlit installed: `pip install streamlit plotly pandas`
4. Use `--data-dir` argument to specify path

### Issue: High memory usage

**Solutions:**
1. Disable distribution tracking: remove `--log-distributions`
2. Reduce window size in `RewardMetricsTracker`
3. Increase logging interval (log every N steps)
4. Clear metrics buffer periodically

---

## Examples

### Example 1: Basic Monitoring

```bash
python scripts/train_grpo_monitored.py \
    --num-steps 100 \
    --model-id google/gemma-3-1b-it
```

**Access:**
- W&B: Automatic (if API key set)
- TensorBoard: `tensorboard --logdir=/tmp/tensorboard/grpo_rewards`
- Web: `streamlit run src/monitoring/web_dashboard.py`

### Example 2: Detailed Analysis

```bash
python scripts/train_grpo_monitored.py \
    --num-steps 500 \
    --use-lora \
    --enable-alerts \
    --log-distributions \
    --monitoring-dir ./detailed_run
```

**Features enabled:**
- Real-time alerts for issues
- Full reward distributions
- Custom monitoring directory

### Example 3: Offline Monitoring

```bash
# Training (W&B and TensorBoard disabled)
python scripts/train_grpo_monitored.py \
    --num-steps 200 \
    --no-wandb \
    --no-tensorboard \
    --monitoring-dir ./offline_run

# Later: View with web dashboard
streamlit run src/monitoring/web_dashboard.py -- --data-dir ./offline_run
```

### Example 4: Programmatic Analysis

```python
import json
from pathlib import Path
from src.monitoring import RewardAnalyzer

# Load monitoring data
data_dir = Path('./monitoring_data')
reward_history = []

for step_file in sorted(data_dir.glob('step_*.json')):
    with open(step_file) as f:
        step_data = json.load(f)
        # Extract rewards by function
        # ... process step_data ...

# Analyze
analyzer = RewardAnalyzer()
report = analyzer.analyze_quality(reward_history, steps)

print(f"Quality Score: {report.quality_score:.1f}/100")
for issue in report.issues:
    print(f"❌ {issue}")
for rec in report.recommendations:
    print(f"💡 {rec}")
```

---

## FAQ

**Q: Can I use this with non-GRPO training?**

A: Yes! The monitoring system is general-purpose. Just integrate `RewardMetricsTracker` and `DashboardManager` into your training loop.

**Q: How do I compare multiple training runs?**

A: Use W&B grouping, TensorBoard run comparison, or load multiple JSON exports into the web dashboard.

**Q: Can I add custom reward functions?**

A: Yes! Just add your function to the `reward_fns` list and include its name in `reward_function_names`.

**Q: What's the performance overhead?**

A: Minimal (<1% typically). Most overhead is from I/O (logging). Disable `--log-distributions` for faster training.

**Q: Can I export monitoring data?**

A: Yes! JSON exports are in `monitoring_data/`. Use `final_metrics.json` for complete summary.

---

## Contributing

To extend the monitoring system:

1. **Add new metrics**: Extend `RewardMetrics` dataclass
2. **Add new analyzers**: Create methods in `RewardAnalyzer`
3. **Add new visualizations**: Edit `web_dashboard.py`
4. **Add new alerts**: Use `RealtimeMonitor.add_threshold_alert()`

---

## References

- **GRPO Paper**: [Link to paper if available]
- **W&B Docs**: https://docs.wandb.ai
- **TensorBoard Guide**: https://www.tensorflow.org/tensorboard
- **Streamlit Docs**: https://docs.streamlit.io

---

## License

Same as parent project.

---

## Support

For issues or questions:
1. Check this documentation
2. Review examples in `scripts/`
3. Check monitoring code in `src/monitoring/`
4. Open an issue on GitHub
