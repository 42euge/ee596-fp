# Reward Monitoring System

Comprehensive monitoring and visualization system for GRPO reward signals.

## Quick Start

```bash
# Install dependencies
pip install -r requirements-monitoring.txt

# Run demo
python examples/monitoring_demo.py

# View dashboards
tensorboard --logdir=/tmp/tensorboard/grpo_rewards
streamlit run src/monitoring/web_dashboard.py -- --data-dir ./demo_monitoring_data
```

## Features

- ✅ Per-reward-function tracking
- ✅ Quality metrics (SNR, entropy, variance)
- ✅ Multi-backend dashboards (W&B, TensorBoard, Web)
- ✅ Real-time alerting
- ✅ Statistical analysis and trend detection
- ✅ Reward hacking detection

## Components

| Module | Purpose |
|--------|---------|
| `metrics_tracker.py` | Core metrics tracking and aggregation |
| `dashboard_manager.py` | Multi-backend dashboard management |
| `reward_analyzer.py` | Statistical analysis and quality scoring |
| `realtime_monitor.py` | Real-time alerting system |
| `tensorboard_config.py` | TensorBoard configuration |
| `web_dashboard.py` | Streamlit web dashboard |

## Usage

### Training with Monitoring

```bash
python scripts/train_grpo_monitored.py \
    --num-steps 100 \
    --enable-alerts \
    --log-distributions
```

### Programmatic API

```python
from src.monitoring import RewardMetricsTracker, DashboardManager

# Initialize
tracker = RewardMetricsTracker(
    reward_function_names=['func1', 'func2'],
    window_size=100
)

dashboard = DashboardManager(
    project_name='my-project',
    enable_wandb=True,
    enable_tensorboard=True
)

# Track step
metrics = tracker.track_step(
    step=42,
    rewards_by_function={'func1': [...], 'func2': [...]},
    kl_divergence=0.05
)

# Log to dashboards
dashboard.log_metrics(step, metrics)
```

## Documentation

See [docs/MONITORING.md](../../docs/MONITORING.md) for complete documentation.

## Dashboards

### W&B
- Cloud-based experiment tracking
- Custom charts and comparisons
- Team collaboration

### TensorBoard
- Local visualization
- Custom layouts
- Histogram support

### Web Dashboard
- Interactive Plotly charts
- Real-time updates
- Custom analysis

## Metrics Tracked

- **Aggregate**: mean, std, min, max, total
- **Quality**: signal-to-noise ratio, entropy, variance
- **Per-function**: individual scores and contributions
- **Training dynamics**: KL divergence, policy entropy, advantages

## Examples

See `examples/monitoring_demo.py` for a complete demo.
