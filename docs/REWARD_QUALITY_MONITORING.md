# Reward Quality Monitoring System

Automated quality assessment system for detecting reward hacks and pathologies in reinforcement learning training.

## Overview

The reward quality monitoring system provides comprehensive tools for:
- **Reward Hacking Detection**: Identifies when models exploit reward functions without achieving intended behavior
- **Statistical Anomaly Detection**: Detects unusual patterns in reward distributions
- **Response Quality Metrics**: Measures diversity, coherence, and content quality
- **Real-time Monitoring**: Integrates with Weights & Biases for live tracking
- **Automatic Interventions**: Alerts and logging when pathologies are detected

## Quick Start

### Basic Usage

```python
from tunrex.datasets import (
    create_default_monitor,
    match_format_exactly,
    check_answer,
)

# Create a monitor instance
monitor = create_default_monitor(
    wandb_run=wandb.run,  # Optional W&B integration
    enable_interventions=True
)

# During training, monitor each batch
responses = [
    "<reasoning>My reasoning</reasoning><answer>42</answer>",
    "<reasoning>Another reasoning</reasoning><answer>100</answer>",
]
rewards = {
    'format': [3.0, 3.0],
    'accuracy': [1.5, 0.0],
}

metrics = monitor.monitor_batch(
    responses=responses,
    rewards=rewards,
    step=current_step
)

# Access quality metrics
print(f"Format compliance: {metrics.format_compliance_rate:.2%}")
print(f"Suspected gaming: {metrics.suspected_format_gaming:.2%}")
```

### Integration with Training Script

Add to `scripts/train_grpo.py`:

```python
# After W&B initialization (around line 180)
from tunrex.datasets import create_default_monitor

print("\n[*] Initializing reward quality monitor...")
reward_monitor = create_default_monitor(
    wandb_run=wandb.run if wandb_enabled else None,
    enable_interventions=True,
    window_size=1000
)

# Note: Full integration requires custom training loop or tunix modifications
# For now, use post-training analysis (see below)
```

## Detected Pathologies

### 1. Reward Saturation

**What it detects**: Model consistently receives maximum or minimum rewards

**Why it matters**: Indicates training has plateaued or reward is too easy/hard

**Example**:
```
[HIGH] reward_saturation_high: Reward 'format' saturating at maximum: 95.3% of samples at max value 3.0
```

**Mitigation**:
- Adjust reward scaling
- Add more challenging reward components
- Check if task is too easy

### 2. Variance Collapse

**What it detects**: All rewards become nearly identical (low coefficient of variation)

**Why it matters**: Training signal degraded, model not differentiating between good/bad outputs

**Example**:
```
[CRITICAL] variance_collapse: Reward 'accuracy' variance collapsed: CV=0.008, std=0.012, mean=1.500
```

**Mitigation**:
- Investigate model mode collapse
- Increase temperature/exploration
- Check reward function diversity

### 3. Format Gaming

**What it detects**: Correct format tags with minimal or nonsense content

**Why it matters**: Classic reward hacking - model learns form without substance

**Example**:
```
[CRITICAL] format_gaming: Suspected format gaming: 45.2% of formatted responses have low-quality content
```

**Detection criteria**:
- Has correct `<reasoning>` and `<answer>` tags
- BUT: reasoning < 20 chars OR answer < 2 chars OR repetitive content

**Mitigation**:
- Add content quality rewards
- Increase weight on accuracy/semantic rewards
- Add minimum length requirements to reward

### 4. Excessive Repetition

**What it detects**: High n-gram repetition within responses

**Why it matters**: Model generating repetitive, low-quality text

**Example**:
```
[HIGH] excessive_repetition: avg=62.3%, 8/10 responses highly repetitive
```

**Mitigation**:
- Adjust sampling parameters (temperature, top-p)
- Add repetition penalties
- Check for degenerate model states

### 5. Diversity Collapse (Mode Collapse)

**What it detects**: Many identical or near-identical responses

**Why it matters**: Model stuck in limited output modes

**Example**:
```
[CRITICAL] diversity_collapse: only 8.5% unique responses in recent 500 samples
```

**Mitigation**:
- Increase temperature
- Add diversity bonuses to reward
- Check for optimization issues

### 6. Degenerate Outputs

**What it detects**: Empty tags, missing content, malformed responses

**Why it matters**: Basic generation failure

**Example**:
```
[HIGH] degenerate_outputs: 35.0% responses are empty or malformed
```

**Mitigation**:
- Check model stability
- Verify generation parameters
- Investigate tokenizer issues

### 7. Reward Spikes

**What it detects**: Individual rewards far from historical distribution (>3 std devs)

**Why it matters**: Could indicate bugs or unusual inputs

**Example**:
```
[MEDIUM] reward_spike: Unusual reward spike in 'accuracy': value=100.0, z-score=5.23
```

## Quality Metrics

The system tracks comprehensive quality metrics:

### Format Quality
- `format_compliance_rate`: % with correct `<reasoning>` and `<answer>` tags
- `tag_correctness_rate`: % with properly paired tags

### Content Quality
- `avg_reasoning_length`: Average character count in reasoning sections
- `avg_answer_length`: Average character count in answer sections
- `repetition_rate`: Average n-gram repetition score (0-1)
- `empty_content_rate`: % of responses with empty/trivial content

### Diversity Metrics
- `unique_responses_ratio`: Ratio of unique responses to total
- `lexical_diversity`: Type-token ratio across all responses

### Pathology Indicators
- `suspected_format_gaming`: Rate of good format + bad content
- `suspected_reward_hacking`: Overall hacking indicator

### Reward Statistics
For each reward component:
- `mean`, `std`, `min`, `max`, `median`

## Configuration

### Custom Thresholds

```python
from tunrex.datasets import RewardQualityAssessor

custom_thresholds = {
    'reward_saturation_high': 0.95,  # 95% samples at max
    'reward_saturation_low': 0.95,   # 95% samples at min
    'variance_collapse_threshold': 0.01,  # CV < 0.01
    'format_gaming_threshold': 0.3,  # 30% gaming rate
    'repetition_threshold': 0.5,     # 50% repetition
    'empty_content_threshold': 0.2,  # 20% empty
    'diversity_collapse_threshold': 0.1,  # 10% unique
    'reward_spike_std': 3.0,  # 3 std deviations
}

assessor = RewardQualityAssessor(
    window_size=1000,
    min_samples_for_analysis=100,
    alert_thresholds=custom_thresholds
)
```

### Intervention Settings

```python
from tunrex.datasets import InterventionConfig, RewardQualityMonitor

config = InterventionConfig(
    enable_interventions=True,
    intervention_severity='medium',  # Alert on: low|medium|high|critical
    log_to_console=True,
    log_to_file=True,
    send_to_wandb=True,
    alert_log_file='/tmp/reward_alerts.jsonl',
    metrics_log_file='/tmp/reward_metrics.jsonl',
    alert_aggregation_window=100,  # Steps
    max_alerts_per_window=10,  # Rate limiting
)

monitor = RewardQualityMonitor(assessor, config, wandb_run)
```

## Post-Training Analysis

### Generate Report from Logs

```python
from tunrex.datasets import RewardQualityDashboard

dashboard = RewardQualityDashboard(
    metrics_log_file='/tmp/reward_quality_metrics.jsonl',
    alert_log_file='/tmp/reward_quality_alerts.jsonl'
)

# Generate HTML report
dashboard.generate_report('/tmp/reward_quality_report.html')

# Get metrics as DataFrame (requires pandas)
metrics_df = dashboard.get_metrics_dataframe()
```

### Analyze Saved Responses

```python
from tunrex.datasets import RewardQualityAssessor

# Load saved responses and rewards
responses = load_responses_from_checkpoint(...)
rewards = compute_rewards(responses, ...)

# Analyze
assessor = RewardQualityAssessor()
metrics, alerts = assessor.assess_batch(responses, rewards)

# Print alerts
for alert in alerts:
    print(f"[{alert.severity.upper()}] {alert.message}")

# Get summary
summary = assessor.get_summary_statistics()
print(json.dumps(summary, indent=2))
```

## Advanced: Wrapped Reward Functions

For deeper integration, wrap reward functions:

```python
from tunrex.datasets import create_monitored_reward_functions

# Original reward functions
reward_fns = [
    match_format_exactly,
    check_answer,
    check_numbers,
]

# Create monitored versions
monitored_fns, monitor = create_monitored_reward_functions(
    reward_functions=reward_fns,
    reward_names=['format', 'accuracy', 'numbers'],
    wandb_run=wandb.run,
    enable_interventions=True
)

# Use monitored functions in training
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=monitored_fns,  # Use wrapped versions
    algo_config=grpo_config,
)

# After training, flush pending data
for fn in monitored_fns:
    fn.flush()
```

## Weights & Biases Integration

When `wandb_run` is provided, metrics are automatically logged under the `reward_quality/` namespace:

**Metrics logged**:
- `reward_quality/format_compliance_rate`
- `reward_quality/suspected_format_gaming`
- `reward_quality/unique_responses_ratio`
- `reward_quality/alerts_critical` (count)
- `reward_quality/alerts_high` (count)
- `reward_quality/reward_<name>_mean`
- `reward_quality/reward_<name>_std`
- And more...

**Alert visualization**:
Recent alerts are logged as HTML blocks for easy viewing in W&B dashboard.

## Example: Full Training Integration

```python
#!/usr/bin/env python3
"""Training with reward quality monitoring."""

import wandb
from tunrex.datasets import (
    create_default_monitor,
    get_train_val_test_datasets,
    match_format_exactly,
    check_answer,
)
from tunix.rl.grpo.grpo_learner import GRPOLearner

# Initialize W&B
wandb.init(project="tunix-grpo", name="monitored-training")

# Create quality monitor
monitor = create_default_monitor(
    wandb_run=wandb.run,
    enable_interventions=True
)

# Load data
train_dataset, val_dataset, _ = get_train_val_test_datasets(...)

# Setup trainer (standard GRPO)
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[match_format_exactly, check_answer],
    algo_config=grpo_config,
)

# Train
grpo_trainer.train(train_dataset, val_dataset)

# Note: Full integration requires custom training loop
# or modifications to tunix library to expose batch data

# Get final summary
summary = monitor.get_monitoring_summary()
print(f"\nMonitoring Summary:")
print(f"  Total samples: {summary['total_samples']}")
print(f"  Critical alerts: {summary['monitoring']['total_alerts_by_severity']['critical']}")

wandb.finish()
```

## Testing

Run the test suite:

```bash
# Run all quality monitoring tests
pytest tests/test_reward_quality.py -v

# Run specific test
pytest tests/test_reward_quality.py::TestRewardQualityAssessor::test_format_gaming_detection -v
```

## Best Practices

1. **Start Early**: Enable monitoring from the beginning of training
2. **Review Alerts**: Check alert logs regularly, don't ignore warnings
3. **Tune Thresholds**: Adjust thresholds based on your task and model
4. **Combine Metrics**: Look for combinations of pathologies (e.g., high format compliance + low diversity)
5. **Validate Manually**: Sample and manually inspect flagged responses
6. **Iterate on Rewards**: Use insights to improve reward function design
7. **Track Trends**: Monitor metrics over time, not just absolute values

## Troubleshooting

### No alerts being generated

- Check `min_samples_for_analysis` (default: 100) - need enough samples
- Verify `enable_interventions=True`
- Check `intervention_severity` threshold

### Too many alerts

- Increase thresholds for specific pathologies
- Use `max_alerts_per_window` to rate-limit
- Reduce `intervention_severity` (e.g., 'high' instead of 'medium')

### High memory usage

- Reduce `window_size` (default: 1000)
- Process in smaller batches
- Clear history periodically with `assessor.reward_history.clear()`

### W&B not logging

- Verify `wandb_run` is passed correctly
- Check `send_to_wandb=True` in config
- Ensure W&B is initialized before creating monitor

## API Reference

See docstrings in:
- `TunRex/src/tunrex/datasets/reward_quality.py` - Core assessment
- `TunRex/src/tunrex/datasets/reward_monitor.py` - Monitoring and intervention
- `TunRex/src/tunrex/datasets/reward_wrapper.py` - Function wrappers

## Citation

If you use this system in your research, please cite:

```bibtex
@software{tunrex_reward_quality,
  title={Automated Reward Quality Assessment for RLHF},
  author={TunRex Contributors},
  year={2024},
  url={https://github.com/yourusername/ee596-fp}
}
```
