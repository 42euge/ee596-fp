# Reward Quality Assessment System

Automated detection of reward hacks and pathologies in RLHF training.

## What It Does

The reward quality assessment system monitors your training process in real-time to detect:

- **Reward Hacking**: Model exploiting reward functions (e.g., correct format with nonsense content)
- **Mode Collapse**: All responses becoming identical or similar
- **Reward Saturation**: Getting stuck at max/min rewards
- **Degenerate Outputs**: Empty tags, repetitive text, malformed responses
- **Statistical Anomalies**: Unusual reward distributions

## Quick Start

### 1. Basic Monitoring

```python
from tunrex.datasets import create_default_monitor

# Create monitor
monitor = create_default_monitor(wandb_run=wandb.run)

# Monitor a batch
metrics = monitor.monitor_batch(
    responses=model_responses,
    rewards={'format': [3.0, 2.5], 'accuracy': [1.5, 0.0]},
    step=current_step
)
```

### 2. Post-Training Analysis

```bash
# Analyze from saved responses
python scripts/analyze_reward_quality.py \
    --responses responses.jsonl \
    --report quality_report.txt

# Analyze from checkpoint
python scripts/analyze_reward_quality.py \
    --checkpoint /path/to/checkpoint \
    --test-data ./data/test \
    --num-samples 1000
```

### 3. Run Tests

```bash
pytest tests/test_reward_quality.py -v
```

## Key Components

### Core Assessment (`reward_quality.py`)
- `RewardQualityAssessor`: Main assessment engine
- Detects 7 types of pathologies
- Computes comprehensive quality metrics

### Monitoring (`reward_monitor.py`)
- `RewardQualityMonitor`: Real-time monitoring with alerts
- W&B integration for dashboards
- Automatic intervention system
- Log file generation

### Wrappers (`reward_wrapper.py`)
- `MonitoredRewardFunction`: Wrap existing reward functions
- `RewardFunctionRegistry`: Manage multiple reward components
- Helper utilities for batch processing

## Detected Pathologies

1. **Reward Saturation** (High/Low) - Training plateaued
2. **Variance Collapse** - Rewards all similar
3. **Format Gaming** - Correct tags, bad content ⚠️
4. **Excessive Repetition** - Repetitive outputs
5. **Diversity Collapse** - Mode collapse
6. **Degenerate Outputs** - Empty/malformed responses
7. **Reward Spikes** - Statistical anomalies

## Documentation

- **Full Guide**: [`/docs/REWARD_QUALITY_MONITORING.md`](../docs/REWARD_QUALITY_MONITORING.md)
  - Detailed API reference
  - Configuration options
  - Integration examples
  - Troubleshooting

- **Analysis Script**: `scripts/analyze_reward_quality.py --help`

## Example: Detecting Format Gaming

The system detects the classic reward hack where models learn to produce correctly formatted outputs without meaningful content:

```
Response: "<reasoning>x</reasoning><answer>y</answer>"
Format Reward: 3.0 ✓  (correct tags!)
Content Quality: ✗  (too short, nonsense)

Alert: [CRITICAL] format_gaming: 45.2% of formatted responses have low-quality content
```

**Detection criteria**:
- Has correct `<reasoning>` and `<answer>` tags
- BUT: reasoning < 20 chars OR answer < 2 chars OR highly repetitive

**Mitigation**:
- Add content quality rewards (minimum length, semantic coherence)
- Increase weight on accuracy/semantic similarity rewards
- Adjust format reward to require minimum content

## Integration with Training

### Option 1: Wrapper Functions (Recommended)

```python
from tunrex.datasets import create_monitored_reward_functions

monitored_fns, monitor = create_monitored_reward_functions(
    reward_functions=[match_format_exactly, check_answer],
    reward_names=['format', 'accuracy'],
    wandb_run=wandb.run
)

# Use in GRPO
grpo_trainer = GRPOLearner(
    reward_fns=monitored_fns,  # ✓ Automatic monitoring
    ...
)
```

### Option 2: Manual Monitoring

```python
monitor = create_default_monitor(wandb_run=wandb.run)

# In your training loop
for batch in dataset:
    responses = generate(batch)
    rewards = compute_rewards(responses)
    monitor.monitor_batch(responses, rewards, step=global_step)
```

### Option 3: Post-Training Analysis

If you can't modify training code, analyze afterwards:

```bash
python scripts/analyze_reward_quality.py --checkpoint /path/to/model
```

## W&B Dashboard

When integrated with Weights & Biases, you get:

- Real-time quality metric plots
- Alert feed showing pathologies as they occur
- Reward distribution histograms
- Diversity and format tracking

All metrics logged under `reward_quality/*` namespace.

## Architecture

```
RewardQualityAssessor (Core)
    ├─ Compute quality metrics
    ├─ Detect pathologies
    └─ Track statistics
         │
         ↓
RewardQualityMonitor (Wrapper)
    ├─ Integrate with training
    ├─ Log to W&B
    ├─ Generate alerts
    └─ Save to files
         │
         ↓
MonitoredRewardFunction (Optional)
    └─ Wrap individual reward functions
```

## File Structure

```
TunRex/src/tunrex/datasets/
├── reward_quality.py      # Core assessment logic
├── reward_monitor.py      # Monitoring & intervention
└── reward_wrapper.py      # Function wrappers

scripts/
└── analyze_reward_quality.py  # Standalone analysis tool

tests/
└── test_reward_quality.py     # Comprehensive test suite

docs/
└── REWARD_QUALITY_MONITORING.md  # Full documentation
```

## Citation

```bibtex
@software{tunrex_reward_quality,
  title={Automated Reward Quality Assessment for RLHF},
  author={TunRex Contributors},
  year={2024}
}
```

## Contributing

When adding new pathology detectors:

1. Add detection method to `RewardQualityAssessor`
2. Add alert type to severity mapping
3. Add tests to `test_reward_quality.py`
4. Document in `REWARD_QUALITY_MONITORING.md`

## Support

- Full docs: `docs/REWARD_QUALITY_MONITORING.md`
- Tests: `pytest tests/test_reward_quality.py -v`
- Issues: File in main repo issue tracker
