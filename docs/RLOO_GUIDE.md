# RLOO (REINFORCE Leave-One-Out) Guide

## Overview

RLOO is an alternative advantage estimator to GRPO, designed for more stable training with **noisy or subjective rewards**. It's particularly well-suited for:

- 📝 **Rubric-based rewards** (continuous scores from LLM judges)
- 🎨 **Creative tasks** (writing quality, reasoning assessment)
- 📊 **Non-verifiable problems** (where binary correct/wrong doesn't apply)

## Key Differences from GRPO

| Aspect | GRPO | RLOO |
|--------|------|------|
| **Advantage Formula** | `A_i = (R_i - mean(R)) / std(R)` | `A_i = R_i - mean(R_j where j != i)` |
| **Normalization** | Uses std normalization | No std normalization (more stable) |
| **KL Penalty** | Separate loss term | Can fold into reward: `R' = R - β*KL` |
| **Robustness** | Sensitive to outliers | More robust to noise |
| **Best For** | Binary rewards (math, code) | Continuous rewards (rubrics, quality) |

## Quick Start

### Basic Usage

Train with RLOO instead of GRPO by adding `--advantage-estimator rloo`:

```bash
python scripts/train_grpo.py \
  --advantage-estimator rloo \
  --num-generations 4 \
  --beta 0.08 \
  --rubric-file data/rubrics/reasoning_quality.yaml \
  --num-steps 1000
```

### With KL in Reward (Recommended)

For maximum stability, fold KL directly into the reward:

```bash
python scripts/train_grpo.py \
  --advantage-estimator rloo \
  --kl-in-reward \
  --num-generations 4 \
  --beta 0.08 \
  --rubric-file data/rubrics/reasoning_quality.yaml
```

### With Advantage Clipping

Prevent outlier advantages from dominating training:

```bash
python scripts/train_grpo.py \
  --advantage-estimator rloo \
  --kl-in-reward \
  --advantage-clip 5.0 \
  --num-generations 4
```

## Configuration Parameters

### Required

- `--advantage-estimator rloo` - Use RLOO instead of GRPO
- `--num-generations K` - Number of samples per prompt (must be ≥ 2)
  - **Recommended**: 4-8 for RLOO (higher K = more stable baselines)

### Optional (RLOO-specific)

- `--kl-in-reward` - Fold KL into reward instead of separate loss
  - **Default**: False (uses separate KL loss like GRPO)
  - **Recommended**: True (matches RLOO paper, more stable)

- `--advantage-clip VALUE` - Clip advantages to `[-VALUE, VALUE]`
  - **Default**: None (no clipping)
  - **Recommended**: 5.0-10.0 when using noisy rewards

### Standard RL Parameters

- `--beta` - KL coefficient (default: 0.08)
- `--epsilon` - Policy ratio clipping (default: 0.2)
- `--learning-rate` - Learning rate (default: 3e-6)
- `--temperature` - Sampling temperature (default: 0.9)

## Mathematical Details

### Leave-One-Out Baseline

RLOO uses a **leave-one-out** baseline for each sample:

```
For K generations {R_1, R_2, ..., R_K} per prompt:

Advantage for generation i:
  A_i = R_i - mean(R_j where j ≠ i)
      = R_i - (sum(R) - R_i) / (K - 1)
```

**Why this is better than GRPO's std normalization:**

1. **No std amplification**: Outlier rewards don't inflate advantages
2. **Individualized baselines**: Each sample gets its own baseline
3. **No zero-mean constraint**: Advantages reflect true reward differences

### KL Integration

When `--kl-in-reward` is enabled:

```python
# Modified reward
R'_i = R_i - β * KL(π_θ(·|x) || π_ref(·|x))

# Then compute RLOO advantages on R'
A_i = R'_i - mean(R'_j where j ≠ i)
```

This is more stable than GRPO's approach of using KL as a separate loss term.

## Example: Training on Rubric-Based Rewards

### Problem Statement

You're training a model to "show its work" on math problems. Instead of binary correct/wrong, you use a rubric that scores:
- Clarity (0-3 points)
- Correctness (0-3 points)
- Completeness (0-3 points)

Total reward is continuous: 0-9 points.

### Why RLOO?

With GRPO:
- Std normalization amplifies noise from subjective rubric scores
- Model may exploit reward model inconsistencies (reward hacking)
- Training can be unstable with high variance

With RLOO:
- Leave-one-out baseline is more stable with noisy scores
- KL penalty prevents excessive policy drift
- Better generalization to held-out problems

### Training Command

```bash
python scripts/train_grpo.py \
  --advantage-estimator rloo \
  --kl-in-reward \
  --num-generations 8 \
  --beta 0.1 \
  --advantage-clip 5.0 \
  --rubric-file data/rubrics/math_reasoning.yaml \
  --rubric-weight 1.0 \
  --num-steps 2000 \
  --learning-rate 3e-6 \
  --batch-size 2 \
  --use-lora \
  --lora-rank 64
```

### Expected Behavior

- **First 500 steps**: Model learns basic rubric patterns
- **Steps 500-1500**: Reward increases steadily, KL stays bounded
- **After 1500**: Convergence, model shows consistent reasoning quality

Monitor in W&B:
- `reward/rubric_*` - Per-criterion scores
- `kl_divergence` - Should stay < 5.0 with β=0.1
- `advantage_*` - Should be stable, no extreme outliers

## Comparison: GRPO vs RLOO

### When to Use GRPO

✅ **Use GRPO when:**
- You have binary rewards (correct/wrong, pass/fail)
- Rewards come from a deterministic verifier (unit tests, math checker)
- You want to match existing GRPO benchmarks
- You're working on code/math tasks with ground truth

### When to Use RLOO

✅ **Use RLOO when:**
- You have continuous rewards (scores 0-10, rubric grades)
- Rewards come from subjective sources (LLM judges, human feedback)
- You're seeing reward hacking or instability with GRPO
- You're working on creative/reasoning tasks without ground truth
- You want KL-integrated rewards for stability

### Benchmark Results (from Ahmadian et al. 2024)

On preference optimization tasks:
- **RLOO**: 67.3% win rate vs reference
- **PPO**: 63.1% win rate
- **GRPO**: ~65% win rate (estimated)

RLOO achieves better sample efficiency with fewer rollouts.

## Troubleshooting

### Issue: Training is unstable with RLOO

**Solution 1**: Increase `num_generations`
```bash
--num-generations 8  # Higher K = more stable LOO baselines
```

**Solution 2**: Enable advantage clipping
```bash
--advantage-clip 5.0  # Prevent outlier rewards from dominating
```

**Solution 3**: Adjust KL coefficient
```bash
--beta 0.15  # Higher β = stronger KL constraint
```

### Issue: Reward doesn't improve

**Solution**: Check rubric weights and baseline rewards
```bash
# Make sure rubric rewards are scaled appropriately
--rubric-weight 1.0

# Verify baseline rewards are non-zero
# (Check W&B logs for initial reward distributions)
```

### Issue: OOM (out of memory)

**Solution**: Reduce `num_generations` or `batch_size`
```bash
--num-generations 4  # RLOO works with K=4
--batch-size 1       # Reduce batch size
```

## Implementation Details

### Code Structure

```
scripts/
  rloo_learner.py          # RLOOLearner, RLOOConfig, compute_rloo_advantages
  train_grpo.py            # Main training script (supports both GRPO/RLOO)
  training_config.py       # create_rloo_config()
  utils.py                 # CLI argument parsing

tests/
  unit/test_rloo.py        # Unit tests for RLOO
```

### Key Functions

- `RLOOConfig` - Configuration dataclass
- `compute_rloo_advantages(rewards, kl, beta, kl_in_reward, advantage_clip)` - Core RLOO logic
- `RLOOLearner.train(train_ds, val_ds)` - Training loop (wraps GRPO)

### Extending RLOO

To customize RLOO for your use case:

```python
from scripts.rloo_learner import create_rloo_learner, RLOOConfig

# Custom config
config = RLOOConfig(
    num_generations=8,
    beta=0.12,
    kl_in_reward=True,
    advantage_clip=10.0,
)

# Create learner
learner = create_rloo_learner(
    rl_cluster=rl_cluster,
    reward_fns=your_reward_functions,
    **config.__dict__
)

# Train
learner.train(train_ds, val_ds)
```

## References

1. **Ahmadian et al. 2024**: ["Back to Basics: Revisiting REINFORCE Style Optimization"](https://arxiv.org/abs/2402.14740)
   - Original RLOO paper, shows advantages over PPO/DPO

2. **verl RLOO implementation**: https://github.com/volcengine/verl
   - Production RLOO implementation in PyTorch

3. **swift RLOO docs**: https://github.com/modelscope/swift
   - Alternative implementation with good documentation

4. **PRIME (uses RLOO)**: https://arxiv.org/abs/2410.12297
   - State-of-the-art preference learning, uses RLOO internally

## FAQ

**Q: Can I use RLOO with non-rubric rewards?**
A: Yes! RLOO works with any reward function. It's just particularly good with noisy/continuous rewards.

**Q: What's the minimum `num_generations` for RLOO?**
A: K ≥ 2, but K=4-8 is recommended for stable baselines.

**Q: Does RLOO work with LoRA?**
A: Yes, RLOO is compatible with all GRPO features (LoRA, checkpointing, W&B logging, etc.)

**Q: How does RLOO compare to PPO?**
A: RLOO is simpler (no value function), more sample-efficient, and more stable with noisy rewards.

**Q: Can I mix GRPO and RLOO in the same run?**
A: Not recommended. Choose one algorithm per training run.

## Support

For issues or questions:
- Check existing issues: https://github.com/42euge/ee596-fp/issues
- Create new issue with `[RLOO]` prefix
- Include training command, W&B logs, and error messages
