# Weights & Biases Metrics Guide

This document describes the metrics and visualizations logged to W&B during GRPO training.

## Setup

W&B logging is enabled automatically when `WANDB_API_KEY` is set. Disable with `--no-wandb`.

```bash
export WANDB_API_KEY=your_key
python scripts/train_grpo.py --num-steps 100 --wandb-project my-project
```

## Run Configuration

Each run logs these configuration parameters:

| Config | Description |
|--------|-------------|
| `model_id` | HuggingFace model ID (e.g., `google/gemma-3-1b-it`) |
| `num_steps` | Total training steps |
| `learning_rate` | Peak learning rate |
| `batch_size` | Micro batch size |
| `use_lora` | Whether LoRA is enabled |
| `lora_rank` | LoRA rank (if enabled) |
| `lora_alpha` | LoRA alpha (if enabled) |
| `num_generations` | Generations per prompt (G in GRPO) |
| `beta` | KL divergence coefficient |
| `epsilon` | PPO clipping epsilon |
| `temperature` | Sampling temperature |
| `max_prompt_length` | Maximum prompt tokens |
| `max_generation_steps` | Maximum generation tokens |
| `weight_decay` | AdamW weight decay |
| `max_grad_norm` | Gradient clipping norm |
| `warmup_fraction` | Warmup steps fraction |
| `num_tpu_cores` | Number of TPU cores |
| `has_tpu` | Whether TPU was detected |
| `rubric_file` | Rubric YAML path (if used) |
| `rubric_weight` | Rubric reward weight (if used) |

## Reward Metrics

### Per-Reward Function Metrics

For each reward function, the following metrics are logged:

| Metric | Description |
|--------|-------------|
| `rewards/{name}_mean` | Mean score for current batch |
| `rewards/{name}_max` | Maximum score in batch |
| `rewards/{name}_min` | Minimum score in batch |
| `rewards/{name}_hist` | Histogram of recent scores (every N steps) |
| `rewards/{name}_recent_mean` | Rolling mean of last 10 scores |

Default reward functions:
- `format_exact` - Exact format match (`<reasoning>...</reasoning><answer>...</answer>`)
- `format_approx` - Approximate format match
- `answer_check` - Correct answer verification
- `number_check` - Numeric answer validation

### Aggregate Metrics

| Metric | Description |
|--------|-------------|
| `rewards/total_mean` | Sum of all reward function means |
| `train/step` | Current training step |

## Rubric Metrics

When using `--rubric-file`, additional metrics are logged:

### Per-Criterion Tracking

| Metric | Description |
|--------|-------------|
| `rewards/rubric_mean` | Mean rubric score |
| `rewards/rubric_max` | Max rubric score |
| `rewards/rubric_min` | Min rubric score |
| `criteria/{rubric_name}/{criterion_name}_mean` | Per-criterion mean (every N steps) |

### Rubric Summary Table

At the end of training, a table `rubric_criteria_summary` is logged with:

| Column | Description |
|--------|-------------|
| criterion | Full criterion path |
| mean | Mean score across all samples |
| std | Standard deviation |
| min | Minimum score |
| max | Maximum score |
| samples | Number of samples scored |

## Final Summary Metrics

Logged at training completion:

| Metric | Description |
|--------|-------------|
| `final/{name}_mean` | Final mean for each reward |
| `final/{name}_std` | Final standard deviation |
| `final/{name}_max` | Final maximum score |
| `final/{name}_min` | Final minimum score |
| `final/{name}_total_samples` | Total samples processed |

## Recommended Dashboard Panels

### Training Progress
- Line chart: `rewards/total_mean` vs step
- Line chart: All `rewards/*_mean` metrics overlaid

### Reward Breakdown
- Bar chart: Compare `final/*_mean` across reward functions
- Histogram: `rewards/*_hist` for score distributions

### Rubric Analysis (if using rubrics)
- Table: `rubric_criteria_summary`
- Line chart: `criteria/*/*_mean` to track criterion improvement
- Heatmap: Criterion scores over time

### System Metrics
- W&B automatically logs system metrics (GPU/TPU utilization, memory)

## Example Queries

Filter runs by configuration:
```
config.use_lora == true AND config.num_steps >= 100
```

Compare rubric vs non-rubric runs:
```
config.rubric_file != null
```

## Troubleshooting

**No metrics appearing:**
- Verify `WANDB_API_KEY` is set
- Check for `--no-wandb` flag
- Look for "W&B initialized" in logs

**Missing rubric metrics:**
- Ensure `--rubric-file` points to valid YAML
- Check "Loaded rubric" message in logs

**Histograms not showing:**
- Histograms log every 10 steps by default
- Need at least 10 samples to generate
