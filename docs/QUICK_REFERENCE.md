# Reward Model Pipeline - Quick Reference

One-page cheat sheet for the reward model development pipeline.

## Installation

```bash
# Quick start (all-in-one)
make quickstart

# Or step-by-step
make install        # Install dependencies
make setup          # Set up dev environment
make prepare-gsm8k  # Prepare dataset
```

## Common Commands

### Dataset

```bash
# Prepare datasets
make prepare-gsm8k
make prepare-openrubrics

# Or with CLI
python scripts/reward_pipeline.py dataset prepare --config gsm8k
```

### Training

```bash
# Quick test (10 steps)
make train-small

# Basic training (100 steps)
make train

# Full training (1000 steps)
make train-full

# Custom training
python scripts/reward_pipeline.py train \
  --steps 2000 \
  --lr 1e-5 \
  --batch-size 2 \
  --dataset gsm8k \
  --run-name my-experiment
```

### Evaluation

```bash
# Quick eval (100 samples)
make evaluate-quick

# Full evaluation
make evaluate

# Custom evaluation
python scripts/reward_pipeline.py evaluate \
  --checkpoint ./checkpoints/step_1000 \
  --dataset gsm8k \
  --split test \
  --num-samples 500
```

### Monitoring

```bash
# List available runs
make monitor

# Monitor specific run
make monitor RUN=my_run_name

# Compare runs
make monitor-compare RUNS="run1 run2 run3"

# View latest logs
make logs
```

### Deployment

```bash
# Deploy checkpoint
make deploy \
  CHECKPOINT=./checkpoints/step_1000 \
  REPO_ID=username/my-model

# Deploy as private
make deploy-private \
  CHECKPOINT=./checkpoints/step_1000 \
  REPO_ID=username/my-model
```

## Development Workflow

```bash
# 1. Make changes to code
vim src/main.py

# 2. Format and test
make dev  # Runs: format, lint, test

# 3. Commit (pre-commit hooks run automatically)
git add .
git commit -m "Add feature"

# 4. Push
git push
```

## Code Quality

```bash
make format         # Format with black + isort
make lint           # Run flake8 + mypy
make lint-security  # Run bandit
make test           # Run all tests
make test-quick     # Run quick tests
make pre-commit     # All pre-commit checks
```

## CLI Quick Reference

### reward_pipeline.py

```bash
# Dataset
python scripts/reward_pipeline.py dataset prepare --config <name>

# Training
python scripts/reward_pipeline.py train [--steps N] [--lr X] [--run-name NAME]

# Evaluation
python scripts/reward_pipeline.py evaluate [--checkpoint PATH] [--dataset NAME]

# Deployment
python scripts/reward_pipeline.py deploy --checkpoint PATH --repo-id REPO

# Monitoring
python scripts/reward_pipeline.py monitor --wandb-project PROJECT [--run-name NAME]
```

### Individual Scripts

```bash
# Dataset preparation
python scripts/prepare_dataset.py --config gsm8k --output-dir data/gsm8k

# Model evaluation
python scripts/evaluate_model.py --checkpoint PATH --output results.json

# Checkpoint deployment
python scripts/deploy_checkpoint.py --checkpoint PATH --repo-id REPO

# Training monitoring
python scripts/monitor_training.py --run-name NAME

# Training
python scripts/train_grpo.py --num-steps 1000 --use-lora
```

## Environment Variables

```bash
# HuggingFace
export HF_TOKEN=hf_xxxxx

# Weights & Biases
export WANDB_API_KEY=xxxxx
export WANDB_PROJECT=my-project

# Google Cloud (for TPU)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
```

## File Locations

```
ee596-fp/
├── scripts/
│   ├── reward_pipeline.py       # Main CLI
│   ├── prepare_dataset.py       # Dataset prep
│   ├── evaluate_model.py        # Evaluation
│   ├── deploy_checkpoint.py     # Deployment
│   ├── monitor_training.py      # Monitoring
│   └── train_grpo.py            # Training
├── checkpoints/                 # Model checkpoints
├── logs/                        # Training logs & results
├── data/                        # Prepared datasets
└── docs/                        # Documentation
```

## Common Issues & Fixes

### Out of Memory

```bash
# Reduce batch size
python scripts/reward_pipeline.py train --batch-size 1
```

### Import Errors

```bash
# Reinstall
make clean && make install
```

### Dataset Not Found

```bash
# Prepare dataset first
make prepare-gsm8k
```

### W&B Not Logging

```bash
# Login
wandb login
```

### HF Upload Failed

```bash
# Set token
export HF_TOKEN=hf_xxxxx
```

### Pre-commit Fails

```bash
# Format first
make format

# Then commit
git commit
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `help` | Show all available targets |
| `quickstart` | Full setup + dataset prep |
| `install` | Install dependencies |
| `setup` | Set up dev environment |
| `test` | Run all tests |
| `format` | Format code |
| `lint` | Run linters |
| `train` | Start training |
| `evaluate` | Evaluate model |
| `deploy` | Deploy checkpoint |
| `monitor` | Monitor training |
| `clean` | Clean generated files |
| `status` | Show project status |

## GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `tpu-training.yml` | PR, manual | Quick TPU validation |
| `tpu-training-full.yml` | Manual | Full TPU training |
| `auto-evaluation.yml` | PR, schedule, manual | Automated evaluation |

## Metrics Guide

| Metric | Range | Meaning |
|--------|-------|---------|
| Exact Accuracy | 0-1 | Perfect answer match |
| Partial Accuracy | 0-1 | Answer within tolerance |
| Numerical Accuracy | 0-1 | Correct number extracted |
| Format Compliance | 0-1 | Output format correct |

## Reward Function Scores

| Function | Score Range | Purpose |
|----------|-------------|---------|
| `check_answer()` | -1 to 3 | Answer correctness |
| `check_numbers()` | 0 to 1.5 | Numerical correctness |
| `match_format_exactly()` | 0 or 3 | Exact format match |
| `match_format_approximately()` | 0 to 1.5 | Approximate format |

## Tips & Best Practices

✅ **Do:**
- Start with `make train-small` to validate setup
- Use meaningful run names (e.g., `exp1_lr1e5`)
- Track all experiments in W&B
- Validate datasets before training
- Monitor training in real-time
- Compare multiple experiments
- Deploy with descriptive model cards

❌ **Don't:**
- Train without validating dataset first
- Use default run names
- Skip evaluation before deployment
- Commit large checkpoint files
- Push to main without PR
- Deploy without testing

## Example Workflows

### Quick Experiment

```bash
make quickstart
make train
make evaluate-quick
```

### Full Development Cycle

```bash
# Setup (once)
make setup
make prepare-gsm8k

# Experiment
python scripts/reward_pipeline.py train --steps 1000 --run-name exp1
make monitor RUN=exp1

# Evaluate
python scripts/reward_pipeline.py evaluate --checkpoint ./checkpoints/exp1/step_1000

# Deploy
make deploy CHECKPOINT=./checkpoints/exp1/step_1000 REPO_ID=username/model

# Commit
git add .
git commit -m "Add exp1 results"
git push
```

### Hyperparameter Search

```bash
# Run multiple experiments
for lr in 1e-5 3e-6 1e-6; do
  python scripts/reward_pipeline.py train \
    --steps 500 \
    --lr $lr \
    --run-name exp_lr${lr}
done

# Compare results
make monitor-compare RUNS="exp_lr1e-5 exp_lr3e-6 exp_lr1e-6"

# Evaluate best
make evaluate CHECKPOINT=./checkpoints/exp_lr3e-6/step_500
```

## Resources

- **Full Guide:** [docs/PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
- **CI/CD Setup:** [docs/CICD_SETUP.md](CICD_SETUP.md)
- **TunRex Docs:** [TunRex/README.md](../TunRex/README.md)
- **W&B:** https://wandb.ai/
- **HuggingFace:** https://huggingface.co/

## Help

```bash
# CLI help
python scripts/reward_pipeline.py --help
python scripts/reward_pipeline.py train --help

# Makefile help
make help

# Project status
make status

# Version info
make version
```
