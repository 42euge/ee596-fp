# Reward Model Development Pipeline Guide

Complete guide to using the automated reward model development pipelines.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Pipeline Components](#pipeline-components)
4. [Workflows](#workflows)
5. [CLI Reference](#cli-reference)
6. [Makefile Reference](#makefile-reference)
7. [GitHub Actions](#github-actions)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Overview

The reward model development pipeline automates the entire lifecycle of reward model development:

```
Dataset Preparation → Training → Evaluation → Deployment → Monitoring
```

### Key Features

- ✅ **Automated dataset preparation** with validation
- ✅ **One-command training** on local or TPU
- ✅ **Automated evaluation** with comprehensive metrics
- ✅ **One-click deployment** to HuggingFace Hub
- ✅ **Real-time monitoring** with dashboards
- ✅ **CI/CD integration** for automated testing
- ✅ **Code quality checks** with pre-commit hooks

## Quick Start

### 1. Initial Setup

```bash
# Clone repository
git clone https://github.com/42euge/ee596-fp.git
cd ee596-fp

# Quick start (installs dependencies, sets up environment, prepares dataset)
make quickstart
```

### 2. Run Quick Training Test

```bash
# Validate training setup
make train-small
```

### 3. Prepare Dataset

```bash
# Prepare GSM8K dataset
make prepare-gsm8k

# Or use CLI directly
python scripts/reward_pipeline.py dataset prepare --config gsm8k
```

### 4. Start Training

```bash
# Local training (100 steps)
make train

# Full training (1000 steps)
make train-full

# Or use CLI with custom parameters
python scripts/reward_pipeline.py train \
  --steps 1000 \
  --lr 3e-6 \
  --batch-size 1 \
  --dataset gsm8k \
  --wandb-project my-project
```

### 5. Evaluate Model

```bash
# Quick evaluation (100 samples)
make evaluate-quick

# Full evaluation
make evaluate

# Or use CLI
python scripts/reward_pipeline.py evaluate \
  --checkpoint ./checkpoints/step_1000 \
  --dataset gsm8k \
  --split test
```

### 6. Monitor Training

```bash
# List available runs
make monitor

# Monitor specific run
make monitor RUN=train_gsm8k_20241223_120000

# Compare multiple runs
make monitor-compare RUNS="run1 run2 run3"
```

### 7. Deploy Checkpoint

```bash
# Deploy to HuggingFace Hub
make deploy CHECKPOINT=./checkpoints/step_1000 REPO_ID=username/my-reward-model

# Or use CLI
python scripts/reward_pipeline.py deploy \
  --checkpoint ./checkpoints/step_1000 \
  --repo-id username/my-reward-model
```

## Pipeline Components

### 1. Dataset Preparation Pipeline

**Script:** `scripts/prepare_dataset.py`

Handles dataset loading, preprocessing, validation, and splitting.

**Features:**
- Multiple data sources (HuggingFace, Kaggle, TFDS)
- Automatic validation checks
- Train/val/test splitting
- Statistics generation

**Usage:**

```bash
# Prepare GSM8K
python scripts/prepare_dataset.py \
  --config gsm8k \
  --output-dir data/gsm8k \
  --validate

# Prepare OpenRubrics
python scripts/prepare_dataset.py \
  --config openrubrics \
  --output-dir data/openrubrics \
  --validate

# Custom dataset
python scripts/prepare_dataset.py \
  --config custom \
  --custom-config my_config.json \
  --output-dir data/custom
```

**Output:**
- `stats.json` - Dataset statistics
- `config.json` - Configuration used
- Processed dataset files

### 2. Training Orchestration

**Script:** `scripts/reward_pipeline.py train`

Manages training runs with experiment tracking.

**Features:**
- Local or TPU training
- LoRA fine-tuning support
- Weights & Biases integration
- Checkpoint management
- Resume from checkpoint

**Usage:**

```bash
# Basic training
python scripts/reward_pipeline.py train --steps 1000

# With custom parameters
python scripts/reward_pipeline.py train \
  --steps 1000 \
  --lr 3e-6 \
  --batch-size 1 \
  --lora-rank 64 \
  --dataset gsm8k \
  --wandb-project my-project \
  --run-name my-experiment

# Resume from checkpoint
python scripts/reward_pipeline.py train \
  --steps 2000 \
  --resume-from ./checkpoints/step_1000
```

**Output:**
- Training logs in `logs/`
- Checkpoints in `checkpoints/`
- W&B run with metrics

### 3. Evaluation Pipeline

**Script:** `scripts/evaluate_model.py`

Comprehensive model evaluation with multiple metrics.

**Features:**
- Multiple evaluation metrics
- Detailed failure analysis
- Export to JSON/CSV
- Batch processing

**Metrics Tracked:**
- Exact accuracy
- Partial accuracy (numerical tolerance)
- Numerical accuracy
- Format compliance

**Usage:**

```bash
# Evaluate checkpoint
python scripts/evaluate_model.py \
  --checkpoint ./checkpoints/step_1000 \
  --dataset gsm8k \
  --split test \
  --output logs/eval_results.json

# Quick evaluation (100 samples)
python scripts/evaluate_model.py \
  --checkpoint ./checkpoints/step_1000 \
  --dataset gsm8k \
  --num-samples 100 \
  --output logs/eval_quick.json

# Base model evaluation
python scripts/evaluate_model.py \
  --dataset gsm8k \
  --output logs/eval_base.json \
  --no-model
```

**Output:**

```json
{
  "dataset": "gsm8k",
  "split": "test",
  "checkpoint": "./checkpoints/step_1000",
  "metrics": {
    "exact_accuracy": 0.75,
    "partial_accuracy": 0.82,
    "numerical_accuracy": 0.78,
    "format_compliance": 0.95
  },
  "examples": [...]
}
```

### 4. Deployment Pipeline

**Script:** `scripts/deploy_checkpoint.py`

Automated deployment to HuggingFace Hub.

**Features:**
- Checkpoint validation
- Auto-generated model cards
- Repository management
- Tagging and metadata

**Usage:**

```bash
# Deploy checkpoint
python scripts/deploy_checkpoint.py \
  --checkpoint ./checkpoints/step_1000 \
  --repo-id username/my-reward-model

# Private repository
python scripts/deploy_checkpoint.py \
  --checkpoint ./checkpoints/step_1000 \
  --repo-id username/my-reward-model \
  --private

# Custom model card
python scripts/deploy_checkpoint.py \
  --checkpoint ./checkpoints/step_1000 \
  --repo-id username/my-reward-model \
  --model-card my_model_card.md \
  --tags reward-model grpo gemma

# Validate only (no deployment)
python scripts/deploy_checkpoint.py \
  --checkpoint ./checkpoints/step_1000 \
  --repo-id username/my-reward-model \
  --validate-only
```

### 5. Monitoring Dashboard

**Script:** `scripts/monitor_training.py`

Real-time training metrics visualization.

**Features:**
- Parse training logs
- ASCII plots for metrics
- W&B integration
- Run comparison
- Export to CSV/JSON

**Usage:**

```bash
# Monitor from log file
python scripts/monitor_training.py \
  --run-name train_gsm8k_20241223_120000

# Monitor from W&B
python scripts/monitor_training.py \
  --run-name my-wandb-run \
  --wandb \
  --wandb-project reward-model-dev

# Compare multiple runs
python scripts/monitor_training.py \
  --compare run1 run2 run3

# Export metrics
python scripts/monitor_training.py \
  --run-name train_gsm8k_20241223_120000 \
  --export metrics.json \
  --format json
```

**Output:**

```
================================================================================
Training Run: train_gsm8k_20241223_120000
================================================================================
Total Steps: 1000

LOSS:
  Final: 0.1234
  Min: 0.0987
  Max: 0.5432
  Mean: 0.2345

REWARD:
  Final: 2.3456
  Min: 1.2345
  Max: 2.8901
  Mean: 2.1234

LOSS (0.0987 to 0.5432)
────────────────────────────────────────────────────────────────────────────────
                                                                        ●
                                                                      ●
                                                                    ●
                                                                  ●
                                                              ●●●
                                                          ●●●●
                                                      ●●●●
                                                  ●●●●
                                              ●●●●
                                          ●●●●
                                      ●●●●
                                  ●●●●
                              ●●●●
                          ●●●●
                      ●●●●
                  ●●●●
              ●●●●
          ●●●●
      ●●●●
  ●●●●
────────────────────────────────────────────────────────────────────────────────
```

## Workflows

### Complete Development Workflow

```bash
# 1. Setup (one-time)
make setup

# 2. Prepare dataset
make prepare-gsm8k

# 3. Develop and test code
make dev  # Format, lint, test

# 4. Start training
make train-full

# 5. Monitor progress
make monitor RUN=<run_name>

# 6. Evaluate results
make evaluate

# 7. Deploy if satisfied
make deploy CHECKPOINT=./checkpoints/step_1000 REPO_ID=username/model

# 8. Commit changes
git add .
git commit -m "Add reward model training"
git push
```

### Experiment Iteration Workflow

```bash
# Run experiment 1
python scripts/reward_pipeline.py train \
  --steps 1000 \
  --lr 1e-5 \
  --run-name exp1_lr1e5

# Run experiment 2 (in parallel)
python scripts/reward_pipeline.py train \
  --steps 1000 \
  --lr 3e-6 \
  --run-name exp2_lr3e6

# Compare results
make monitor-compare RUNS="exp1_lr1e5 exp2_lr3e6"

# Evaluate best model
make evaluate CHECKPOINT=./checkpoints/exp2_lr3e6/step_1000
```

### Continuous Evaluation Workflow

Automated via GitHub Actions (`.github/workflows/auto-evaluation.yml`):

1. Triggered on PR or schedule
2. Prepares dataset
3. Runs evaluation
4. Posts results as PR comment
5. Uploads results as artifacts

## CLI Reference

### reward_pipeline.py

Main CLI for all pipeline operations.

```bash
python scripts/reward_pipeline.py <command> [options]
```

**Commands:**

- `dataset prepare` - Prepare dataset
- `train` - Start training
- `evaluate` - Evaluate model
- `deploy` - Deploy checkpoint
- `monitor` - Open monitoring dashboard

**Global Options:**

- `--help` - Show help message

**Dataset Prepare Options:**

```bash
python scripts/reward_pipeline.py dataset prepare \
  --config <gsm8k|openrubrics|custom> \
  --output-dir <path> \
  [--no-validate] \
  [--custom-config <path>]
```

**Train Options:**

```bash
python scripts/reward_pipeline.py train \
  [--steps <int>] \
  [--lr <float>] \
  [--batch-size <int>] \
  [--tpu <type>] \
  [--no-lora] \
  [--lora-rank <int>] \
  [--wandb-project <name>] \
  [--run-name <name>] \
  [--dataset <name>] \
  [--resume-from <path>]
```

**Evaluate Options:**

```bash
python scripts/reward_pipeline.py evaluate \
  [--checkpoint <path>] \
  [--dataset <name>] \
  [--split <train|val|test>] \
  [--output <path>] \
  [--num-samples <int>]
```

**Deploy Options:**

```bash
python scripts/reward_pipeline.py deploy \
  --checkpoint <path> \
  --repo-id <username/model> \
  [--private] \
  [--commit-message <msg>]
```

## Makefile Reference

Convenient shortcuts for common tasks.

### Setup & Installation

```bash
make install        # Install dependencies
make install-dev    # Install dev dependencies
make setup          # Full development setup
make quickstart     # Quick start (setup + prepare dataset)
```

### Code Quality

```bash
make format         # Format code with black/isort
make lint           # Run linters
make lint-security  # Security checks
make test           # Run all tests
make test-quick     # Run quick tests
make dev            # format + lint + test
make pre-commit     # Pre-commit checks
```

### Dataset Pipeline

```bash
make prepare-gsm8k       # Prepare GSM8K
make prepare-openrubrics # Prepare OpenRubrics
```

### Training Pipeline

```bash
make train        # Basic training (100 steps)
make train-small  # Quick validation
make train-full   # Full training (1000 steps)
```

### Evaluation Pipeline

```bash
make evaluate       # Full evaluation
make evaluate-quick # Quick evaluation (100 samples)
```

### Deployment Pipeline

```bash
make deploy CHECKPOINT=<path> REPO_ID=<repo>        # Deploy
make deploy-private CHECKPOINT=<path> REPO_ID=<repo> # Deploy private
```

### Monitoring

```bash
make monitor RUN=<name>        # Monitor specific run
make monitor-compare RUNS="..."  # Compare runs
make logs                       # Tail latest log
make status                     # Show project status
```

### Utilities

```bash
make clean      # Clean generated files
make clean-all  # Clean all data
make version    # Show version info
make help       # Show all targets
```

## GitHub Actions

### Auto Evaluation Workflow

**File:** `.github/workflows/auto-evaluation.yml`

Automatically evaluates models on:
- Pull requests
- Pushes to main/claude branches
- Daily schedule
- Manual trigger

**Manual Trigger:**

1. Go to Actions tab
2. Select "Automated Model Evaluation"
3. Click "Run workflow"
4. Fill in parameters:
   - Checkpoint path (optional)
   - Dataset (gsm8k/openrubrics)
   - Split (train/val/test)
   - Number of samples (optional)
   - Upload results to HF Hub (optional)

**PR Comments:**

Evaluation results are automatically posted as PR comments:

```markdown
## 🤖 Automated Evaluation Results

**Dataset:** gsm8k (test split)
**Checkpoint:** base_model
**Examples:** 1319

### Metrics

| Metric | Value | Count |
|--------|-------|-------|
| Exact Accuracy | 75.23% | 992/1319 |
| Partial Accuracy | 82.41% | 1087/1319 |
| Numerical Accuracy | 78.09% | 1030/1319 |
| Format Compliance | 95.45% | 1259/1319 |
```

### TPU Training Workflows

**Files:**
- `.github/workflows/tpu-training.yml` - Quick validation
- `.github/workflows/tpu-training-full.yml` - Full training

See [CICD_SETUP.md](CICD_SETUP.md) for details.

## Best Practices

### Dataset Preparation

1. **Always validate datasets** before training
2. **Check statistics** to ensure data quality
3. **Use appropriate splits** (70/15/15 train/val/test)
4. **Version your datasets** with git or DVC

### Training

1. **Start with small runs** to validate setup
2. **Use meaningful run names** (e.g., `exp1_lr1e5_bs4`)
3. **Track experiments** in Weights & Biases
4. **Save checkpoints frequently** (every 100 steps)
5. **Monitor metrics** during training

### Evaluation

1. **Evaluate on multiple datasets** for robustness
2. **Use held-out test sets** for final evaluation
3. **Analyze failure cases** to improve model
4. **Compare with baselines** for context
5. **Report multiple metrics** (not just accuracy)

### Deployment

1. **Validate checkpoints** before deployment
2. **Write descriptive model cards** with metrics
3. **Tag models appropriately** for discovery
4. **Version your models** (e.g., v1.0, v1.1)
5. **Test deployed models** before announcing

### Monitoring

1. **Check logs regularly** during training
2. **Watch for overfitting** (val loss increasing)
3. **Compare experiments** to find best hyperparameters
4. **Export metrics** for offline analysis
5. **Set up alerts** for failed runs

### Code Quality

1. **Run pre-commit hooks** before committing
2. **Write tests** for new features
3. **Format code** consistently with black
4. **Document functions** with docstrings
5. **Review code** before merging

## Troubleshooting

### Dataset Preparation Issues

**Problem:** Dataset validation fails

```bash
# Check dataset statistics
python scripts/prepare_dataset.py --config gsm8k --output-dir data/gsm8k

# Inspect output
cat data/gsm8k/stats.json
```

**Problem:** Out of memory during dataset loading

```python
# Use smaller batch size in config
config = TunRexConfig.gsm8k()
config.batch_size = 1  # Reduce batch size
```

### Training Issues

**Problem:** Training crashes with OOM

```bash
# Reduce batch size
python scripts/reward_pipeline.py train --batch-size 1

# Or use gradient accumulation
python scripts/reward_pipeline.py train --batch-size 1 --gradient-accumulation-steps 4
```

**Problem:** Loss not decreasing

1. Check learning rate (try 1e-5, 3e-6, 1e-6)
2. Verify dataset quality
3. Check model initialization
4. Review reward function

**Problem:** W&B not logging

```bash
# Check W&B login
wandb login

# Verify API key
echo $WANDB_API_KEY

# Test connection
wandb status
```

### Evaluation Issues

**Problem:** Evaluation very slow

```bash
# Use smaller sample size
python scripts/reward_pipeline.py evaluate --num-samples 100

# Or use batch processing
python scripts/evaluate_model.py --batch-size 4
```

**Problem:** Low accuracy

1. Check prompt formatting
2. Verify answer extraction logic
3. Review reward function scores
4. Compare with baseline model

### Deployment Issues

**Problem:** HuggingFace upload fails

```bash
# Check HF token
echo $HF_TOKEN

# Login to HF
huggingface-cli login

# Test upload
huggingface-cli upload-file test.txt username/test-repo test.txt
```

**Problem:** Checkpoint validation fails

```bash
# Check checkpoint structure
ls -lh checkpoints/step_1000/

# Validate manually
python scripts/deploy_checkpoint.py \
  --checkpoint ./checkpoints/step_1000 \
  --repo-id username/model \
  --validate-only
```

### Monitoring Issues

**Problem:** Can't find log files

```bash
# Check logs directory
ls -lh logs/

# Set correct path
python scripts/monitor_training.py --log-dir /path/to/logs --run-name my_run
```

**Problem:** Metrics not parsing

1. Check log format matches expected pattern
2. Try parsing from W&B instead: `--wandb`
3. Export raw metrics: `--export metrics.json`

### General Issues

**Problem:** Import errors

```bash
# Reinstall dependencies
make clean
make install

# Check Python version
python --version  # Should be 3.11+
```

**Problem:** Permission denied

```bash
# Make scripts executable
chmod +x scripts/*.py scripts/*.sh

# Or use make
make setup
```

**Problem:** Git pre-commit hooks fail

```bash
# Format code
make format

# Fix linting issues
make lint

# Skip hooks (not recommended)
git commit --no-verify
```

## Additional Resources

- [CI/CD Setup Guide](CICD_SETUP.md)
- [TunRex Documentation](../TunRex/README.md)
- [Weights & Biases Docs](https://docs.wandb.ai/)
- [HuggingFace Hub Docs](https://huggingface.co/docs/hub/)

## Support

For issues or questions:

1. Check this guide
2. Check [GitHub Issues](https://github.com/42euge/ee596-fp/issues)
3. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Environment info (`make version`)
