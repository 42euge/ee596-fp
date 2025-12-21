# CI/CD TPU Training Setup

## Overview
This document describes how to set up and maintain the GitHub Actions CI/CD pipeline for TPU training.

There are two workflows available:
1. **TPU Training CI** (`tpu-training.yml`) - Quick validation runs for PRs
2. **TPU Training (Full)** (`tpu-training-full.yml`) - Full training runs with W&B support

## Quick Setup (Coding Agent Prompt)

Use this prompt to ask the coding agent to set up CI/CD:

```
Set up the GCP service account and GitHub secrets for TPU CI/CD:
1. Create service account github-tpu-ci in project kaggle-euge
2. Grant roles/tpu.admin and roles/compute.instanceAdmin.v1
3. Create a key and upload to GitHub secret GCP_SA_KEY for repo 42euge/ee596-fp
```

---

## Manual Setup Commands

### 1. Create Service Account

```bash
gcloud iam service-accounts create github-tpu-ci \
  --project=kaggle-euge \
  --display-name="GitHub TPU CI"
```

### 2. Grant Required Roles

```bash
# TPU admin - create/delete TPU VMs
gcloud projects add-iam-policy-binding kaggle-euge \
  --member="serviceAccount:github-tpu-ci@kaggle-euge.iam.gserviceaccount.com" \
  --role="roles/tpu.admin" --quiet

# Compute admin - SSH into VMs
gcloud projects add-iam-policy-binding kaggle-euge \
  --member="serviceAccount:github-tpu-ci@kaggle-euge.iam.gserviceaccount.com" \
  --role="roles/compute.instanceAdmin.v1" --quiet

# Service Account User - required to create TPU VMs (uses default compute SA)
gcloud iam service-accounts add-iam-policy-binding \
  "969416305790-compute@developer.gserviceaccount.com" \
  --project="kaggle-euge" \
  --member="serviceAccount:github-tpu-ci@kaggle-euge.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

### 3. Create Key and Upload to GitHub

```bash
# One-liner: create key and pipe to GitHub secret
gcloud iam service-accounts keys create /dev/stdout \
  --iam-account=github-tpu-ci@kaggle-euge.iam.gserviceaccount.com \
  2>/dev/null | gh secret set GCP_SA_KEY --repo 42euge/ee596-fp
```

### 4. (Optional) Set HuggingFace Token

```bash
# Interactive - will prompt for value
gh secret set HF_TOKEN --repo 42euge/ee596-fp
```

### 5. (Optional) Set Weights & Biases API Key

For experiment tracking with W&B:

```bash
# Get your API key from https://wandb.ai/authorize
gh secret set WANDB_API_KEY --repo 42euge/ee596-fp
```

> **Note:** WANDB_API_KEY is already configured for this repository.

---

## Verify Setup

```bash
# List secrets
gh secret list --repo 42euge/ee596-fp

# Current configuration:
# GCP_SA_KEY      ✓ configured
# HF_TOKEN        ✓ configured
# WANDB_API_KEY   ✓ configured
```

---

## Trigger the Workflows

### TPU Training CI (Quick Validation)

For PR validation and quick smoke tests:

**Manual Trigger:**
1. Go to https://github.com/42euge/ee596-fp/actions
2. Select "TPU Training CI"
3. Click "Run workflow"
4. Choose TPU type and number of steps

**Via CLI:**
```bash
gh workflow run tpu-training.yml --repo 42euge/ee596-fp \
  -f num_steps=10 \
  -f tpu_type=v5litepod-4
```

### TPU Training (Full) - With W&B Support

For full training runs with experiment tracking:

**Manual Trigger:**
1. Go to https://github.com/42euge/ee596-fp/actions
2. Select "TPU Training (Full)"
3. Click "Run workflow"
4. Configure training parameters:
   - `num_steps`: Number of training steps (default: 100)
   - `tpu_type`: TPU accelerator type
   - `model_id`: HuggingFace model ID (default: google/gemma-3-1b-it)
   - `learning_rate`: Learning rate (default: 3e-6)
   - `batch_size`: Batch size (default: 1)
   - `run_name`: W&B run name (optional)
   - `use_lora`: Enable LoRA training (default: true)

**Via CLI:**
```bash
# Basic run
gh workflow run tpu-training-full.yml --repo 42euge/ee596-fp \
  -f num_steps=100 \
  -f tpu_type=v5litepod-4

# Full configuration
gh workflow run tpu-training-full.yml --repo 42euge/ee596-fp \
  -f num_steps=500 \
  -f tpu_type=v5litepod-8 \
  -f model_id="google/gemma-3-1b-it" \
  -f learning_rate="5e-6" \
  -f batch_size=2 \
  -f run_name="experiment-v1" \
  -f use_lora=true
```

**View W&B Dashboard:**
After starting a run, the training logs will be available at:
https://wandb.ai/eugeniorrivera-university-of-washington/tunix-grpo-ci

---

## Weights & Biases (W&B)

### Project Info

| Setting | Value |
|---------|-------|
| Project | `tunix-grpo-ci` |
| Entity | `eugeniorrivera-university-of-washington` |
| Dashboard | https://wandb.ai/eugeniorrivera-university-of-washington/tunix-grpo-ci |

### W&B CLI Setup (Local)

```bash
# Install wandb
uv pip install wandb

# Login (saves to ~/.netrc)
.venv/bin/wandb login YOUR_API_KEY

# Check status
.venv/bin/wandb status
```

### Useful CLI Commands

```bash
# View runs in browser
open https://wandb.ai/eugeniorrivera-university-of-washington/tunix-grpo-ci

# List local runs pending sync
.venv/bin/wandb sync --list

# Sync offline runs
.venv/bin/wandb sync ./wandb/run-XXXX

# Download run artifacts
.venv/bin/wandb artifact get eugeniorrivera-university-of-washington/tunix-grpo-ci/run-XXXX

# Export run data
.venv/bin/wandb export runs --project tunix-grpo-ci
```

### Metrics Logged

The training script logs the following metrics to W&B:

| Metric | Description |
|--------|-------------|
| `train/loss` | Training loss |
| `train/reward` | Average reward |
| `train/kl_divergence` | KL divergence from reference model |
| `train/learning_rate` | Current learning rate |
| `eval/accuracy` | Evaluation accuracy |
| `eval/format_accuracy` | Format compliance rate |

### Creating a New Project

Projects are created automatically when you first log to them:

```python
import wandb

run = wandb.init(
    project='my-new-project',
    name='run-name',
    config={'lr': 1e-4, 'steps': 100}
)
# ... training code ...
run.finish()
```

---

## Configuration Reference

| Setting | Value | Location |
|---------|-------|----------|
| GCP Project | `kaggle-euge` | `.github/workflows/tpu-training*.yml` |
| TPU Zone | `us-central1-a` | `.github/workflows/tpu-training*.yml` |
| Service Account | `github-tpu-ci@kaggle-euge.iam.gserviceaccount.com` | GCP IAM |
| GitHub Repo | `42euge/ee596-fp` | GitHub |
| W&B Project | `tunix-grpo-ci` | `.github/workflows/tpu-training-full.yml` |

### Training Script Parameters

The full training workflow uses `scripts/train_grpo.py` which accepts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-steps` | 100 | Number of training steps |
| `--model-id` | google/gemma-3-1b-it | HuggingFace model ID |
| `--learning-rate` | 3e-6 | Learning rate |
| `--batch-size` | 1 | Micro batch size |
| `--use-lora` | (flag) | Enable LoRA training |
| `--lora-rank` | 64 | LoRA rank |
| `--lora-alpha` | 64.0 | LoRA alpha |
| `--num-generations` | 2 | GRPO generations per prompt |
| `--beta` | 0.08 | KL divergence coefficient |
| `--epsilon` | 0.2 | Clipping epsilon |
| `--temperature` | 0.9 | Sampling temperature |
| `--wandb-project` | tunix-grpo | W&B project name |
| `--run-name` | (auto) | W&B run name |
| `--no-wandb` | (flag) | Disable W&B logging |

---

## Troubleshooting

### Missing serviceAccountUser Permission
If you see this error:
```
ERROR: The principal making this API call needs to be granted the iam.serviceAccountUser role
```

Run:
```bash
gcloud iam service-accounts add-iam-policy-binding \
  "969416305790-compute@developer.gserviceaccount.com" \
  --project="kaggle-euge" \
  --member="serviceAccount:github-tpu-ci@kaggle-euge.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

### Python Version Mismatch
If you see this error:
```
ERROR: Could not find a version that satisfies the requirement google-tunix
```

The TPU VM needs Python 3.11+. The setup script (`scripts/setup_tpu_vm.sh`) handles this by installing Python 3.11 from deadsnakes PPA.

### Rotate Service Account Key
If the key is compromised or expired:

```bash
# Delete old keys (list first)
gcloud iam service-accounts keys list \
  --iam-account=github-tpu-ci@kaggle-euge.iam.gserviceaccount.com

# Delete specific key
gcloud iam service-accounts keys delete KEY_ID \
  --iam-account=github-tpu-ci@kaggle-euge.iam.gserviceaccount.com

# Create new key and upload
gcloud iam service-accounts keys create /dev/stdout \
  --iam-account=github-tpu-ci@kaggle-euge.iam.gserviceaccount.com \
  2>/dev/null | gh secret set GCP_SA_KEY --repo 42euge/ee596-fp
```

### Delete Service Account
```bash
gcloud iam service-accounts delete github-tpu-ci@kaggle-euge.iam.gserviceaccount.com \
  --project=kaggle-euge
```

### Manual TPU Cleanup
If a TPU VM is left running after a failed workflow:

```bash
# List TPUs
gcloud compute tpus tpu-vm list --zone=us-central1-a --project=kaggle-euge

# Delete specific TPU
gcloud compute tpus tpu-vm delete TPU_NAME --zone=us-central1-a --project=kaggle-euge
```
