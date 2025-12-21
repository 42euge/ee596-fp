# CI/CD TPU Training Setup

## Overview
This document describes how to set up and maintain the GitHub Actions CI/CD pipeline for TPU training.

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

---

## Verify Setup

```bash
# List secrets
gh secret list --repo 42euge/ee596-fp

# Should show:
# GCP_SA_KEY    <timestamp>
# HF_TOKEN      <timestamp>  (if set)
```

---

## Trigger the Workflow

### Manual Trigger
1. Go to https://github.com/42euge/ee596-fp/actions
2. Select "TPU Training CI"
3. Click "Run workflow"
4. Choose TPU type and number of steps

### Via CLI
```bash
gh workflow run tpu-training.yml --repo 42euge/ee596-fp \
  -f num_steps=10 \
  -f tpu_type=v5litepod-4
```

---

## Configuration Reference

| Setting | Value | Location |
|---------|-------|----------|
| GCP Project | `kaggle-euge` | `.github/workflows/tpu-training.yml` |
| TPU Zone | `us-central1-a` | `.github/workflows/tpu-training.yml` |
| Service Account | `github-tpu-ci@kaggle-euge.iam.gserviceaccount.com` | GCP IAM |
| GitHub Repo | `42euge/ee596-fp` | GitHub |

---

## Troubleshooting

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
