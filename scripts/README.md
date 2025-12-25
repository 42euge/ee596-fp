# Scripts

Training and utility scripts for GRPO (Group Relative Policy Optimization) on TPU.

## Scripts

### `train_grpo.py`

Full GRPO training script with W&B logging, checkpointing, and LoRA support.

```bash
python scripts/train_grpo.py --num-steps 100 --model-id google/gemma-3-1b-it --use-lora
```

**Key options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--num-steps` | 100 | Number of training steps |
| `--model-id` | google/gemma-3-1b-it | HuggingFace model ID |
| `--learning-rate` | 3e-6 | Learning rate |
| `--batch-size` | 1 | Micro batch size |
| `--use-lora` | false | Enable LoRA training |
| `--lora-rank` | 64 | LoRA rank |
| `--rubric-file` | None | Path to rubric YAML for reward |
| `--no-wandb` | false | Disable W&B logging |

**Environment variables:**
- `HF_TOKEN` - HuggingFace token for gated models
- `WANDB_API_KEY` - Weights & Biases API key

### `train_small.py`

Minimal validation script to test TPU environment and dependencies.

```bash
python scripts/train_small.py --dry-run  # Check environment only
python scripts/train_small.py --num-steps 10  # Quick validation run
```

### `setup_tpu_vm.sh`

Setup script for TPU VM environment. Installs Python 3.11, uv, and all dependencies.

```bash
./scripts/setup_tpu_vm.sh
```

### `utils.py`

Shared utilities:
- `parse_args()` - CLI argument parser for training scripts
- `check_tpu_availability()` - Detect TPU devices

## Typical Workflow

1. **Test rubric locally:**
   ```bash
   python tests/test_rubric_reward.py --rubric-file rubrics/my_rubric.yaml
   ```

2. **Validate TPU setup:**
   ```bash
   python scripts/train_small.py --dry-run
   ```

3. **Run full training:**
   ```bash
   python scripts/train_grpo.py \
     --num-steps 500 \
     --use-lora \
     --rubric-file rubrics/my_rubric.yaml \
     --wandb-project my-project
   ```

## GitHub Actions

These scripts are used by the TPU training workflows:
- `.github/workflows/tpu-training-full.yml` - Full training on GCP TPU
- `.github/workflows/tpu-training-docker.yml` - Training using Docker image
