# Gemma3-1B Reasoning Model with GRPO Fine-tuning

This project fine-tunes Google's Gemma3-1B model using Group Relative Policy Optimization (GRPO) to improve step-by-step reasoning capabilities. The model learns to produce structured reasoning traces with explicit `<reasoning>` and `<answer>` sections.

## Project Overview

**Goal**: Enhance Gemma3-1B's reasoning abilities through reinforcement learning, training it to:
- Think through problems step-by-step
- Show clear reasoning processes
- Provide structured answers

**Approach**:
- **Base Model**: Gemma3-1B-IT (instruction-tuned)
- **Fine-tuning Method**: GRPO with LoRA adapters
- **Training Data**: OpenRubrics dataset with rubric-based reward signals
- **Reward Functions**: Rubric-as-Reward (RaR) scoring + format compliance

## 🚀 Automated Development Pipelines

This project includes comprehensive automation for the entire reward model development lifecycle:

- ✅ **One-command setup** with `make quickstart`
- ✅ **Automated dataset preparation** with validation
- ✅ **Training orchestration** on local or TPU with W&B tracking
- ✅ **Automated evaluation** with comprehensive metrics
- ✅ **One-click deployment** to HuggingFace Hub
- ✅ **Real-time monitoring** with training dashboards
- ✅ **CI/CD integration** with GitHub Actions
- ✅ **Code quality checks** with pre-commit hooks

**Quick Start:**
```bash
# Full setup (install dependencies, prepare dataset)
make quickstart

# Start training
make train

# Evaluate model
make evaluate

# Monitor training
make monitor RUN=<run_name>

# Deploy checkpoint
make deploy CHECKPOINT=./checkpoints/step_1000 REPO_ID=username/model
```

**Documentation:**
- 📖 [Complete Pipeline Guide](docs/PIPELINE_GUIDE.md) - Full documentation
- 📋 [Quick Reference](docs/QUICK_REFERENCE.md) - Cheat sheet
- 🔧 [CI/CD Setup](docs/CICD_SETUP.md) - GitHub Actions setup

## Repository Structure

```
├── README.md              # This file
├── Makefile               # Development automation (make quickstart, make train, etc.)
├── requirements.txt       # Python dependencies
├── .pre-commit-config.yaml # Code quality hooks
├── src/
│   ├── __init__.py       # Package initialization
│   ├── main.py           # Entry point for inference/evaluation
│   ├── model.py          # Model loading and inference code
│   ├── config.py         # Hyperparameters and configuration
│   └── utils.py          # Helper functions (data loading, rewards, etc.)
├── scripts/               # Automation pipelines
│   ├── reward_pipeline.py    # Main CLI for all pipelines
│   ├── prepare_dataset.py    # Dataset preparation automation
│   ├── train_grpo.py         # GRPO training script
│   ├── evaluate_model.py     # Model evaluation automation
│   ├── deploy_checkpoint.py  # HuggingFace deployment automation
│   ├── monitor_training.py   # Training metrics dashboard
│   └── setup_tpu_vm.sh       # TPU environment setup
├── TunRex/                # Dataset toolkit (git subtree)
│   └── src/tunrex/datasets/  # Dataset loading, rewards, evaluation
├── .github/workflows/     # CI/CD automation
│   ├── auto-evaluation.yml      # Automated evaluation on PRs
│   ├── tpu-training.yml         # Quick TPU validation
│   └── tpu-training-full.yml    # Full TPU training
├── docs/
│   ├── PIPELINE_GUIDE.md  # Complete pipeline documentation
│   ├── QUICK_REFERENCE.md # Quick reference cheat sheet
│   └── CICD_SETUP.md      # GitHub Actions setup guide
├── demo/
│   └── demo.py           # Interactive demo script
├── data/                  # Prepared datasets (generated)
├── checkpoints/           # Saved model weights (generated)
└── logs/                  # Training logs and evaluation results (generated)
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ee596-fp
```

### 2. Install Dependencies

**Using uv (recommended):**
```bash
uv sync
```

To activate the virtual environment:
```bash
source .venv/bin/activate
```

Or run commands directly without activating:
```bash
uv run python demo/demo.py
```

**Using pip:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Authenticate with HuggingFace (Required)

Gemma is a gated model. You must accept the license and authenticate:

1. Visit [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) and accept the license
2. Create a HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Login via CLI:
```bash
huggingface-cli login
```

### 4. Download Pre-trained Model (Optional)

If you have fine-tuned LoRA weights, place them in the `checkpoints/` directory:
```
checkpoints/
└── lora/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

**Pre-trained Model Link**: [TODO: Add Google Drive/HuggingFace link]

## How to Run

### Interactive Demo

Run the demo script for an interactive reasoning session:

```bash
python demo/demo.py
```

With fine-tuned checkpoint:
```bash
python demo/demo.py --checkpoint ./checkpoints/lora
```

### Run Example Problems

See the model solve example problems across different categories:

```bash
python demo/demo.py --examples
```

### Device Selection

The demo automatically detects the best available device (CUDA > MPS > CPU). You can override this:

```bash
# Force CPU
python demo/demo.py --device cpu

# Force CUDA (if available)
python demo/demo.py --device cuda

# Force MPS (Apple Silicon)
python demo/demo.py --device mps
```

### Evaluation on GSM8K

Evaluate the model on the GSM8K math benchmark:

```bash
python -m src.main --mode evaluate --num-samples 100 --output results/eval.json
```

## Expected Output

When you run the demo, you should see output like:

```
======================================================================
  Gemma3-1B Reasoning Model - Demo
  Fine-tuned with GRPO for improved step-by-step reasoning
======================================================================

🖥️  Device: mps
📁 Using base model (no fine-tuned checkpoint)

⏳ Loading model (this may take a minute)...
✅ Model loaded successfully!

💬 Interactive Mode
   Enter your questions below. Type 'quit' or 'exit' to stop.

❓ Your question: How many apples does John have if he starts with 5 and buys 3 more?

⏳ Thinking...

----------------------------------------------------------------------
Question: How many apples does John have if he starts with 5 and buys 3 more?
----------------------------------------------------------------------

📝 REASONING:
   Let's solve this step by step. John starts with 5 apples.
   He buys 3 more apples. To find the total number of apples,
   we add: 5 + 3 = 8 apples.

✅ ANSWER:
   8

----------------------------------------------------------------------
```

## Configuration

Key hyperparameters in `src/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| LoRA Rank | 64 | Low-rank adaptation dimension |
| LoRA Alpha | 64.0 | Scaling factor for LoRA |
| Learning Rate | 3e-6 | Training learning rate |
| Temperature | 0.9 | Generation temperature during training |
| Beta (KL) | 0.08 | KL divergence penalty coefficient |
| Max Generation | 512 | Maximum tokens to generate |

## Training (Advanced)

Training requires a TPU environment (Google Colab or Kaggle recommended). Use the training notebook at `demo/train_colab.ipynb` for the full training pipeline using Google's Tunix library.

Key training components:
- **GRPO**: Group Relative Policy Optimization for RL fine-tuning
- **Rubric-as-Reward**: Uses rubric overlap and reference similarity for reward signals
- **LoRA**: Parameter-efficient fine-tuning with low-rank adapters

### Creating and Using Checkpoints

The training notebook automatically saves checkpoints during training:

**Checkpoint Configuration:**
- Checkpoints are saved every 100 training steps via Orbax CheckpointManager
- The 3 most recent checkpoints are kept (`max_to_keep=3`)
- Set `SAVE_TO_DRIVE=True` in the notebook to persist checkpoints to Google Drive
- Checkpoints are saved to: `{CHECKPOINT_DIR}/actor/{step}/`

**To use checkpoints locally:**

1. After training completes, download `checkpoint_export.zip` from Google Drive (if using `SAVE_TO_DRIVE=True`)
2. Extract to your local `checkpoints/` directory:
   ```bash
   unzip checkpoint_export.zip -d checkpoints/
   ```
3. Run the demo with your checkpoint:
   ```bash
   python demo/demo.py --checkpoint ./checkpoints/actor/<step>/model_params
   ```

**Checkpoint Directory Structure:**
```
checkpoints/
└── actor/
    ├── 100/
    │   └── model_params/
    ├── 200/
    │   └── model_params/
    └── 300/
        └── model_params/
```

## Model Architecture

- **Base Model**: Gemma3-1B-IT (1 billion parameters)
- **Architecture**: Decoder-only transformer
- **Fine-tuning**: LoRA adapters on attention layers
- **Prompt Format**: Gemma chat template with `<start_of_turn>` tokens

## Acknowledgments

- **Google DeepMind**: Gemma3 model and Tunix training library
- **OpenRubrics Dataset**: Training data with rubric-based evaluations
- **GSM8K Dataset**: Math reasoning evaluation benchmark
- **GRPO Paper**: Group Relative Policy Optimization methodology

## References

- [Gemma Model Card](https://ai.google.dev/gemma)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Rubric-as-Reward Paper](https://arxiv.org/pdf/2507.17746)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
