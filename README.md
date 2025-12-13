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

## Repository Structure

```
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── src/
│   ├── __init__.py       # Package initialization
│   ├── main.py           # Entry point for inference/evaluation
│   ├── model.py          # Model loading and inference code
│   ├── config.py         # Hyperparameters and configuration
│   └── utils.py          # Helper functions (data loading, rewards, etc.)
├── demo/
│   └── demo.py           # Interactive demo script
├── data/                  # Dataset files (download separately)
├── checkpoints/           # Saved model weights
└── results/               # Generated outputs and evaluation results
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ee596-fp
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Authenticate with HuggingFace (Required)

Gemma is a gated model. You must accept the license and authenticate:

1. Visit [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it) and accept the license
2. Create a HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Login via CLI:
```bash
huggingface-cli login
```

### 5. Download Pre-trained Model (Optional)

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

Training requires a TPU environment (Google Colab recommended). See the original notebook `reference.ipynb` for the full training pipeline using Google's Tunix library.

Key training components:
- **GRPO**: Group Relative Policy Optimization for RL fine-tuning
- **Rubric-as-Reward**: Uses rubric overlap and reference similarity for reward signals
- **LoRA**: Parameter-efficient fine-tuning with low-rank adapters

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
