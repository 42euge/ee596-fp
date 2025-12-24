"""
Common helper utilities for CLI scripts in this repository.
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse GRPO training CLI arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training on TPU")

    # Training parameters
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of training steps")
    parser.add_argument("--model-id", type=str, default="google/gemma-3-1b-it",
                        help="HuggingFace model ID")
    parser.add_argument("--learning-rate", type=float, default=3e-6,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Micro batch size")
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA for training")
    parser.add_argument("--lora-rank", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=64.0,
                        help="LoRA alpha")

    # GRPO parameters
    parser.add_argument("--num-generations", type=int, default=2,
                        help="Number of generations per prompt (G in GRPO)")
    parser.add_argument("--beta", type=float, default=0.08,
                        help="KL divergence coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="Clipping epsilon")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature during training")

    # Generation parameters
    parser.add_argument("--max-prompt-length", type=int, default=256,
                        help="Maximum prompt length")
    parser.add_argument("--max-generation-steps", type=int, default=768,
                        help="Maximum generation steps")

    # Optimizer parameters
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=0.1,
                        help="Maximum gradient norm for clipping")
    parser.add_argument("--warmup-fraction", type=float, default=0.1,
                        help="Fraction of steps for warmup")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="/tmp/grpo_checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save checkpoint every N steps")
    parser.add_argument("--max-checkpoints", type=int, default=3,
                        help="Maximum number of checkpoints to keep")

    # W&B parameters
    parser.add_argument("--wandb-project", type=str, default="tunix-grpo",
                        help="W&B project name")
    parser.add_argument("--run-name", type=str, default="",
                        help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging")

    # Data parameters
    parser.add_argument("--train-fraction", type=float, default=0.9,
                        help="Fraction of data for training")
    parser.add_argument("--eval-every", type=int, default=50,
                        help="Evaluate every N steps")

    # Rubric parameters
    parser.add_argument("--rubric-file", type=str, default=None,
                        help="Path to rubric YAML file for rubric-based reward")
    parser.add_argument("--rubric-weight", type=float, default=1.0,
                        help="Weight for rubric reward function")

    return parser.parse_args()


def check_tpu_availability():
    """Check if TPU is available and return device info."""
    import jax

    devices = jax.devices()
    print(f"JAX devices: {devices}")

    tpu_devices = [d for d in devices if d.platform == "tpu"]
    if not tpu_devices:
        print("WARNING: No TPU devices found. Running on CPU/GPU.")
        return len(devices), False

    print(f"Found {len(tpu_devices)} TPU core(s)")
    return len(tpu_devices), True
