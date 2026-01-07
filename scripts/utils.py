"""
Common helper utilities for CLI scripts in this repository.
"""

import argparse
import os
import sys


def patch_tunix():
    """Fix tunix.rl.utils.is_positive_integer to handle int type."""
    from tunix.rl import utils as rl_utils

    def is_positive_integer_fixed(value, name: str):
        if value is not None:
            is_int = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
            if not is_int or value <= 0:
                raise ValueError(f"{name} must be a positive integer. Got: {value}")

    rl_utils.is_positive_integer = is_positive_integer_fixed


def get_mesh_config(num_devices: int):
    """Get mesh configuration based on number of devices."""
    if num_devices >= 4:
        return (1, 4)
    elif num_devices == 1:
        return (1, 1)
    else:
        return (1, num_devices)


def create_mesh(num_devices: int):
    """Create JAX mesh for distributed training."""
    import jax

    mesh_counts = get_mesh_config(num_devices)
    print(f"  Mesh config: {mesh_counts}")

    return jax.make_mesh(
        mesh_counts,
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2
    )


def init_wandb(args, num_devices: int, has_tpu: bool):
    """Initialize Weights & Biases logging.

    Returns:
        wandb run object if enabled, None otherwise
    """
    if args.no_wandb or not os.environ.get("WANDB_API_KEY"):
        print("  W&B disabled (no API key or --no-wandb flag)")
        return None

    try:
        import wandb

        run_name = args.run_name or f"grpo-{args.model_id.split('/')[-1]}-steps{args.num_steps}"

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_id": args.model_id,
                "num_steps": args.num_steps,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "use_lora": args.use_lora,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "num_generations": args.num_generations,
                "beta": args.beta,
                "epsilon": args.epsilon,
                "temperature": args.temperature,
                "max_prompt_length": args.max_prompt_length,
                "max_generation_steps": args.max_generation_steps,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,
                "warmup_fraction": args.warmup_fraction,
                "num_tpu_cores": num_devices,
                "has_tpu": has_tpu,
                "rubric_file": args.rubric_file or None,
                "rubric_weight": args.rubric_weight if args.rubric_file else None,
                "advantage_estimator": args.advantage_estimator,
                "kl_in_reward": args.kl_in_reward if args.advantage_estimator == "rloo" else None,
                "advantage_clip": args.advantage_clip if args.advantage_estimator == "rloo" else None,
            }
        )
        print(f"  W&B initialized: {wandb.run.url}")
        return wandb.run
    except Exception as e:
        print(f"  WARNING: Failed to initialize W&B: {e}")
        return None


def finish_wandb(reward_logger=None):
    """Finish W&B run and log final summaries."""
    try:
        import wandb
        if wandb.run is None:
            return

        if reward_logger:
            reward_logger.log_summary()
            if hasattr(reward_logger, 'log_rubric_summary'):
                reward_logger.log_rubric_summary()

        wandb.finish()
    except Exception:
        pass


def check_jax_deps():
    """Check JAX dependencies are available."""
    try:
        import jax
        import flax
        import optax
        print(f"  JAX version: {jax.__version__}")
        print(f"  Flax version: {flax.__version__}")
        print(f"  Optax version: {optax.__version__}")
        return True
    except ImportError as e:
        print(f"ERROR: Missing JAX dependencies: {e}")
        return False


def check_tunix_deps():
    """Check tunix and tunrex imports."""
    try:
        from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
        from tunrex.datasets import get_train_val_test_datasets
        print("  Tunix/TunRex imports: OK")
        return True
    except ImportError as e:
        print(f"ERROR: Failed to import dependencies: {e}")
        return False


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

    # RLOO parameters
    parser.add_argument("--advantage-estimator", type=str, default="grpo",
                        choices=["grpo", "rloo"],
                        help="Advantage estimator: 'grpo' (default) or 'rloo'")
    parser.add_argument("--kl-in-reward", action="store_true",
                        help="For RLOO: fold KL directly into reward (R' = R - β*KL)")
    parser.add_argument("--advantage-clip", type=float, default=None,
                        help="For RLOO: clip advantages to prevent outliers")

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
