#!/usr/bin/env python3
"""
GRPO Training script for TPU with W&B logging.

This script performs full GRPO training with:
- Configurable hyperparameters
- Weights & Biases integration for experiment tracking
- Checkpoint saving
- LoRA support

Usage:
    python scripts/train_grpo.py --num-steps 100 --model-id google/gemma-3-1b-it
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Monkey-patch tunix bug: is_positive_integer calls .is_integer() which only works on float
def _patch_tunix_utils():
    """Fix tunix.rl.utils.is_positive_integer to handle int type."""
    from tunix.rl import utils as rl_utils

    def is_positive_integer_fixed(value, name: str):
        if value is not None:
            is_int = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
            if not is_int or value <= 0:
                raise ValueError(f"{name} must be a positive integer. Got: {value}")

    rl_utils.is_positive_integer = is_positive_integer_fixed

_patch_tunix_utils()


def parse_args():
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


def get_mesh_config(num_devices):
    """Get mesh configuration based on number of devices."""
    if num_devices == 8:
        return (1, 4)
    elif num_devices == 4:
        return (1, 4)
    elif num_devices == 1:
        return (1, 1)
    else:
        # Default: try to split evenly
        return (1, num_devices)


def main():
    args = parse_args()

    print("=" * 60)
    print("GRPO Training with W&B Logging")
    print("=" * 60)

    # Step 1: Check environment
    print("\n[1/8] Checking environment...")

    try:
        import jax
        import flax
        import optax
        print(f"  JAX version: {jax.__version__}")
        print(f"  Flax version: {flax.__version__}")
        print(f"  Optax version: {optax.__version__}")
    except ImportError as e:
        print(f"ERROR: Missing JAX dependencies: {e}")
        sys.exit(1)

    # Step 2: Check TPU
    print("\n[2/8] Checking TPU availability...")
    num_devices, has_tpu = check_tpu_availability()

    # Step 3: Initialize W&B
    print("\n[3/8] Initializing W&B...")
    wandb_enabled = not args.no_wandb and os.environ.get("WANDB_API_KEY")

    if wandb_enabled:
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
                }
            )
            print(f"  W&B initialized: {wandb.run.url}")
        except Exception as e:
            print(f"  WARNING: Failed to initialize W&B: {e}")
            wandb_enabled = False
    else:
        print("  W&B disabled (no API key or --no-wandb flag)")

    # Step 4: Import training dependencies
    print("\n[4/8] Loading training dependencies...")
    try:
        from flax import nnx
        from huggingface_hub import snapshot_download
        import jax.numpy as jnp
        from orbax import checkpoint as ocp
        import qwix
        from tunix.generate import sampler as sampler_lib
        from tunix.generate import tokenizer_adapter as tokenizer_lib
        from tunix.models.gemma3 import model as gemma_lib
        from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
        from tunix.rl import rl_cluster as rl_cluster_lib
        from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
        from tunix.rl.rollout import base_rollout
        from tunix.sft import metrics_logger
        print("  Tunix imports: OK")
    except ImportError as e:
        print(f"ERROR: Failed to import dependencies: {e}")
        sys.exit(1)

    try:
        from tunrex.datasets import (
            get_train_val_test_datasets,
            get_system_prompt,
            DEFAULT_TEMPLATE,
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        )
        print("  TunRex imports: OK")
    except ImportError as e:
        print(f"ERROR: Failed to import TunRex: {e}")
        sys.exit(1)

    # Step 5: Download model
    print(f"\n[5/8] Downloading model: {args.model_id}...")

    # Get HF token for gated models
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("  Using HF_TOKEN for authentication")
    else:
        print("  WARNING: HF_TOKEN not set, gated models may fail to download")

    ignore_patterns = ["*.pth"]
    local_model_path = snapshot_download(
        repo_id=args.model_id,
        ignore_patterns=ignore_patterns,
        token=hf_token
    )
    print(f"  Model downloaded to: {local_model_path}")

    # Get EOS tokens
    EOS_TOKENS = []
    generation_config_path = os.path.join(local_model_path, "generation_config.json")
    if os.path.exists(generation_config_path):
        with open(generation_config_path, "r") as f:
            gen_config = json.load(f)
        EOS_TOKENS = gen_config.get("eos_token_id", [])
        print(f"  EOS tokens: {EOS_TOKENS}")

    # Step 6: Load model and create mesh
    print("\n[6/8] Loading model and creating mesh...")

    # Determine model config
    if "gemma-3-270m" in args.model_id:
        model_config = gemma_lib.ModelConfig.gemma3_270m()
    elif "gemma-3-1b" in args.model_id:
        model_config = gemma_lib.ModelConfig.gemma3_1b_it()
    else:
        print(f"WARNING: Unknown model, defaulting to gemma3_1b_it config")
        model_config = gemma_lib.ModelConfig.gemma3_1b_it()

    mesh_counts = get_mesh_config(num_devices)
    print(f"  Mesh config: {mesh_counts}")

    mesh = jax.make_mesh(
        mesh_counts,
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2
    )

    with mesh:
        # Load reference model
        print("  Loading reference model...")
        reference_model = params_safetensors_lib.create_model_from_safe_tensors(
            local_model_path, model_config, mesh
        )

        # Load/create policy model
        if args.use_lora:
            print(f"  Creating LoRA model (rank={args.lora_rank}, alpha={args.lora_alpha})...")

            lora_provider = qwix.LoraProvider(
                module_path=(
                    ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
                    ".*attn_vec_einsum"
                ),
                rank=args.lora_rank,
                alpha=args.lora_alpha,
            )

            base_for_lora = params_safetensors_lib.create_model_from_safe_tensors(
                local_model_path, model_config, mesh
            )
            model_input = base_for_lora.get_model_input()
            policy_model = qwix.apply_lora_to_model(
                base_for_lora, lora_provider, **model_input
            )

            # Shard LoRA model
            state = nnx.state(policy_model)
            pspecs = nnx.get_partition_spec(state)
            sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
            nnx.update(policy_model, sharded_state)
        else:
            print("  Using full model (no LoRA)...")
            policy_model = params_safetensors_lib.create_model_from_safe_tensors(
                local_model_path, model_config, mesh
            )

    # Load tokenizer
    GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"
    tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
    if tokenizer.eos_id() not in EOS_TOKENS:
        EOS_TOKENS.append(tokenizer.eos_id())

    # Step 7: Load dataset
    print("\n[7/8] Loading dataset...")
    SYSTEM_PROMPT = get_system_prompt(0)

    num_batches = args.num_steps  # One batch per step
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        train_data_dir="./data/train",
        test_data_dir="./data/test",
        source="tfds",
        batch_size=args.batch_size,
        num_batches=num_batches,
        num_test_batches=min(64, num_batches // 10 + 1),
        train_fraction=args.train_fraction,
        num_epochs=1,
        template=DEFAULT_TEMPLATE,
        system_prompt=SYSTEM_PROMPT,
    )
    print(f"  Train batches: {len(train_dataset)}")
    print(f"  Val batches: {len(val_dataset) if val_dataset else 0}")
    print(f"  Test batches: {len(test_dataset)}")

    # Step 8: Configure and run training
    print("\n[8/8] Starting training...")

    # Checkpointing options
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=args.save_interval,
        max_to_keep=args.max_checkpoints
    )

    # Metrics logger with W&B
    run_name = args.run_name or f"grpo-{args.model_id.split('/')[-1]}"
    metrics_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir="/tmp/tensorboard/grpo",
        project_name=args.wandb_project,
        run_name=run_name,
        flush_every_n_steps=20
    )

    # Optimizer with warmup and cosine decay
    warmup_steps = int(args.warmup_fraction * args.num_steps)
    optimizer = optax.adamw(
        learning_rate=optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=args.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=args.num_steps,
            end_value=0.0,
        ),
        b1=0.9,
        b2=0.99,
        weight_decay=args.weight_decay,
    )

    if args.max_grad_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_norm=args.max_grad_norm),
            optimizer,
        )

    # Training config
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine='vanilla',
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=args.eval_every,
            max_steps=args.num_steps,
            mini_batch_size=args.batch_size,
            train_micro_batch_size=args.batch_size,
            metrics_logging_options=metrics_logging_options,
            checkpoint_root_directory=args.checkpoint_dir,
            checkpointing_options=checkpointing_options,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=args.max_generation_steps,
            max_prompt_length=args.max_prompt_length,
            kv_cache_size=args.max_prompt_length + args.max_generation_steps + 256,
            temperature=args.temperature,
            top_p=1.0,
            top_k=50,
            eos_tokens=EOS_TOKENS,
        ),
    )

    grpo_config = GRPOConfig(
        num_generations=args.num_generations,
        num_iterations=1,
        beta=args.beta,
        epsilon=args.epsilon,
    )

    # Create RL cluster and trainer
    with mesh:
        rl_cluster = rl_cluster_lib.RLCluster(
            actor=policy_model,
            reference=reference_model,
            tokenizer=tokenizer,
            cluster_config=cluster_config,
        )

        grpo_trainer = GRPOLearner(
            rl_cluster=rl_cluster,
            reward_fns=[
                match_format_exactly,
                match_format_approximately,
                check_answer,
                check_numbers,
            ],
            algo_config=grpo_config,
        )

        # Run training
        print("\n" + "=" * 60)
        print("Starting GRPO Training")
        print("=" * 60)
        print(f"  Steps: {args.num_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  LoRA: {args.use_lora}")
        print(f"  Checkpoints: {args.checkpoint_dir}")
        print("=" * 60 + "\n")

        grpo_trainer.train(train_dataset, val_dataset)

    # Finish W&B
    if wandb_enabled:
        import wandb
        wandb.finish()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Checkpoints saved to: {args.checkpoint_dir}")

    sys.exit(0)


if __name__ == "__main__":
    main()
