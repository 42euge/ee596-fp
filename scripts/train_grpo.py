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

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import parse_args, check_tpu_availability


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
                    "rubric_file": args.rubric_file or None,
                    "rubric_weight": args.rubric_weight if args.rubric_file else None,
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

        # Build reward functions list with W&B logging
        base_reward_fns = [
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ]
        reward_names = [
            "format_exact",
            "format_approx",
            "answer_check",
            "number_check",
        ]

        # Load rubric if specified
        rubricset = None
        if args.rubric_file:
            from src.rubrics import load_rubricset_from_yaml
            print(f"  Loading rubric from: {args.rubric_file}")
            rubricset = load_rubricset_from_yaml(args.rubric_file)
            print(f"  Loaded rubric: {rubricset.name} with {len(rubricset.rubrics)} rubric(s)")

        # Wrap reward functions with W&B logging
        reward_logger = None
        if wandb_enabled:
            try:
                from src.wandb_rewards import create_logged_reward_fns
                reward_fns, reward_logger = create_logged_reward_fns(
                    base_reward_fns,
                    reward_names,
                    rubricset=rubricset,
                    rubric_weight=args.rubric_weight if args.rubric_file else 1.0,
                    log_every_n_steps=10,
                )
                print(f"  W&B reward logging enabled for {len(reward_fns)} reward functions")
            except ImportError as e:
                print(f"  WARNING: Could not enable W&B reward logging: {e}")
                reward_fns = base_reward_fns
                if rubricset:
                    from src.rubrics import create_grpo_reward_function
                    reward_fns.append(create_grpo_reward_function(rubricset, weight=args.rubric_weight))
        else:
            reward_fns = base_reward_fns
            if rubricset:
                from src.rubrics import create_grpo_reward_function
                reward_fns.append(create_grpo_reward_function(rubricset, weight=args.rubric_weight))

        grpo_trainer = GRPOLearner(
            rl_cluster=rl_cluster,
            reward_fns=reward_fns,
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

    # Log reward summary and finish W&B
    if wandb_enabled:
        import wandb
        if reward_logger:
            reward_logger.log_summary()
            if hasattr(reward_logger, 'log_rubric_summary'):
                reward_logger.log_rubric_summary()
        wandb.finish()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Checkpoints saved to: {args.checkpoint_dir}")

    sys.exit(0)


if __name__ == "__main__":
    main()
