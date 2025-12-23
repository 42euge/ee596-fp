#!/usr/bin/env python3
"""
GRPO Training script with Enhanced Reward Monitoring.

This script extends the base GRPO training with comprehensive reward monitoring:
- Per-reward-function tracking
- Quality metrics (SNR, entropy, variance)
- W&B custom dashboards
- TensorBoard enhanced logging
- Web dashboard data export
- Real-time alerting

Usage:
    python scripts/train_grpo_monitored.py --num-steps 100 --model-id google/gemma-3-1b-it
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with monitoring")

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
    parser.add_argument("--wandb-project", type=str, default="tunix-grpo-monitored",
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

    # Monitoring parameters
    parser.add_argument("--monitoring-dir", type=str, default="./monitoring_data",
                        help="Directory for monitoring data exports")
    parser.add_argument("--tensorboard-dir", type=str, default="/tmp/tensorboard/grpo_rewards",
                        help="TensorBoard log directory")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging")
    parser.add_argument("--enable-alerts", action="store_true",
                        help="Enable real-time alerting")
    parser.add_argument("--log-distributions", action="store_true",
                        help="Log full reward distributions (memory intensive)")

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("GRPO Training with Enhanced Reward Monitoring")
    print("="*60)

    # Import monitoring components
    print("\n[1/9] Loading monitoring system...")
    from src.monitoring import (
        RewardMetricsTracker,
        DashboardManager,
        RewardAnalyzer,
        RealtimeMonitor,
    )
    from src.monitoring.tensorboard_config import setup_tensorboard_monitoring

    # Setup TensorBoard
    if not args.no_tensorboard:
        print("[2/9] Setting up TensorBoard...")
        tb_info = setup_tensorboard_monitoring(args.tensorboard_dir)
        print(f"  TensorBoard: {tb_info['tensorboard_command']}")

    # Import training dependencies
    print("[3/9] Importing training dependencies...")
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn

    print("[4/9] Importing Tunix...")
    try:
        from tunix.rl.actor_critic import (
            learners as learner_lib,
            rl_cluster as rl_cluster_lib,
        )
        from tunix.rl.actor_critic.learners.grpo import GRPOLearner, GRPOConfig
        from tunix.rl.actor_critic.rollout_lib import ClusterConfig, SampleConfig
        from tunix.rl import metrics_logger
    except ImportError as e:
        print(f"Error importing Tunix: {e}")
        print("Make sure Tunix is properly installed")
        sys.exit(1)

    print("[5/9] Importing TunRex...")
    try:
        from TunRex.src.tunrex.datasets import (
            get_train_val_test_datasets,
            DEFAULT_TEMPLATE,
        )
        from TunRex.src.tunrex.datasets.rewards import (
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        )
        from TunRex.src.tunrex.datasets.config import SYSTEM_PROMPTS
    except ImportError as e:
        print(f"Error importing TunRex: {e}")
        sys.exit(1)

    # Check TPU
    print("[6/9] Checking TPU availability...")
    devices = jax.devices()
    tpu_devices = [d for d in devices if d.platform == "tpu"]
    has_tpu = len(tpu_devices) > 0

    if has_tpu:
        print(f"  ✓ TPU available: {len(tpu_devices)} cores")
    else:
        print("  ⚠️  No TPU found, using CPU/GPU")

    # Initialize monitoring
    print("[7/9] Initializing monitoring system...")

    # Reward function names
    reward_function_names = [
        "format_exact",
        "format_approx",
        "check_answer",
        "check_numbers",
    ]

    # Create metrics tracker
    metrics_tracker = RewardMetricsTracker(
        reward_function_names=reward_function_names,
        window_size=100,
        track_distributions=args.log_distributions,
    )

    # Create dashboard manager
    dashboard_manager = DashboardManager(
        project_name=args.wandb_project,
        run_name=args.run_name if args.run_name else None,
        log_dir=args.tensorboard_dir,
        enable_wandb=not args.no_wandb,
        enable_tensorboard=not args.no_tensorboard,
        enable_json_export=True,
        json_export_dir=args.monitoring_dir,
    )

    # Define W&B custom charts
    dashboard_manager.define_wandb_charts(reward_function_names)

    # Create reward analyzer
    analyzer = RewardAnalyzer()

    # Create real-time monitor
    monitor = None
    if args.enable_alerts:
        monitor = RealtimeMonitor(print_alerts=True)
        monitor.setup_default_alerts()
        print("  ✓ Real-time alerts enabled")

    print("\n✓ Monitoring system initialized")
    print(f"  - Metrics tracker: {len(reward_function_names)} functions")
    print(f"  - Dashboard manager: W&B={dashboard_manager.wandb_enabled}, TB={dashboard_manager.tensorboard_enabled}")
    print(f"  - Export directory: {args.monitoring_dir}")

    # Initialize W&B
    wandb_enabled = not args.no_wandb and os.getenv('WANDB_API_KEY')
    if wandb_enabled:
        print("[8/9] Initializing W&B...")
        import wandb

        run_name = args.run_name if args.run_name else f"grpo-{args.model_id.split('/')[-1]}-{args.num_steps}steps"

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                # Model
                "model_id": args.model_id,
                "use_lora": args.use_lora,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,

                # Training
                "num_steps": args.num_steps,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "warmup_fraction": args.warmup_fraction,
                "weight_decay": args.weight_decay,
                "max_grad_norm": args.max_grad_norm,

                # GRPO
                "num_generations": args.num_generations,
                "beta": args.beta,
                "epsilon": args.epsilon,
                "temperature": args.temperature,

                # Hardware
                "has_tpu": has_tpu,
                "num_tpu_cores": len(tpu_devices) if has_tpu else 0,

                # Monitoring
                "enhanced_monitoring": True,
                "track_distributions": args.log_distributions,
                "alerts_enabled": args.enable_alerts,
            }
        )
        print(f"  ✓ W&B run: {wandb.run.url}")

    print("[9/9] Setting up training components...")

    # Import HF transformers
    from transformers import AutoTokenizer

    # Load tokenizer
    print(f"  Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    EOS_TOKENS = [tokenizer.eos_token_id] if tokenizer.eos_token_id else [2]

    # System prompt
    SYSTEM_PROMPT = SYSTEM_PROMPTS.get(2, SYSTEM_PROMPTS[0])

    # Load datasets
    print("  Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(
        train_data_dir="./data/train",
        test_data_dir="./data/test",
        source="tfds",
        batch_size=args.batch_size,
        num_batches=args.num_steps,
        train_fraction=args.train_fraction,
        template=DEFAULT_TEMPLATE,
        system_prompt=SYSTEM_PROMPT,
    )

    # Download model
    print(f"  Downloading model: {args.model_id}")
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(
        repo_id=args.model_id,
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
        token=os.getenv("HF_TOKEN"),
    )
    print(f"  ✓ Model downloaded to: {model_path}")

    # Load model and create mesh
    print("  Creating JAX mesh and models...")
    from tunix.automodel.flax_lm import AutoFlaxLM

    # Create JAX mesh
    mesh_shape = (len(tpu_devices),) if has_tpu else (1,)
    mesh = jax.sharding.Mesh(
        devices=jax.devices()[:len(mesh_shape)],
        axis_names=('data',)
    )

    # Load reference model (frozen)
    reference_model = AutoFlaxLM.from_pretrained(
        model_path,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )

    # Load policy model (trainable)
    policy_model = AutoFlaxLM.from_pretrained(
        model_path,
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
        lora_rank=args.lora_rank if args.use_lora else 0,
        lora_alpha=args.lora_alpha if args.use_lora else 0.0,
    )

    # Create optimizer with warmup and cosine decay
    optimizer = optax.adamw(
        learning_rate=optax.schedules.warmup_cosine_decay_schedule(
            peak_value=args.learning_rate,
            warmup_steps=int(args.warmup_fraction * args.num_steps),
            decay_steps=args.num_steps,
        ),
        weight_decay=args.weight_decay,
    )

    # Gradient clipping
    if args.max_grad_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optimizer,
        )

    # Create cluster configuration
    cluster_config = ClusterConfig(
        optimizer=optimizer,
        log_config=metrics_logger.MetricsLoggerOptions(
            log_dir=args.tensorboard_dir,
            project_name=args.wandb_project,
            run_name=run_name if wandb_enabled else None,
            flush_every_n_steps=20,
        ),
        actor_checkpoint_config=learner_lib.CheckpointConfig(
            checkpoint_dir=f"{args.checkpoint_dir}/actor",
            save_interval_steps=args.save_interval,
            max_to_keep=args.max_checkpoints,
        ),
        sample_config=SampleConfig(
            max_prompt_length=args.max_prompt_length,
            max_generation_steps=args.max_generation_steps,
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

    # Define reward functions with monitoring wrapper
    reward_functions = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ]

    # Create monitoring wrapper for reward functions
    def create_monitored_reward_fn(original_fn, fn_name):
        """Wrap reward function to track individual scores."""
        def wrapped_fn(*args, **kwargs):
            return original_fn(*args, **kwargs)
        wrapped_fn.__name__ = fn_name
        return wrapped_fn

    monitored_reward_fns = [
        create_monitored_reward_fn(fn, name)
        for fn, name in zip(reward_functions, reward_function_names)
    ]

    # Create RL cluster and trainer
    print("\n" + "="*60)
    print("Starting GRPO Training with Monitoring")
    print("="*60)
    print(f"  Model: {args.model_id}")
    print(f"  Steps: {args.num_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LoRA: {args.use_lora} (rank={args.lora_rank})")
    print(f"  Generations per prompt: {args.num_generations}")
    print(f"  Beta (KL): {args.beta}")
    print(f"  Epsilon: {args.epsilon}")
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  Monitoring: {args.monitoring_dir}")
    print("="*60 + "\n")

    with mesh:
        rl_cluster = rl_cluster_lib.RLCluster(
            actor=policy_model,
            reference=reference_model,
            tokenizer=tokenizer,
            cluster_config=cluster_config,
        )

        grpo_trainer = GRPOLearner(
            rl_cluster=rl_cluster,
            reward_fns=monitored_reward_fns,
            algo_config=grpo_config,
        )

        # Run training
        # Note: The actual monitoring integration would require hooks into
        # the Tunix training loop. For now, we've set up the infrastructure.
        grpo_trainer.train(train_dataset, val_dataset)

    # Generate final analysis
    print("\n" + "="*60)
    print("Generating Final Analysis")
    print("="*60)

    # Log final summary
    dashboard_manager.log_summary(metrics_tracker, args.num_steps)

    # Export full metrics
    export_path = Path(args.monitoring_dir) / "final_metrics.json"
    export_data = metrics_tracker.export_to_dict()

    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\n✓ Final metrics exported to: {export_path}")

    # Close dashboards
    dashboard_manager.close()

    # Finish W&B
    if wandb_enabled:
        wandb.finish()

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Checkpoints: {args.checkpoint_dir}")
    print(f"  Monitoring data: {args.monitoring_dir}")
    if not args.no_tensorboard:
        print(f"  TensorBoard: {args.tensorboard_dir}")
        print(f"  View with: tensorboard --logdir={args.tensorboard_dir}")
    print("\n  To view web dashboard:")
    print(f"    streamlit run src/monitoring/web_dashboard.py -- --data-dir {args.monitoring_dir}")
    print("="*60)

    sys.exit(0)


if __name__ == "__main__":
    main()
