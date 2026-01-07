"""Training configuration utilities for GRPO."""

import optax
from orbax import checkpoint as ocp

from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger


def load_datasets(args):
    """Load train/val/test datasets based on args."""
    from tunrex.datasets import get_train_val_test_datasets, get_system_prompt, DEFAULT_TEMPLATE

    return get_train_val_test_datasets(
        train_data_dir="./data/train",
        test_data_dir="./data/test",
        source="tfds",
        batch_size=args.batch_size,
        num_batches=args.num_steps,
        num_test_batches=min(64, args.num_steps // 10 + 1),
        train_fraction=args.train_fraction,
        num_epochs=1,
        template=DEFAULT_TEMPLATE,
        system_prompt=get_system_prompt(0),
    )


def create_optimizer(args):
    """Create AdamW optimizer with warmup and cosine decay."""
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

    return optimizer


def create_cluster_config(args, mesh, eos_tokens: list):
    """Create RL cluster configuration.

    Args:
        args: Training arguments
        mesh: JAX mesh
        eos_tokens: List of EOS token IDs

    Returns:
        ClusterConfig for RL training
    """
    optimizer = create_optimizer(args)

    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=args.save_interval,
        max_to_keep=args.max_checkpoints
    )

    run_name = args.run_name or f"grpo-{args.model_id.split('/')[-1]}"
    metrics_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir="/tmp/tensorboard/grpo",
        project_name=args.wandb_project,
        run_name=run_name,
        flush_every_n_steps=20
    )

    return rl_cluster_lib.ClusterConfig(
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
            eos_tokens=eos_tokens,
        ),
    )


def create_grpo_config(args):
    """Create GRPO algorithm configuration."""
    return GRPOConfig(
        num_generations=args.num_generations,
        num_iterations=1,
        beta=args.beta,
        epsilon=args.epsilon,
    )


def create_rloo_config(args):
    """Create RLOO algorithm configuration."""
    from scripts.rloo_learner import RLOOConfig

    return RLOOConfig(
        num_generations=args.num_generations,
        num_iterations=1,
        beta=args.beta,
        epsilon=args.epsilon,
        kl_in_reward=args.kl_in_reward,
        advantage_clip=args.advantage_clip,
    )
