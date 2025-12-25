#!/usr/bin/env python3
"""GRPO Training script for TPU with W&B logging."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import (
    parse_args, patch_tunix, check_tpu_availability, check_jax_deps,
    check_tunix_deps, create_mesh, init_wandb, finish_wandb,
)
from scripts.model_utils import download_model, load_models
from scripts.training_config import create_cluster_config, create_grpo_config, load_datasets
from scripts.reward_utils import setup_rewards


def main():
    args = parse_args()
    patch_tunix()

    print("=" * 60)
    print("GRPO Training with W&B Logging")
    print("=" * 60)

    # Check dependencies
    print("\n[1/5] Checking environment...")
    if not check_jax_deps():
        sys.exit(1)

    print("\n[2/5] Checking TPU...")
    num_devices, has_tpu = check_tpu_availability()

    print("\n[3/5] Initializing W&B...")
    wandb_run = init_wandb(args, num_devices, has_tpu)

    if not check_tunix_deps():
        sys.exit(1)

    # Download model
    print(f"\n[4/5] Downloading model: {args.model_id}...")
    model_path, eos_tokens = download_model(args.model_id)

    # Create mesh and load models
    print("\n[5/5] Loading model and starting training...")
    mesh = create_mesh(num_devices)

    with mesh:
        policy_model, reference_model, tokenizer, eos_tokens = load_models(
            model_path, mesh, args, eos_tokens
        )

        train_ds, val_ds, test_ds = load_datasets(args)
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds) if val_ds else 0}, Test: {len(test_ds)}")

        # Create configs
        cluster_config = create_cluster_config(args, mesh, eos_tokens)
        grpo_config = create_grpo_config(args)

        # Setup rewards
        reward_fns, reward_logger = setup_rewards(args, wandb_enabled=wandb_run is not None)

        # Create trainer
        from tunix.rl import rl_cluster as rl_cluster_lib
        from tunix.rl.grpo.grpo_learner import GRPOLearner

        rl_cluster = rl_cluster_lib.RLCluster(
            actor=policy_model,
            reference=reference_model,
            tokenizer=tokenizer,
            cluster_config=cluster_config,
        )

        trainer = GRPOLearner(
            rl_cluster=rl_cluster,
            reward_fns=reward_fns,
            algo_config=grpo_config,
        )

        # Train
        print("\n" + "=" * 60)
        print(f"Training: {args.num_steps} steps, lr={args.learning_rate}, LoRA={args.use_lora}")
        print("=" * 60 + "\n")

        trainer.train(train_ds, val_ds)

    finish_wandb(reward_logger)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE - Checkpoints: {args.checkpoint_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
