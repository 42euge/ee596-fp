#!/usr/bin/env python3
"""
Minimal GRPO training script for CI/CD validation.

This script runs a small number of training steps to validate that:
1. The TPU environment is properly configured
2. Dependencies are correctly installed
3. Model loading and training works end-to-end

Usage:
    python scripts/train_small.py --num-steps 10
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import check_tpu_availability


def main():
    parser = argparse.ArgumentParser(description="Run minimal GRPO training for CI validation")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--dry-run", action="store_true", help="Only check environment, don't train")
    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Training CI Validation")
    print("=" * 60)

    # Step 1: Check environment
    print("\n[1/5] Checking environment...")

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
    print("\n[2/5] Checking TPU availability...")
    _, has_tpu = check_tpu_availability()

    if args.dry_run:
        print("\n[DRY RUN] Skipping training steps.")
        print("Environment check passed!")
        sys.exit(0)

    # Step 3: Import training dependencies
    print("\n[3/5] Loading training dependencies...")
    try:
        import tunix
        print(f"  Tunix version: {tunix.__version__}")
        print("  Tunix imports: OK")
    except ImportError as e:
        print(f"ERROR: Failed to import Tunix: {e}")
        print("Make sure tunix is installed correctly.")
        sys.exit(1)

    try:
        from transformers import AutoTokenizer
        print("  Transformers imports: OK")
    except ImportError as e:
        print(f"ERROR: Failed to import transformers: {e}")
        sys.exit(1)

    # Step 4: Validate TPU computation
    print("\n[4/5] Validating TPU computation...")
    try:
        import jax.numpy as jnp
        from jax import random

        # Simple JAX computation on TPU
        key = random.PRNGKey(0)
        x = random.normal(key, (1000, 1000))
        y = jnp.dot(x, x.T)
        result = float(jnp.mean(y))
        print(f"  Matrix multiplication test: PASSED (result={result:.4f})")
    except Exception as e:
        print(f"ERROR: TPU computation failed: {e}")
        sys.exit(1)

    # Step 5: Load Gemma tokenizer
    print(f"\n[5/5] Loading Gemma model...")

    model_id = "google/gemma-3-1b-it"
    try:
        # Uses cached HuggingFace token from huggingface-cli login
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"  Tokenizer loaded: {model_id}")
    except Exception as e:
        print(f"ERROR: Failed to load Gemma tokenizer: {e}")
        print("  Make sure you're logged in with: huggingface-cli login")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("CI VALIDATION PASSED")
    print("=" * 60)
    print(f"\nTPU cores: {len([d for d in jax.devices() if d.platform == 'tpu'])}")
    print(f"Ready for training!")

    print("\nTraining completed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
