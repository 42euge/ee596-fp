#!/usr/bin/env python3
"""Download models and datasets at Docker build time.

This script is called during the Layer 2 Docker build to pre-cache
models from HuggingFace and datasets from the configured source.

Usage:
    python download_assets.py --models gemma-3-1b-it --dataset-source tfds
    python download_assets.py --models both --dataset-source huggingface
"""

import argparse
import os
import sys

# Add app root to path for imports
sys.path.insert(0, "/app")


MODEL_ID_MAP = {
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-270m": "google/gemma-3-270m",
}


def download_models(model_specs: list[str]) -> None:
    """Download models from HuggingFace using model_utils.download_model.

    Args:
        model_specs: List of model specs (gemma-3-1b-it, gemma-3-270m, both)
    """
    from scripts.model_utils import download_model

    models_to_download = []
    for spec in model_specs:
        spec = spec.strip()
        if spec == "both":
            models_to_download.extend(["gemma-3-1b-it", "gemma-3-270m"])
        elif spec in MODEL_ID_MAP:
            models_to_download.append(spec)
        elif "/" in spec:
            # Allow full HuggingFace IDs
            models_to_download.append(spec)
        elif spec:
            print(f"WARNING: Unknown model spec '{spec}', skipping")

    for model_spec in models_to_download:
        full_id = MODEL_ID_MAP.get(model_spec, model_spec)
        print(f"\nDownloading model: {full_id}")
        try:
            local_path, eos_tokens = download_model(full_id)
            print(f"  Downloaded to: {local_path}")
            print(f"  EOS tokens: {eos_tokens}")
        except Exception as e:
            print(f"  ERROR downloading {full_id}: {e}")
            # Continue with other models


def download_datasets(source: str) -> None:
    """Pre-download datasets based on source.

    Uses the dataset loading functions to trigger downloads and cache them.

    Args:
        source: Data source (tfds, huggingface, kaggle)
    """
    if source == "tfds":
        print("\nDownloading GSM8K from TensorFlow Datasets...")
        try:
            from tunrex.datasets import (
                get_train_val_test_datasets,
                get_system_prompt,
                DEFAULT_TEMPLATE,
            )

            # This will trigger the download and cache it
            train_ds, val_ds, test_ds = get_train_val_test_datasets(
                train_data_dir="/app/data/train",
                test_data_dir="/app/data/test",
                source="tfds",
                batch_size=1,
                num_batches=1,  # Just need to trigger download
                num_test_batches=1,
                train_fraction=1.0,
                num_epochs=1,
                template=DEFAULT_TEMPLATE,
                system_prompt=get_system_prompt(0),
            )
            print(f"  Downloaded: {len(train_ds)} train, {len(test_ds)} test batches")
        except Exception as e:
            print(f"  ERROR downloading TFDS: {e}")
            raise

    elif source == "huggingface":
        print("\nDownloading GSM8K from HuggingFace...")
        try:
            from tunrex.datasets.loaders import load_from_huggingface

            train_ds = load_from_huggingface("gsm8k", split="train", subset="main")
            test_ds = load_from_huggingface("gsm8k", split="test", subset="main")
            print(f"  Downloaded: {len(train_ds)} train, {len(test_ds)} test examples")
        except Exception as e:
            print(f"  ERROR downloading from HuggingFace: {e}")
            raise

    elif source == "kaggle":
        print("\nDownloading GSM8K from Kaggle...")
        try:
            from tunrex.datasets.loaders import download_kaggle_dataset

            download_kaggle_dataset("/app/data/gsm8k")
            print("  Downloaded to /app/data/gsm8k")
        except Exception as e:
            print(f"  ERROR downloading from Kaggle: {e}")
            raise
    else:
        raise ValueError(f"Unknown dataset source: {source}")


def main():
    parser = argparse.ArgumentParser(
        description="Download models and datasets for Docker build"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gemma-3-1b-it",
        help="Comma-separated models: gemma-3-1b-it, gemma-3-270m, both",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="tfds",
        choices=["tfds", "huggingface", "kaggle"],
        help="Dataset source",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model downloads",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip dataset downloads",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Docker Asset Download")
    print("=" * 60)
    print(f"Models: {args.models}")
    print(f"Dataset source: {args.dataset_source}")
    print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")

    # Download models
    if not args.skip_models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        download_models(models)
    else:
        print("\nSkipping model downloads")

    # Download datasets
    if not args.skip_datasets:
        download_datasets(args.dataset_source)
    else:
        print("\nSkipping dataset downloads")

    print("\n" + "=" * 60)
    print("Asset download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
