#!/usr/bin/env python3
"""
Automated Dataset Preparation Pipeline

Handles:
- Loading datasets from multiple sources (HuggingFace, Kaggle, TFDS)
- Preprocessing and validation
- Train/val/test splitting
- Statistics generation
- Format validation
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TunRex.src.tunrex.datasets import TunRexConfig, TunRex
from TunRex.src.tunrex.datasets.loaders import (
    load_from_huggingface,
    load_from_kaggle,
    load_openrubrics,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreparationPipeline:
    """Automated dataset preparation with validation"""

    PRESET_CONFIGS = {
        "gsm8k": TunRexConfig.gsm8k,
        "openrubrics": TunRexConfig.openrubrics,
    }

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        config_name: str,
        validate: bool = True,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prepare dataset with specified configuration

        Args:
            config_name: Name of preset config or 'custom'
            validate: Run validation checks
            custom_config: Custom configuration dict

        Returns:
            Dataset statistics and metadata
        """
        logger.info(f"Preparing dataset: {config_name}")

        # Get configuration
        if config_name in self.PRESET_CONFIGS:
            config = self.PRESET_CONFIGS[config_name]()
            logger.info(f"Using preset config: {config_name}")
        elif custom_config:
            config = TunRexConfig(**custom_config)
            logger.info("Using custom configuration")
        else:
            raise ValueError(f"Unknown config: {config_name}. Use preset or provide custom_config")

        # Initialize TunRex
        logger.info("Loading dataset...")
        tunrex = TunRex(config)

        # Prepare datasets (train/val/test splits)
        logger.info("Preparing train/val/test splits...")
        datasets = tunrex.prepare_datasets()

        # Collect statistics
        stats = {
            "config_name": config_name,
            "source": config.source,
            "splits": {},
            "validation": {},
        }

        for split_name, dataset in datasets.items():
            logger.info(f"Processing {split_name} split...")

            # Count examples
            num_examples = 0
            for _ in dataset:
                num_examples += 1

            stats["splits"][split_name] = {
                "num_examples": num_examples,
            }

            logger.info(f"  {split_name}: {num_examples} examples")

        # Run validation checks
        if validate:
            logger.info("Running validation checks...")
            validation_results = self._validate_dataset(datasets, config)
            stats["validation"] = validation_results

            if not validation_results["passed"]:
                logger.warning("Validation checks failed!")
                for error in validation_results["errors"]:
                    logger.warning(f"  - {error}")
            else:
                logger.info("All validation checks passed!")

        # Save statistics
        stats_file = self.output_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Statistics saved to: {stats_file}")

        # Save config
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            # Convert config to dict
            config_dict = {
                "source": config.source,
                "name": config.name,
                "subset": config.subset,
                "split": config.split,
                "reasoning_start": config.reasoning_start,
                "reasoning_end": config.reasoning_end,
                "solution_start": config.solution_start,
                "solution_end": config.solution_end,
            }
            json.dump(config_dict, f, indent=2)

        logger.info(f"Config saved to: {config_file}")

        return stats

    def _validate_dataset(
        self,
        datasets: Dict[str, Any],
        config: TunRexConfig,
    ) -> Dict[str, Any]:
        """
        Run validation checks on prepared dataset

        Checks:
        - All splits have examples
        - Required fields are present
        - Format tags are correct
        - No empty or malformed examples
        """
        validation = {
            "passed": True,
            "errors": [],
            "warnings": [],
        }

        # Check 1: All splits have examples
        for split_name, dataset in datasets.items():
            count = 0
            for _ in dataset:
                count += 1
                if count > 0:  # At least one example
                    break

            if count == 0:
                validation["passed"] = False
                validation["errors"].append(f"{split_name} split is empty")

        # Check 2: Sample a few examples and validate format
        if "train" in datasets:
            logger.info("Sampling examples for validation...")
            sample_count = 0
            max_samples = 10

            for example in datasets["train"]:
                sample_count += 1
                if sample_count > max_samples:
                    break

                # Check required fields
                if "question" not in example and "prompt" not in example:
                    validation["warnings"].append(
                        f"Example missing 'question' or 'prompt' field"
                    )

                if "answer" not in example:
                    validation["warnings"].append(
                        f"Example missing 'answer' field"
                    )

                # Check for empty values
                for key, value in example.items():
                    if value is None or (isinstance(value, str) and not value.strip()):
                        validation["warnings"].append(
                            f"Example has empty value for field: {key}"
                        )

        # Check 3: Verify format tags are configured
        if not config.reasoning_start or not config.solution_start:
            validation["warnings"].append(
                "Format tags (reasoning_start/solution_start) not configured"
            )

        return validation

    def generate_report(self, stats: Dict[str, Any]) -> str:
        """Generate human-readable report"""
        lines = [
            "=" * 60,
            "Dataset Preparation Report",
            "=" * 60,
            f"Config: {stats['config_name']}",
            f"Source: {stats['source']}",
            "",
            "Splits:",
        ]

        for split_name, split_stats in stats["splits"].items():
            lines.append(f"  {split_name}: {split_stats['num_examples']} examples")

        if "validation" in stats:
            lines.append("")
            lines.append("Validation:")
            validation = stats["validation"]

            if validation["passed"]:
                lines.append("  ✓ All checks passed")
            else:
                lines.append("  ✗ Some checks failed")

            if validation.get("errors"):
                lines.append("")
                lines.append("Errors:")
                for error in validation["errors"]:
                    lines.append(f"  - {error}")

            if validation.get("warnings"):
                lines.append("")
                lines.append("Warnings:")
                for warning in validation["warnings"]:
                    lines.append(f"  - {warning}")

        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Automated dataset preparation")
    parser.add_argument(
        "--config",
        default="gsm8k",
        help="Dataset config (gsm8k, openrubrics, or custom)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for prepared data"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Run validation checks"
    )
    parser.add_argument(
        "--no-validate",
        dest="validate",
        action="store_false",
        help="Skip validation checks"
    )
    parser.add_argument(
        "--custom-config",
        type=Path,
        help="Path to custom config JSON file"
    )

    args = parser.parse_args()

    # Load custom config if provided
    custom_config = None
    if args.custom_config:
        with open(args.custom_config) as f:
            custom_config = json.load(f)

    # Run pipeline
    pipeline = DatasetPreparationPipeline(args.output_dir)

    try:
        stats = pipeline.prepare(
            config_name=args.config,
            validate=args.validate,
            custom_config=custom_config,
        )

        # Print report
        report = pipeline.generate_report(stats)
        print(report)

        # Exit with error if validation failed
        if args.validate and not stats["validation"]["passed"]:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()
