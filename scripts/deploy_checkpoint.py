#!/usr/bin/env python3
"""
Automated Checkpoint Deployment Pipeline

Handles:
- Uploading checkpoints to HuggingFace Hub
- Creating model cards with metrics
- Tagging releases
- Managing model repositories
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointDeploymentPipeline:
    """Automated checkpoint deployment to HuggingFace Hub"""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def deploy_to_huggingface(
        self,
        repo_id: str,
        private: bool = False,
        commit_message: Optional[str] = None,
        model_card: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> Dict[str, Any]:
        """
        Deploy checkpoint to HuggingFace Hub

        Args:
            repo_id: Repository ID (username/model-name)
            private: Make repository private
            commit_message: Custom commit message
            model_card: Custom model card content
            tags: Model tags for discovery

        Returns:
            Deployment information
        """
        logger.info(f"Deploying checkpoint to HuggingFace Hub: {repo_id}")

        try:
            from huggingface_hub import HfApi, create_repo, upload_folder

            # Initialize API
            api = HfApi()

            # Check for HF token
            token = os.environ.get("HF_TOKEN")
            if not token:
                logger.warning("HF_TOKEN not found in environment. Login may be required.")

            # Create repository if it doesn't exist
            try:
                logger.info(f"Creating repository: {repo_id}")
                create_repo(
                    repo_id=repo_id,
                    private=private,
                    exist_ok=True,
                    token=token,
                )
                logger.info("Repository created/verified")
            except Exception as e:
                logger.warning(f"Repository creation failed: {e}")

            # Generate model card if not provided
            if model_card is None:
                model_card = self._generate_model_card(repo_id, tags)

            # Save model card
            model_card_path = self.checkpoint_path / "README.md"
            with open(model_card_path, 'w') as f:
                f.write(model_card)

            logger.info(f"Model card saved to: {model_card_path}")

            # Upload checkpoint folder
            logger.info("Uploading checkpoint files...")

            commit_message = commit_message or f"Upload checkpoint from {self.checkpoint_path.name}"

            upload_folder(
                folder_path=str(self.checkpoint_path),
                repo_id=repo_id,
                commit_message=commit_message,
                token=token,
            )

            logger.info("Upload complete!")

            return {
                "status": "deployed",
                "repo_id": repo_id,
                "url": f"https://huggingface.co/{repo_id}",
                "checkpoint_path": str(self.checkpoint_path),
                "private": private,
            }

        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            raise
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise

    def _generate_model_card(
        self,
        repo_id: str,
        tags: Optional[list] = None
    ) -> str:
        """Generate a basic model card"""

        if tags is None:
            tags = ["text-generation", "gemma", "reward-model", "grpo"]

        # Try to load training config or metrics if available
        config_file = self.checkpoint_path / "config.json"
        metrics_file = self.checkpoint_path / "metrics.json"

        config_info = ""
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
                config_info = f"**Training Configuration:**\n```json\n{json.dumps(config, indent=2)}\n```\n\n"

        metrics_info = ""
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
                metrics_info = f"**Evaluation Metrics:**\n```json\n{json.dumps(metrics, indent=2)}\n```\n\n"

        card = f"""---
license: apache-2.0
base_model: google/gemma-3-1b-it
tags:
{chr(10).join(f'- {tag}' for tag in tags)}
---

# {repo_id.split('/')[-1]}

This model is a fine-tuned version of `google/gemma-3-1b-it` using Group Relative Policy Optimization (GRPO) with LoRA adapters.

## Model Description

- **Base Model:** google/gemma-3-1b-it
- **Fine-tuning Method:** GRPO (Group Relative Policy Optimization)
- **Adapter Type:** LoRA (Low-Rank Adaptation)
- **Training Framework:** JAX/Flax with Google Tunix

## Training Details

{config_info}

## Evaluation Results

{metrics_info}

## Usage

```python
from src.model import GemmaModel

# Load model with checkpoint
model = GemmaModel(checkpoint_path="{repo_id}")

# Generate
question = "What is 2+2?"
output = model.infer(question)
print(output)
```

## Training Procedure

This model was trained using the automated reward model development pipeline:

```bash
# Prepare dataset
python scripts/reward_pipeline.py dataset prepare --config gsm8k

# Train model
python scripts/reward_pipeline.py train --steps 1000 --use-lora

# Evaluate
python scripts/reward_pipeline.py evaluate --checkpoint ./checkpoints/step_1000

# Deploy
python scripts/reward_pipeline.py deploy --checkpoint ./checkpoints/step_1000 --repo-id {repo_id}
```

## Citation

If you use this model, please cite:

```bibtex
@misc{{{repo_id.replace('/', '-')},
  author = {{Your Name}},
  title = {{{repo_id.split('/')[-1]}}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## License

This model is released under the Apache 2.0 License, following the base model license.
"""

        return card

    def validate_checkpoint(self) -> Dict[str, Any]:
        """
        Validate checkpoint structure

        Returns:
            Validation results
        """
        logger.info("Validating checkpoint structure...")

        validation = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "files_found": [],
        }

        # Check for required files
        required_files = [
            "adapter_config.json",  # LoRA config
            "adapter_model.safetensors",  # LoRA weights
        ]

        optional_files = [
            "config.json",  # Training config
            "metrics.json",  # Evaluation metrics
            "tokenizer_config.json",  # Tokenizer config
        ]

        for filename in required_files:
            filepath = self.checkpoint_path / filename
            if filepath.exists():
                validation["files_found"].append(filename)
            else:
                validation["passed"] = False
                validation["errors"].append(f"Required file missing: {filename}")

        for filename in optional_files:
            filepath = self.checkpoint_path / filename
            if filepath.exists():
                validation["files_found"].append(filename)
            else:
                validation["warnings"].append(f"Optional file missing: {filename}")

        # Check file sizes
        for filepath in self.checkpoint_path.iterdir():
            if filepath.is_file():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                logger.info(f"  {filepath.name}: {size_mb:.2f} MB")

        if validation["passed"]:
            logger.info("Checkpoint validation passed!")
        else:
            logger.warning("Checkpoint validation failed!")
            for error in validation["errors"]:
                logger.warning(f"  - {error}")

        return validation


def main():
    parser = argparse.ArgumentParser(description="Deploy checkpoint to HuggingFace Hub")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repository ID (username/model-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--commit-message",
        help="Custom commit message"
    )
    parser.add_argument(
        "--model-card",
        type=Path,
        help="Path to custom model card (README.md)"
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Model tags for discovery"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate checkpoint, don't deploy"
    )

    args = parser.parse_args()

    # Run deployment
    pipeline = CheckpointDeploymentPipeline(args.checkpoint)

    try:
        # Validate checkpoint
        validation = pipeline.validate_checkpoint()

        if not validation["passed"]:
            logger.error("Checkpoint validation failed. Cannot deploy.")
            sys.exit(1)

        if args.validate_only:
            logger.info("Validation complete (--validate-only flag set)")
            return

        # Load custom model card if provided
        model_card = None
        if args.model_card:
            with open(args.model_card) as f:
                model_card = f.read()

        # Deploy
        result = pipeline.deploy_to_huggingface(
            repo_id=args.repo_id,
            private=args.private,
            commit_message=args.commit_message,
            model_card=model_card,
            tags=args.tags,
        )

        # Print result
        print(json.dumps(result, indent=2))

        logger.info(f"✓ Successfully deployed to: {result['url']}")

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()
