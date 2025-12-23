#!/usr/bin/env python3
"""
Unified CLI for Reward Model Development Pipeline

This tool reduces toil by automating the entire reward model development lifecycle:
- Dataset preparation and validation
- Training orchestration
- Evaluation and metrics
- Checkpoint deployment
- Monitoring and visualization

Usage:
    python reward_pipeline.py dataset prepare --config gsm8k
    python reward_pipeline.py train --steps 1000 --tpu v5litepod-4
    python reward_pipeline.py evaluate --checkpoint ./checkpoints/step_1000
    python reward_pipeline.py deploy --checkpoint ./checkpoints/step_1000 --repo username/model
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RewardPipeline:
    """Main pipeline orchestrator for reward model development"""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"

        # Create directories if they don't exist
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

    def prepare_dataset(
        self,
        config: str = "gsm8k",
        validate: bool = True,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Prepare and validate dataset for training

        Args:
            config: Dataset configuration (gsm8k, openrubrics, custom)
            validate: Run validation checks
            output_dir: Output directory for processed data

        Returns:
            Dictionary with dataset statistics
        """
        logger.info(f"Preparing dataset with config: {config}")

        output_dir = output_dir or self.data_dir / config
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run dataset preparation script
        cmd = [
            sys.executable,
            str(self.project_root / "scripts" / "prepare_dataset.py"),
            "--config", config,
            "--output-dir", str(output_dir),
        ]

        if validate:
            cmd.append("--validate")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Dataset preparation failed: {result.stderr}")
            raise RuntimeError(f"Dataset preparation failed: {result.stderr}")

        logger.info(f"Dataset prepared successfully in {output_dir}")

        # Load and return statistics
        stats_file = output_dir / "stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                return json.load(f)

        return {"status": "completed", "output_dir": str(output_dir)}

    def start_training(
        self,
        num_steps: int = 100,
        learning_rate: float = 3e-6,
        batch_size: int = 1,
        tpu_type: Optional[str] = None,
        use_lora: bool = True,
        lora_rank: int = 64,
        wandb_project: str = "reward-model-dev",
        run_name: Optional[str] = None,
        dataset: str = "gsm8k",
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a training run (local or on TPU)

        Args:
            num_steps: Number of training steps
            learning_rate: Learning rate
            batch_size: Batch size per device
            tpu_type: TPU type (None for local, v5litepod-4, etc.)
            use_lora: Use LoRA fine-tuning
            lora_rank: LoRA rank
            wandb_project: Weights & Biases project name
            run_name: Custom run name
            dataset: Dataset to use (gsm8k, openrubrics)
            resume_from: Checkpoint to resume from

        Returns:
            Training run information
        """
        if run_name is None:
            run_name = f"train_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting training run: {run_name}")

        if tpu_type:
            # Start TPU training via GitHub Actions or gcloud
            return self._start_tpu_training(
                num_steps=num_steps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                tpu_type=tpu_type,
                use_lora=use_lora,
                lora_rank=lora_rank,
                wandb_project=wandb_project,
                run_name=run_name,
                dataset=dataset,
                resume_from=resume_from,
            )
        else:
            # Start local training
            return self._start_local_training(
                num_steps=num_steps,
                learning_rate=learning_rate,
                batch_size=batch_size,
                use_lora=use_lora,
                lora_rank=lora_rank,
                wandb_project=wandb_project,
                run_name=run_name,
                dataset=dataset,
                resume_from=resume_from,
            )

    def _start_local_training(
        self,
        num_steps: int,
        learning_rate: float,
        batch_size: int,
        use_lora: bool,
        lora_rank: int,
        wandb_project: str,
        run_name: str,
        dataset: str,
        resume_from: Optional[str],
    ) -> Dict[str, Any]:
        """Start training on local machine"""
        logger.info("Starting local training...")

        cmd = [
            sys.executable,
            str(self.project_root / "scripts" / "train_grpo.py"),
            "--num-steps", str(num_steps),
            "--learning-rate", str(learning_rate),
            "--batch-size", str(batch_size),
            "--wandb-project", wandb_project,
            "--run-name", run_name,
            "--dataset", dataset,
        ]

        if use_lora:
            cmd.extend(["--use-lora", "--lora-rank", str(lora_rank)])

        if resume_from:
            cmd.extend(["--resume-from", resume_from])

        # Create log file
        log_file = self.logs_dir / f"{run_name}.log"

        logger.info(f"Training logs will be saved to: {log_file}")
        logger.info(f"Command: {' '.join(cmd)}")

        # Start training in background
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

        return {
            "status": "started",
            "run_name": run_name,
            "pid": process.pid,
            "log_file": str(log_file),
            "command": " ".join(cmd)
        }

    def _start_tpu_training(
        self,
        num_steps: int,
        learning_rate: float,
        batch_size: int,
        tpu_type: str,
        use_lora: bool,
        lora_rank: int,
        wandb_project: str,
        run_name: str,
        dataset: str,
        resume_from: Optional[str],
    ) -> Dict[str, Any]:
        """Start training on TPU via gcloud"""
        logger.info(f"Starting TPU training on {tpu_type}...")

        # This would typically create a TPU VM and run training
        # For now, we'll provide instructions
        logger.warning("TPU training requires manual setup or GitHub Actions workflow")
        logger.info("To run on TPU:")
        logger.info("1. Use GitHub Actions workflow: .github/workflows/tpu-training-full.yml")
        logger.info("2. Or manually create TPU VM and run scripts/train_grpo.py")

        return {
            "status": "manual_setup_required",
            "tpu_type": tpu_type,
            "instructions": "Use GitHub Actions or manual TPU VM setup"
        }

    def evaluate(
        self,
        checkpoint: Optional[str] = None,
        dataset: str = "gsm8k",
        split: str = "test",
        output_file: Optional[Path] = None,
        num_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a model checkpoint

        Args:
            checkpoint: Path to checkpoint directory
            dataset: Dataset to evaluate on
            split: Dataset split (train, val, test)
            output_file: Output file for results
            num_samples: Number of samples to evaluate (None for all)

        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating checkpoint: {checkpoint or 'base model'}")

        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.logs_dir / f"eval_{dataset}_{split}_{timestamp}.json"

        cmd = [
            sys.executable,
            str(self.project_root / "scripts" / "evaluate_model.py"),
            "--dataset", dataset,
            "--split", split,
            "--output", str(output_file),
        ]

        if checkpoint:
            cmd.extend(["--checkpoint", checkpoint])

        if num_samples:
            cmd.extend(["--num-samples", str(num_samples)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Evaluation failed: {result.stderr}")
            raise RuntimeError(f"Evaluation failed: {result.stderr}")

        logger.info(f"Evaluation completed. Results saved to: {output_file}")

        # Load and return results
        with open(output_file) as f:
            results = json.load(f)

        # Print summary
        logger.info("Evaluation Summary:")
        for metric, value in results.get("metrics", {}).items():
            logger.info(f"  {metric}: {value:.4f}")

        return results

    def deploy_checkpoint(
        self,
        checkpoint: str,
        repo_id: str,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Deploy checkpoint to HuggingFace Hub

        Args:
            checkpoint: Path to checkpoint directory
            repo_id: HuggingFace repo ID (username/model-name)
            private: Make repo private
            commit_message: Custom commit message

        Returns:
            Deployment information
        """
        logger.info(f"Deploying checkpoint to {repo_id}")

        cmd = [
            sys.executable,
            str(self.project_root / "scripts" / "deploy_checkpoint.py"),
            "--checkpoint", checkpoint,
            "--repo-id", repo_id,
        ]

        if private:
            cmd.append("--private")

        if commit_message:
            cmd.extend(["--commit-message", commit_message])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Deployment failed: {result.stderr}")
            raise RuntimeError(f"Deployment failed: {result.stderr}")

        logger.info(f"Successfully deployed to https://huggingface.co/{repo_id}")

        return {
            "status": "deployed",
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}"
        }

    def monitor(
        self,
        wandb_project: str = "reward-model-dev",
        run_name: Optional[str] = None,
    ) -> None:
        """
        Open monitoring dashboard for training runs

        Args:
            wandb_project: Weights & Biases project name
            run_name: Specific run name to monitor
        """
        logger.info("Opening monitoring dashboard...")

        if run_name:
            # Open specific run in W&B
            url = f"https://wandb.ai/{wandb_project}/runs/{run_name}"
        else:
            # Open project overview
            url = f"https://wandb.ai/{wandb_project}"

        logger.info(f"Opening: {url}")
        subprocess.run(["open", url], check=False)


def main():
    parser = argparse.ArgumentParser(
        description="Reward Model Development Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")

    prepare_parser = dataset_subparsers.add_parser("prepare", help="Prepare dataset")
    prepare_parser.add_argument("--config", default="gsm8k", help="Dataset config")
    prepare_parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    prepare_parser.add_argument("--output-dir", type=Path, help="Output directory")

    # Train command
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument("--steps", type=int, default=100, help="Training steps")
    train_parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    train_parser.add_argument("--tpu", type=str, help="TPU type (e.g., v5litepod-4)")
    train_parser.add_argument("--no-lora", action="store_true", help="Disable LoRA")
    train_parser.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    train_parser.add_argument("--wandb-project", default="reward-model-dev", help="W&B project")
    train_parser.add_argument("--run-name", help="Custom run name")
    train_parser.add_argument("--dataset", default="gsm8k", help="Dataset name")
    train_parser.add_argument("--resume-from", help="Checkpoint to resume from")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--checkpoint", help="Checkpoint path")
    eval_parser.add_argument("--dataset", default="gsm8k", help="Dataset name")
    eval_parser.add_argument("--split", default="test", help="Dataset split")
    eval_parser.add_argument("--output", type=Path, help="Output file")
    eval_parser.add_argument("--num-samples", type=int, help="Number of samples")

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy checkpoint")
    deploy_parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    deploy_parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    deploy_parser.add_argument("--private", action="store_true", help="Private repo")
    deploy_parser.add_argument("--commit-message", help="Commit message")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Open monitoring dashboard")
    monitor_parser.add_argument("--wandb-project", default="reward-model-dev", help="W&B project")
    monitor_parser.add_argument("--run-name", help="Specific run name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    pipeline = RewardPipeline()

    try:
        if args.command == "dataset":
            if args.dataset_command == "prepare":
                result = pipeline.prepare_dataset(
                    config=args.config,
                    validate=not args.no_validate,
                    output_dir=args.output_dir
                )
                print(json.dumps(result, indent=2))

        elif args.command == "train":
            result = pipeline.start_training(
                num_steps=args.steps,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                tpu_type=args.tpu,
                use_lora=not args.no_lora,
                lora_rank=args.lora_rank,
                wandb_project=args.wandb_project,
                run_name=args.run_name,
                dataset=args.dataset,
                resume_from=args.resume_from,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "evaluate":
            result = pipeline.evaluate(
                checkpoint=args.checkpoint,
                dataset=args.dataset,
                split=args.split,
                output_file=args.output,
                num_samples=args.num_samples,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "deploy":
            result = pipeline.deploy_checkpoint(
                checkpoint=args.checkpoint,
                repo_id=args.repo_id,
                private=args.private,
                commit_message=args.commit_message,
            )
            print(json.dumps(result, indent=2))

        elif args.command == "monitor":
            pipeline.monitor(
                wandb_project=args.wandb_project,
                run_name=args.run_name,
            )

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
