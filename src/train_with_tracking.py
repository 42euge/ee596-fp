"""
Example training script with experiment tracking integration.

This demonstrates how to integrate the experiment tracker into a training workflow.
For actual TPU training, use scripts/train_grpo.py with tracking modifications.
"""

import argparse
import time
from pathlib import Path

from experiment_tracker import ExperimentTracker, ExperimentConfig
from evaluation.benchmark_registry import BenchmarkRegistry
import evaluation.benchmarks  # Auto-register benchmarks


def train_model(config: ExperimentConfig, tracker: ExperimentTracker) -> str:
    """
    Mock training function for demonstration.

    In real usage, this would be replaced with actual GRPO training.

    Args:
        config: Experiment configuration
        tracker: Experiment tracker

    Returns:
        Path to checkpoint
    """
    print("\n" + "="*80)
    print("TRAINING PHASE")
    print("="*80)

    num_steps = config.num_steps

    for step in range(num_steps):
        # Simulate training step
        time.sleep(0.01)

        # Mock metrics
        loss = 1.0 - (step / num_steps) * 0.8
        reward = (step / num_steps) * 5.0
        lr = config.learning_rate * (1 - step / num_steps)

        # Log metrics
        tracker.log_metrics({
            "train/loss": loss,
            "train/reward": reward,
            "train/learning_rate": lr,
        }, step=step)

        # Log checkpoint
        if step % 50 == 0:
            checkpoint_path = f"/tmp/checkpoint_step_{step}"
            tracker.log_checkpoint(checkpoint_path, step, size_bytes=1024*1024)

        # Print progress
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}: loss={loss:.3f}, reward={reward:.3f}")

    # Final checkpoint
    checkpoint_path = f"/tmp/checkpoint_final"
    tracker.log_checkpoint(checkpoint_path, num_steps, size_bytes=1024*1024)

    print(f"\nTraining completed! Final checkpoint: {checkpoint_path}")

    return checkpoint_path


def evaluate_model(checkpoint_path: str, tracker: ExperimentTracker):
    """
    Mock evaluation function for demonstration.

    In real usage, this would load the model and run actual evaluation.

    Args:
        checkpoint_path: Path to model checkpoint
        tracker: Experiment tracker
    """
    print("\n" + "="*80)
    print("EVALUATION PHASE")
    print("="*80)

    # Mock model class with generate method
    class MockModel:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path

        def generate(self, question, **kwargs):
            """Mock generation."""
            # Simple mock: return a structured response
            return (
                "<reasoning>This is mock reasoning for the question.</reasoning>"
                "<answer>42</answer>"
            )

    # Create mock model
    model = MockModel(checkpoint_path)

    # Evaluate on GSM8K
    print("\nEvaluating on GSM8K benchmark...")

    # Mock evaluation result
    from evaluation.benchmarks.base import EvaluationResult, SampleResult

    # Create mock results
    per_sample_results = []
    for i in range(100):
        # Simulate varying accuracy
        is_correct = (i % 3 != 0)  # ~67% accuracy
        per_sample_results.append(
            SampleResult(
                sample_id=f"gsm8k_{i}",
                question=f"Mock question {i}",
                gold_answer=42,
                predicted_answer=42 if is_correct else 43,
                reasoning="Mock reasoning",
                is_correct=is_correct,
                format_correct=True,
                generation_time=0.5,
                metadata={}
            )
        )

    result = EvaluationResult(
        benchmark_name="gsm8k",
        num_samples=100,
        metrics={
            "accuracy": 0.67,
            "partial_accuracy": 0.73,
            "format_accuracy": 0.95,
            "avg_generation_time": 0.5,
        },
        per_sample_results=per_sample_results
    )

    # Log evaluation results
    tracker.log_evaluation("gsm8k", result.to_dict())

    print(f"  Accuracy: {result.metrics['accuracy']:.1%}")
    print(f"  Partial Accuracy: {result.metrics['partial_accuracy']:.1%}")
    print(f"  Format Accuracy: {result.metrics['format_accuracy']:.1%}")


def main():
    """Main training workflow with experiment tracking."""
    parser = argparse.ArgumentParser(description="Train with experiment tracking")

    # Model args
    parser.add_argument("--base_model", default="google/gemma-3-1b-it")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)

    # Training args
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=64)

    # GRPO args
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Tracking args
    parser.add_argument("--experiment_name", help="Custom experiment name")
    parser.add_argument("--notes", help="Experiment notes")
    parser.add_argument("--db_path", default="experiments.db", help="Database path")
    parser.add_argument("--backends", nargs="+", default=["local"], help="Tracking backends")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")

    args = parser.parse_args()

    # Create configuration
    config = ExperimentConfig(
        base_model=args.base_model,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        beta=args.beta,
        temperature=args.temperature,
    )

    # Initialize tracker
    tracker = ExperimentTracker(
        backends=args.backends,
        db_path=args.db_path
    )

    # Start experiment
    experiment_id = tracker.start_experiment(
        config=config,
        notes=args.notes,
        experiment_name=args.experiment_name
    )

    print(f"\n{'='*80}")
    print(f"Experiment ID: {experiment_id}")
    print(f"{'='*80}")
    print("\nConfiguration:")
    print(f"  Model: {config.base_model}")
    print(f"  LoRA: rank={config.lora_rank}, alpha={config.lora_alpha}")
    print(f"  Steps: {config.num_steps}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  GRPO Beta: {config.beta}")

    try:
        # Train model
        checkpoint_path = train_model(config, tracker)

        # Evaluate model
        if not args.skip_eval:
            evaluate_model(checkpoint_path, tracker)

        # Finish experiment
        tracker.finish_experiment(status="completed")

        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"\nExperiment ID: {experiment_id}")
        print(f"Database: {args.db_path}")
        print("\nNext steps:")
        print(f"  - View leaderboard: python -m src.analysis.leaderboard --db {args.db_path}")
        print(f"  - View details: python -m src.analysis.leaderboard --db {args.db_path} --details {experiment_id}")
        print(f"  - Compare experiments: python -m src.analysis.compare <exp1> <exp2> --db {args.db_path}")

    except Exception as e:
        print(f"\nError during experiment: {e}")
        tracker.finish_experiment(status="failed")
        raise


if __name__ == "__main__":
    main()
