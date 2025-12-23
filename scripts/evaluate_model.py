#!/usr/bin/env python3
"""
Automated Model Evaluation Pipeline

Handles:
- Loading checkpoints (base model or fine-tuned)
- Running evaluation on test sets
- Computing multiple metrics (accuracy, format compliance, etc.)
- Generating detailed reports
- Exporting results to JSON/CSV
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TunRex.src.tunrex.datasets import TunRexConfig, TunRex
from TunRex.src.tunrex.datasets.evaluate import evaluate
from TunRex.src.tunrex.datasets.rewards import (
    check_answer,
    check_numbers,
    match_format_exactly,
    match_format_approximately,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluationPipeline:
    """Automated model evaluation with comprehensive metrics"""

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer"""
        logger.info("Loading model...")

        try:
            from src.model import GemmaModel

            if self.checkpoint_path:
                logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
                self.model = GemmaModel(checkpoint_path=self.checkpoint_path)
            else:
                logger.info("Loading base model (no checkpoint)")
                self.model = GemmaModel()

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def evaluate_dataset(
        self,
        dataset_name: str = "gsm8k",
        split: str = "test",
        num_samples: Optional[int] = None,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset

        Args:
            dataset_name: Dataset to evaluate (gsm8k, openrubrics)
            split: Dataset split (train, val, test)
            num_samples: Number of samples to evaluate (None for all)
            batch_size: Batch size for inference

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating on {dataset_name} ({split} split)")

        # Load dataset
        if dataset_name == "gsm8k":
            config = TunRexConfig.gsm8k()
        elif dataset_name == "openrubrics":
            config = TunRexConfig.openrubrics()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        tunrex = TunRex(config)
        datasets = tunrex.prepare_datasets()

        if split not in datasets:
            raise ValueError(f"Split '{split}' not found in dataset")

        dataset = datasets[split]

        # Collect examples
        examples = []
        for i, example in enumerate(dataset):
            if num_samples and i >= num_samples:
                break
            examples.append(example)

        logger.info(f"Evaluating {len(examples)} examples...")

        # Run evaluation
        results = {
            "dataset": dataset_name,
            "split": split,
            "checkpoint": self.checkpoint_path or "base_model",
            "num_examples": len(examples),
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "examples": [],
        }

        # Metrics tracking
        total = len(examples)
        exact_correct = 0
        partial_correct = 0
        format_correct = 0
        numerical_correct = 0

        for i, example in enumerate(examples):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{total}")

            # Get question and answer
            question = example.get("question") or example.get("prompt", "")
            reference_answer = example.get("answer", "")

            # Generate model output
            try:
                if self.model:
                    output = self.model.infer(question)[0]
                else:
                    # Fallback: use placeholder
                    output = "<reasoning>Test</reasoning><answer>Test</answer>"

                # Extract answer from output
                import re
                answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
                predicted_answer = answer_match.group(1).strip() if answer_match else ""

                # Compute metrics
                exact_score = check_answer(output, reference_answer, config)
                numerical_score = check_numbers(output, reference_answer, config)
                format_score = match_format_exactly(output, config)

                # Update counters
                if exact_score > 2.0:  # High score means correct
                    exact_correct += 1
                if exact_score > 0.5:  # Partial credit
                    partial_correct += 1
                if numerical_score > 0:
                    numerical_correct += 1
                if format_score > 0:
                    format_correct += 1

                # Store example result
                results["examples"].append({
                    "id": i,
                    "question": question,
                    "reference_answer": reference_answer,
                    "predicted_answer": predicted_answer,
                    "model_output": output,
                    "exact_score": exact_score,
                    "numerical_score": numerical_score,
                    "format_score": format_score,
                    "correct": exact_score > 2.0,
                })

            except Exception as e:
                logger.warning(f"Error evaluating example {i}: {e}")
                results["examples"].append({
                    "id": i,
                    "error": str(e),
                })

        # Compute final metrics
        results["metrics"] = {
            "exact_accuracy": exact_correct / total if total > 0 else 0.0,
            "partial_accuracy": partial_correct / total if total > 0 else 0.0,
            "numerical_accuracy": numerical_correct / total if total > 0 else 0.0,
            "format_compliance": format_correct / total if total > 0 else 0.0,
            "total_examples": total,
            "exact_correct": exact_correct,
            "partial_correct": partial_correct,
            "numerical_correct": numerical_correct,
            "format_correct": format_correct,
        }

        logger.info("Evaluation complete!")
        logger.info(f"Exact Accuracy: {results['metrics']['exact_accuracy']:.2%}")
        logger.info(f"Partial Accuracy: {results['metrics']['partial_accuracy']:.2%}")
        logger.info(f"Format Compliance: {results['metrics']['format_compliance']:.2%}")

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report"""
        metrics = results["metrics"]

        lines = [
            "=" * 80,
            "Model Evaluation Report",
            "=" * 80,
            f"Dataset: {results['dataset']} ({results['split']} split)",
            f"Checkpoint: {results['checkpoint']}",
            f"Timestamp: {results['timestamp']}",
            f"Examples Evaluated: {results['num_examples']}",
            "",
            "Metrics:",
            f"  Exact Accuracy:       {metrics['exact_accuracy']:.2%} ({metrics['exact_correct']}/{metrics['total_examples']})",
            f"  Partial Accuracy:     {metrics['partial_accuracy']:.2%} ({metrics['partial_correct']}/{metrics['total_examples']})",
            f"  Numerical Accuracy:   {metrics['numerical_accuracy']:.2%} ({metrics['numerical_correct']}/{metrics['total_examples']})",
            f"  Format Compliance:    {metrics['format_compliance']:.2%} ({metrics['format_correct']}/{metrics['total_examples']})",
            "",
        ]

        # Show some example failures
        failures = [ex for ex in results["examples"] if not ex.get("correct", False)]
        if failures:
            lines.append("Sample Failures:")
            for i, failure in enumerate(failures[:5]):  # Show up to 5 failures
                if "error" in failure:
                    lines.append(f"\n  Example {failure['id']}: ERROR - {failure['error']}")
                else:
                    lines.append(f"\n  Example {failure['id']}:")
                    lines.append(f"    Question: {failure['question'][:100]}...")
                    lines.append(f"    Reference: {failure['reference_answer']}")
                    lines.append(f"    Predicted: {failure['predicted_answer']}")
            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Automated model evaluation")
    parser.add_argument(
        "--checkpoint",
        help="Path to model checkpoint (None for base model)"
    )
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        help="Dataset to evaluate (gsm8k, openrubrics)"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split (train, val, test)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to evaluate (None for all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model loading (for testing pipeline)"
    )

    args = parser.parse_args()

    # Run evaluation
    pipeline = ModelEvaluationPipeline(checkpoint_path=args.checkpoint)

    try:
        if not args.no_model:
            pipeline.load_model()

        results = pipeline.evaluate_dataset(
            dataset_name=args.dataset,
            split=args.split,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
        )

        # Save results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {args.output}")

        # Print report
        report = pipeline.generate_report(results)
        print(report)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
