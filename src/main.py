"""
Main entry point for Gemma3-1B GRPO Fine-tuning

Usage:
    python -m src.main --mode inference --checkpoint ./checkpoints/lora
    python -m src.main --mode evaluate --checkpoint ./checkpoints/lora
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from .config import Config, get_default_config, format_prompt, get_system_prompt
from .model import GemmaModel, load_model, get_device
from .utils import (
    load_gsm8k_dataset,
    evaluate_accuracy,
    extract_reasoning_and_answer,
)


def run_inference(
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    interactive: bool = True,
    questions: Optional[list] = None,
    output_file: Optional[str] = None,
):
    """Run inference with the model.

    Args:
        checkpoint_path: Path to LoRA checkpoint
        device: Device to use
        interactive: Whether to run in interactive mode
        questions: List of questions (for batch mode)
        output_file: Output file path
    """
    print("Loading model...")
    model = load_model(checkpoint_path=checkpoint_path, device=device)

    results = []

    if interactive:
        print("\n" + "=" * 60)
        print("Gemma3-1B Reasoning Model - Interactive Mode")
        print("=" * 60)
        print("Enter your questions (type 'quit' to exit):\n")

        while True:
            try:
                question = input("\n> Question: ").strip()
                if question.lower() in ["quit", "exit", "q"]:
                    break
                if not question:
                    continue

                print("\nThinking...")
                result = model.solve(question, temperature=0.7)

                print("\n" + "-" * 40)
                print("REASONING:")
                print(result["reasoning"] or "(No reasoning section found)")
                print("\nANSWER:")
                print(result["answer"] or "(No answer section found)")
                print("-" * 40)

                results.append({
                    "question": question,
                    "response": result["response"],
                    "reasoning": result["reasoning"],
                    "answer": result["answer"],
                })

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Batch mode
        if questions:
            for i, question in enumerate(questions):
                print(f"\nProcessing question {i+1}/{len(questions)}...")
                result = model.solve(question, temperature=0.7)
                results.append({
                    "question": question,
                    "response": result["response"],
                    "reasoning": result["reasoning"],
                    "answer": result["answer"],
                })
                print(f"  Answer: {result['answer']}")

    # Save results
    if output_file and results:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def run_evaluation(
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    data_dir: str = "./data",
    num_samples: int = 100,
    output_file: Optional[str] = None,
):
    """Evaluate model on GSM8K test set.

    Args:
        checkpoint_path: Path to LoRA checkpoint
        device: Device to use
        data_dir: Directory containing data
        num_samples: Number of samples to evaluate
        output_file: Output file for detailed results
    """
    print("Loading model...")
    model = load_model(checkpoint_path=checkpoint_path, device=device)

    print(f"\nLoading test data from {data_dir}...")
    try:
        test_data = load_gsm8k_dataset(
            data_dir=data_dir,
            split="test",
            max_examples=num_samples,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease download the GSM8K dataset first.")
        return

    print(f"Evaluating on {len(test_data)} examples...")

    predictions = []
    ground_truths = []
    detailed_results = []

    for i, item in enumerate(test_data):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(test_data)}")

        result = model.solve(
            item["question"],
            temperature=1e-4,  # Greedy decoding for evaluation
            do_sample=False,
        )

        predictions.append(result["response"])
        ground_truths.append(item["answer"])

        detailed_results.append({
            "question": item["question"],
            "ground_truth": item["answer"],
            "prediction": result["answer"],
            "reasoning": result["reasoning"],
            "full_response": result["response"],
        })

    # Calculate metrics
    metrics = evaluate_accuracy(predictions, ground_truths)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples:      {metrics['total']}")
    print(f"Correct:            {metrics['correct']} ({metrics['accuracy']:.2f}%)")
    print(f"Partial correct:    {metrics['partial_accuracy']:.2f}%")
    print(f"Format correct:     {metrics['format_accuracy']:.2f}%")
    print("=" * 60)

    # Save detailed results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "results": detailed_results,
            }, f, indent=2)
        print(f"\nDetailed results saved to: {output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Gemma3-1B GRPO Fine-tuning - Inference & Evaluation"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["inference", "evaluate"],
        default="inference",
        help="Mode: inference (interactive) or evaluate (GSM8K test)",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to LoRA checkpoint directory",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for inference",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing dataset files",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for evaluation",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )

    parser.add_argument(
        "--questions",
        type=str,
        nargs="+",
        default=None,
        help="Questions to answer (batch mode)",
    )

    args = parser.parse_args()

    if args.mode == "inference":
        interactive = args.questions is None
        run_inference(
            checkpoint_path=args.checkpoint,
            device=args.device,
            interactive=interactive,
            questions=args.questions,
            output_file=args.output,
        )
    elif args.mode == "evaluate":
        run_evaluation(
            checkpoint_path=args.checkpoint,
            device=args.device,
            data_dir=args.data_dir,
            num_samples=args.num_samples,
            output_file=args.output,
        )


if __name__ == "__main__":
    main()
