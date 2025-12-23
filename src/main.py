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

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from .config import Config, get_default_config, format_prompt, get_system_prompt
from .model import GemmaModel, load_model, get_device
from .utils import (
    load_gsm8k_dataset,
    evaluate_accuracy,
    extract_reasoning_and_answer,
)
from .logger import get_logger

logger = get_logger(__name__)


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
    logger.info("Loading model for inference...")
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
            logger.info(f"Processing {len(questions)} questions in batch mode")

            # Use tqdm progress bar if available
            iterator = tqdm(questions, desc="Processing questions") if HAS_TQDM else questions

            for i, question in enumerate(iterator):
                if not HAS_TQDM:
                    logger.info(f"Processing question {i+1}/{len(questions)}")

                result = model.solve(question, temperature=0.7)
                results.append({
                    "question": question,
                    "response": result["response"],
                    "reasoning": result["reasoning"],
                    "answer": result["answer"],
                })

                if not HAS_TQDM:
                    logger.info(f"  Answer: {result['answer']}")

    # Save results
    if output_file and results:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {output_file}")

    return results


def run_evaluation(
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    data_dir: str = "./data",
    num_samples: int = 100,
    output_file: Optional[str] = None,
    batch_size: int = 8,
):
    """Evaluate model on GSM8K test set.

    Args:
        checkpoint_path: Path to LoRA checkpoint
        device: Device to use
        data_dir: Directory containing data
        num_samples: Number of samples to evaluate
        output_file: Output file for detailed results
        batch_size: Batch size for evaluation (higher = faster but more memory)
    """
    logger.info("Loading model for evaluation...")
    model = load_model(checkpoint_path=checkpoint_path, device=device)

    logger.info(f"Loading test data from {data_dir}...")
    try:
        test_data = load_gsm8k_dataset(
            data_dir=data_dir,
            split="test",
            max_examples=num_samples,
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.info("Please download the GSM8K dataset first.")
        return

    logger.info(f"Evaluating on {len(test_data)} examples with batch_size={batch_size}")

    predictions = []
    ground_truths = []
    detailed_results = []

    # Process in batches for better performance
    for batch_start in range(0, len(test_data), batch_size):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch = test_data[batch_start:batch_end]

        # Prepare batch of prompts
        from .config import format_prompt, get_system_prompt
        prompts = [
            format_prompt(item["question"], system_prompt=get_system_prompt(2))
            for item in batch
        ]

        # Use progress bar if available
        if HAS_TQDM and batch_start == 0:
            # Initialize progress bar on first batch
            pbar = tqdm(total=len(test_data), desc="Evaluating")

        # Batch generation
        responses = model.generate_batch(
            prompts,
            temperature=1e-4,  # Greedy decoding for evaluation
            do_sample=False,
        )

        # Process batch results
        for i, (item, response) in enumerate(zip(batch, responses)):
            reasoning, answer = extract_reasoning_and_answer(response)

            predictions.append(response)
            ground_truths.append(item["answer"])

            detailed_results.append({
                "question": item["question"],
                "ground_truth": item["answer"],
                "prediction": answer,
                "reasoning": reasoning,
                "full_response": response,
            })

        # Update progress
        if HAS_TQDM:
            pbar.update(len(batch))
        else:
            logger.info(f"Progress: {batch_end}/{len(test_data)}")

    if HAS_TQDM:
        pbar.close()

    # Calculate metrics
    metrics = evaluate_accuracy(predictions, ground_truths)

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples:      {metrics['total']}")
    logger.info(f"Correct:            {metrics['correct']} ({metrics['accuracy']:.2f}%)")
    logger.info(f"Partial correct:    {metrics['partial_accuracy']:.2f}%")
    logger.info(f"Format correct:     {metrics['format_accuracy']:.2f}%")
    logger.info("=" * 60)

    # Save detailed results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump({
                "metrics": metrics,
                "results": detailed_results,
            }, f, indent=2)
        logger.info(f"Detailed results saved to: {output_file}")

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

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (higher = faster but more memory)",
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
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
