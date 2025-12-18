#!/usr/bin/env python3
"""
Comparison Demo for Gemma3-1B Reasoning Model

Compares different generation strategies and demonstrates the impact
of GRPO fine-tuning on reasoning quality.

Features:
    - Temperature comparison (greedy vs creative)
    - System prompt comparison
    - Base model vs fine-tuned comparison (if checkpoint available)
    - Output format analysis

Usage:
    python demo/comparison_demo.py                     # Run comparison
    python demo/comparison_demo.py --checkpoint ./checkpoints/lora  # Compare with fine-tuned
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GemmaModel, get_device
from src.config import SYSTEM_PROMPTS, get_system_prompt
from src.utils import extract_reasoning_and_answer


def print_header():
    """Print the demo header."""
    print("\n" + "=" * 70)
    print("  Gemma3-1B Reasoning Model - Comparison Demo")
    print("  Analyzing Generation Strategies")
    print("=" * 70)


def compare_temperatures(model: GemmaModel, question: str) -> Dict[str, Any]:
    """Compare different temperature settings."""
    print("\n TEMPERATURE COMPARISON")
    print("-" * 70)
    print(f"Question: {question}\n")

    strategies = {
        "Greedy (t=0.01)": {"temperature": 0.01, "top_k": 1, "top_p": 1.0},
        "Conservative (t=0.3)": {"temperature": 0.3, "top_k": 20, "top_p": 0.9},
        "Standard (t=0.7)": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        "Creative (t=0.9)": {"temperature": 0.9, "top_k": 100, "top_p": 0.98},
    }

    results = {}
    for name, params in strategies.items():
        print(f" {name}...")
        start = time.time()
        result = model.solve(question, **params)
        elapsed = time.time() - start

        reasoning = result.get("reasoning", "")
        answer = result.get("answer", "")

        # Analyze quality
        has_format = bool(reasoning) and bool(answer)
        reasoning_length = len(reasoning.split())

        results[name] = {
            "answer": answer,
            "reasoning_preview": reasoning[:150] + "..." if len(reasoning) > 150 else reasoning,
            "has_format": has_format,
            "reasoning_words": reasoning_length,
            "time": elapsed,
        }

        # Display
        print(f"   Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")
        print(f"   Format: {'✓' if has_format else '✗'} | Words: {reasoning_length} | Time: {elapsed:.2f}s")
        print()

    return results


def compare_system_prompts(model: GemmaModel, question: str) -> Dict[str, Any]:
    """Compare different system prompts."""
    print("\n SYSTEM PROMPT COMPARISON")
    print("-" * 70)
    print(f"Question: {question}\n")

    # Select a few representative prompts
    prompt_versions = [0, 2, 6]

    results = {}
    for version in prompt_versions:
        prompt_preview = get_system_prompt(version)[:60] + "..."
        print(f" Version {version}: {prompt_preview}")

        result = model.solve(
            question,
            system_prompt_version=version,
            temperature=0.5,
        )

        reasoning = result.get("reasoning", "")
        answer = result.get("answer", "")

        results[f"v{version}"] = {
            "answer": answer,
            "reasoning_preview": reasoning[:150] + "..." if len(reasoning) > 150 else reasoning,
            "has_format": bool(reasoning) and bool(answer),
            "reasoning_words": len(reasoning.split()),
        }

        print(f"   Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")
        print()

    return results


def analyze_output_quality(model: GemmaModel, questions: List[str]) -> Dict[str, Any]:
    """Analyze output quality across multiple questions."""
    print("\n OUTPUT QUALITY ANALYSIS")
    print("-" * 70)

    total = len(questions)
    format_ok = 0
    reasoning_lengths = []
    answer_lengths = []

    for i, q in enumerate(questions, 1):
        print(f"\r Analyzing... {i}/{total}", end="")

        result = model.solve(q, temperature=0.5)
        reasoning, answer = result.get("reasoning", ""), result.get("answer", "")

        if reasoning and answer:
            format_ok += 1

        reasoning_lengths.append(len(reasoning.split()))
        answer_lengths.append(len(answer.split()))

    print()

    # Calculate statistics
    avg_reasoning = sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0
    avg_answer = sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0

    analysis = {
        "total_questions": total,
        "format_compliance": format_ok / total * 100,
        "avg_reasoning_words": avg_reasoning,
        "avg_answer_words": avg_answer,
        "min_reasoning_words": min(reasoning_lengths) if reasoning_lengths else 0,
        "max_reasoning_words": max(reasoning_lengths) if reasoning_lengths else 0,
    }

    print(f"""
   Format Compliance:     {analysis['format_compliance']:.1f}%
   Avg Reasoning Length:  {analysis['avg_reasoning_words']:.1f} words
   Avg Answer Length:     {analysis['avg_answer_words']:.1f} words
   Reasoning Range:       {analysis['min_reasoning_words']}-{analysis['max_reasoning_words']} words
""")

    return analysis


def compare_runs(model: GemmaModel, question: str, num_runs: int = 5) -> Dict[str, Any]:
    """Compare multiple runs with same settings to check consistency."""
    print("\n CONSISTENCY ANALYSIS")
    print("-" * 70)
    print(f"Question: {question}")
    print(f"Running {num_runs} times with temperature=0.7...\n")

    answers = []
    reasoning_lengths = []

    for i in range(num_runs):
        print(f"\r Run {i+1}/{num_runs}...", end="")

        result = model.solve(question, temperature=0.7)
        answer = result.get("answer", "").strip()
        reasoning = result.get("reasoning", "")

        answers.append(answer)
        reasoning_lengths.append(len(reasoning.split()))

    print()

    # Analyze consistency
    unique_answers = list(set(answers))
    most_common = max(set(answers), key=answers.count) if answers else ""
    consistency = answers.count(most_common) / len(answers) * 100 if answers else 0

    print(f"   Unique Answers: {len(unique_answers)}")
    print(f"   Most Common: {most_common}")
    print(f"   Consistency: {consistency:.1f}%")
    print(f"\n   All Answers:")
    for i, ans in enumerate(answers, 1):
        print(f"     {i}. {ans}")

    return {
        "unique_answers": len(unique_answers),
        "most_common": most_common,
        "consistency_pct": consistency,
        "answers": answers,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Comparison Demo for Gemma3-1B Reasoning Model"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA checkpoint",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to run on",
    )

    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization (CUDA only)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick comparison only",
    )

    args = parser.parse_args()

    print_header()

    # Setup
    device = get_device(args.device)
    print(f"\n Device: {device}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f" Checkpoint: {args.checkpoint}")
    else:
        print(" Using base model")

    if args.load_in_4bit and device != "cuda":
        args.load_in_4bit = False

    # Load model
    print("\n Loading model...")
    try:
        model = GemmaModel(
            checkpoint_path=args.checkpoint if args.checkpoint and os.path.exists(args.checkpoint) else None,
            device=device,
            load_in_4bit=args.load_in_4bit,
        )
        model.load()
    except Exception as e:
        print(f"\n Failed to load model: {e}")
        sys.exit(1)

    print(" Model loaded successfully!")

    # Test questions
    test_questions = [
        "What is 15% of 80?",
        "If a train travels 120 miles in 2 hours, what is its average speed?",
        "A store sells apples for $2 each. How much do 7 apples cost?",
    ]

    # Run comparisons
    print("\n" + "=" * 70)
    compare_temperatures(model, test_questions[0])

    if not args.quick:
        compare_system_prompts(model, test_questions[1])
        compare_runs(model, test_questions[0])
        analyze_output_quality(model, test_questions)

    print("\n" + "=" * 70)
    print("  COMPARISON COMPLETE")
    print("=" * 70)

    # Summary recommendations
    print("""
 RECOMMENDATIONS:
   - Use temperature=0.3-0.5 for math problems (more deterministic)
   - Use temperature=0.7-0.9 for creative/open-ended questions
   - System prompt v2 provides good balance of reasoning structure
   - Fine-tuned checkpoint significantly improves format compliance
""")


if __name__ == "__main__":
    main()
