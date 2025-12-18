#!/usr/bin/env python3
"""
Demo script for Gemma3-1B Reasoning Model

This script demonstrates the fine-tuned Gemma3-1B model's reasoning capabilities.
It supports running on CUDA, MPS (Apple Silicon), or CPU.

Features:
    - Interactive question answering
    - Multiple example categories (Math, Logic, Science, Creative, Domain-specific)
    - Streaming output support
    - Generation strategy selection
    - Batch processing

Usage:
    python demo/demo.py                           # Interactive mode
    python demo/demo.py --checkpoint ./checkpoints/lora  # With fine-tuned weights
    python demo/demo.py --examples                # Run example problems
    python demo/demo.py --stream                  # Enable streaming output
    python demo/demo.py --batch questions.txt     # Batch processing

See also:
    - demo.ipynb: Jupyter notebook with comprehensive examples
    - benchmark_demo.py: GSM8K evaluation benchmark
    - domain_demo.py: Domain-specific assistant demos
    - comparison_demo.py: Generation strategy comparison
    - grpo_explainer.py: GRPO training explanation
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GemmaModel, load_model, get_device
from src.config import format_prompt, get_system_prompt


# Example problems for demonstration - expanded with more categories
EXAMPLE_PROBLEMS = [
    # Math examples
    {
        "category": "Math - Arithmetic",
        "question": "A store sells apples for $2 each and oranges for $3 each. If Sarah buys 4 apples and 5 oranges, how much does she spend in total?",
    },
    {
        "category": "Math - Distance/Speed",
        "question": "A train travels at 60 miles per hour. How far will it travel in 2.5 hours?",
    },
    {
        "category": "Math - Percentages",
        "question": "A bookstore has 120 books. They sell 35% of them in the first week. How many books are left?",
    },
    # Logic examples
    {
        "category": "Logic - Syllogism",
        "question": "If all cats have tails, and Whiskers is a cat, what can we conclude about Whiskers?",
    },
    {
        "category": "Logic - Combinatorics",
        "question": "In a room with 5 people, each person shakes hands with every other person exactly once. How many handshakes occur in total?",
    },
    # Science examples
    {
        "category": "Science - Physics",
        "question": "Why does ice float on water instead of sinking?",
    },
    {
        "category": "Science - Optics",
        "question": "What causes the sky to appear blue during the day?",
    },
    # Creative examples
    {
        "category": "Creative - Hypothetical",
        "question": "Imagine a world where plants could communicate with humans. How might this change agriculture?",
    },
    # Domain-specific examples (matching Tunix mobile demos)
    {
        "category": "Coding - Algorithm",
        "question": "I have a Python list of numbers and want to find all pairs that sum to a target value. What's an efficient approach?",
    },
    {
        "category": "Finance - Analysis",
        "question": "A company has revenue of $1M, COGS of $400K, and operating expenses of $300K. What is the operating profit margin?",
    },
]


# Generation strategy presets
GENERATION_STRATEGIES = {
    "greedy": {"temperature": 0.01, "top_k": 1, "top_p": 1.0, "desc": "Deterministic, consistent output"},
    "conservative": {"temperature": 0.3, "top_k": 20, "top_p": 0.9, "desc": "Low randomness, focused"},
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95, "desc": "Balanced (default)"},
    "creative": {"temperature": 0.9, "top_k": 100, "top_p": 0.98, "desc": "High diversity, exploratory"},
}


def print_header():
    """Print the demo header."""
    print("\n" + "=" * 70)
    print("  Gemma3-1B Reasoning Model - Demo")
    print("  Fine-tuned with GRPO for improved step-by-step reasoning")
    print("=" * 70)


def print_result(question: str, result: dict, category: str = None):
    """Pretty print a result."""
    print("\n" + "-" * 70)
    if category:
        print(f"Category: {category}")
    print(f"Question: {question}")
    print("-" * 70)

    print("\n📝 REASONING:")
    reasoning = result.get("reasoning", "")
    if reasoning:
        # Word wrap long reasoning
        words = reasoning.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > 70:
                print(f"   {line}")
                line = word
            else:
                line = f"{line} {word}" if line else word
        if line:
            print(f"   {line}")
    else:
        print("   (No reasoning section found)")

    print("\n✅ ANSWER:")
    answer = result.get("answer", "")
    if answer:
        print(f"   {answer}")
    else:
        print("   (No answer section found)")

    print("-" * 70)


def run_examples(model: GemmaModel, strategy: str = "standard"):
    """Run the example problems."""
    gen_params = GENERATION_STRATEGIES.get(strategy, GENERATION_STRATEGIES["standard"])

    print(f"\n Running example problems (strategy: {strategy})...\n")

    for i, example in enumerate(EXAMPLE_PROBLEMS, 1):
        print(f"\n[Example {i}/{len(EXAMPLE_PROBLEMS)}]")
        result = model.solve(
            example["question"],
            temperature=gen_params["temperature"],
            top_k=gen_params["top_k"],
            top_p=gen_params["top_p"],
        )
        print_result(example["question"], result, category=example["category"])

        # Pause between examples
        if i < len(EXAMPLE_PROBLEMS):
            input("\nPress Enter for next example...")


def run_interactive(model: GemmaModel, strategy: str = "standard", stream: bool = False):
    """Run interactive mode."""
    gen_params = GENERATION_STRATEGIES.get(strategy, GENERATION_STRATEGIES["standard"])

    print("\n Interactive Mode")
    print(f"   Strategy: {strategy} ({gen_params['desc']})")
    print("   Commands: 'quit', 'examples', 'strategy', 'help'\n")

    current_strategy = strategy

    while True:
        try:
            question = input("\n Your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\n Goodbye!")
                break

            if question.lower() == "examples":
                run_examples(model, current_strategy)
                continue

            if question.lower() == "strategy":
                print("\n Available strategies:")
                for name, params in GENERATION_STRATEGIES.items():
                    marker = "" if name == current_strategy else " "
                    print(f"   {marker} {name}: {params['desc']}")
                new_strat = input("\n   Enter strategy name: ").strip().lower()
                if new_strat in GENERATION_STRATEGIES:
                    current_strategy = new_strat
                    gen_params = GENERATION_STRATEGIES[new_strat]
                    print(f"    Strategy changed to: {new_strat}")
                continue

            if question.lower() == "help":
                print_help()
                continue

            if not question:
                continue

            print("\n Thinking...")
            start_time = time.time()

            result = model.solve(
                question,
                temperature=gen_params["temperature"],
                top_k=gen_params["top_k"],
                top_p=gen_params["top_p"],
            )

            elapsed = time.time() - start_time
            print_result(question, result)
            print(f"   Time: {elapsed:.2f}s")

        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")


def run_batch(model: GemmaModel, questions: List[str], strategy: str = "standard", output_file: Optional[str] = None):
    """Run batch processing on a list of questions."""
    import json

    gen_params = GENERATION_STRATEGIES.get(strategy, GENERATION_STRATEGIES["standard"])
    results = []

    print(f"\n Processing {len(questions)} questions...")

    for i, question in enumerate(questions, 1):
        print(f"\r   [{i}/{len(questions)}] Processing...", end="")

        result = model.solve(
            question.strip(),
            temperature=gen_params["temperature"],
            top_k=gen_params["top_k"],
            top_p=gen_params["top_p"],
        )

        results.append({
            "question": question.strip(),
            "reasoning": result.get("reasoning", ""),
            "answer": result.get("answer", ""),
        })

    print("\n Done!")

    # Output results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"   Results saved to: {output_file}")
    else:
        # Print results
        for i, res in enumerate(results, 1):
            print(f"\n--- Question {i} ---")
            print(f"Q: {res['question']}")
            print(f"A: {res['answer']}")

    return results


def print_help():
    """Print help information."""
    print("""
 COMMANDS:
   quit/exit/q  - Exit the demo
   examples     - Run example problems
   strategy     - Change generation strategy
   help         - Show this help

 DEMO SCRIPTS:
   demo.ipynb         - Jupyter notebook (full examples)
   benchmark_demo.py  - GSM8K evaluation
   domain_demo.py     - Domain-specific demos
   comparison_demo.py - Strategy comparison
   grpo_explainer.py  - GRPO training explanation

 TIPS:
   - Use 'greedy' strategy for math (more consistent)
   - Use 'creative' strategy for open-ended questions
   - Load fine-tuned checkpoint for best results
""")


def main():
    parser = argparse.ArgumentParser(
        description="Demo for Gemma3-1B Reasoning Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo/demo.py                              # Interactive mode
  python demo/demo.py --examples                   # Run example problems
  python demo/demo.py --strategy greedy            # Use greedy decoding
  python demo/demo.py --batch questions.txt        # Batch processing
  python demo/demo.py --checkpoint ./checkpoints/lora  # With fine-tuned weights

See also:
  demo.ipynb         - Jupyter notebook with comprehensive examples
  benchmark_demo.py  - GSM8K evaluation benchmark
  domain_demo.py     - Domain-specific assistant demos
  comparison_demo.py - Generation strategy comparison
  grpo_explainer.py  - GRPO training explanation
        """
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
        help="Device to run on (default: auto-detect)",
    )

    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run example problems and exit",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="standard",
        choices=list(GENERATION_STRATEGIES.keys()),
        help="Generation strategy (default: standard)",
    )

    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Path to file with questions (one per line) for batch processing",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for batch results (JSON format)",
    )

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output (simulated)",
    )

    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization (requires bitsandbytes, CUDA only)",
    )

    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Use 8-bit quantization (requires bitsandbytes, CUDA only)",
    )

    args = parser.parse_args()

    print_header()

    # Detect device
    device = get_device(args.device)
    print(f"\n Device: {device}")

    # Check for checkpoint
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f" Checkpoint: {args.checkpoint}")
        else:
            print(f" Warning: Checkpoint not found at {args.checkpoint}")
            print("   Using base model weights only.")
            args.checkpoint = None
    else:
        print(" Using base model (no fine-tuned checkpoint)")

    # Show strategy
    strat_info = GENERATION_STRATEGIES.get(args.strategy, GENERATION_STRATEGIES["standard"])
    print(f" Strategy: {args.strategy} ({strat_info['desc']})")

    # Check quantization compatibility
    if (args.load_in_4bit or args.load_in_8bit) and device != "cuda":
        print(" Warning: Quantization only supported on CUDA. Disabling.")
        args.load_in_4bit = False
        args.load_in_8bit = False

    # Load model
    print("\n Loading model (this may take a minute)...")
    try:
        model = GemmaModel(
            checkpoint_path=args.checkpoint,
            device=device,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
        model.load()
    except Exception as e:
        print(f"\n Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you have enough RAM/VRAM")
        print("  2. Try running on CPU: --device cpu")
        print("  3. Try with quantization: --load-in-4bit (CUDA only)")
        sys.exit(1)

    print(" Model loaded successfully!")

    # Run demo based on mode
    if args.batch:
        # Batch processing mode
        if not os.path.exists(args.batch):
            print(f"\n Error: Batch file not found: {args.batch}")
            sys.exit(1)

        with open(args.batch, "r") as f:
            questions = [line.strip() for line in f if line.strip()]

        run_batch(model, questions, strategy=args.strategy, output_file=args.output)

    elif args.examples:
        # Example problems mode
        run_examples(model, args.strategy)

    else:
        # Interactive mode
        run_interactive(model, strategy=args.strategy, stream=args.stream)


if __name__ == "__main__":
    main()
