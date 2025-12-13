#!/usr/bin/env python3
"""
Demo script for Gemma3-1B Reasoning Model

This script demonstrates the fine-tuned Gemma3-1B model's reasoning capabilities.
It supports running on CUDA, MPS (Apple Silicon), or CPU.

Usage:
    python demo/demo.py                           # Interactive mode
    python demo/demo.py --checkpoint ./checkpoints/lora  # With fine-tuned weights
    python demo/demo.py --examples                # Run example problems
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GemmaModel, load_model, get_device
from src.config import format_prompt, get_system_prompt


# Example problems for demonstration
EXAMPLE_PROBLEMS = [
    {
        "category": "Math",
        "question": "A store sells apples for $2 each and oranges for $3 each. If Sarah buys 4 apples and 5 oranges, how much does she spend in total?",
    },
    {
        "category": "Math",
        "question": "A train travels at 60 miles per hour. How far will it travel in 2.5 hours?",
    },
    {
        "category": "Logic",
        "question": "If all cats have tails, and Whiskers is a cat, what can we conclude about Whiskers?",
    },
    {
        "category": "Science",
        "question": "Why does ice float on water instead of sinking?",
    },
    {
        "category": "Creative",
        "question": "Imagine a world where plants could communicate with humans. How might this change agriculture?",
    },
]


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


def run_examples(model: GemmaModel):
    """Run the example problems."""
    print("\n🎯 Running example problems...\n")

    for i, example in enumerate(EXAMPLE_PROBLEMS, 1):
        print(f"\n[Example {i}/{len(EXAMPLE_PROBLEMS)}]")
        result = model.solve(
            example["question"],
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        print_result(example["question"], result, category=example["category"])

        # Pause between examples
        if i < len(EXAMPLE_PROBLEMS):
            input("\nPress Enter for next example...")


def run_interactive(model: GemmaModel):
    """Run interactive mode."""
    print("\n💬 Interactive Mode")
    print("   Enter your questions below. Type 'quit' or 'exit' to stop.")
    print("   Type 'examples' to see example problems.\n")

    while True:
        try:
            question = input("\n❓ Your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if question.lower() == "examples":
                run_examples(model)
                continue

            if not question:
                continue

            print("\n⏳ Thinking...")
            result = model.solve(
                question,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            print_result(question, result)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Demo for Gemma3-1B Reasoning Model"
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
    print(f"\n🖥️  Device: {device}")

    # Check for checkpoint
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"📁 Checkpoint: {args.checkpoint}")
        else:
            print(f"⚠️  Warning: Checkpoint not found at {args.checkpoint}")
            print("   Using base model weights only.")
            args.checkpoint = None
    else:
        print("📁 Using base model (no fine-tuned checkpoint)")

    # Check quantization compatibility
    if (args.load_in_4bit or args.load_in_8bit) and device != "cuda":
        print("⚠️  Warning: Quantization only supported on CUDA. Disabling.")
        args.load_in_4bit = False
        args.load_in_8bit = False

    # Load model
    print("\n⏳ Loading model (this may take a minute)...")
    try:
        model = GemmaModel(
            checkpoint_path=args.checkpoint,
            device=device,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
        model.load()
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you have enough RAM/VRAM")
        print("  2. Try running on CPU: --device cpu")
        print("  3. Try with quantization: --load-in-4bit (CUDA only)")
        sys.exit(1)

    print("✅ Model loaded successfully!")

    # Run demo
    if args.examples:
        run_examples(model)
    else:
        run_interactive(model)


if __name__ == "__main__":
    main()
