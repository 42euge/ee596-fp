#!/usr/bin/env python3
"""
Domain-Specific Demo for Gemma3-1B Reasoning Model

Demonstrates the model's capabilities as a domain expert assistant,
following the Tunix mobile deployment demo pattern.

Use Cases:
    - Medical Assistant
    - Legal Advisor
    - Coding Helper
    - Financial Analyst
    - Educational Tutor

Usage:
    python demo/domain_demo.py                 # Interactive domain selection
    python demo/domain_demo.py --domain medical  # Specific domain
    python demo/domain_demo.py --all           # Run all domain examples
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GemmaModel, get_device
from src.config import format_prompt


# Domain-specific example problems
DOMAIN_EXAMPLES = {
    "medical": {
        "name": "Medical Assistant",
        "description": "Healthcare reasoning and differential diagnosis",
        "emoji": "",
        "examples": [
            {
                "question": "A 45-year-old patient presents with chest pain that worsens with deep breathing, fever of 101°F, and a dry cough. What conditions should be considered?",
                "context": "Consider common and serious causes of pleuritic chest pain."
            },
            {
                "question": "A patient has a BMI of 32, fasting glucose of 110 mg/dL, and blood pressure of 145/95. What lifestyle modifications would you recommend?",
                "context": "Focus on evidence-based interventions for metabolic syndrome risk factors."
            },
            {
                "question": "What are the key differences between Type 1 and Type 2 diabetes in terms of pathophysiology and treatment approach?",
                "context": "Explain the mechanisms and therapeutic strategies."
            },
        ]
    },
    "legal": {
        "name": "Legal Advisor",
        "description": "Legal reasoning and procedural guidance",
        "emoji": "",
        "examples": [
            {
                "question": "A tenant has not paid rent for 3 months. What are the general steps a landlord should follow before proceeding with an eviction?",
                "context": "Provide general procedural guidance (not legal advice)."
            },
            {
                "question": "What are the key elements that must be proven in a breach of contract case?",
                "context": "Explain the fundamental requirements for contract breach claims."
            },
            {
                "question": "Explain the difference between negligence and gross negligence in personal injury cases.",
                "context": "Clarify the legal standards and implications."
            },
        ]
    },
    "coding": {
        "name": "Coding Helper",
        "description": "Programming problem-solving and algorithm design",
        "emoji": "",
        "examples": [
            {
                "question": "I have a Python list of numbers and want to find all pairs that sum to a target value. What's the most efficient approach?",
                "context": "Consider time and space complexity trade-offs."
            },
            {
                "question": "Explain the difference between a stack and a queue, and give an example use case for each.",
                "context": "Provide clear examples and implementation considerations."
            },
            {
                "question": "How would you design a rate limiter for an API that allows 100 requests per minute per user?",
                "context": "Consider different rate limiting algorithms and their trade-offs."
            },
        ]
    },
    "finance": {
        "name": "Financial Analyst",
        "description": "Financial calculations and analysis",
        "emoji": "",
        "examples": [
            {
                "question": "A company has revenue of $1M, COGS of $400K, operating expenses of $300K, and pays 25% in taxes. Calculate the net profit margin.",
                "context": "Show the step-by-step calculation."
            },
            {
                "question": "If an investment grows from $10,000 to $15,000 over 3 years, what is the compound annual growth rate (CAGR)?",
                "context": "Explain the formula and calculation."
            },
            {
                "question": "A bond has a face value of $1000, coupon rate of 5%, and current price of $950. What is its current yield?",
                "context": "Calculate and explain the yield concept."
            },
        ]
    },
    "education": {
        "name": "Educational Tutor",
        "description": "Teaching concepts with clear explanations",
        "emoji": "",
        "examples": [
            {
                "question": "Explain the Pythagorean theorem to a middle school student and give a real-world example.",
                "context": "Use simple language and relatable examples."
            },
            {
                "question": "What causes the seasons on Earth? Why is it summer in the Northern Hemisphere when it's winter in the Southern Hemisphere?",
                "context": "Explain with visual concepts a student can understand."
            },
            {
                "question": "How does photosynthesis work? Break it down into simple steps.",
                "context": "Make it accessible for a high school biology student."
            },
        ]
    },
}


def print_header():
    """Print the demo header."""
    print("\n" + "=" * 70)
    print("  Gemma3-1B Reasoning Model - Domain Expert Demo")
    print("  Showcasing Specialized Assistant Capabilities")
    print("=" * 70)


def print_domain_menu():
    """Print available domains."""
    print("\n Available Domains:\n")
    for key, domain in DOMAIN_EXAMPLES.items():
        print(f"  {domain['emoji']} {key:12} - {domain['description']}")
    print()


def print_result(question: str, result: Dict[str, Any], domain: str = None):
    """Pretty print a result."""
    print("\n" + "-" * 70)
    if domain:
        emoji = DOMAIN_EXAMPLES.get(domain, {}).get('emoji', '')
        print(f"{emoji} Domain: {DOMAIN_EXAMPLES.get(domain, {}).get('name', domain)}")
    print(f"\n Question: {question}")
    print("-" * 70)

    print("\n REASONING:")
    reasoning = result.get("reasoning", "")
    if reasoning:
        # Word wrap
        words = reasoning.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > 68:
                print(f"   {line}")
                line = word
            else:
                line = f"{line} {word}" if line else word
        if line:
            print(f"   {line}")
    else:
        print("   (No reasoning section found)")

    print("\n ANSWER:")
    answer = result.get("answer", "")
    if answer:
        # Word wrap answer too
        words = answer.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > 68:
                print(f"   {line}")
                line = word
            else:
                line = f"{line} {word}" if line else word
        if line:
            print(f"   {line}")
    else:
        print("   (No answer section found)")

    print("-" * 70)


def run_domain_examples(model: GemmaModel, domain: str, interactive: bool = False):
    """Run examples for a specific domain."""
    if domain not in DOMAIN_EXAMPLES:
        print(f" Unknown domain: {domain}")
        return

    domain_info = DOMAIN_EXAMPLES[domain]
    print(f"\n{domain_info['emoji']} {domain_info['name']}")
    print(f"   {domain_info['description']}\n")

    examples = domain_info["examples"]

    for i, example in enumerate(examples, 1):
        print(f"[Example {i}/{len(examples)}]")

        # Show context if available
        if example.get("context"):
            print(f"   Context: {example['context']}")

        print(" Thinking...")

        result = model.solve(
            example["question"],
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )

        print_result(example["question"], result, domain)

        if interactive and i < len(examples):
            try:
                input("\nPress Enter for next example (or Ctrl+C to skip)...")
            except KeyboardInterrupt:
                print("\n Skipping remaining examples...")
                break


def run_interactive_domain(model: GemmaModel, domain: str):
    """Run interactive mode for a specific domain."""
    domain_info = DOMAIN_EXAMPLES.get(domain, {"name": domain, "emoji": ""})

    print(f"\n{domain_info['emoji']} {domain_info['name']} - Interactive Mode")
    print("   Enter your questions. Type 'back' to return to menu, 'quit' to exit.\n")

    while True:
        try:
            question = input(f"\n{domain_info['emoji']} Your question: ").strip()

            if question.lower() in ["back", "menu"]:
                return "menu"

            if question.lower() in ["quit", "exit", "q"]:
                return "quit"

            if not question:
                continue

            print("\n Thinking...")
            result = model.solve(
                question,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
            )
            print_result(question, result, domain)

        except KeyboardInterrupt:
            print("\n")
            return "menu"


def main():
    parser = argparse.ArgumentParser(
        description="Domain-Specific Demo for Gemma3-1B Reasoning Model"
    )

    parser.add_argument(
        "--domain",
        type=str,
        choices=list(DOMAIN_EXAMPLES.keys()),
        default=None,
        help="Specific domain to demo",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run examples from all domains",
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

    args = parser.parse_args()

    print_header()

    # Detect device
    device = get_device(args.device)
    print(f"\n Device: {device}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f" Checkpoint: {args.checkpoint}")
    else:
        print(" Using base model (no fine-tuned checkpoint)")

    # Quantization check
    if args.load_in_4bit and device != "cuda":
        print(" Warning: Quantization only supported on CUDA. Disabling.")
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

    # Run mode
    if args.all:
        # Run all domain examples
        for domain in DOMAIN_EXAMPLES:
            run_domain_examples(model, domain, interactive=True)
            print("\n" + "=" * 70)

    elif args.domain:
        # Run specific domain
        run_domain_examples(model, args.domain, interactive=True)

        # Offer interactive mode
        print("\n Switch to interactive mode? (y/n): ", end="")
        if input().strip().lower() == "y":
            run_interactive_domain(model, args.domain)

    else:
        # Interactive menu mode
        while True:
            print_domain_menu()
            print("   Enter domain name, 'examples', or 'quit': ", end="")

            try:
                choice = input().strip().lower()

                if choice in ["quit", "exit", "q"]:
                    print("\n Goodbye!")
                    break

                if choice == "examples":
                    # Quick sample from each domain
                    for domain in DOMAIN_EXAMPLES:
                        domain_info = DOMAIN_EXAMPLES[domain]
                        example = domain_info["examples"][0]
                        print(f"\n{domain_info['emoji']} {domain_info['name']}:")
                        print(f"   Q: {example['question'][:60]}...")
                        result = model.solve(example["question"], temperature=0.7)
                        print(f"   A: {result.get('answer', 'N/A')[:100]}...")
                    continue

                if choice in DOMAIN_EXAMPLES:
                    action = run_interactive_domain(model, choice)
                    if action == "quit":
                        print("\n Goodbye!")
                        break
                else:
                    print(f" Unknown option: {choice}")

            except KeyboardInterrupt:
                print("\n\n Goodbye!")
                break


if __name__ == "__main__":
    main()
