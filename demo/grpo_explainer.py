#!/usr/bin/env python3
"""
GRPO Training Explainer Demo

Interactive explanation of Group Relative Policy Optimization (GRPO)
and how it's used to fine-tune the Gemma3-1B model for reasoning.

This demo shows:
    - How GRPO generates and compares multiple responses
    - How rewards are calculated (format, accuracy, rubric)
    - How the model learns from relative rankings

Usage:
    python demo/grpo_explainer.py           # Run explanation demo
    python demo/grpo_explainer.py --live    # Live demonstration with model
"""

import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    Config, get_default_config,
    REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END
)
from src.utils import (
    extract_reasoning_and_answer,
    format_reward,
    rubric_overlap_score,
)


def print_header():
    """Print the demo header."""
    print("\n" + "=" * 70)
    print("  GRPO Training Explainer")
    print("  Understanding Group Relative Policy Optimization")
    print("=" * 70)


def explain_grpo():
    """Explain the GRPO algorithm."""
    explanation = """

 WHAT IS GRPO?

 Group Relative Policy Optimization (GRPO) is a reinforcement learning
 algorithm for fine-tuning language models. Unlike PPO which uses a
 separate critic network, GRPO is "critic-free" and uses group-based
 comparisons.

 ┌─────────────────────────────────────────────────────────────────────┐
 │                      GRPO TRAINING LOOP                            │
 │                                                                     │
 │  ┌─────────────┐    ┌─────────────────────────────────────────┐    │
 │  │   Prompt    │───▶│  Generate N Responses (e.g., N=4)       │    │
 │  └─────────────┘    └─────────────────────────────────────────┘    │
 │                                     │                               │
 │                                     ▼                               │
 │              ┌───────────────────────────────────────────┐         │
 │              │         Score Each Response               │         │
 │              │  R = Format + Accuracy + Rubric Reward    │         │
 │              └───────────────────────────────────────────┘         │
 │                                     │                               │
 │                                     ▼                               │
 │              ┌───────────────────────────────────────────┐         │
 │              │    Compute Advantages (Relative Ranking)  │         │
 │              │    A = (R - mean(R)) / std(R)             │         │
 │              └───────────────────────────────────────────┘         │
 │                                     │                               │
 │                                     ▼                               │
 │              ┌───────────────────────────────────────────┐         │
 │              │       Update Policy (Increase prob of     │         │
 │              │       high-advantage responses)           │         │
 │              └───────────────────────────────────────────┘         │
 │                                                                     │
 └─────────────────────────────────────────────────────────────────────┘

"""
    print(explanation)
    input("Press Enter to continue...")


def explain_rewards():
    """Explain the reward functions."""
    print("\n REWARD FUNCTIONS")
    print("-" * 70)

    print("""
 GRPO uses multiple reward signals to guide learning:

 1. FORMAT REWARD (-2 to +2)
    ━━━━━━━━━━━━━━━━━━━━━━━━
    Checks for proper reasoning structure:
    - Has <reasoning>...</reasoning> tags?  (+1)
    - Has <answer>...</answer> tags?        (+1)
    - Reasoning is substantive (>50 chars)? (+0.5)
    - Answer is concise (<100 chars)?       (+0.5)

 2. ACCURACY REWARD (0 or 1.5)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    For verifiable problems (math):
    - Correct numerical answer?             (+1.5)
    - Incorrect or missing?                 (0)

 3. RUBRIC-AS-REWARD (RaR) (0-20)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Based on OpenRubrics dataset:
    - Rubric term overlap (0-10)
    - Reference response similarity (0-5)
    - Target score alignment (0-5)

""")
    input("Press Enter to see examples...")


def demonstrate_format_reward():
    """Demonstrate format reward calculation."""
    print("\n FORMAT REWARD EXAMPLES")
    print("-" * 70)

    examples = [
        {
            "name": "Perfect Format",
            "text": f"{REASONING_START}Let me think step by step. First, I need to add 5 and 3, which gives 8. Then I multiply by 2 to get 16.{REASONING_END}{SOLUTION_START}16{SOLUTION_END}",
        },
        {
            "name": "Missing Answer Tags",
            "text": f"{REASONING_START}The answer is found by adding 5 and 3.{REASONING_END}The answer is 8.",
        },
        {
            "name": "No Tags",
            "text": "The answer is 8 because 5 + 3 = 8.",
        },
        {
            "name": "Short Reasoning",
            "text": f"{REASONING_START}5+3=8{REASONING_END}{SOLUTION_START}8{SOLUTION_END}",
        },
    ]

    for ex in examples:
        print(f"\n Example: {ex['name']}")
        print(f"   Text: {ex['text'][:60]}...")

        # Calculate reward
        scores = format_reward([""], [ex["text"]])
        print(f"   Format Reward: {scores[0]:.1f}")

        reasoning, answer = extract_reasoning_and_answer(ex["text"])
        print(f"   Reasoning found: {'Yes' if reasoning else 'No'}")
        print(f"   Answer found: {'Yes' if answer else 'No'}")

    input("\nPress Enter to continue...")


def demonstrate_rubric_reward():
    """Demonstrate rubric overlap scoring."""
    print("\n RUBRIC OVERLAP EXAMPLES")
    print("-" * 70)

    rubric = """
    The response should:
    1. Show step-by-step calculation
    2. Explain the mathematical operations used
    3. Arrive at the correct numerical answer
    4. Be clear and organized
    """

    responses = [
        {
            "name": "High Overlap",
            "text": "Step 1: I'll calculate this step by step. The mathematical operation we need is addition. Step 2: 5 + 3 = 8. This gives us the correct numerical answer of 8, presented in a clear and organized way.",
        },
        {
            "name": "Medium Overlap",
            "text": "To solve this, we add the numbers: 5 + 3 = 8. The answer is 8.",
        },
        {
            "name": "Low Overlap",
            "text": "8",
        },
    ]

    print(f" Rubric: {rubric[:60]}...")

    for resp in responses:
        score = rubric_overlap_score(resp["text"], rubric)
        print(f"\n {resp['name']}:")
        print(f"   Response: {resp['text'][:50]}...")
        print(f"   Rubric Score: {score:.2f}/10")

    input("\nPress Enter to continue...")


def explain_advantages():
    """Explain advantage calculation."""
    print("\n ADVANTAGE CALCULATION")
    print("-" * 70)

    print("""
 GRPO computes advantages using GROUP NORMALIZATION:

 Given rewards R = [r1, r2, r3, r4] for a group of 4 responses:

   1. Calculate mean: μ = (r1 + r2 + r3 + r4) / 4
   2. Calculate std:  σ = sqrt(variance(R))
   3. Normalize:      Ai = (ri - μ) / σ

 Example:
   Raw Rewards:  R = [3.5, 1.2, 2.8, 4.1]
   Mean:         μ = 2.9
   Std:          σ = 1.08

   Advantages:
     A1 = (3.5 - 2.9) / 1.08 = +0.56  (above average)
     A2 = (1.2 - 2.9) / 1.08 = -1.57  (below average)
     A3 = (2.8 - 2.9) / 1.08 = -0.09  (near average)
     A4 = (4.1 - 2.9) / 1.08 = +1.11  (best in group)

 The model learns to INCREASE probability of high-advantage responses
 and DECREASE probability of low-advantage responses.

 ┌──────────────────────────────────────────────────────────┐
 │  Response  │  Reward  │  Advantage  │      Action        │
 ├────────────┼──────────┼─────────────┼────────────────────┤
 │     R4     │   4.1    │   +1.11     │  Increase prob  ▲  │
 │     R1     │   3.5    │   +0.56     │  Increase prob  ▲  │
 │     R3     │   2.8    │   -0.09     │  Near neutral   ─  │
 │     R2     │   1.2    │   -1.57     │  Decrease prob  ▼  │
 └──────────────────────────────────────────────────────────┘

""")
    input("Press Enter to continue...")


def show_training_config():
    """Show training configuration."""
    print("\n TRAINING CONFIGURATION")
    print("-" * 70)

    config = get_default_config()

    print(f"""
 LoRA Configuration:
   Rank:                  {config.lora.rank}
   Alpha:                 {config.lora.alpha}

 GRPO Configuration:
   Temperature:           {config.grpo.temperature}
   Top-k:                 {config.grpo.top_k}
   Num Generations:       {config.grpo.num_generations}
   Beta (KL penalty):     {config.grpo.beta}
   Epsilon (clipping):    {config.grpo.epsilon}

 Training Configuration:
   Learning Rate:         {config.training.learning_rate}
   Micro Batch Size:      {config.training.train_micro_batch_size}
   Num Batches:           {config.training.num_batches}
   Warmup Ratio:          {config.training.warmup_ratio}

 Data Configuration:
   Dataset:               OpenRubrics
   Max Examples:          {config.data.openrubrics_max_examples}
""")


def live_demo():
    """Run live demonstration with model."""
    print("\n LIVE GRPO SIMULATION")
    print("-" * 70)

    try:
        from src.model import GemmaModel, get_device

        device = get_device("auto")
        print(f" Loading model on {device}...")

        model = GemmaModel(device=device)
        model.load()

        question = "What is 7 + 8?"
        print(f"\n Generating 4 responses to: '{question}'")
        print(" (Simulating GRPO group generation)\n")

        responses = []
        rewards = []

        for i in range(4):
            result = model.solve(question, temperature=0.9)
            response = result.get("response", "")
            responses.append(response)

            # Calculate reward
            fmt_reward = format_reward([""], [response])[0]
            rewards.append(fmt_reward)

            answer = result.get("answer", "N/A")
            print(f"   Response {i+1}: Answer='{answer}', Format Reward={fmt_reward:.1f}")

        # Calculate advantages
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        std_r = max(std_r, 0.1)  # Avoid division by zero

        advantages = [(r - mean_r) / std_r for r in rewards]

        print(f"\n Advantage Calculation:")
        print(f"   Mean Reward: {mean_r:.2f}")
        print(f"   Std Reward:  {std_r:.2f}")

        for i, (r, a) in enumerate(zip(rewards, advantages)):
            direction = "▲" if a > 0 else "▼" if a < 0 else "─"
            print(f"   Response {i+1}: Reward={r:.1f}, Advantage={a:+.2f} {direction}")

        best_idx = advantages.index(max(advantages))
        print(f"\n Best Response: #{best_idx + 1}")
        print(f"   (Model would learn to generate more responses like this)")

    except Exception as e:
        print(f" Could not load model: {e}")
        print(" Run with a loaded model to see live demonstration.")


def main():
    parser = argparse.ArgumentParser(
        description="GRPO Training Explainer Demo"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live demonstration with model",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip interactive pauses",
    )

    args = parser.parse_args()

    # Override input if quick mode
    if args.quick:
        global input
        input = lambda x: None

    print_header()
    explain_grpo()
    explain_rewards()
    demonstrate_format_reward()
    demonstrate_rubric_reward()
    explain_advantages()
    show_training_config()

    if args.live:
        live_demo()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
 Key Takeaways:
   1. GRPO is critic-free RL using group comparisons
   2. Rewards combine format, accuracy, and rubric signals
   3. Advantages are normalized within each group
   4. Model learns from relative rankings, not absolute scores
   5. Results in improved reasoning structure and accuracy

 Resources:
   - GRPO Paper: https://arxiv.org/abs/2402.03300
   - Tunix Library: https://github.com/google/tunix
   - Rubric-as-Reward: https://arxiv.org/pdf/2507.17746
""")


if __name__ == "__main__":
    main()
