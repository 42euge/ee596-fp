#!/usr/bin/env python3
"""
PRIME RL Demo Script

Demonstrates the core functionality of PRIME RL:
1. Parsing reasoning into steps
2. Evaluating individual steps
3. Calculating trajectory rewards
4. Using PRIME RL reward functions

Run: python examples/prime_rl_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prime_rl import (
    PRIMEConfig,
    StepParser,
    StepEvaluator,
    ProcessRewardCalculator,
    prime_rl_reward,
    StepParsingStrategy,
    StepEvaluationMethod,
    RewardAggregation,
    format_steps_for_display,
)


def demo_step_parsing():
    """Demonstrate step parsing with different strategies."""
    print("=" * 80)
    print("DEMO 1: Step Parsing")
    print("=" * 80)

    completion = """<reasoning>
Step 1: Identify the problem - we need to calculate 15 + 27
Step 2: Add the two numbers: 15 + 27 = 42
Step 3: Verify the calculation is correct
Step 4: State the final answer
</reasoning>
<answer>42</answer>"""

    print("\nOriginal completion:")
    print(completion)

    # Parse with numbered strategy
    config = PRIMEConfig(step_parsing_strategy=StepParsingStrategy.NUMBERED)
    parser = StepParser(config)
    steps = parser.parse(completion)

    print(f"\n✓ Parsed into {len(steps)} steps:")
    print(format_steps_for_display(steps))


def demo_step_evaluation():
    """Demonstrate step evaluation."""
    print("\n" + "=" * 80)
    print("DEMO 2: Step Evaluation")
    print("=" * 80)

    from src.prime_rl.step_parser import ParsedStep

    # Create test steps
    steps = [
        ParsedStep(0, "Identify the variables: x = 5, y = 3"),
        ParsedStep(1, "Apply the formula: z = x + y"),
        ParsedStep(2, "Calculate: z = 5 + 3 = 8"),
        ParsedStep(3, "Therefore, the answer is 8"),
    ]

    # Evaluate with hybrid method
    config = PRIMEConfig(step_evaluation_method=StepEvaluationMethod.HYBRID)
    evaluator = StepEvaluator(config)

    print("\nEvaluating steps with HYBRID method:")
    for step in steps:
        reward = evaluator.evaluate_step(step)
        status = "✓" if reward.reward > 0.6 else "~" if reward.reward > 0.4 else "✗"
        print(f"\n{status} Step {step.index + 1}: {step.text[:60]}...")
        print(f"   Reward: {reward.reward:.3f}")
        print(f"   Method: {reward.evaluation_method}")


def demo_trajectory_rewards():
    """Demonstrate trajectory reward calculation."""
    print("\n" + "=" * 80)
    print("DEMO 3: Trajectory Reward Calculation")
    print("=" * 80)

    completion = """<reasoning>
Step 1: We need to find the sum of 25 and 17
Step 2: Breaking it down: 25 + 17
Step 3: 20 + 10 = 30, and 5 + 7 = 12
Step 4: Therefore: 30 + 12 = 42
Step 5: Verify: 25 + 17 = 42 ✓
</reasoning>
<answer>42</answer>"""

    # Test different aggregation strategies
    strategies = [
        RewardAggregation.DISCOUNTED_SUM,
        RewardAggregation.MEAN,
        RewardAggregation.WEIGHTED_MEAN,
    ]

    print("\nComparing aggregation strategies:")
    print("-" * 80)

    for strategy in strategies:
        config = PRIMEConfig(
            reward_aggregation=strategy,
            gamma=0.95
        )

        calculator = ProcessRewardCalculator(config)
        trajectory = calculator.calculate_trajectory_reward(
            prompt="What is 25 + 17?",
            completion=completion,
            final_answer_reward=1.0,  # Assume correct answer
            answer="42"
        )

        print(f"\n{strategy.value.upper()}:")
        print(f"  Steps: {trajectory.num_steps}")
        print(f"  Mean step reward: {trajectory.mean_step_reward:.3f}")
        print(f"  Aggregated reward: {trajectory.aggregated_reward:.3f}")
        print(f"  Total reward: {trajectory.total_reward:.3f}")


def demo_grpo_reward_function():
    """Demonstrate GRPO-compatible reward function."""
    print("\n" + "=" * 80)
    print("DEMO 4: GRPO-Compatible Reward Function")
    print("=" * 80)

    # Batch of examples
    prompts = [
        "What is 5 × 3?",
        "Calculate 100 - 37",
        "What is 2^3?",
    ]

    completions = [
        """<reasoning>
Step 1: Multiply 5 by 3
Step 2: 5 × 3 = 15
</reasoning>
<answer>15</answer>""",

        """<reasoning>
Step 1: Subtract 37 from 100
Step 2: 100 - 37 = 63
</reasoning>
<answer>63</answer>""",

        """<reasoning>
Step 1: Calculate 2 to the power of 3
Step 2: 2 × 2 × 2 = 8
</reasoning>
<answer>8</answer>""",
    ]

    answers = ["15", "63", "8"]

    # Calculate rewards
    config = PRIMEConfig(
        step_evaluation_method=StepEvaluationMethod.HYBRID,
        reward_aggregation=RewardAggregation.DISCOUNTED_SUM,
        gamma=0.95
    )

    print("\nCalculating PRIME RL rewards for batch of 3 examples...")
    rewards = prime_rl_reward(
        prompts=prompts,
        completions=completions,
        answer=answers,
        config=config
    )

    print("\nResults:")
    print("-" * 80)
    for i, (prompt, reward) in enumerate(zip(prompts, rewards)):
        print(f"\nExample {i + 1}: {prompt}")
        print(f"  Reward: {reward:.3f}")


def demo_process_vs_outcome():
    """Demonstrate combining process and outcome rewards."""
    print("\n" + "=" * 80)
    print("DEMO 5: Process vs Outcome Rewards")
    print("=" * 80)

    # Example with correct answer but flawed reasoning
    completion_flawed = """<reasoning>
Step 1: I'll just guess
Step 2: The answer is probably 4
</reasoning>
<answer>4</answer>"""

    # Example with correct reasoning
    completion_correct = """<reasoning>
Step 1: Add 2 + 2
Step 2: 2 + 2 = 4
Step 3: Verify: 4 is correct
</reasoning>
<answer>4</answer>"""

    prompt = "What is 2 + 2?"
    answer = "4"

    config = PRIMEConfig(
        combine_with_outcome_rewards=True,
        outcome_reward_weight=0.5,  # 50/50 process and outcome
    )

    calculator = ProcessRewardCalculator(config)

    print("\nExample 1: Flawed reasoning, correct answer")
    traj1 = calculator.calculate_trajectory_reward(
        prompt=prompt,
        completion=completion_flawed,
        final_answer_reward=1.0,  # Correct answer
        answer=answer
    )

    print(f"  Process reward: {traj1.aggregated_reward:.3f}")
    print(f"  Outcome reward: {traj1.final_answer_reward:.3f}")
    print(f"  Total reward: {traj1.total_reward:.3f}")

    print("\nExample 2: Good reasoning, correct answer")
    traj2 = calculator.calculate_trajectory_reward(
        prompt=prompt,
        completion=completion_correct,
        final_answer_reward=1.0,  # Correct answer
        answer=answer
    )

    print(f"  Process reward: {traj2.aggregated_reward:.3f}")
    print(f"  Outcome reward: {traj2.final_answer_reward:.3f}")
    print(f"  Total reward: {traj2.total_reward:.3f}")

    print("\n📊 Analysis:")
    print(f"  Both have correct final answer (outcome reward = 1.0)")
    print(f"  But example 2 has better process reward: {traj2.aggregated_reward:.3f} vs {traj1.aggregated_reward:.3f}")
    print(f"  PRIME RL encourages good reasoning: {traj2.total_reward:.3f} vs {traj1.total_reward:.3f}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "PRIME RL DEMONSTRATION" + " " * 36 + "║")
    print("║" + " " * 10 + "Process-based Reinforcement with Intermediate Model Evaluation" + " " * 5 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    try:
        demo_step_parsing()
        demo_step_evaluation()
        demo_trajectory_rewards()
        demo_grpo_reward_function()
        demo_process_vs_outcome()

        print("\n" + "=" * 80)
        print("✓ All demos completed successfully!")
        print("=" * 80)
        print("\nFor more information, see:")
        print("  - Documentation: docs/PRIME_RL.md")
        print("  - Tests: tests/test_prime_rl.py")
        print("  - Training script: scripts/train_prime_rl.py")
        print()

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
