#!/usr/bin/env python3
"""
Example of using the Reward Monitoring system.

This script demonstrates how to integrate reward hack detection
into your GRPO training pipeline.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reward_monitoring import RewardHackDetector, DetectionConfig
from src.reward_monitoring_integration import RewardFunctionMonitor, log_detections_to_wandb

# Import your reward functions
# For this example, we'll create simple mock reward functions
def mock_format_reward(response: str, question: str, answer: str) -> float:
    """Mock format reward - checks for reasoning and answer tags."""
    if '<reasoning>' in response and '</reasoning>' in response:
        if '<answer>' in response and '</answer>' in response:
            return 3.0
    return 0.0


def mock_accuracy_reward(response: str, question: str, answer: str) -> float:
    """Mock accuracy reward - checks if answer matches."""
    import re
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if match:
        predicted = match.group(1).strip()
        if predicted == answer:
            return 3.0
    return 0.0


def example_basic_usage():
    """Basic usage example."""
    print("=" * 70)
    print("BASIC USAGE EXAMPLE")
    print("=" * 70)

    # Create detection config
    config = DetectionConfig(
        reward_zscore_threshold=3.0,
        min_reasoning_length=20,
        max_ngram_repetition_ratio=0.3,
    )

    # Create detector
    detector = RewardHackDetector(config)

    # Example responses
    responses = [
        # Normal response
        "<reasoning>Let me solve this step by step. First, I'll identify the key information. Then I'll apply the formula.</reasoning><answer>42</answer>",

        # Too short (will trigger detection)
        "<reasoning>Easy.</reasoning><answer>42</answer>",

        # Repetitive (will trigger detection)
        "<reasoning>solve solve solve solve solve solve solve solve solve solve</reasoning><answer>42</answer>",

        # Format gaming (will trigger detection)
        "<reasoning>x</reasoning><answer>42</answer>",
    ]

    questions = ["What is 6 * 7?"] * len(responses)
    answers = ["42"] * len(responses)

    print("\nAnalyzing responses...\n")

    for i, (response, question, answer) in enumerate(zip(responses, questions, answers)):
        # Compute reward components
        format_reward = mock_format_reward(response, question, answer)
        accuracy_reward = mock_accuracy_reward(response, question, answer)
        total_reward = format_reward + accuracy_reward

        # Analyze for anomalies
        detections = detector.analyze_step(
            response=response,
            total_reward=total_reward,
            reward_components={
                'format': format_reward,
                'accuracy': accuracy_reward,
            },
        )

        print(f"Response {i+1}:")
        print(f"  Reward: {total_reward:.2f} (format: {format_reward:.2f}, accuracy: {accuracy_reward:.2f})")
        print(f"  Detections: {len(detections)}")

        for detection in detections:
            print(f"    [{detection.severity.upper()}] {detection.detection_type}: {detection.message}")

        print()

    # Print summary
    print(detector.get_summary_metrics())


def example_with_monitor():
    """Example using the RewardFunctionMonitor."""
    print("\n" + "=" * 70)
    print("REWARD FUNCTION MONITOR EXAMPLE")
    print("=" * 70)

    # Create monitor
    monitor = RewardFunctionMonitor(
        reward_fns=[mock_format_reward, mock_accuracy_reward],
        reward_fn_names=['format', 'accuracy'],
        wandb_enabled=False,  # Disable W&B for this example
    )

    # Simulate training steps
    print("\nSimulating training steps...\n")

    for step in range(10):
        # Mock data
        responses = [
            f"<reasoning>Step {step}: Analyzing the problem carefully with detailed reasoning that is long enough.</reasoning><answer>42</answer>",
            f"<reasoning>Quick solution for step {step}.</reasoning><answer>42</answer>",
        ]
        questions = ["What is the answer?"] * 2
        answers = ["42"] * 2

        # Compute rewards with monitoring
        results = monitor.compute_rewards(
            responses=responses,
            questions=questions,
            answers=answers,
            step=step,
            kl_divergence=0.1,  # Mock KL divergence
            gradient_norm=1.5,  # Mock gradient norm
            loss=2.3,  # Mock loss
        )

        # Log metrics periodically
        if step % 5 == 0:
            metrics = monitor.get_metrics_for_logging()
            print(f"Step {step} metrics:")
            for key, value in list(metrics.items())[:5]:  # Print first 5 metrics
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Print final summary
    print("\n" + monitor.get_summary_report())


def example_anomaly_detection():
    """Example showing various anomaly types."""
    print("\n" + "=" * 70)
    print("ANOMALY DETECTION EXAMPLES")
    print("=" * 70)

    detector = RewardHackDetector()

    anomaly_examples = [
        {
            'name': 'Mode Collapse',
            'responses': ["Same response"] * 25,  # Identical responses
            'description': 'Detecting when model generates identical responses repeatedly',
        },
        {
            'name': 'Excessive Repetition',
            'responses': ["the the the the the the the the the the the the the the"],
            'description': 'Detecting excessive token repetition',
        },
        {
            'name': 'Format Gaming',
            'responses': ["<reasoning>x</reasoning><answer>y</answer>"],
            'description': 'Getting format rewards without substantive reasoning',
        },
        {
            'name': 'Response Too Short',
            'responses': ["<answer>42</answer>"],
            'description': 'Response missing required components',
        },
    ]

    for example in anomaly_examples:
        print(f"\n{example['name']}:")
        print(f"  Description: {example['description']}")

        for response in example['responses'][:1]:  # Process first response
            detections = detector.analyze_step(
                response=response,
                total_reward=3.0,  # High format reward
                reward_components={'format': 3.0, 'accuracy': 0.0},
            )

            if detections:
                for detection in detections:
                    print(f"  ✓ Detected: [{detection.severity}] {detection.message}")
            else:
                print("  ✗ No detection (processing more samples needed)")

        # Process remaining samples for mode collapse
        if example['name'] == 'Mode Collapse':
            for response in example['responses'][1:]:
                detector.analyze_step(
                    response=response,
                    total_reward=3.0,
                    reward_components={'format': 3.0, 'accuracy': 0.0},
                )
            # Check again after all samples
            detections = detector.analyze_step(
                response=example['responses'][0],
                total_reward=3.0,
                reward_components={'format': 3.0, 'accuracy': 0.0},
            )
            if detections:
                for detection in detections:
                    if detection.detection_type == 'mode_collapse':
                        print(f"  ✓ Detected after {len(example['responses'])} samples: [{detection.severity}] {detection.message}")


def example_training_dynamics():
    """Example showing training dynamics monitoring."""
    print("\n" + "=" * 70)
    print("TRAINING DYNAMICS MONITORING")
    print("=" * 70)

    detector = RewardHackDetector()

    dynamics_examples = [
        {
            'name': 'Exploding Gradients',
            'gradient_norm': 100.0,
            'kl_divergence': 0.1,
            'loss': 2.0,
        },
        {
            'name': 'Vanishing Gradients',
            'gradient_norm': 1e-8,
            'kl_divergence': 0.1,
            'loss': 2.0,
        },
        {
            'name': 'Excessive KL Divergence',
            'gradient_norm': 1.0,
            'kl_divergence': 10.0,
            'loss': 2.0,
        },
        {
            'name': 'Insufficient Exploration',
            'gradient_norm': 1.0,
            'kl_divergence': 0.0001,
            'loss': 2.0,
        },
    ]

    for example in dynamics_examples:
        print(f"\n{example['name']}:")

        detections = detector.analyze_step(
            response="<reasoning>Normal response with adequate reasoning.</reasoning><answer>42</answer>",
            total_reward=5.0,
            reward_components={'format': 3.0, 'accuracy': 2.0},
            gradient_norm=example.get('gradient_norm'),
            kl_divergence=example.get('kl_divergence'),
            loss=example.get('loss'),
        )

        for detection in detections:
            if detection.detection_type in ['gradient_norm', 'kl_divergence']:
                print(f"  ✓ Detected: [{detection.severity}] {detection.message}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("REWARD MONITORING SYSTEM EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate the reward hack detection system.")
    print("They show how to detect various problematic behaviors during training.")
    print("=" * 70)

    # Run examples
    example_basic_usage()
    example_with_monitor()
    example_anomaly_detection()
    example_training_dynamics()

    print("\n" + "=" * 70)
    print("EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nFor integration with GRPO training, see:")
    print("  scripts/train_grpo_monitored.py")
    print("=" * 70 + "\n")
