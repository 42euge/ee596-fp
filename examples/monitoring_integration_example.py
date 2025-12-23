#!/usr/bin/env python3
"""
Example: Integrating Reward Monitoring into GRPO Training

This example shows how to add comprehensive reward monitoring to your GRPO training script.
It demonstrates the minimal changes needed to add full observability.

Usage:
    python examples/monitoring_integration_example.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Example of integrating monitoring into GRPO training."""

    # ========================================================================
    # STEP 1: Import monitoring
    # ========================================================================
    from tunrex.monitoring import setup_grpo_monitoring
    from tunrex.datasets.rewards import (
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    )

    print("=" * 80)
    print("GRPO Training with Reward Monitoring - Example")
    print("=" * 80)

    # ========================================================================
    # STEP 2: Setup monitoring BEFORE creating GRPO trainer
    # ========================================================================
    print("\n📊 Setting up reward monitoring...")

    monitoring = setup_grpo_monitoring(
        reward_functions=[
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ],
        # Optional: provide custom names
        reward_names=["format_exact", "format_approx", "answer", "numbers"],
        # Enable W&B logging (requires wandb.init() to be called first)
        use_wandb=False,  # Set to True if W&B is initialized
        # Enable anomaly detection
        enable_anomaly_detection=True,
        # Logging configuration
        log_frequency=10,  # Log metrics every 10 steps
        summary_frequency=50,  # Print summary every 50 steps
        visualization_frequency=100,  # Create plots every 100 steps
        # Output directory
        output_dir="/tmp/reward_monitoring_example",
        # Verbose output
        verbose=True,
    )

    # ========================================================================
    # STEP 3: Get wrapped reward functions
    # ========================================================================
    wrapped_reward_functions = monitoring.get_wrapped_reward_functions()

    print("\n✓ Monitoring setup complete!")
    print(f"  - Monitoring {len(wrapped_reward_functions)} reward functions")
    print(f"  - Output directory: /tmp/reward_monitoring_example")

    # ========================================================================
    # STEP 4: Pass wrapped functions to GRPO trainer
    # ========================================================================
    # In your actual training script, you would do:
    #
    # grpo_trainer = GRPOLearner(
    #     rl_cluster=rl_cluster,
    #     reward_fns=wrapped_reward_functions,  # <-- Use wrapped functions
    #     algo_config=grpo_config,
    # )

    # ========================================================================
    # STEP 5: Simulate training loop
    # ========================================================================
    print("\n" + "=" * 80)
    print("Simulating Training Loop")
    print("=" * 80)

    import random
    import time

    num_steps = 200

    for step in range(num_steps):
        # Simulate training step...
        time.sleep(0.01)  # Simulate computation

        # Simulate reward values (in real training, these come from GRPO)
        simulated_rewards = {
            "format_exact": random.choice([0.0, 3.0]),
            "format_approx": random.uniform(-0.5, 2.5),
            "answer": random.choice([-1.0, 0.5, 1.5, 3.0]),
            "numbers": random.choice([0.0, 1.5]),
        }

        # Introduce some anomalies for demonstration
        if step == 50:
            simulated_rewards["answer"] = -10.0  # Outlier
        if step >= 100 and step < 120:
            simulated_rewards["format_exact"] = 0.0  # Flatline period

        # ================================================================
        # STEP 6: Update monitoring each step
        # ================================================================
        monitoring.update_step(step, reward_values=simulated_rewards)

    # ========================================================================
    # STEP 7: Finalize monitoring at end of training
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Complete - Finalizing Monitoring")
    print("=" * 80)

    monitoring.finalize()

    print("\n✓ Example complete!")
    print("\nGenerated files:")
    print("  - /tmp/reward_monitoring_example/final_dashboard.png")
    print("  - /tmp/reward_monitoring_example/final_summary.html")
    print("  - /tmp/reward_monitoring_example/reward_history.png")
    print("  - /tmp/reward_monitoring_example/reward_distributions.png")
    print("  - /tmp/reward_monitoring_example/monitoring_data.json")
    print("\nOpen final_summary.html in your browser to see the full report!")


def example_minimal_integration():
    """Minimal example showing just the essential changes."""

    print("\n" + "=" * 80)
    print("MINIMAL INTEGRATION EXAMPLE")
    print("=" * 80)

    print("""
To add monitoring to your existing training script, make these 3 changes:

1. Import and setup monitoring:

    from tunrex.monitoring import setup_grpo_monitoring
    from tunrex.datasets.rewards import match_format_exactly, ...

    monitoring = setup_grpo_monitoring(
        reward_functions=[match_format_exactly, ...],
        use_wandb=True,
    )

2. Use wrapped reward functions:

    grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=monitoring.get_wrapped_reward_functions(),  # <-- Changed
        algo_config=grpo_config,
    )

3. Update monitoring in training loop:

    for step in range(num_steps):
        # ... training code ...
        monitoring.update_step(step)  # <-- Added

    monitoring.finalize()  # <-- Added at the end

That's it! Monitoring is now active.
""")


if __name__ == "__main__":
    # Run full example
    main()

    # Show minimal integration
    example_minimal_integration()
