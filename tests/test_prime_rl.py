"""
Tests for PRIME RL (Process-based Reinforcement with Intermediate Model Evaluation)

Tests cover:
- Step parsing with different strategies
- Step evaluation with multiple methods
- Process reward calculation
- Trajectory-based rewards
- Integration with GRPO reward functions
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prime_rl import (
    PRIMEConfig,
    StepParser,
    StepEvaluator,
    ProcessRewardCalculator,
    StepParsingStrategy,
    StepEvaluationMethod,
    RewardAggregation,
    parse_steps,
    prime_rl_reward,
    create_prime_rl_reward_suite,
)


class TestStepParser(unittest.TestCase):
    """Test step parsing functionality."""

    def setUp(self):
        """Set up test cases."""
        self.numbered_completion = """<reasoning>
Step 1: Identify the variables
Step 2: Apply the formula
Step 3: Calculate the result
Step 4: Verify the answer
</reasoning>
<answer>42</answer>"""

        self.line_based_completion = """<reasoning>
First, we identify the variables.
Then, we apply the formula.
Next, we calculate the result.
Finally, we verify the answer.
</reasoning>
<answer>42</answer>"""

    def test_numbered_parsing(self):
        """Test parsing numbered steps."""
        config = PRIMEConfig(step_parsing_strategy=StepParsingStrategy.NUMBERED)
        parser = StepParser(config)

        steps = parser.parse(self.numbered_completion)

        self.assertEqual(len(steps), 4)
        self.assertIn("Identify the variables", steps[0].text)
        self.assertIn("Apply the formula", steps[1].text)

    def test_line_based_parsing(self):
        """Test parsing line-based steps."""
        config = PRIMEConfig(step_parsing_strategy=StepParsingStrategy.LINE_BASED)
        parser = StepParser(config)

        steps = parser.parse(self.line_based_completion)

        self.assertGreater(len(steps), 0)
        self.assertTrue(any("variables" in step.text for step in steps))

    def test_sentence_based_parsing(self):
        """Test sentence-based parsing."""
        config = PRIMEConfig(step_parsing_strategy=StepParsingStrategy.SENTENCE_BASED)
        parser = StepParser(config)

        completion = "First, add 2 and 2. Then, multiply by 3. Finally, subtract 1."
        steps = parser.parse(completion)

        self.assertGreaterEqual(len(steps), 3)

    def test_step_metadata(self):
        """Test that steps have correct metadata."""
        config = PRIMEConfig(step_parsing_strategy=StepParsingStrategy.NUMBERED)
        parser = StepParser(config)

        completion = "Step 1: Calculate 2 + 2 = 4"
        steps = parser.parse(completion)

        self.assertEqual(len(steps), 1)
        self.assertTrue(steps[0].is_calculation)

    def test_min_step_length_filter(self):
        """Test filtering by minimum step length."""
        config = PRIMEConfig(
            step_parsing_strategy=StepParsingStrategy.LINE_BASED,
            min_step_length=20
        )
        parser = StepParser(config)

        completion = "Short.\nThis is a longer step with more content.\nTiny."
        steps = parser.parse(completion)

        # Only the longer step should remain
        self.assertEqual(len(steps), 1)
        self.assertIn("longer step", steps[0].text)


class TestStepEvaluator(unittest.TestCase):
    """Test step evaluation functionality."""

    def setUp(self):
        """Set up test cases."""
        self.config = PRIMEConfig(
            step_evaluation_method=StepEvaluationMethod.RULE_BASED
        )

    def test_rule_based_evaluation(self):
        """Test rule-based step evaluation."""
        from src.prime_rl.step_parser import ParsedStep

        evaluator = StepEvaluator(self.config)

        # Good step with calculation
        step = ParsedStep(
            index=0,
            text="Calculate: 2 + 2 = 4"
        )

        reward = evaluator.evaluate_step(step)

        self.assertIsNotNone(reward)
        self.assertGreater(reward.reward, 0.0)
        self.assertEqual(reward.step_index, 0)

    def test_symbolic_evaluation(self):
        """Test symbolic evaluation (if sympy available)."""
        config = PRIMEConfig(
            step_evaluation_method=StepEvaluationMethod.SYMBOLIC,
            enable_symbolic_solver=True
        )

        from src.prime_rl.step_parser import ParsedStep

        evaluator = StepEvaluator(config)

        # Correct equation
        step = ParsedStep(
            index=0,
            text="Therefore, 2 + 2 = 4"
        )

        reward = evaluator.evaluate_step(step)

        self.assertIsNotNone(reward)
        # Should return some score (even if sympy not available)
        self.assertIsNotNone(reward.reward)

    def test_hybrid_evaluation(self):
        """Test hybrid evaluation combining multiple methods."""
        config = PRIMEConfig(
            step_evaluation_method=StepEvaluationMethod.HYBRID
        )

        from src.prime_rl.step_parser import ParsedStep

        evaluator = StepEvaluator(config)

        step = ParsedStep(
            index=0,
            text="Using the formula, we get x = 10"
        )

        reward = evaluator.evaluate_step(step)

        self.assertIsNotNone(reward)
        self.assertIn("evaluations", reward.details)


class TestProcessRewardCalculator(unittest.TestCase):
    """Test process reward calculation."""

    def setUp(self):
        """Set up test cases."""
        self.config = PRIMEConfig(
            step_parsing_strategy=StepParsingStrategy.NUMBERED,
            step_evaluation_method=StepEvaluationMethod.RULE_BASED,
            reward_aggregation=RewardAggregation.DISCOUNTED_SUM,
            gamma=0.95
        )

        self.completion = """<reasoning>
Step 1: Identify that we need to solve 2 + 2
Step 2: Add the numbers: 2 + 2 = 4
Step 3: Verify the result is correct
</reasoning>
<answer>4</answer>"""

    def test_trajectory_reward_calculation(self):
        """Test calculating trajectory reward."""
        calculator = ProcessRewardCalculator(self.config)

        trajectory = calculator.calculate_trajectory_reward(
            prompt="What is 2 + 2?",
            completion=self.completion,
            final_answer_reward=1.0,
            question="What is 2 + 2?",
            answer="4"
        )

        self.assertIsNotNone(trajectory)
        self.assertEqual(len(trajectory.steps), 3)
        self.assertEqual(len(trajectory.step_rewards), 3)
        self.assertIsNotNone(trajectory.total_reward)

    def test_reward_aggregation_strategies(self):
        """Test different reward aggregation strategies."""
        strategies = [
            RewardAggregation.SUM,
            RewardAggregation.DISCOUNTED_SUM,
            RewardAggregation.MEAN,
            RewardAggregation.WEIGHTED_MEAN,
        ]

        for strategy in strategies:
            config = PRIMEConfig(
                reward_aggregation=strategy,
                gamma=0.95
            )

            calculator = ProcessRewardCalculator(config)
            trajectory = calculator.calculate_trajectory_reward(
                prompt="Test",
                completion=self.completion,
                final_answer_reward=1.0
            )

            self.assertIsNotNone(trajectory.total_reward)
            self.assertIsInstance(trajectory.total_reward, float)

    def test_combine_with_outcome_reward(self):
        """Test combining process and outcome rewards."""
        config = PRIMEConfig(
            combine_with_outcome_rewards=True,
            outcome_reward_weight=0.5
        )

        calculator = ProcessRewardCalculator(config)
        trajectory = calculator.calculate_trajectory_reward(
            prompt="Test",
            completion=self.completion,
            final_answer_reward=1.0
        )

        # Total reward should be combination of process and outcome
        self.assertIsNotNone(trajectory.total_reward)

    def test_normalize_rewards(self):
        """Test reward normalization."""
        calculator = ProcessRewardCalculator(self.config)

        # Create multiple trajectories
        trajectories = [
            calculator.calculate_trajectory_reward(
                prompt=f"Test {i}",
                completion=self.completion,
                final_answer_reward=float(i) / 10.0
            )
            for i in range(5)
        ]

        # Normalize
        normalized = calculator.normalize_rewards(trajectories)

        self.assertEqual(len(normalized), len(trajectories))

        # Check that mean is approximately 0
        mean_reward = sum(t.total_reward for t in normalized) / len(normalized)
        self.assertAlmostEqual(mean_reward, 0.0, places=1)


class TestRewardFunctions(unittest.TestCase):
    """Test GRPO-compatible reward functions."""

    def test_prime_rl_reward_basic(self):
        """Test basic PRIME RL reward function."""
        prompts = ["What is 2 + 2?"]
        completions = ["""<reasoning>
Step 1: Add 2 + 2
Step 2: Result is 4
</reasoning>
<answer>4</answer>"""]

        config = PRIMEConfig()
        rewards = prime_rl_reward(
            prompts=prompts,
            completions=completions,
            answer=["4"],
            config=config
        )

        self.assertEqual(len(rewards), 1)
        self.assertIsInstance(rewards[0], float)

    def test_prime_rl_reward_batch(self):
        """Test PRIME RL reward with batch of examples."""
        prompts = [
            "What is 2 + 2?",
            "What is 3 + 3?",
            "What is 5 + 5?",
        ]
        completions = [
            "Step 1: Calculate 2 + 2 = 4\nAnswer: 4",
            "Step 1: Calculate 3 + 3 = 6\nAnswer: 6",
            "Step 1: Calculate 5 + 5 = 10\nAnswer: 10",
        ]
        answers = ["4", "6", "10"]

        config = PRIMEConfig()
        rewards = prime_rl_reward(
            prompts=prompts,
            completions=completions,
            answer=answers,
            config=config
        )

        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(isinstance(r, float) for r in rewards))

    def test_create_reward_suite(self):
        """Test creating suite of PRIME RL rewards."""
        config = PRIMEConfig()
        reward_fns = create_prime_rl_reward_suite(
            config=config,
            include_format=True,
            include_accuracy=True,
            include_pure_prime=True
        )

        # Should have 3 reward functions
        self.assertEqual(len(reward_fns), 3)

        # All should be callable
        self.assertTrue(all(callable(fn) for fn in reward_fns))


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation."""

    def test_default_config(self):
        """Test default configuration is valid."""
        config = PRIMEConfig()

        self.assertIsNotNone(config.gamma)
        self.assertGreater(config.gamma, 0.0)
        self.assertLessEqual(config.gamma, 1.0)

    def test_gamma_validation(self):
        """Test gamma validation."""
        with self.assertRaises(AssertionError):
            PRIMEConfig(gamma=1.5)  # > 1.0

        with self.assertRaises(AssertionError):
            PRIMEConfig(gamma=-0.1)  # < 0.0

    def test_custom_delimiter_validation(self):
        """Test custom delimiter validation."""
        with self.assertRaises(AssertionError):
            # Should fail if custom delimiter not provided
            PRIMEConfig(
                step_parsing_strategy=StepParsingStrategy.CUSTOM_DELIMITER,
                custom_delimiter=None
            )

        # Should succeed with delimiter
        config = PRIMEConfig(
            step_parsing_strategy=StepParsingStrategy.CUSTOM_DELIMITER,
            custom_delimiter="---"
        )
        self.assertEqual(config.custom_delimiter, "---")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Configuration
        config = PRIMEConfig(
            step_parsing_strategy=StepParsingStrategy.NUMBERED,
            step_evaluation_method=StepEvaluationMethod.HYBRID,
            reward_aggregation=RewardAggregation.DISCOUNTED_SUM,
            gamma=0.95
        )

        # Input
        prompts = ["Solve: 2 + 2 = ?"]
        completions = ["""<reasoning>
Step 1: Identify the problem as addition
Step 2: Add 2 + 2 = 4
Step 3: Verify the result
</reasoning>
<answer>4</answer>"""]
        answers = ["4"]

        # Calculate rewards
        rewards = prime_rl_reward(
            prompts=prompts,
            completions=completions,
            answer=answers,
            config=config
        )

        # Verify output
        self.assertEqual(len(rewards), 1)
        self.assertIsInstance(rewards[0], float)

    def test_multiple_parsing_strategies(self):
        """Test that different parsing strategies work."""
        completion = """<reasoning>
Step 1: First step
Step 2: Second step
Step 3: Third step
</reasoning>"""

        strategies = [
            StepParsingStrategy.NUMBERED,
            StepParsingStrategy.LINE_BASED,
            StepParsingStrategy.SENTENCE_BASED,
        ]

        for strategy in strategies:
            config = PRIMEConfig(step_parsing_strategy=strategy)
            steps = parse_steps(completion, config)

            # All strategies should extract some steps
            self.assertGreater(len(steps), 0, f"Failed for {strategy}")


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == "__main__":
    run_tests()
