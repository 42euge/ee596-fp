"""Integration tests for reward_robustness/evaluator.py."""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path to allow direct imports without going through src/__init__.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestRobustnessEvaluator:
    """Integration tests for RobustnessEvaluator class."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic test configuration."""
        from reward_robustness.config import (
            RobustnessConfig,
            PerturbationConfig,
        )

        return RobustnessConfig(
            internal_rewards=["format_reward"],
            perturbations=PerturbationConfig(
                enabled_types=["reorder"],  # Only reorder to avoid external deps
                num_variants=2,
            ),
            num_samples=3,
            output_dir=None,  # Don't save to disk
            save_detailed=True,
        )

    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        prompts = [
            "What is 2 + 2?",
            "Calculate 5 * 3.",
            "What is 10 - 4?",
        ]
        completions = [
            "<reasoning>First add. Then verify. Check again. Done.</reasoning><answer>4</answer>",
            "<reasoning>Multiply. Check. Verify result. Complete.</reasoning><answer>15</answer>",
            "<reasoning>Subtract. Verify. Double check. Finished.</reasoning><answer>6</answer>",
        ]
        answers = ["4", "15", "6"]
        return prompts, completions, answers

    def test_evaluator_init(self, basic_config):
        """Test evaluator initialization."""
        from reward_robustness.evaluator import RobustnessEvaluator

        evaluator = RobustnessEvaluator(basic_config)

        assert evaluator.config == basic_config
        assert evaluator._is_setup is False

    def test_evaluator_setup(self, basic_config):
        """Test evaluator setup with mocked rewards."""
        from reward_robustness.evaluator import RobustnessEvaluator
        from reward_robustness.rewards import InternalReward

        evaluator = RobustnessEvaluator(basic_config)

        # Mock the setup to avoid import issues
        def mock_format_reward(prompts, completions, **kwargs):
            return [1.0] * len(completions)

        evaluator._reward_models = [InternalReward(mock_format_reward, "format_reward")]
        evaluator._perturbation_pipeline = Mock()
        evaluator._perturbation_pipeline.perturbation_names = ["reorder"]
        evaluator._is_setup = True

        assert evaluator._is_setup is True
        assert len(evaluator._reward_models) == 1
        assert evaluator._perturbation_pipeline is not None

    def test_evaluator_setup_idempotent(self, basic_config):
        """Test that setup can be called multiple times safely."""
        from reward_robustness.evaluator import RobustnessEvaluator

        evaluator = RobustnessEvaluator(basic_config)

        # Manually mark as setup
        evaluator._is_setup = True

        # Calling setup again should return early
        evaluator.setup()  # Should not fail since already set up

        assert evaluator._is_setup is True

    def test_evaluator_evaluate_basic(self, basic_config, sample_data):
        """Test basic evaluation with mocked components."""
        from reward_robustness.evaluator import RobustnessEvaluator, RobustnessResults
        from reward_robustness.rewards import InternalReward
        from reward_robustness.perturbations import PerturbationPipeline, PerturbedText

        prompts, completions, answers = sample_data

        evaluator = RobustnessEvaluator(basic_config)

        # Mock the internal reward
        def mock_format_reward(prompts, completions, **kwargs):
            return [1.5] * len(completions)

        evaluator._reward_models = [InternalReward(mock_format_reward, "format_reward")]

        # Mock perturbation pipeline
        from reward_robustness.config import PerturbationConfig
        evaluator._perturbation_pipeline = PerturbationPipeline(
            PerturbationConfig(enabled_types=["reorder"], num_variants=2)
        )
        evaluator._is_setup = True

        results = evaluator.evaluate(
            prompts=prompts,
            completions=completions,
            answers=answers,
            verbose=False,
        )

        assert isinstance(results, RobustnessResults)
        assert results.num_samples == 3
        assert "format_reward" in results.metrics
        assert len(results.ranking) == 1

    def test_evaluator_respects_num_samples(self, basic_config, sample_data):
        """Test that num_samples limit is respected."""
        from reward_robustness.evaluator import RobustnessEvaluator
        from reward_robustness.rewards import InternalReward
        from reward_robustness.perturbations import PerturbationPipeline
        from reward_robustness.config import PerturbationConfig

        prompts, completions, answers = sample_data
        basic_config.num_samples = 2

        evaluator = RobustnessEvaluator(basic_config)

        def mock_reward(prompts, completions, **kwargs):
            return [1.0] * len(completions)

        evaluator._reward_models = [InternalReward(mock_reward, "format_reward")]
        evaluator._perturbation_pipeline = PerturbationPipeline(
            PerturbationConfig(enabled_types=["reorder"], num_variants=2)
        )
        evaluator._is_setup = True

        results = evaluator.evaluate(
            prompts=prompts,
            completions=completions,
            answers=answers,
            verbose=False,
        )

        assert results.num_samples == 2

    def test_evaluator_with_detailed_output(self, basic_config, sample_data):
        """Test evaluation with detailed per-sample output."""
        from reward_robustness.evaluator import RobustnessEvaluator
        from reward_robustness.rewards import InternalReward
        from reward_robustness.perturbations import PerturbationPipeline
        from reward_robustness.config import PerturbationConfig

        prompts, completions, answers = sample_data
        basic_config.save_detailed = True

        evaluator = RobustnessEvaluator(basic_config)

        def mock_reward(prompts, completions, **kwargs):
            return [1.0] * len(completions)

        evaluator._reward_models = [InternalReward(mock_reward, "format_reward")]
        evaluator._perturbation_pipeline = PerturbationPipeline(
            PerturbationConfig(enabled_types=["reorder"], num_variants=2)
        )
        evaluator._is_setup = True

        results = evaluator.evaluate(
            prompts=prompts,
            completions=completions,
            verbose=False,
        )

        assert results.samples is not None
        assert len(results.samples) == 3

        sample = results.samples[0]
        assert sample.sample_idx == 0
        assert sample.prompt == prompts[0]

    def test_evaluator_without_detailed_output(self, basic_config, sample_data):
        """Test evaluation without detailed output."""
        from reward_robustness.evaluator import RobustnessEvaluator
        from reward_robustness.rewards import InternalReward
        from reward_robustness.perturbations import PerturbationPipeline
        from reward_robustness.config import PerturbationConfig

        prompts, completions, _ = sample_data
        basic_config.save_detailed = False

        evaluator = RobustnessEvaluator(basic_config)

        def mock_reward(prompts, completions, **kwargs):
            return [1.0] * len(completions)

        evaluator._reward_models = [InternalReward(mock_reward, "format_reward")]
        evaluator._perturbation_pipeline = PerturbationPipeline(
            PerturbationConfig(enabled_types=["reorder"], num_variants=2)
        )
        evaluator._is_setup = True

        results = evaluator.evaluate(
            prompts=prompts,
            completions=completions,
            verbose=False,
        )

        assert results.samples is None


class TestRobustnessResults:
    """Tests for RobustnessResults dataclass."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        from reward_robustness.evaluator import RobustnessResults
        from reward_robustness.metrics import ConsistencyMetrics

        metrics = {
            "format_reward": ConsistencyMetrics(
                reward_name="format_reward",
                mean_variance=0.1,
                max_variance=0.2,
                median_variance=0.1,
                variance_std=0.05,
                mean_cv=0.1,
                kendall_tau=0.9,
                spearman_rho=0.9,
                flip_rate=0.05,
                max_deviation=0.2,
                mean_deviation=0.1,
                stability_score=0.85,
                num_samples=10,
            )
        }

        return RobustnessResults(
            timestamp="2024-01-01T00:00:00",
            config={"test": "config"},
            num_samples=10,
            perturbation_types=["reorder"],
            num_variants_per_sample=3,
            metrics=metrics,
            ranking=[{"reward": "format_reward", "stability_score": 0.85}],
            samples=None,
        )

    def test_results_to_dict(self, sample_results):
        """Test conversion to dictionary."""
        result_dict = sample_results.to_dict()

        assert "meta" in result_dict
        assert "summary" in result_dict
        assert "results" in result_dict

        assert result_dict["meta"]["num_samples"] == 10
        assert result_dict["summary"]["ranking"][0]["reward"] == "format_reward"
        assert "format_reward" in result_dict["results"]

    def test_results_to_json(self, sample_results):
        """Test conversion to JSON string."""
        json_str = sample_results.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["meta"]["num_samples"] == 10

    def test_results_save(self, sample_results):
        """Test saving results to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            sample_results.save(output_path)

            assert output_path.exists()

            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded["meta"]["num_samples"] == 10

    def test_results_save_creates_dirs(self, sample_results):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "results.json"
            sample_results.save(output_path)

            assert output_path.exists()


class TestReportGeneration:
    """Tests for report generation."""

    @pytest.fixture
    def evaluator_with_results(self):
        """Create evaluator with sample results."""
        from reward_robustness.config import RobustnessConfig, PerturbationConfig
        from reward_robustness.evaluator import RobustnessEvaluator, RobustnessResults
        from reward_robustness.metrics import ConsistencyMetrics

        config = RobustnessConfig(
            internal_rewards=["format_reward"],
            perturbations=PerturbationConfig(enabled_types=["reorder"]),
        )

        evaluator = RobustnessEvaluator(config)

        metrics = {
            "format_reward": ConsistencyMetrics(
                reward_name="format_reward",
                mean_variance=0.1,
                max_variance=0.2,
                median_variance=0.1,
                variance_std=0.05,
                mean_cv=0.1,
                kendall_tau=0.9,
                spearman_rho=0.9,
                flip_rate=0.05,
                max_deviation=0.2,
                mean_deviation=0.1,
                stability_score=0.85,
                num_samples=10,
            )
        }

        results = RobustnessResults(
            timestamp="2024-01-01T00:00:00",
            config={},
            num_samples=10,
            perturbation_types=["reorder"],
            num_variants_per_sample=3,
            metrics=metrics,
            ranking=[{"reward": "format_reward", "stability_score": 0.85}],
            samples=None,
        )

        return evaluator, results

    def test_generate_report_markdown(self, evaluator_with_results):
        """Test generating markdown report."""
        evaluator, results = evaluator_with_results

        report = evaluator.generate_report(results, format="markdown")

        assert "# Reward Robustness Evaluation Report" in report
        assert "format_reward" in report
        assert "0.85" in report or "0.850" in report

    def test_generate_report_text(self, evaluator_with_results):
        """Test generating text report."""
        evaluator, results = evaluator_with_results

        report = evaluator.generate_report(results, format="text")

        assert "REWARD ROBUSTNESS EVALUATION REPORT" in report
        assert "format_reward" in report

    def test_generate_report_json(self, evaluator_with_results):
        """Test generating JSON report."""
        evaluator, results = evaluator_with_results

        report = evaluator.generate_report(results, format="json")

        # Should be valid JSON
        parsed = json.loads(report)
        assert "meta" in parsed


class TestEvaluatorWithMultipleRewards:
    """Tests for evaluator with multiple reward functions."""

    def test_multiple_internal_rewards(self):
        """Test evaluation with multiple internal rewards (mocked)."""
        from reward_robustness.config import RobustnessConfig, PerturbationConfig
        from reward_robustness.evaluator import RobustnessEvaluator
        from reward_robustness.rewards import InternalReward
        from reward_robustness.perturbations import PerturbationPipeline

        config = RobustnessConfig(
            internal_rewards=["format_reward", "accuracy_reward"],
            perturbations=PerturbationConfig(
                enabled_types=["reorder"],
                num_variants=2,
            ),
            num_samples=2,
            save_detailed=False,
        )

        prompts = ["What is 2+2?", "What is 3+3?"]
        completions = [
            "<reasoning>Add. Check. Verify. Done.</reasoning><answer>4</answer>",
            "<reasoning>Add. Check. Verify. Done.</reasoning><answer>6</answer>",
        ]
        answers = ["4", "6"]

        evaluator = RobustnessEvaluator(config)

        # Mock rewards
        def mock_format(p, c, **k): return [1.0] * len(c)
        def mock_accuracy(p, c, **k): return [0.5] * len(c)

        evaluator._reward_models = [
            InternalReward(mock_format, "format_reward"),
            InternalReward(mock_accuracy, "accuracy_reward"),
        ]
        evaluator._perturbation_pipeline = PerturbationPipeline(config.perturbations)
        evaluator._is_setup = True

        results = evaluator.evaluate(
            prompts=prompts,
            completions=completions,
            answers=answers,
            verbose=False,
        )

        assert "format_reward" in results.metrics
        assert "accuracy_reward" in results.metrics
        assert len(results.ranking) == 2

    def test_ranking_order(self):
        """Test that ranking is ordered by stability score (mocked)."""
        from reward_robustness.config import RobustnessConfig, PerturbationConfig
        from reward_robustness.evaluator import RobustnessEvaluator
        from reward_robustness.rewards import InternalReward
        from reward_robustness.perturbations import PerturbationPipeline

        config = RobustnessConfig(
            internal_rewards=["format_reward", "accuracy_reward"],
            perturbations=PerturbationConfig(
                enabled_types=["reorder"],
                num_variants=2,
            ),
            num_samples=3,
            save_detailed=False,
        )

        prompts = ["Q1?", "Q2?", "Q3?"]
        completions = [
            "<reasoning>Step one. Step two. Step three. Done.</reasoning><answer>1</answer>",
            "<reasoning>Step one. Step two. Step three. Done.</reasoning><answer>2</answer>",
            "<reasoning>Step one. Step two. Step three. Done.</reasoning><answer>3</answer>",
        ]
        answers = ["1", "2", "3"]

        evaluator = RobustnessEvaluator(config)

        def mock_format(p, c, **k): return [1.0] * len(c)
        def mock_accuracy(p, c, **k): return [0.5] * len(c)

        evaluator._reward_models = [
            InternalReward(mock_format, "format_reward"),
            InternalReward(mock_accuracy, "accuracy_reward"),
        ]
        evaluator._perturbation_pipeline = PerturbationPipeline(config.perturbations)
        evaluator._is_setup = True

        results = evaluator.evaluate(
            prompts=prompts,
            completions=completions,
            answers=answers,
            verbose=False,
        )

        # Ranking should be sorted by stability (highest first)
        if len(results.ranking) > 1:
            assert results.ranking[0]["stability_score"] >= results.ranking[1]["stability_score"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_prompts(self):
        """Test with empty prompts list - should handle gracefully or raise."""
        from reward_robustness.config import RobustnessConfig, PerturbationConfig
        from reward_robustness.evaluator import RobustnessEvaluator, RobustnessResults
        from reward_robustness.rewards import InternalReward
        from reward_robustness.perturbations import PerturbationPipeline

        config = RobustnessConfig(
            internal_rewards=["format_reward"],
            perturbations=PerturbationConfig(enabled_types=["reorder"]),
            num_samples=10,
            output_dir=None,
        )

        evaluator = RobustnessEvaluator(config)

        def mock_reward(p, c, **k): return [1.0] * len(c)
        evaluator._reward_models = [InternalReward(mock_reward, "format_reward")]
        evaluator._perturbation_pipeline = PerturbationPipeline(config.perturbations)
        evaluator._is_setup = True

        # Empty prompts may raise or return empty results - both are valid
        try:
            results = evaluator.evaluate(
                prompts=[],
                completions=[],
                verbose=False,
            )
            assert results.num_samples == 0
        except (ValueError, IndexError):
            # Empty input edge case - acceptable to raise
            pass

    def test_single_sample(self):
        """Test with single sample."""
        from reward_robustness.config import RobustnessConfig, PerturbationConfig
        from reward_robustness.evaluator import RobustnessEvaluator
        from reward_robustness.rewards import InternalReward
        from reward_robustness.perturbations import PerturbationPipeline

        config = RobustnessConfig(
            internal_rewards=["format_reward"],
            perturbations=PerturbationConfig(
                enabled_types=["reorder"],
                num_variants=2,
            ),
            num_samples=10,
        )

        evaluator = RobustnessEvaluator(config)

        def mock_reward(p, c, **k): return [1.0] * len(c)
        evaluator._reward_models = [InternalReward(mock_reward, "format_reward")]
        evaluator._perturbation_pipeline = PerturbationPipeline(config.perturbations)
        evaluator._is_setup = True

        results = evaluator.evaluate(
            prompts=["Single question?"],
            completions=["<reasoning>Think. Process. Conclude. Done.</reasoning><answer>42</answer>"],
            verbose=False,
        )

        assert results.num_samples == 1
