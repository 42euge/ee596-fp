"""Unit tests for reward_robustness/metrics.py."""

import pytest
import sys
from pathlib import Path
import numpy as np

# Add src to path to allow direct imports without going through src/__init__.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestVarianceMetrics:
    """Tests for variance computation functions."""

    def test_compute_variance_metrics_basic(self):
        """Test basic variance computation."""
        from reward_robustness.metrics import compute_variance_metrics

        original = np.array([1.0, 2.0, 3.0])
        # Each sample has 3 perturbations with slight variations
        perturbed = np.array([
            [1.1, 0.9, 1.0],  # variance around 1.0
            [2.1, 1.9, 2.0],  # variance around 2.0
            [3.1, 2.9, 3.0],  # variance around 3.0
        ])

        metrics = compute_variance_metrics(original, perturbed)

        assert "mean_variance" in metrics
        assert "max_variance" in metrics
        assert "median_variance" in metrics
        assert "variance_std" in metrics
        assert "mean_cv" in metrics
        assert "sample_variances" in metrics

        # All variances should be small (around 0.01)
        assert metrics["mean_variance"] < 0.1
        assert len(metrics["sample_variances"]) == 3

    def test_compute_variance_metrics_zero_variance(self):
        """Test when all perturbed scores are identical."""
        from reward_robustness.metrics import compute_variance_metrics

        original = np.array([1.0, 2.0])
        perturbed = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])

        metrics = compute_variance_metrics(original, perturbed)

        assert metrics["mean_variance"] == 0.0
        assert metrics["max_variance"] == 0.0

    def test_compute_variance_metrics_high_variance(self):
        """Test with high variance perturbations."""
        from reward_robustness.metrics import compute_variance_metrics

        original = np.array([1.0, 2.0])
        perturbed = np.array([
            [0.0, 2.0, 1.0],  # high variance
            [0.0, 4.0, 2.0],  # high variance
        ])

        metrics = compute_variance_metrics(original, perturbed)

        assert metrics["mean_variance"] > 0.5


class TestRankingMetrics:
    """Tests for ranking correlation metrics."""

    @pytest.fixture(autouse=True)
    def check_scipy(self):
        """Skip tests if scipy is not installed."""
        try:
            from scipy.stats import kendalltau, spearmanr
        except ImportError:
            pytest.skip("scipy not installed")

    def test_compute_ranking_metrics_perfect_correlation(self):
        """Test with perfectly correlated scores."""
        from reward_robustness.metrics import compute_ranking_metrics

        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Perturbations maintain same ranking
        perturbed = np.array([
            [1.1, 1.0, 0.9],
            [2.1, 2.0, 1.9],
            [3.1, 3.0, 2.9],
            [4.1, 4.0, 3.9],
            [5.1, 5.0, 4.9],
        ])

        metrics = compute_ranking_metrics(original, perturbed)

        assert "kendall_tau" in metrics
        assert "spearman_rho" in metrics
        # Should be close to 1.0 (perfect correlation)
        assert metrics["kendall_tau"] > 0.9
        assert metrics["spearman_rho"] > 0.9

    def test_compute_ranking_metrics_reversed(self):
        """Test with reversed ranking."""
        from reward_robustness.metrics import compute_ranking_metrics

        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Perturbations reverse the ranking
        perturbed = np.array([
            [5.0, 5.0, 5.0],
            [4.0, 4.0, 4.0],
            [3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
        ])

        metrics = compute_ranking_metrics(original, perturbed)

        # Should be close to -1.0 (negative correlation)
        assert metrics["kendall_tau"] < -0.9
        assert metrics["spearman_rho"] < -0.9

    def test_compute_ranking_metrics_single_sample(self):
        """Test with single sample (edge case)."""
        from reward_robustness.metrics import compute_ranking_metrics

        original = np.array([1.0])
        perturbed = np.array([[1.1, 0.9, 1.0]])

        metrics = compute_ranking_metrics(original, perturbed)

        # Should handle gracefully
        assert metrics["kendall_tau"] == 1.0
        assert metrics["spearman_rho"] == 1.0

    def test_compute_ranking_metrics_constant_scores(self):
        """Test when all scores are the same."""
        from reward_robustness.metrics import compute_ranking_metrics

        original = np.array([1.0, 1.0, 1.0])
        perturbed = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ])

        metrics = compute_ranking_metrics(original, perturbed)

        # Should return 1.0 for constant scores (perfect consistency by definition)
        assert metrics["kendall_tau"] == 1.0
        assert metrics["spearman_rho"] == 1.0


class TestDeviationMetrics:
    """Tests for deviation computation."""

    def test_compute_deviation_metrics_basic(self):
        """Test basic deviation computation."""
        from reward_robustness.metrics import compute_deviation_metrics

        original = np.array([1.0, 2.0, 3.0])
        perturbed = np.array([
            [1.1, 0.9, 1.0],
            [2.2, 1.8, 2.0],
            [3.3, 2.7, 3.0],
        ])

        metrics = compute_deviation_metrics(original, perturbed)

        assert "max_deviation" in metrics
        assert "mean_deviation" in metrics
        assert "sample_deviations" in metrics

        # Max deviation should be 0.3 (from 3.0 to 3.3 or 2.7)
        assert abs(metrics["max_deviation"] - 0.3) < 0.01

    def test_compute_deviation_metrics_zero_deviation(self):
        """Test when there's no deviation."""
        from reward_robustness.metrics import compute_deviation_metrics

        original = np.array([1.0, 2.0])
        perturbed = np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ])

        metrics = compute_deviation_metrics(original, perturbed)

        assert metrics["max_deviation"] == 0.0
        assert metrics["mean_deviation"] == 0.0


class TestFlipRate:
    """Tests for flip rate computation."""

    def test_compute_flip_rate_no_flips(self):
        """Test when no scores flip across threshold."""
        from reward_robustness.metrics import compute_flip_rate

        original = np.array([1.0, 2.0, 3.0])  # All positive
        perturbed = np.array([
            [0.5, 0.8, 1.2],  # All positive
            [1.5, 1.8, 2.2],  # All positive
            [2.5, 2.8, 3.2],  # All positive
        ])

        flip_rate = compute_flip_rate(original, perturbed, threshold=0.0)

        assert flip_rate == 0.0

    def test_compute_flip_rate_all_flip(self):
        """Test when all samples flip."""
        from reward_robustness.metrics import compute_flip_rate

        original = np.array([1.0, 2.0, 3.0])  # All positive
        perturbed = np.array([
            [-0.5, 0.8, 1.2],  # First flips to negative
            [-0.5, 1.8, 2.2],  # First flips to negative
            [-0.5, 2.8, 3.2],  # First flips to negative
        ])

        flip_rate = compute_flip_rate(original, perturbed, threshold=0.0)

        assert flip_rate == 1.0

    def test_compute_flip_rate_partial(self):
        """Test with some samples flipping."""
        from reward_robustness.metrics import compute_flip_rate

        original = np.array([1.0, 2.0])
        perturbed = np.array([
            [-0.5, 0.8, 1.2],  # First perturbation flips
            [1.5, 1.8, 2.2],   # No flips
        ])

        flip_rate = compute_flip_rate(original, perturbed, threshold=0.0)

        assert flip_rate == 0.5  # 1 out of 2 samples flipped

    def test_compute_flip_rate_custom_threshold(self):
        """Test with custom threshold."""
        from reward_robustness.metrics import compute_flip_rate

        original = np.array([1.5, 0.5])  # Both above threshold=1.0? No, 0.5 is below
        perturbed = np.array([
            [1.6, 1.4, 1.5],  # All above threshold
            [0.4, 0.6, 0.5],  # All below threshold
        ])

        flip_rate = compute_flip_rate(original, perturbed, threshold=1.0)

        # First sample: original 1.5 > 1.0, all perturbed > 1.0 -> no flip
        # Second sample: original 0.5 < 1.0, all perturbed < 1.0 -> no flip
        assert flip_rate == 0.0


class TestStabilityScore:
    """Tests for overall stability score computation."""

    def test_compute_stability_score_perfect(self):
        """Test stability score for perfectly consistent rewards."""
        from reward_robustness.metrics import compute_stability_score

        variance_metrics = {
            "mean_variance": 0.0,
            "max_variance": 0.0,
        }
        ranking_metrics = {
            "kendall_tau": 1.0,
            "spearman_rho": 1.0,
        }
        deviation_metrics = {
            "mean_deviation": 0.0,
            "max_deviation": 0.0,
        }
        flip_rate = 0.0

        score = compute_stability_score(
            variance_metrics, ranking_metrics, deviation_metrics, flip_rate
        )

        assert score == 1.0

    def test_compute_stability_score_poor(self):
        """Test stability score for poor consistency."""
        from reward_robustness.metrics import compute_stability_score

        variance_metrics = {
            "mean_variance": 2.0,  # High variance
            "max_variance": 5.0,
        }
        ranking_metrics = {
            "kendall_tau": 0.0,  # No correlation
            "spearman_rho": 0.0,
        }
        deviation_metrics = {
            "mean_deviation": 2.0,  # High deviation
            "max_deviation": 5.0,
        }
        flip_rate = 0.5  # Half flip

        score = compute_stability_score(
            variance_metrics, ranking_metrics, deviation_metrics, flip_rate
        )

        assert score < 0.5

    def test_compute_stability_score_bounded(self):
        """Test that stability score is bounded between 0 and 1."""
        from reward_robustness.metrics import compute_stability_score

        # Even with extreme values, should be bounded
        variance_metrics = {"mean_variance": 100.0, "max_variance": 100.0}
        ranking_metrics = {"kendall_tau": -1.0, "spearman_rho": -1.0}
        deviation_metrics = {"mean_deviation": 100.0, "max_deviation": 100.0}

        score = compute_stability_score(
            variance_metrics, ranking_metrics, deviation_metrics, flip_rate=1.0
        )

        assert 0.0 <= score <= 1.0


class TestConsistencyMetrics:
    """Tests for the main compute_consistency_metrics function."""

    def test_compute_consistency_metrics_basic(self):
        """Test basic consistency metrics computation."""
        from reward_robustness.metrics import (
            compute_consistency_metrics,
            ConsistencyMetrics,
        )

        original = [1.0, 2.0, 3.0, 4.0, 5.0]
        perturbed = [
            [1.1, 0.9, 1.0, 1.05, 0.95],
            [2.1, 1.9, 2.0, 2.05, 1.95],
            [3.1, 2.9, 3.0, 3.05, 2.95],
            [4.1, 3.9, 4.0, 4.05, 3.95],
            [5.1, 4.9, 5.0, 5.05, 4.95],
        ]

        metrics = compute_consistency_metrics(
            reward_name="test_reward",
            original_scores=original,
            perturbed_scores=perturbed,
        )

        assert isinstance(metrics, ConsistencyMetrics)
        assert metrics.reward_name == "test_reward"
        assert metrics.num_samples == 5
        assert metrics.stability_score > 0.5  # Should be relatively stable

    def test_compute_consistency_metrics_with_details(self):
        """Test that sample details are included when requested."""
        from reward_robustness.metrics import compute_consistency_metrics

        original = [1.0, 2.0]
        perturbed = [[1.1, 0.9], [2.1, 1.9]]

        metrics = compute_consistency_metrics(
            reward_name="test",
            original_scores=original,
            perturbed_scores=perturbed,
            include_details=True,
        )

        assert metrics.sample_variances is not None
        assert metrics.sample_deviations is not None
        assert len(metrics.sample_variances) == 2

    def test_compute_consistency_metrics_without_details(self):
        """Test that sample details are excluded when not requested."""
        from reward_robustness.metrics import compute_consistency_metrics

        original = [1.0, 2.0]
        perturbed = [[1.1, 0.9], [2.1, 1.9]]

        metrics = compute_consistency_metrics(
            reward_name="test",
            original_scores=original,
            perturbed_scores=perturbed,
            include_details=False,
        )

        assert metrics.sample_variances is None
        assert metrics.sample_deviations is None


class TestCompareMetrics:
    """Tests for compare_metrics function."""

    def test_compare_metrics_ranking(self):
        """Test that metrics are properly ranked by stability."""
        from reward_robustness.metrics import (
            ConsistencyMetrics,
            compare_metrics,
        )

        metrics_a = ConsistencyMetrics(
            reward_name="reward_a",
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
            stability_score=0.9,
            num_samples=100,
        )

        metrics_b = ConsistencyMetrics(
            reward_name="reward_b",
            mean_variance=0.5,
            max_variance=1.0,
            median_variance=0.5,
            variance_std=0.2,
            mean_cv=0.5,
            kendall_tau=0.5,
            spearman_rho=0.5,
            flip_rate=0.2,
            max_deviation=1.0,
            mean_deviation=0.5,
            stability_score=0.5,
            num_samples=100,
        )

        ranking = compare_metrics([metrics_a, metrics_b])

        # reward_a should rank first (higher stability)
        assert ranking[0][0] == "reward_a"
        assert ranking[0][1] == 0.9
        assert ranking[1][0] == "reward_b"
        assert ranking[1][1] == 0.5

    def test_compare_metrics_empty_list(self):
        """Test with empty metrics list."""
        from reward_robustness.metrics import compare_metrics

        ranking = compare_metrics([])

        assert ranking == []
