"""
Rubric Comparator - Tools for comparing and benchmarking multiple rubric designs

This module provides statistical comparison and A/B testing capabilities for rubric designs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
import json
from pathlib import Path

from .evaluator import EvaluationResult
from .designer import BaseRubric


@dataclass
class ComparisonResult:
    """Results from comparing multiple rubrics"""

    rubric_names: List[str]
    results: List[EvaluationResult]

    # Comparative statistics
    rankings: Dict[str, int] = field(default_factory=dict)
    relative_performance: Dict[str, float] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)

    # Best rubric
    best_rubric: Optional[str] = None
    best_score: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Return comparison summary"""
        return {
            "rubric_names": self.rubric_names,
            "rankings": self.rankings,
            "best_rubric": self.best_rubric,
            "best_score": self.best_score,
            "relative_performance": self.relative_performance,
            "score_statistics": {
                name: {
                    "mean": result.mean_score,
                    "std": result.std_score,
                    "median": result.median_score,
                }
                for name, result in zip(self.rubric_names, self.results)
            },
        }

    def save(self, path: str):
        """Save comparison results to JSON"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "summary": self.summary(),
            "statistical_tests": self.statistical_tests,
            "detailed_results": [
                result.summary() for result in self.results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


class RubricComparator:
    """
    Compare multiple rubric designs statistically.

    This class provides:
    - Side-by-side comparison of rubric scores
    - Statistical significance testing
    - Ranking and relative performance metrics
    - Visualization utilities

    Example:
        comparator = RubricComparator()
        comparison = comparator.compare([result1, result2, result3])
        print(f"Best rubric: {comparison.best_rubric}")
        print(f"Rankings: {comparison.rankings}")
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level for statistical tests
        """
        self.alpha = alpha

    def compare(
        self,
        results: List[EvaluationResult],
        metric: str = "mean_score",
    ) -> ComparisonResult:
        """
        Compare multiple evaluation results.

        Args:
            results: List of EvaluationResult objects to compare
            metric: Metric to use for comparison ("mean_score", "median_score", etc.)

        Returns:
            ComparisonResult with rankings and statistical tests
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 results to compare")

        rubric_names = [r.rubric_name for r in results]

        # Extract scores for comparison
        scores_by_rubric = {
            r.rubric_name: [s.total for s in r.scores]
            for r in results
        }

        # Rank rubrics by metric
        metric_values = {r.rubric_name: getattr(r, metric) for r in results}
        rankings = self._rank_rubrics(metric_values)

        # Find best rubric
        best_rubric = max(metric_values, key=metric_values.get)
        best_score = metric_values[best_rubric]

        # Compute relative performance (normalized to best)
        relative_performance = {
            name: value / best_score if best_score > 0 else 0.0
            for name, value in metric_values.items()
        }

        # Statistical tests
        statistical_tests = {}

        # Pairwise t-tests
        pairwise_tests = self._pairwise_ttests(scores_by_rubric)
        statistical_tests["pairwise_ttests"] = pairwise_tests

        # ANOVA (if more than 2 rubrics)
        if len(results) > 2:
            anova_result = self._anova_test(scores_by_rubric)
            statistical_tests["anova"] = anova_result

        # Effect sizes (Cohen's d)
        effect_sizes = self._compute_effect_sizes(scores_by_rubric)
        statistical_tests["effect_sizes"] = effect_sizes

        comparison = ComparisonResult(
            rubric_names=rubric_names,
            results=results,
            rankings=rankings,
            relative_performance=relative_performance,
            statistical_tests=statistical_tests,
            best_rubric=best_rubric,
            best_score=best_score,
            metadata={"metric": metric, "alpha": self.alpha},
        )

        return comparison

    def _rank_rubrics(self, metric_values: Dict[str, float]) -> Dict[str, int]:
        """Rank rubrics by metric value (1 = best)"""
        sorted_items = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
        return {name: rank + 1 for rank, (name, _) in enumerate(sorted_items)}

    def _pairwise_ttests(
        self,
        scores_by_rubric: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform pairwise t-tests between all rubric pairs"""
        results = {}

        rubric_names = list(scores_by_rubric.keys())
        for i, name1 in enumerate(rubric_names):
            for name2 in rubric_names[i + 1:]:
                scores1 = scores_by_rubric[name1]
                scores2 = scores_by_rubric[name2]

                # Perform two-sample t-test
                try:
                    t_stat, p_value = stats.ttest_ind(scores1, scores2)
                    significant = p_value < self.alpha

                    pair_key = f"{name1} vs {name2}"
                    results[pair_key] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": significant,
                        "mean_diff": float(np.mean(scores1) - np.mean(scores2)),
                    }
                except Exception as e:
                    results[f"{name1} vs {name2}"] = {"error": str(e)}

        return results

    def _anova_test(
        self,
        scores_by_rubric: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Perform one-way ANOVA across all rubrics"""
        try:
            score_groups = list(scores_by_rubric.values())
            f_stat, p_value = stats.f_oneway(*score_groups)

            return {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
                "interpretation": (
                    "Significant differences exist between rubrics"
                    if p_value < self.alpha
                    else "No significant differences between rubrics"
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    def _compute_effect_sizes(
        self,
        scores_by_rubric: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute Cohen's d effect sizes between rubric pairs"""
        results = {}

        rubric_names = list(scores_by_rubric.keys())
        for i, name1 in enumerate(rubric_names):
            for name2 in rubric_names[i + 1:]:
                scores1 = np.array(scores_by_rubric[name1])
                scores2 = np.array(scores_by_rubric[name2])

                # Cohen's d
                try:
                    mean_diff = np.mean(scores1) - np.mean(scores2)
                    pooled_std = np.sqrt(
                        (np.var(scores1) + np.var(scores2)) / 2
                    )

                    if pooled_std > 0:
                        cohens_d = mean_diff / pooled_std
                    else:
                        cohens_d = 0.0

                    # Interpret effect size
                    if abs(cohens_d) < 0.2:
                        interpretation = "negligible"
                    elif abs(cohens_d) < 0.5:
                        interpretation = "small"
                    elif abs(cohens_d) < 0.8:
                        interpretation = "medium"
                    else:
                        interpretation = "large"

                    pair_key = f"{name1} vs {name2}"
                    results[pair_key] = {
                        "cohens_d": float(cohens_d),
                        "interpretation": interpretation,
                    }
                except Exception as e:
                    results[f"{name1} vs {name2}"] = {"error": str(e)}

        return results

    def find_best_rubric(
        self,
        results: List[EvaluationResult],
        criteria: str = "mean_score",
        min_samples: int = 30,
    ) -> Tuple[str, EvaluationResult]:
        """
        Find the best rubric based on criteria.

        Args:
            results: List of evaluation results
            criteria: Metric to optimize ("mean_score", "median_score", etc.)
            min_samples: Minimum samples required for valid comparison

        Returns:
            Tuple of (rubric_name, result)
        """
        valid_results = [r for r in results if r.num_samples >= min_samples]

        if not valid_results:
            raise ValueError(f"No results with at least {min_samples} samples")

        best_result = max(valid_results, key=lambda r: getattr(r, criteria))
        return best_result.rubric_name, best_result


def compare_rubrics(
    results: List[EvaluationResult],
    alpha: float = 0.05,
    output_path: Optional[str] = None,
) -> ComparisonResult:
    """
    Convenience function to compare rubric evaluation results.

    Args:
        results: List of EvaluationResult objects
        alpha: Significance level for statistical tests
        output_path: Optional path to save comparison results

    Returns:
        ComparisonResult
    """
    comparator = RubricComparator(alpha=alpha)
    comparison = comparator.compare(results)

    if output_path:
        comparison.save(output_path)
        print(f"Comparison results saved to {output_path}")

    return comparison


def benchmark_rubric(
    rubric: BaseRubric,
    benchmark_results: List[EvaluationResult],
    metric: str = "mean_score",
) -> Dict[str, Any]:
    """
    Benchmark a rubric against existing results.

    Args:
        rubric: The rubric to benchmark (with its evaluation result)
        benchmark_results: List of baseline evaluation results
        metric: Metric to compare

    Returns:
        Dictionary with benchmark statistics
    """
    # Find the result for the given rubric
    rubric_result = None
    for r in benchmark_results:
        if r.rubric_name == rubric.name:
            rubric_result = r
            break

    if rubric_result is None:
        raise ValueError(f"No result found for rubric {rubric.name}")

    # Get metric values for all rubrics
    metric_values = [getattr(r, metric) for r in benchmark_results]
    rubric_value = getattr(rubric_result, metric)

    # Compute percentile
    percentile = stats.percentileofscore(metric_values, rubric_value)

    # Compute z-score
    mean_value = np.mean(metric_values)
    std_value = np.std(metric_values)
    z_score = (rubric_value - mean_value) / std_value if std_value > 0 else 0.0

    return {
        "rubric_name": rubric.name,
        "metric": metric,
        "value": rubric_value,
        "percentile": percentile,
        "z_score": float(z_score),
        "mean_baseline": mean_value,
        "std_baseline": std_value,
        "rank": sorted(metric_values, reverse=True).index(rubric_value) + 1,
        "total_rubrics": len(benchmark_results),
    }
