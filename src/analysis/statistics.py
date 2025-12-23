"""Statistical analysis for experiment comparison."""

import json
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..experiment_tracker import LocalBackend


def compute_significance(
    experiment_id_1: str,
    experiment_id_2: str,
    db_path: str = "experiments.db",
    benchmark: str = "gsm8k",
    metric: str = "accuracy",
    method: str = "bootstrap",
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Dict:
    """Compute statistical significance of difference between two experiments.

    Args:
        experiment_id_1: First experiment ID
        experiment_id_2: Second experiment ID
        db_path: Path to database
        benchmark: Benchmark name
        metric: Metric to compare
        method: Statistical test method ('bootstrap', 't-test', 'permutation')
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary with test results
    """
    backend = LocalBackend(db_path)

    # Get per-sample results for both experiments
    results_1 = _get_per_sample_metric(backend, experiment_id_1, benchmark, metric)
    results_2 = _get_per_sample_metric(backend, experiment_id_2, benchmark, metric)

    if not results_1 or not results_2:
        return {
            "error": "Could not find per-sample results for one or both experiments"
        }

    # Convert to numpy arrays
    values_1 = np.array(results_1)
    values_2 = np.array(results_2)

    # Compute difference
    mean_1 = np.mean(values_1)
    mean_2 = np.mean(values_2)
    difference = mean_2 - mean_1

    result = {
        "experiment_1": experiment_id_1,
        "experiment_2": experiment_id_2,
        "metric": metric,
        "mean_1": float(mean_1),
        "mean_2": float(mean_2),
        "difference": float(difference),
        "method": method,
    }

    if method == "bootstrap":
        # Bootstrap test
        ci_lower, ci_upper, p_value = bootstrap_comparison(
            values_1, values_2, n_bootstrap, confidence_level
        )

        result.update({
            "confidence_level": confidence_level,
            "confidence_interval": [float(ci_lower), float(ci_upper)],
            "p_value": float(p_value),
            "is_significant": 0 not in [ci_lower, ci_upper],
        })

    elif method == "t-test":
        # Independent t-test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(values_2, values_1)

        result.update({
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "is_significant": p_value < (1 - confidence_level),
        })

    elif method == "permutation":
        # Permutation test
        p_value = permutation_test(values_1, values_2, n_permutations=n_bootstrap)

        result.update({
            "p_value": float(p_value),
            "is_significant": p_value < (1 - confidence_level),
        })

    else:
        result["error"] = f"Unknown method: {method}"

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(values_1)**2 + np.std(values_2)**2) / 2)
    if pooled_std > 0:
        cohens_d = difference / pooled_std
        result["cohens_d"] = float(cohens_d)
        result["effect_size"] = _interpret_cohens_d(cohens_d)

    return result


def bootstrap_comparison(
    values_1: np.ndarray,
    values_2: np.ndarray,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """Bootstrap comparison of two samples.

    Args:
        values_1: First sample values
        values_2: Second sample values
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level

    Returns:
        (ci_lower, ci_upper, p_value)
    """
    n1, n2 = len(values_1), len(values_2)

    # Compute observed difference
    observed_diff = np.mean(values_2) - np.mean(values_1)

    # Bootstrap
    diffs = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample_1 = np.random.choice(values_1, size=n1, replace=True)
        sample_2 = np.random.choice(values_2, size=n2, replace=True)

        # Compute difference
        diff = np.mean(sample_2) - np.mean(sample_1)
        diffs.append(diff)

    diffs = np.array(diffs)

    # Confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(diffs, alpha / 2 * 100)
    ci_upper = np.percentile(diffs, (1 - alpha / 2) * 100)

    # P-value (two-tailed)
    if observed_diff >= 0:
        p_value = 2 * np.mean(diffs <= 0)
    else:
        p_value = 2 * np.mean(diffs >= 0)

    return ci_lower, ci_upper, min(p_value, 1.0)


def permutation_test(
    values_1: np.ndarray,
    values_2: np.ndarray,
    n_permutations: int = 10000
) -> float:
    """Permutation test for difference in means.

    Args:
        values_1: First sample values
        values_2: Second sample values
        n_permutations: Number of permutations

    Returns:
        P-value
    """
    # Observed difference
    observed_diff = abs(np.mean(values_2) - np.mean(values_1))

    # Combine samples
    combined = np.concatenate([values_1, values_2])
    n1 = len(values_1)

    # Permutation test
    count = 0
    for _ in range(n_permutations):
        # Shuffle and split
        np.random.shuffle(combined)
        perm_1 = combined[:n1]
        perm_2 = combined[n1:]

        # Compute difference
        perm_diff = abs(np.mean(perm_2) - np.mean(perm_1))

        if perm_diff >= observed_diff:
            count += 1

    p_value = count / n_permutations
    return p_value


def _get_per_sample_metric(
    backend: LocalBackend,
    experiment_id: str,
    benchmark: str,
    metric: str
) -> Optional[List[float]]:
    """Get per-sample metric values.

    Args:
        backend: LocalBackend instance
        experiment_id: Experiment ID
        benchmark: Benchmark name
        metric: Metric name (e.g., 'accuracy')

    Returns:
        List of per-sample values, or None if not found
    """
    # Get evaluation results
    eval_results = backend.get_evaluation_results(experiment_id, benchmark)

    if not eval_results:
        return None

    # Get the most recent evaluation
    eval_result = eval_results[0]
    metrics_json = eval_result["metrics_json"]

    # Check if per-sample results are available
    if "per_sample_results" not in metrics_json:
        # Fall back to aggregate metric
        if metric in metrics_json:
            # Return single value repeated (not ideal but works)
            num_samples = metrics_json.get("num_samples", 1)
            return [metrics_json[metric]] * num_samples
        return None

    # Extract per-sample metric
    per_sample_results = metrics_json["per_sample_results"]

    if metric == "accuracy":
        # Binary: 1 if correct, 0 otherwise
        return [1.0 if r["is_correct"] else 0.0 for r in per_sample_results]
    elif metric == "format_accuracy":
        return [1.0 if r["format_correct"] else 0.0 for r in per_sample_results]
    elif metric == "generation_time":
        return [r["generation_time"] for r in per_sample_results]
    else:
        # Try to extract from metadata
        return [r.get(metric, 0.0) for r in per_sample_results]


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        Effect size interpretation
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def format_significance_result(result: Dict) -> str:
    """Format significance test result.

    Args:
        result: Result from compute_significance()

    Returns:
        Formatted string
    """
    lines = []

    lines.append(f"\n{'='*80}")
    lines.append("Statistical Significance Test")
    lines.append(f"{'='*80}")

    if "error" in result:
        lines.append(f"\nError: {result['error']}")
        lines.append(f"{'='*80}\n")
        return "\n".join(lines)

    lines.append(f"\nExperiment 1: {result['experiment_1']}")
    lines.append(f"Experiment 2: {result['experiment_2']}")
    lines.append(f"Metric:       {result['metric']}")
    lines.append(f"Method:       {result['method']}")

    lines.append(f"\nResults:")
    lines.append(f"  Mean 1:     {result['mean_1']:.4f}")
    lines.append(f"  Mean 2:     {result['mean_2']:.4f}")
    lines.append(f"  Difference: {result['difference']:.4f}")

    if "confidence_interval" in result:
        ci = result["confidence_interval"]
        lines.append(f"  95% CI:     [{ci[0]:.4f}, {ci[1]:.4f}]")

    lines.append(f"  P-value:    {result['p_value']:.4f}")

    if "cohens_d" in result:
        lines.append(f"  Cohen's d:  {result['cohens_d']:.3f} ({result['effect_size']})")

    lines.append(f"\nSignificant: {'Yes' if result['is_significant'] else 'No'}")

    if result['is_significant']:
        if result['difference'] > 0:
            lines.append("Experiment 2 is significantly better than Experiment 1.")
        else:
            lines.append("Experiment 1 is significantly better than Experiment 2.")
    else:
        lines.append("No significant difference detected.")

    lines.append(f"{'='*80}\n")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute statistical significance")
    parser.add_argument("exp1", help="First experiment ID")
    parser.add_argument("exp2", help="Second experiment ID")
    parser.add_argument("--db", default="experiments.db", help="Path to database")
    parser.add_argument("--benchmark", default="gsm8k", help="Benchmark name")
    parser.add_argument("--metric", default="accuracy", help="Metric to compare")
    parser.add_argument("--method", default="bootstrap", choices=["bootstrap", "t-test", "permutation"])
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Number of bootstrap samples")

    args = parser.parse_args()

    result = compute_significance(
        args.exp1, args.exp2,
        db_path=args.db,
        benchmark=args.benchmark,
        metric=args.metric,
        method=args.method,
        n_bootstrap=args.n_bootstrap
    )

    print(format_significance_result(result))
