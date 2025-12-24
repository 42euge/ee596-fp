"""
Consistency metrics for reward robustness evaluation.

Computes statistical metrics measuring how consistent reward scores are
across semantic-preserving perturbations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class ConsistencyMetrics:
    """Results from consistency evaluation for a single reward model."""

    reward_name: str

    # Variance metrics
    mean_variance: float  # Mean variance across samples
    max_variance: float  # Maximum variance observed
    median_variance: float  # Median variance
    variance_std: float  # Std of variances across samples

    # Coefficient of variation (normalized variance)
    mean_cv: float  # Mean coefficient of variation (std/mean)

    # Ranking metrics
    kendall_tau: float  # Kendall's tau rank correlation
    spearman_rho: float  # Spearman rank correlation

    # Stability metrics
    flip_rate: float  # % of samples changing sign/threshold
    max_deviation: float  # Maximum absolute deviation from original
    mean_deviation: float  # Mean absolute deviation from original

    # Overall stability score (0-1, higher is more stable)
    stability_score: float

    # Sample count
    num_samples: int

    # Per-sample details (optional)
    sample_variances: Optional[List[float]] = None
    sample_deviations: Optional[List[float]] = None


@dataclass
class SampleMetrics:
    """Metrics for a single sample's perturbation results."""

    sample_idx: int
    original_score: float
    perturbed_scores: List[float]
    variance: float
    std: float
    mean_perturbed: float
    max_deviation: float
    flipped: bool  # Whether sign changed


def compute_variance_metrics(
    original_scores: np.ndarray,
    perturbed_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute variance-based consistency metrics.

    Args:
        original_scores: Shape (n_samples,) - original scores
        perturbed_scores: Shape (n_samples, n_perturbations) - perturbed scores

    Returns:
        Dictionary of variance metrics
    """
    # Compute variance per sample across perturbations
    variances = np.var(perturbed_scores, axis=1)

    # Coefficient of variation per sample
    means = np.mean(perturbed_scores, axis=1)
    stds = np.std(perturbed_scores, axis=1)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        cvs = np.where(means != 0, stds / np.abs(means), 0)
        cvs = np.nan_to_num(cvs, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "mean_variance": float(np.mean(variances)),
        "max_variance": float(np.max(variances)),
        "median_variance": float(np.median(variances)),
        "variance_std": float(np.std(variances)),
        "mean_cv": float(np.mean(cvs)),
        "sample_variances": variances.tolist(),
    }


def compute_ranking_metrics(
    original_scores: np.ndarray,
    perturbed_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute rank correlation metrics.

    Tests whether the relative ranking of samples is preserved
    when comparing original scores to mean perturbed scores.

    Args:
        original_scores: Shape (n_samples,)
        perturbed_scores: Shape (n_samples, n_perturbations)

    Returns:
        Dictionary with Kendall's tau and Spearman's rho
    """
    try:
        from scipy.stats import kendalltau, spearmanr
    except ImportError:
        # Return NaN if scipy not available
        return {
            "kendall_tau": float("nan"),
            "spearman_rho": float("nan"),
        }

    # Compare original ranking to mean of perturbed scores
    mean_perturbed = np.mean(perturbed_scores, axis=1)

    # Handle edge cases
    if len(original_scores) < 2:
        return {"kendall_tau": 1.0, "spearman_rho": 1.0}

    if np.std(original_scores) == 0 or np.std(mean_perturbed) == 0:
        # All scores are the same - perfect consistency by definition
        return {"kendall_tau": 1.0, "spearman_rho": 1.0}

    tau, _ = kendalltau(original_scores, mean_perturbed)
    rho, _ = spearmanr(original_scores, mean_perturbed)

    return {
        "kendall_tau": float(np.nan_to_num(tau, nan=0.0)),
        "spearman_rho": float(np.nan_to_num(rho, nan=0.0)),
    }


def compute_deviation_metrics(
    original_scores: np.ndarray,
    perturbed_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute deviation metrics comparing original to perturbed scores.

    Args:
        original_scores: Shape (n_samples,)
        perturbed_scores: Shape (n_samples, n_perturbations)

    Returns:
        Dictionary of deviation metrics
    """
    # Absolute deviation from original for each perturbation
    # Shape: (n_samples, n_perturbations)
    deviations = np.abs(perturbed_scores - original_scores[:, np.newaxis])

    # Max deviation per sample, then overall max
    max_per_sample = np.max(deviations, axis=1)
    mean_per_sample = np.mean(deviations, axis=1)

    return {
        "max_deviation": float(np.max(max_per_sample)),
        "mean_deviation": float(np.mean(mean_per_sample)),
        "sample_deviations": mean_per_sample.tolist(),
    }


def compute_flip_rate(
    original_scores: np.ndarray,
    perturbed_scores: np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Compute the rate at which scores flip across threshold.

    A "flip" occurs when the original score and any perturbed score
    are on opposite sides of the threshold.

    Args:
        original_scores: Shape (n_samples,)
        perturbed_scores: Shape (n_samples, n_perturbations)
        threshold: The threshold to check flips against

    Returns:
        Flip rate (0.0 to 1.0)
    """
    original_above = original_scores > threshold

    # Check if any perturbation crosses the threshold
    any_flip = np.zeros(len(original_scores), dtype=bool)

    for i in range(perturbed_scores.shape[1]):
        perturbed_above = perturbed_scores[:, i] > threshold
        any_flip |= original_above != perturbed_above

    return float(np.mean(any_flip))


def compute_stability_score(
    variance_metrics: Dict[str, float],
    ranking_metrics: Dict[str, float],
    deviation_metrics: Dict[str, float],
    flip_rate: float,
    score_range: Tuple[float, float] = (-2.0, 3.0),
) -> float:
    """Compute overall stability score (0-1, higher is better).

    Combines multiple metrics into a single stability score.

    Args:
        variance_metrics: Output from compute_variance_metrics
        ranking_metrics: Output from compute_ranking_metrics
        deviation_metrics: Output from compute_deviation_metrics
        flip_rate: Output from compute_flip_rate
        score_range: Expected (min, max) range of reward scores

    Returns:
        Stability score from 0 to 1
    """
    range_size = score_range[1] - score_range[0]

    # Normalized variance component (lower is better)
    # Normalize by dividing by range^2 (variance units)
    normalized_variance = min(
        1.0, variance_metrics["mean_variance"] / (range_size**2 / 4)
    )
    variance_score = 1.0 - normalized_variance

    # Ranking component (higher is better)
    # Average of tau and rho, already in [0,1] range (roughly)
    kendall = max(0, ranking_metrics.get("kendall_tau", 0))
    spearman = max(0, ranking_metrics.get("spearman_rho", 0))
    ranking_score = (kendall + spearman) / 2

    # Deviation component (lower is better)
    normalized_deviation = min(
        1.0, deviation_metrics["mean_deviation"] / (range_size / 2)
    )
    deviation_score = 1.0 - normalized_deviation

    # Flip rate component (lower is better)
    flip_score = 1.0 - flip_rate

    # Weighted average
    weights = {
        "variance": 0.25,
        "ranking": 0.25,
        "deviation": 0.25,
        "flip": 0.25,
    }

    stability = (
        weights["variance"] * variance_score
        + weights["ranking"] * ranking_score
        + weights["deviation"] * deviation_score
        + weights["flip"] * flip_score
    )

    return float(np.clip(stability, 0.0, 1.0))


def compute_consistency_metrics(
    reward_name: str,
    original_scores: List[float],
    perturbed_scores: List[List[float]],
    threshold: float = 0.0,
    score_range: Tuple[float, float] = (-2.0, 3.0),
    include_details: bool = True,
) -> ConsistencyMetrics:
    """Compute all consistency metrics for a reward model.

    Args:
        reward_name: Name of the reward model
        original_scores: List of original scores (n_samples,)
        perturbed_scores: List of lists of perturbed scores (n_samples, n_perturbations)
        threshold: Threshold for flip rate computation
        score_range: Expected (min, max) range of reward scores
        include_details: Whether to include per-sample details

    Returns:
        ConsistencyMetrics dataclass with all computed metrics
    """
    # Convert to numpy arrays
    original = np.array(original_scores)
    perturbed = np.array(perturbed_scores)

    # Ensure 2D
    if perturbed.ndim == 1:
        perturbed = perturbed.reshape(-1, 1)

    # Compute all metric groups
    variance_metrics = compute_variance_metrics(original, perturbed)
    ranking_metrics = compute_ranking_metrics(original, perturbed)
    deviation_metrics = compute_deviation_metrics(original, perturbed)
    flip_rate = compute_flip_rate(original, perturbed, threshold)

    # Compute overall stability score
    stability = compute_stability_score(
        variance_metrics,
        ranking_metrics,
        deviation_metrics,
        flip_rate,
        score_range,
    )

    return ConsistencyMetrics(
        reward_name=reward_name,
        mean_variance=variance_metrics["mean_variance"],
        max_variance=variance_metrics["max_variance"],
        median_variance=variance_metrics["median_variance"],
        variance_std=variance_metrics["variance_std"],
        mean_cv=variance_metrics["mean_cv"],
        kendall_tau=ranking_metrics["kendall_tau"],
        spearman_rho=ranking_metrics["spearman_rho"],
        flip_rate=flip_rate,
        max_deviation=deviation_metrics["max_deviation"],
        mean_deviation=deviation_metrics["mean_deviation"],
        stability_score=stability,
        num_samples=len(original_scores),
        sample_variances=variance_metrics["sample_variances"] if include_details else None,
        sample_deviations=deviation_metrics["sample_deviations"] if include_details else None,
    )


def compare_metrics(
    metrics_list: List[ConsistencyMetrics],
) -> List[Tuple[str, float]]:
    """Compare multiple reward models by stability score.

    Args:
        metrics_list: List of ConsistencyMetrics for different rewards

    Returns:
        Sorted list of (reward_name, stability_score) tuples, best first
    """
    rankings = [(m.reward_name, m.stability_score) for m in metrics_list]
    return sorted(rankings, key=lambda x: x[1], reverse=True)
