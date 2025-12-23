"""
Reward Signal Analyzer

Provides statistical analysis and quality assessment of reward signals.
Helps researchers understand reward function behavior and identify issues.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats
from dataclasses import dataclass


@dataclass
class RewardQualityReport:
    """Report on reward signal quality."""

    # Overall quality score (0-100)
    quality_score: float

    # Individual quality indicators
    consistency_score: float  # How consistent are rewards over time
    discriminability_score: float  # How well rewards separate good/bad completions
    stability_score: float  # How stable are rewards (low variance in variance)
    coverage_score: float  # How much of reward range is being used

    # Specific issues detected
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]

    # Statistics
    statistics: Dict[str, Any]


class RewardAnalyzer:
    """
    Analyzes reward signal quality and provides insights.

    Features:
    - Quality scoring across multiple dimensions
    - Correlation analysis between reward functions
    - Reward function contribution analysis
    - Anomaly detection and diagnosis
    - Actionable recommendations
    """

    def __init__(self):
        """Initialize the analyzer."""
        pass

    def analyze_quality(
        self,
        reward_history: List[Dict[str, np.ndarray]],
        steps: List[int],
    ) -> RewardQualityReport:
        """
        Perform comprehensive quality analysis of reward signals.

        Args:
            reward_history: List of dicts mapping function names to reward arrays
            steps: Corresponding step numbers

        Returns:
            RewardQualityReport with detailed analysis
        """
        issues = []
        warnings = []
        recommendations = []
        statistics = {}

        # Aggregate all rewards
        all_function_names = list(reward_history[0].keys()) if reward_history else []

        # Compute quality scores
        consistency = self._compute_consistency(reward_history, all_function_names)
        discriminability = self._compute_discriminability(reward_history, all_function_names)
        stability = self._compute_stability(reward_history, all_function_names)
        coverage = self._compute_coverage(reward_history, all_function_names)

        statistics['consistency'] = consistency
        statistics['discriminability'] = discriminability
        statistics['stability'] = stability
        statistics['coverage'] = coverage

        # Detect issues
        if consistency < 0.3:
            issues.append("Very low reward consistency - rewards vary drastically")
            recommendations.append("Consider smoothing rewards or adjusting reward function weights")
        elif consistency < 0.5:
            warnings.append("Low reward consistency")

        if discriminability < 0.3:
            issues.append("Poor reward discriminability - cannot distinguish quality")
            recommendations.append("Increase reward function sensitivity or range")
        elif discriminability < 0.5:
            warnings.append("Moderate reward discriminability")

        if stability < 0.3:
            issues.append("Unstable reward signals - high variance in variance")
            recommendations.append("Check for numerical instabilities or outliers")
        elif stability < 0.5:
            warnings.append("Some reward instability detected")

        if coverage < 0.3:
            warnings.append("Limited reward range coverage - most rewards clustered")
            recommendations.append("Consider expanding reward scales or adding diversity")

        # Compute overall quality score (weighted average)
        weights = {
            'consistency': 0.25,
            'discriminability': 0.35,  # Most important
            'stability': 0.25,
            'coverage': 0.15,
        }

        quality_score = (
            weights['consistency'] * consistency +
            weights['discriminability'] * discriminability +
            weights['stability'] * stability +
            weights['coverage'] * coverage
        ) * 100

        # Add correlation analysis
        correlations = self._analyze_correlations(reward_history, all_function_names)
        statistics['correlations'] = correlations

        # Check for highly correlated functions
        for (func1, func2), corr in correlations.items():
            if abs(corr) > 0.9:
                warnings.append(
                    f"High correlation ({corr:.2f}) between {func1} and {func2} - "
                    "may be redundant"
                )

        return RewardQualityReport(
            quality_score=quality_score,
            consistency_score=consistency * 100,
            discriminability_score=discriminability * 100,
            stability_score=stability * 100,
            coverage_score=coverage * 100,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            statistics=statistics,
        )

    def _compute_consistency(
        self,
        reward_history: List[Dict[str, np.ndarray]],
        function_names: List[str],
    ) -> float:
        """
        Compute reward consistency over time.

        Higher score = more consistent (lower coefficient of variation in means).
        """
        if len(reward_history) < 2:
            return 1.0

        # For each function, compute mean at each step
        function_means_over_time = {name: [] for name in function_names}

        for step_rewards in reward_history:
            for name in function_names:
                if name in step_rewards:
                    rewards = np.array(step_rewards[name]).flatten()
                    function_means_over_time[name].append(np.mean(rewards))

        # Compute coefficient of variation for each function
        cvs = []
        for name, means in function_means_over_time.items():
            if len(means) > 0:
                mean_of_means = np.mean(means)
                std_of_means = np.std(means)
                if mean_of_means != 0:
                    cv = std_of_means / abs(mean_of_means)
                    cvs.append(cv)

        if not cvs:
            return 1.0

        # Lower CV = more consistent, normalize to 0-1 score
        avg_cv = np.mean(cvs)
        consistency = 1.0 / (1.0 + avg_cv)  # Sigmoid-like mapping

        return float(consistency)

    def _compute_discriminability(
        self,
        reward_history: List[Dict[str, np.ndarray]],
        function_names: List[str],
    ) -> float:
        """
        Compute reward discriminability.

        Higher score = better separation between different completions.
        Uses coefficient of variation within batches.
        """
        if not reward_history:
            return 0.0

        # Aggregate all rewards across steps
        all_rewards_per_function = {name: [] for name in function_names}

        for step_rewards in reward_history:
            for name in function_names:
                if name in step_rewards:
                    rewards = np.array(step_rewards[name]).flatten()
                    all_rewards_per_function[name].extend(rewards)

        # Compute CV for each function
        cvs = []
        for name, all_rewards in all_rewards_per_function.items():
            if len(all_rewards) > 0:
                rewards_array = np.array(all_rewards)
                mean = np.mean(rewards_array)
                std = np.std(rewards_array)

                if mean != 0:
                    cv = std / abs(mean)
                    cvs.append(cv)
                elif std > 0:
                    # Zero mean but non-zero std indicates good separation
                    cvs.append(1.0)

        if not cvs:
            return 0.0

        # Higher CV = better discriminability, normalize to 0-1
        avg_cv = np.mean(cvs)
        discriminability = min(1.0, avg_cv / 2.0)  # Cap at 1.0

        return float(discriminability)

    def _compute_stability(
        self,
        reward_history: List[Dict[str, np.ndarray]],
        function_names: List[str],
    ) -> float:
        """
        Compute reward stability.

        Higher score = more stable (consistent variance over time).
        """
        if len(reward_history) < 3:
            return 1.0

        # For each function, compute variance at each step
        function_vars_over_time = {name: [] for name in function_names}

        for step_rewards in reward_history:
            for name in function_names:
                if name in step_rewards:
                    rewards = np.array(step_rewards[name]).flatten()
                    if len(rewards) > 1:
                        function_vars_over_time[name].append(np.var(rewards))

        # Compute stability of variance (low variance in variance = stable)
        stability_scores = []
        for name, variances in function_vars_over_time.items():
            if len(variances) > 1:
                variance_of_variance = np.var(variances)
                mean_variance = np.mean(variances)

                if mean_variance > 0:
                    # Normalized variance of variance
                    cv_of_var = np.sqrt(variance_of_variance) / mean_variance
                    stability = 1.0 / (1.0 + cv_of_var)
                    stability_scores.append(stability)

        if not stability_scores:
            return 1.0

        return float(np.mean(stability_scores))

    def _compute_coverage(
        self,
        reward_history: List[Dict[str, np.ndarray]],
        function_names: List[str],
    ) -> float:
        """
        Compute reward range coverage.

        Higher score = using more of the available reward range.
        """
        if not reward_history:
            return 0.0

        # Aggregate all rewards
        all_rewards_per_function = {name: [] for name in function_names}

        for step_rewards in reward_history:
            for name in function_names:
                if name in step_rewards:
                    rewards = np.array(step_rewards[name]).flatten()
                    all_rewards_per_function[name].extend(rewards)

        # For each function, compute coverage
        coverage_scores = []
        for name, all_rewards in all_rewards_per_function.items():
            if len(all_rewards) > 1:
                rewards_array = np.array(all_rewards)

                # Compute range coverage using percentile spread
                min_val = np.min(rewards_array)
                max_val = np.max(rewards_array)
                range_span = max_val - min_val

                # Compute how spread out values are using IQR
                q25, q75 = np.percentile(rewards_array, [25, 75])
                iqr = q75 - q25

                if range_span > 0:
                    # Coverage = IQR / range (how much of range is actively used)
                    coverage = min(1.0, iqr / range_span)
                    coverage_scores.append(coverage)

        if not coverage_scores:
            return 0.0

        return float(np.mean(coverage_scores))

    def _analyze_correlations(
        self,
        reward_history: List[Dict[str, np.ndarray]],
        function_names: List[str],
    ) -> Dict[Tuple[str, str], float]:
        """
        Analyze correlations between reward functions.

        Returns:
            Dict mapping (func1, func2) pairs to correlation coefficients
        """
        if len(function_names) < 2 or not reward_history:
            return {}

        # Aggregate rewards for each function
        all_rewards_per_function = {name: [] for name in function_names}

        for step_rewards in reward_history:
            for name in function_names:
                if name in step_rewards:
                    rewards = np.array(step_rewards[name]).flatten()
                    all_rewards_per_function[name].extend(rewards)

        # Compute pairwise correlations
        correlations = {}

        for i, func1 in enumerate(function_names):
            for func2 in function_names[i+1:]:
                rewards1 = np.array(all_rewards_per_function[func1])
                rewards2 = np.array(all_rewards_per_function[func2])

                # Ensure same length
                min_len = min(len(rewards1), len(rewards2))
                if min_len > 1:
                    rewards1 = rewards1[:min_len]
                    rewards2 = rewards2[:min_len]

                    # Compute Pearson correlation
                    if np.std(rewards1) > 0 and np.std(rewards2) > 0:
                        corr, _ = stats.pearsonr(rewards1, rewards2)
                        correlations[(func1, func2)] = float(corr)

        return correlations

    def analyze_function_importance(
        self,
        reward_history: List[Dict[str, np.ndarray]],
        function_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze the importance/contribution of each reward function.

        Returns:
            Dict mapping function names to importance metrics
        """
        if not reward_history:
            return {}

        importance = {}

        # Aggregate rewards
        all_rewards_per_function = {name: [] for name in function_names}

        for step_rewards in reward_history:
            for name in function_names:
                if name in step_rewards:
                    rewards = np.array(step_rewards[name]).flatten()
                    all_rewards_per_function[name].extend(rewards)

        # Compute total rewards at each step
        total_rewards = []
        for step_rewards in reward_history:
            step_total = sum(
                np.sum(step_rewards.get(name, [0]))
                for name in function_names
            )
            total_rewards.append(step_total)

        total_sum = sum(total_rewards) if total_rewards else 1.0

        for name in function_names:
            rewards = np.array(all_rewards_per_function[name])

            if len(rewards) > 0:
                # Contribution = sum of rewards / total sum
                contribution = np.sum(rewards) / total_sum if total_sum != 0 else 0.0

                # Variability = std / mean (normalized)
                mean_val = np.mean(rewards)
                std_val = np.std(rewards)
                variability = std_val / abs(mean_val) if mean_val != 0 else 0.0

                # Impact = contribution * variability
                # High impact = contributes a lot AND varies (provides signal)
                impact = contribution * variability

                importance[name] = {
                    'contribution': float(contribution * 100),  # Percentage
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'variability': float(variability),
                    'impact_score': float(impact * 100),
                }

        return importance

    def detect_reward_hacking(
        self,
        reward_history: List[Dict[str, np.ndarray]],
        function_names: List[str],
        performance_metrics: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Detect potential reward hacking (rewards increasing but performance not).

        Args:
            reward_history: Reward signals over time
            function_names: Names of reward functions
            performance_metrics: Optional actual performance (accuracy, etc.) over time

        Returns:
            Dict with reward hacking analysis
        """
        if len(reward_history) < 5:
            return {'status': 'insufficient_data'}

        # Compute total reward trend
        total_rewards_over_time = []
        for step_rewards in reward_history:
            step_total = sum(
                np.mean(step_rewards.get(name, [0]))
                for name in function_names
            )
            total_rewards_over_time.append(step_total)

        # Compute reward trend (linear regression slope)
        steps = np.arange(len(total_rewards_over_time))
        reward_slope, _ = np.polyfit(steps, total_rewards_over_time, 1)

        result = {
            'reward_trend': 'increasing' if reward_slope > 0 else 'decreasing',
            'reward_slope': float(reward_slope),
        }

        # If performance metrics provided, check alignment
        if performance_metrics and len(performance_metrics) == len(total_rewards_over_time):
            perf_slope, _ = np.polyfit(steps, performance_metrics, 1)

            result['performance_trend'] = 'increasing' if perf_slope > 0 else 'decreasing'
            result['performance_slope'] = float(perf_slope)

            # Check for misalignment
            if reward_slope > 0 and perf_slope < 0:
                result['warning'] = 'Potential reward hacking: rewards increasing but performance decreasing'
                result['severity'] = 'high'
            elif reward_slope > 0 and abs(perf_slope) < 0.01:
                result['warning'] = 'Possible reward hacking: rewards increasing but performance flat'
                result['severity'] = 'medium'
            else:
                result['status'] = 'aligned'

        return result
