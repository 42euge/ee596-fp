"""
Automated Quality Assessment for Reward Functions

This module provides comprehensive tools for detecting reward hacks, pathologies,
and quality issues in reinforcement learning from human feedback (RLHF) systems.

Key Features:
- Reward hacking detection (format gaming, content degradation)
- Statistical anomaly detection (saturation, variance collapse)
- Response quality metrics (diversity, coherence, alignment)
- Real-time monitoring and alerting
- Integration with Weights & Biases
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import warnings


@dataclass
class RewardStats:
    """Statistics for a reward component over a window of samples."""
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0
    samples: int = 0

    def to_dict(self) -> Dict[str, float]:
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'median': self.median,
            'samples': self.samples
        }


@dataclass
class PathologyAlert:
    """Alert for detected reward pathology."""
    pathology_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metrics: Dict[str, Any]
    timestamp: Optional[int] = None

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.pathology_type}: {self.message}"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a batch of responses."""
    # Format quality
    format_compliance_rate: float = 0.0
    tag_correctness_rate: float = 0.0

    # Content quality
    avg_reasoning_length: float = 0.0
    avg_answer_length: float = 0.0
    repetition_rate: float = 0.0
    empty_content_rate: float = 0.0

    # Diversity metrics
    unique_responses_ratio: float = 0.0
    lexical_diversity: float = 0.0  # Type-token ratio

    # Reward statistics
    reward_stats: Dict[str, RewardStats] = field(default_factory=dict)

    # Pathology indicators
    suspected_format_gaming: float = 0.0  # Rate of correct format with low-quality content
    suspected_reward_hacking: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'format_compliance_rate': self.format_compliance_rate,
            'tag_correctness_rate': self.tag_correctness_rate,
            'avg_reasoning_length': self.avg_reasoning_length,
            'avg_answer_length': self.avg_answer_length,
            'repetition_rate': self.repetition_rate,
            'empty_content_rate': self.empty_content_rate,
            'unique_responses_ratio': self.unique_responses_ratio,
            'lexical_diversity': self.lexical_diversity,
            'suspected_format_gaming': self.suspected_format_gaming,
            'suspected_reward_hacking': self.suspected_reward_hacking,
        }
        # Add reward stats
        for name, stats in self.reward_stats.items():
            for key, value in stats.to_dict().items():
                result[f'reward_{name}_{key}'] = value
        return result


class RewardQualityAssessor:
    """
    Main class for automated reward quality assessment.

    Monitors reward distributions, detects pathologies, and provides
    alerts for potential issues in the training process.
    """

    def __init__(
        self,
        window_size: int = 1000,
        min_samples_for_analysis: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the quality assessor.

        Args:
            window_size: Number of recent samples to keep for analysis
            min_samples_for_analysis: Minimum samples before running statistical tests
            alert_thresholds: Custom thresholds for pathology detection
        """
        self.window_size = window_size
        self.min_samples_for_analysis = min_samples_for_analysis

        # Default alert thresholds
        self.thresholds = {
            'reward_saturation_high': 0.9,  # % samples at max reward
            'reward_saturation_low': 0.9,   # % samples at min reward
            'variance_collapse_threshold': 0.01,  # std/mean ratio
            'format_gaming_threshold': 0.3,  # Rate of good format + bad content
            'repetition_threshold': 0.5,  # Fraction of repeated tokens
            'empty_content_threshold': 0.2,  # Rate of empty reasoning/answer
            'diversity_collapse_threshold': 0.1,  # Unique responses ratio
            'reward_spike_std': 3.0,  # Std deviations for spike detection
        }
        if alert_thresholds:
            self.thresholds.update(alert_thresholds)

        # Tracking data structures
        self.reward_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.response_history: deque = deque(maxlen=window_size)
        self.alerts: List[PathologyAlert] = []
        self.global_step = 0

    def assess_batch(
        self,
        responses: List[str],
        rewards: Dict[str, List[float]],
        references: Optional[List[str]] = None
    ) -> Tuple[QualityMetrics, List[PathologyAlert]]:
        """
        Assess quality of a batch of responses and their rewards.

        Args:
            responses: List of model-generated responses
            rewards: Dictionary mapping reward component names to lists of rewards
            references: Optional list of reference/ground-truth responses

        Returns:
            Tuple of (QualityMetrics, List of PathologyAlerts)
        """
        batch_size = len(responses)
        alerts = []

        # Update history
        for response in responses:
            self.response_history.append(response)
        for reward_name, reward_values in rewards.items():
            self.reward_history[reward_name].extend(reward_values)

        self.global_step += batch_size

        # Compute quality metrics
        metrics = self._compute_quality_metrics(responses, rewards)

        # Run pathology detectors
        if len(self.response_history) >= self.min_samples_for_analysis:
            alerts.extend(self._detect_reward_saturation(rewards))
            alerts.extend(self._detect_variance_collapse(rewards))
            alerts.extend(self._detect_format_gaming(responses, rewards))
            alerts.extend(self._detect_repetition_pathology(responses))
            alerts.extend(self._detect_diversity_collapse(responses))
            alerts.extend(self._detect_reward_spikes(rewards))
            alerts.extend(self._detect_degenerate_outputs(responses))

        # Store alerts
        for alert in alerts:
            alert.timestamp = self.global_step
        self.alerts.extend(alerts)

        return metrics, alerts

    def _compute_quality_metrics(
        self,
        responses: List[str],
        rewards: Dict[str, List[float]]
    ) -> QualityMetrics:
        """Compute comprehensive quality metrics for responses."""
        metrics = QualityMetrics()

        if not responses:
            return metrics

        # Extract content from responses
        reasoning_lengths = []
        answer_lengths = []
        has_correct_format = 0
        has_correct_tags = 0

        for response in responses:
            # Check format compliance
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

            if reasoning_match and answer_match:
                has_correct_format += 1
                reasoning_content = reasoning_match.group(1).strip()
                answer_content = answer_match.group(1).strip()
                reasoning_lengths.append(len(reasoning_content))
                answer_lengths.append(len(answer_content))
            else:
                reasoning_lengths.append(0)
                answer_lengths.append(0)

            # Check tag presence (even if malformed)
            if '<reasoning>' in response and '</reasoning>' in response:
                has_correct_tags += 1

        metrics.format_compliance_rate = has_correct_format / len(responses)
        metrics.tag_correctness_rate = has_correct_tags / len(responses)
        metrics.avg_reasoning_length = np.mean(reasoning_lengths) if reasoning_lengths else 0
        metrics.avg_answer_length = np.mean(answer_lengths) if answer_lengths else 0

        # Repetition analysis
        repetition_scores = [self._compute_repetition_score(r) for r in responses]
        metrics.repetition_rate = np.mean(repetition_scores)

        # Empty content detection
        empty_count = sum(1 for r in responses if self._is_empty_response(r))
        metrics.empty_content_rate = empty_count / len(responses)

        # Diversity metrics
        unique_responses = len(set(responses))
        metrics.unique_responses_ratio = unique_responses / len(responses)
        metrics.lexical_diversity = self._compute_lexical_diversity(responses)

        # Compute reward statistics (use full history for better estimates)
        for reward_name, reward_values in rewards.items():
            historical_values = list(self.reward_history[reward_name])
            if historical_values:
                stats = RewardStats(
                    mean=np.mean(historical_values),
                    std=np.std(historical_values),
                    min=np.min(historical_values),
                    max=np.max(historical_values),
                    median=np.median(historical_values),
                    samples=len(historical_values)
                )
                metrics.reward_stats[reward_name] = stats

        # Detect format gaming: good format but suspiciously short/empty content
        format_gaming_count = 0
        for i, response in enumerate(responses):
            if has_correct_format and (reasoning_lengths[i] < 20 or answer_lengths[i] < 2):
                format_gaming_count += 1
        metrics.suspected_format_gaming = format_gaming_count / len(responses)

        return metrics

    def _detect_reward_saturation(
        self,
        rewards: Dict[str, List[float]]
    ) -> List[PathologyAlert]:
        """Detect if rewards are saturating at max or min values."""
        alerts = []

        for reward_name, reward_values in rewards.items():
            historical = list(self.reward_history[reward_name])
            if len(historical) < self.min_samples_for_analysis:
                continue

            arr = np.array(historical)
            max_val = arr.max()
            min_val = arr.min()

            if max_val == min_val:
                continue  # Will be caught by variance collapse

            # Check saturation at maximum
            at_max = np.sum(arr == max_val) / len(arr)
            if at_max > self.thresholds['reward_saturation_high']:
                alerts.append(PathologyAlert(
                    pathology_type='reward_saturation_high',
                    severity='high' if at_max > 0.95 else 'medium',
                    message=f"Reward '{reward_name}' saturating at maximum: {at_max*100:.1f}% of samples at max value {max_val}",
                    metrics={'reward_name': reward_name, 'saturation_rate': at_max, 'max_value': max_val}
                ))

            # Check saturation at minimum
            at_min = np.sum(arr == min_val) / len(arr)
            if at_min > self.thresholds['reward_saturation_low']:
                alerts.append(PathologyAlert(
                    pathology_type='reward_saturation_low',
                    severity='high' if at_min > 0.95 else 'medium',
                    message=f"Reward '{reward_name}' saturating at minimum: {at_min*100:.1f}% of samples at min value {min_val}",
                    metrics={'reward_name': reward_name, 'saturation_rate': at_min, 'min_value': min_val}
                ))

        return alerts

    def _detect_variance_collapse(
        self,
        rewards: Dict[str, List[float]]
    ) -> List[PathologyAlert]:
        """Detect if reward variance has collapsed (all rewards similar)."""
        alerts = []

        for reward_name, reward_values in rewards.items():
            historical = list(self.reward_history[reward_name])
            if len(historical) < self.min_samples_for_analysis:
                continue

            arr = np.array(historical)
            mean = arr.mean()
            std = arr.std()

            if mean == 0:
                continue

            cv = std / abs(mean)  # Coefficient of variation

            if cv < self.thresholds['variance_collapse_threshold']:
                alerts.append(PathologyAlert(
                    pathology_type='variance_collapse',
                    severity='critical',
                    message=f"Reward '{reward_name}' variance collapsed: CV={cv:.4f}, std={std:.4f}, mean={mean:.4f}",
                    metrics={'reward_name': reward_name, 'cv': cv, 'std': std, 'mean': mean}
                ))

        return alerts

    def _detect_format_gaming(
        self,
        responses: List[str],
        rewards: Dict[str, List[float]]
    ) -> List[PathologyAlert]:
        """
        Detect format gaming: achieving high format rewards with low-quality content.

        This is a sophisticated reward hack where the model learns to produce
        correctly formatted outputs without meaningful reasoning or answers.
        """
        alerts = []

        gaming_count = 0
        total_with_format = 0

        for response in responses:
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

            if reasoning_match and answer_match:
                total_with_format += 1
                reasoning = reasoning_match.group(1).strip()
                answer = answer_match.group(1).strip()

                # Check for suspiciously low-quality content despite correct format
                is_gaming = (
                    len(reasoning) < 20 or  # Too short reasoning
                    len(answer) < 2 or  # Too short answer
                    self._compute_repetition_score(reasoning) > 0.5 or  # Repetitive
                    reasoning.lower() in ['n/a', 'none', 'idk', '...', 'thinking'] or
                    answer.lower() in ['n/a', 'none', 'idk', '...']
                )

                if is_gaming:
                    gaming_count += 1

        if total_with_format > 0:
            gaming_rate = gaming_count / total_with_format

            if gaming_rate > self.thresholds['format_gaming_threshold']:
                alerts.append(PathologyAlert(
                    pathology_type='format_gaming',
                    severity='critical',
                    message=f"Suspected format gaming: {gaming_rate*100:.1f}% of formatted responses have low-quality content",
                    metrics={'gaming_rate': gaming_rate, 'gaming_count': gaming_count, 'total_formatted': total_with_format}
                ))

        return alerts

    def _detect_repetition_pathology(
        self,
        responses: List[str]
    ) -> List[PathologyAlert]:
        """Detect excessive repetition in responses."""
        alerts = []

        repetition_scores = [self._compute_repetition_score(r) for r in responses]
        avg_repetition = np.mean(repetition_scores)

        if avg_repetition > self.thresholds['repetition_threshold']:
            high_rep_count = sum(1 for s in repetition_scores if s > 0.7)
            alerts.append(PathologyAlert(
                pathology_type='excessive_repetition',
                severity='high',
                message=f"Excessive repetition detected: avg={avg_repetition*100:.1f}%, {high_rep_count}/{len(responses)} responses highly repetitive",
                metrics={'avg_repetition': avg_repetition, 'high_repetition_count': high_rep_count}
            ))

        return alerts

    def _detect_diversity_collapse(
        self,
        responses: List[str]
    ) -> List[PathologyAlert]:
        """Detect if response diversity has collapsed (mode collapse)."""
        alerts = []

        # Use recent history for diversity check
        recent_responses = list(self.response_history)[-min(500, len(self.response_history)):]

        if len(recent_responses) < self.min_samples_for_analysis:
            return alerts

        unique_ratio = len(set(recent_responses)) / len(recent_responses)

        if unique_ratio < self.thresholds['diversity_collapse_threshold']:
            alerts.append(PathologyAlert(
                pathology_type='diversity_collapse',
                severity='critical',
                message=f"Response diversity collapsed: only {unique_ratio*100:.1f}% unique responses in recent {len(recent_responses)} samples",
                metrics={'unique_ratio': unique_ratio, 'total_responses': len(recent_responses), 'unique_responses': len(set(recent_responses))}
            ))

        return alerts

    def _detect_reward_spikes(
        self,
        rewards: Dict[str, List[float]]
    ) -> List[PathologyAlert]:
        """Detect sudden spikes or drops in reward values."""
        alerts = []

        for reward_name, reward_values in rewards.items():
            historical = list(self.reward_history[reward_name])
            if len(historical) < self.min_samples_for_analysis:
                continue

            # Compute rolling statistics
            arr = np.array(historical)
            mean = arr.mean()
            std = arr.std()

            if std == 0:
                continue

            # Check recent batch for spikes
            recent = reward_values
            for value in recent:
                z_score = abs(value - mean) / std
                if z_score > self.thresholds['reward_spike_std']:
                    alerts.append(PathologyAlert(
                        pathology_type='reward_spike',
                        severity='medium',
                        message=f"Unusual reward spike in '{reward_name}': value={value:.3f}, z-score={z_score:.2f}",
                        metrics={'reward_name': reward_name, 'value': value, 'z_score': z_score, 'mean': mean, 'std': std}
                    ))

        return alerts

    def _detect_degenerate_outputs(
        self,
        responses: List[str]
    ) -> List[PathologyAlert]:
        """Detect degenerate outputs (empty, truncated, malformed)."""
        alerts = []

        empty_count = sum(1 for r in responses if self._is_empty_response(r))
        empty_rate = empty_count / len(responses)

        if empty_rate > self.thresholds['empty_content_threshold']:
            alerts.append(PathologyAlert(
                pathology_type='degenerate_outputs',
                severity='high',
                message=f"High rate of degenerate outputs: {empty_rate*100:.1f}% responses are empty or malformed",
                metrics={'empty_rate': empty_rate, 'empty_count': empty_count, 'total': len(responses)}
            ))

        return alerts

    @staticmethod
    def _compute_repetition_score(text: str) -> float:
        """
        Compute repetition score for text (0=no repetition, 1=fully repetitive).

        Uses n-gram repetition detection.
        """
        if not text or len(text) < 10:
            return 0.0

        # Tokenize (simple whitespace)
        tokens = text.split()
        if len(tokens) < 5:
            return 0.0

        # Check 3-gram repetition
        ngram_size = 3
        ngrams = []
        for i in range(len(tokens) - ngram_size + 1):
            ngram = tuple(tokens[i:i+ngram_size])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        # Repetition score: 1 - (unique/total)
        repetition = 1.0 - (unique_ngrams / total_ngrams)
        return repetition

    @staticmethod
    def _is_empty_response(response: str) -> bool:
        """Check if response is empty or degenerate."""
        # Extract content from tags
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)

        if not reasoning_match or not answer_match:
            return True

        reasoning = reasoning_match.group(1).strip()
        answer = answer_match.group(1).strip()

        # Check if content is trivial
        return len(reasoning) < 5 or len(answer) < 1

    @staticmethod
    def _compute_lexical_diversity(responses: List[str]) -> float:
        """
        Compute lexical diversity (type-token ratio) across responses.

        Higher values indicate more diverse vocabulary.
        """
        all_tokens = []
        for response in responses:
            tokens = response.lower().split()
            all_tokens.extend(tokens)

        if not all_tokens:
            return 0.0

        unique_tokens = len(set(all_tokens))
        total_tokens = len(all_tokens)

        return unique_tokens / total_tokens

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all tracked metrics."""
        summary = {
            'total_samples': self.global_step,
            'window_size': len(self.response_history),
            'total_alerts': len(self.alerts),
            'alerts_by_severity': defaultdict(int),
            'alerts_by_type': defaultdict(int),
        }

        # Aggregate alerts
        for alert in self.alerts:
            summary['alerts_by_severity'][alert.severity] += 1
            summary['alerts_by_type'][alert.pathology_type] += 1

        # Reward statistics
        summary['reward_statistics'] = {}
        for reward_name, values in self.reward_history.items():
            if values:
                arr = np.array(list(values))
                summary['reward_statistics'][reward_name] = {
                    'mean': float(arr.mean()),
                    'std': float(arr.std()),
                    'min': float(arr.min()),
                    'max': float(arr.max()),
                    'samples': len(values)
                }

        return summary

    def reset_alerts(self):
        """Clear all stored alerts."""
        self.alerts = []

    def get_recent_alerts(self, n: int = 10) -> List[PathologyAlert]:
        """Get the n most recent alerts."""
        return self.alerts[-n:] if self.alerts else []
