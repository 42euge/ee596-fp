"""
Reward Hack Detection and Behavior Monitoring System

This module provides automated detection of reward hacking and problematic
behaviors during reinforcement learning training.

Key Detection Mechanisms:
1. Statistical anomaly detection in reward distributions
2. Behavioral pattern analysis (mode collapse, repetition, format gaming)
3. Training dynamics monitoring (KL divergence, gradients)
4. Multi-component reward exploitation detection
"""

import numpy as np
from collections import deque, Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration for reward hack detection thresholds."""

    # Statistical anomaly detection
    reward_zscore_threshold: float = 3.0  # Z-score threshold for reward anomalies
    reward_variance_threshold: float = 5.0  # Max variance change factor
    min_samples_for_detection: int = 10  # Minimum samples before detection

    # Response length monitoring
    min_response_length: int = 10  # Minimum acceptable response length
    max_response_length: int = 2048  # Maximum acceptable response length
    length_outlier_threshold: float = 3.0  # Z-score for length outliers

    # Repetition detection
    max_ngram_repetition_ratio: float = 0.3  # Max ratio of repeated n-grams
    ngram_size: int = 3  # Size of n-grams to check
    max_token_repetition_ratio: float = 0.5  # Max ratio of repeated tokens

    # Format gaming detection
    min_reasoning_length: int = 20  # Min chars in reasoning section
    format_quality_ratio_threshold: float = 0.3  # Min ratio of quality to format reward

    # Diversity monitoring
    min_unique_responses_ratio: float = 0.5  # Min ratio of unique responses
    similarity_threshold: float = 0.9  # Threshold for considering responses similar

    # KL divergence monitoring
    kl_divergence_min: float = 0.001  # Too little exploration
    kl_divergence_max: float = 5.0  # Too much divergence

    # Gradient monitoring
    gradient_norm_max: float = 10.0  # Max acceptable gradient norm
    gradient_norm_min: float = 1e-6  # Min gradient norm (vanishing)

    # Reward component balance
    max_component_imbalance: float = 0.9  # Max fraction from single component

    # Moving average window sizes
    short_window_size: int = 20  # Short-term statistics
    long_window_size: int = 100  # Long-term statistics


@dataclass
class DetectionResult:
    """Result from a single detection check."""
    is_anomalous: bool
    severity: str  # 'low', 'medium', 'high', 'critical'
    detection_type: str
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)


class RewardStatistics:
    """Track and analyze reward statistics over time."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.short_window = deque(maxlen=20)

    def update(self, reward: float):
        """Add a new reward value."""
        self.rewards.append(reward)
        self.short_window.append(reward)

    def get_mean(self) -> float:
        """Get mean of all rewards."""
        return np.mean(self.rewards) if self.rewards else 0.0

    def get_std(self) -> float:
        """Get standard deviation of all rewards."""
        return np.std(self.rewards) if len(self.rewards) > 1 else 0.0

    def get_recent_mean(self) -> float:
        """Get mean of recent rewards."""
        return np.mean(self.short_window) if self.short_window else 0.0

    def get_recent_std(self) -> float:
        """Get std of recent rewards."""
        return np.std(self.short_window) if len(self.short_window) > 1 else 0.0

    def detect_anomaly(self, new_reward: float, threshold: float = 3.0) -> bool:
        """Detect if new reward is anomalous using z-score."""
        if len(self.rewards) < 10:
            return False
        mean = self.get_mean()
        std = self.get_std()
        if std < 1e-6:  # Avoid division by zero
            return False
        z_score = abs((new_reward - mean) / std)
        return z_score > threshold

    def detect_distribution_shift(self) -> Tuple[bool, float]:
        """Detect significant shift in reward distribution."""
        if len(self.rewards) < self.window_size:
            return False, 0.0

        # Compare recent vs historical statistics
        recent_mean = self.get_recent_mean()
        historical_mean = self.get_mean()
        historical_std = self.get_std()

        if historical_std < 1e-6:
            return False, 0.0

        shift_score = abs((recent_mean - historical_mean) / historical_std)
        return shift_score > 2.0, shift_score


class ResponseAnalyzer:
    """Analyze generated responses for problematic patterns."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.response_hashes = deque(maxlen=100)

    def analyze_length(self, response: str) -> Optional[DetectionResult]:
        """Check for length anomalies."""
        length = len(response)

        if length < self.config.min_response_length:
            return DetectionResult(
                is_anomalous=True,
                severity='high',
                detection_type='response_length',
                message=f'Response too short: {length} chars (min: {self.config.min_response_length})',
                metrics={'length': length, 'min_expected': self.config.min_response_length}
            )

        if length > self.config.max_response_length:
            return DetectionResult(
                is_anomalous=True,
                severity='medium',
                detection_type='response_length',
                message=f'Response too long: {length} chars (max: {self.config.max_response_length})',
                metrics={'length': length, 'max_expected': self.config.max_response_length}
            )

        return None

    def analyze_repetition(self, response: str) -> Optional[DetectionResult]:
        """Detect excessive token or n-gram repetition."""
        tokens = response.split()

        if len(tokens) < 10:
            return None

        # Check token repetition
        token_counts = Counter(tokens)
        max_token_count = max(token_counts.values())
        token_repetition_ratio = max_token_count / len(tokens)

        if token_repetition_ratio > self.config.max_token_repetition_ratio:
            most_common = token_counts.most_common(1)[0]
            return DetectionResult(
                is_anomalous=True,
                severity='high',
                detection_type='token_repetition',
                message=f'Excessive token repetition: "{most_common[0]}" appears {most_common[1]} times ({token_repetition_ratio:.1%})',
                metrics={'repetition_ratio': token_repetition_ratio, 'most_common_token': most_common[0], 'count': most_common[1]}
            )

        # Check n-gram repetition
        ngrams = [tuple(tokens[i:i+self.config.ngram_size])
                  for i in range(len(tokens) - self.config.ngram_size + 1)]

        if len(ngrams) > 0:
            ngram_counts = Counter(ngrams)
            max_ngram_count = max(ngram_counts.values())
            ngram_repetition_ratio = max_ngram_count / len(ngrams)

            if ngram_repetition_ratio > self.config.max_ngram_repetition_ratio:
                most_common_ngram = ngram_counts.most_common(1)[0]
                return DetectionResult(
                    is_anomalous=True,
                    severity='medium',
                    detection_type='ngram_repetition',
                    message=f'{self.config.ngram_size}-gram repetition detected: {ngram_repetition_ratio:.1%} ratio',
                    metrics={'ngram_repetition_ratio': ngram_repetition_ratio, 'ngram_size': self.config.ngram_size}
                )

        return None

    def analyze_format_gaming(self, response: str, format_reward: float,
                            quality_reward: float) -> Optional[DetectionResult]:
        """Detect if model is gaming format rewards without quality."""
        # Extract reasoning section
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        reasoning_length = len(reasoning_match.group(1).strip()) if reasoning_match else 0

        # Check if getting format reward but with minimal reasoning
        if format_reward > 0 and reasoning_length < self.config.min_reasoning_length:
            return DetectionResult(
                is_anomalous=True,
                severity='high',
                detection_type='format_gaming',
                message=f'Format gaming detected: high format reward ({format_reward}) but minimal reasoning ({reasoning_length} chars)',
                metrics={'format_reward': format_reward, 'reasoning_length': reasoning_length, 'quality_reward': quality_reward}
            )

        # Check reward component balance
        total_reward = format_reward + quality_reward
        if total_reward > 0:
            format_ratio = format_reward / total_reward
            if format_ratio > self.config.format_quality_ratio_threshold and quality_reward == 0:
                return DetectionResult(
                    is_anomalous=True,
                    severity='medium',
                    detection_type='reward_imbalance',
                    message=f'Reward imbalance: {format_ratio:.1%} from format, 0% from quality',
                    metrics={'format_ratio': format_ratio, 'format_reward': format_reward, 'quality_reward': quality_reward}
                )

        return None

    def analyze_diversity(self, response: str) -> Optional[DetectionResult]:
        """Track response diversity to detect mode collapse."""
        # Simple hash-based diversity check
        response_hash = hash(response.strip())
        self.response_hashes.append(response_hash)

        if len(self.response_hashes) < 20:
            return None

        unique_ratio = len(set(self.response_hashes)) / len(self.response_hashes)

        if unique_ratio < self.config.min_unique_responses_ratio:
            return DetectionResult(
                is_anomalous=True,
                severity='critical',
                detection_type='mode_collapse',
                message=f'Potential mode collapse: only {unique_ratio:.1%} unique responses in last {len(self.response_hashes)} samples',
                metrics={'unique_ratio': unique_ratio, 'window_size': len(self.response_hashes)}
            )

        return None


class RewardComponentAnalyzer:
    """Analyze individual reward components for exploitation patterns."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.component_history = {
            'format': deque(maxlen=100),
            'accuracy': deque(maxlen=100),
            'numbers': deque(maxlen=100),
            'other': deque(maxlen=100),
        }

    def update(self, components: Dict[str, float]):
        """Update component history."""
        for key, value in components.items():
            if key in self.component_history:
                self.component_history[key].append(value)

    def detect_exploitation(self) -> Optional[DetectionResult]:
        """Detect if model is exploiting a single reward component."""
        if all(len(hist) < 20 for hist in self.component_history.values()):
            return None

        # Calculate mean contribution of each component
        component_means = {
            key: np.mean(hist) if hist else 0.0
            for key, hist in self.component_history.items()
        }

        total = sum(component_means.values())
        if total < 1e-6:
            return None

        # Check if one component dominates
        component_ratios = {key: val/total for key, val in component_means.items()}
        max_component = max(component_ratios, key=component_ratios.get)
        max_ratio = component_ratios[max_component]

        if max_ratio > self.config.max_component_imbalance:
            return DetectionResult(
                is_anomalous=True,
                severity='high',
                detection_type='component_exploitation',
                message=f'Reward exploitation detected: {max_ratio:.1%} of reward from {max_component} component',
                metrics={'dominant_component': max_component, 'dominance_ratio': max_ratio, **component_ratios}
            )

        return None


class TrainingDynamicsMonitor:
    """Monitor training dynamics for instabilities."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.kl_history = deque(maxlen=100)
        self.gradient_norm_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)

    def update_kl(self, kl_divergence: float):
        """Update KL divergence history."""
        self.kl_history.append(kl_divergence)

    def update_gradient_norm(self, grad_norm: float):
        """Update gradient norm history."""
        self.gradient_norm_history.append(grad_norm)

    def update_loss(self, loss: float):
        """Update loss history."""
        self.loss_history.append(loss)

    def check_kl_divergence(self, current_kl: float) -> Optional[DetectionResult]:
        """Check for KL divergence anomalies."""
        if current_kl < self.config.kl_divergence_min:
            return DetectionResult(
                is_anomalous=True,
                severity='medium',
                detection_type='kl_divergence',
                message=f'KL divergence too low: {current_kl:.6f} (min: {self.config.kl_divergence_min}) - insufficient exploration',
                metrics={'kl_divergence': current_kl, 'threshold_min': self.config.kl_divergence_min}
            )

        if current_kl > self.config.kl_divergence_max:
            return DetectionResult(
                is_anomalous=True,
                severity='high',
                detection_type='kl_divergence',
                message=f'KL divergence too high: {current_kl:.6f} (max: {self.config.kl_divergence_max}) - policy diverging too much',
                metrics={'kl_divergence': current_kl, 'threshold_max': self.config.kl_divergence_max}
            )

        return None

    def check_gradient_norm(self, current_norm: float) -> Optional[DetectionResult]:
        """Check for gradient anomalies."""
        if current_norm < self.config.gradient_norm_min:
            return DetectionResult(
                is_anomalous=True,
                severity='high',
                detection_type='gradient_norm',
                message=f'Vanishing gradients detected: {current_norm:.2e} (min: {self.config.gradient_norm_min:.2e})',
                metrics={'gradient_norm': current_norm, 'threshold_min': self.config.gradient_norm_min}
            )

        if current_norm > self.config.gradient_norm_max:
            return DetectionResult(
                is_anomalous=True,
                severity='critical',
                detection_type='gradient_norm',
                message=f'Exploding gradients detected: {current_norm:.2e} (max: {self.config.gradient_norm_max:.2e})',
                metrics={'gradient_norm': current_norm, 'threshold_max': self.config.gradient_norm_max}
            )

        return None

    def check_loss_plateau(self) -> Optional[DetectionResult]:
        """Detect if loss has plateaued (potential reward hacking)."""
        if len(self.loss_history) < 50:
            return None

        recent_losses = list(self.loss_history)[-20:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)

        # Coefficient of variation
        if loss_mean > 1e-6:
            cv = loss_std / loss_mean
            if cv < 0.01:  # Very low variation
                return DetectionResult(
                    is_anomalous=True,
                    severity='medium',
                    detection_type='loss_plateau',
                    message=f'Loss plateau detected: CV={cv:.4f} over last 20 steps',
                    metrics={'coefficient_of_variation': cv, 'mean_loss': loss_mean, 'std_loss': loss_std}
                )

        return None


class RewardHackDetector:
    """Main detector coordinating all monitoring systems."""

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self.reward_stats = RewardStatistics(window_size=self.config.long_window_size)
        self.response_analyzer = ResponseAnalyzer(self.config)
        self.component_analyzer = RewardComponentAnalyzer(self.config)
        self.dynamics_monitor = TrainingDynamicsMonitor(self.config)

        self.total_detections = 0
        self.detections_by_type = Counter()
        self.detections_by_severity = Counter()

    def analyze_step(self,
                     response: str,
                     total_reward: float,
                     reward_components: Optional[Dict[str, float]] = None,
                     kl_divergence: Optional[float] = None,
                     gradient_norm: Optional[float] = None,
                     loss: Optional[float] = None) -> List[DetectionResult]:
        """
        Analyze a single training step for anomalies.

        Args:
            response: The generated response text
            total_reward: Total reward received
            reward_components: Dict of individual reward components (format, accuracy, etc.)
            kl_divergence: KL divergence from reference policy
            gradient_norm: Gradient norm
            loss: Training loss

        Returns:
            List of DetectionResult objects for any anomalies found
        """
        detections = []

        # Update statistics
        self.reward_stats.update(total_reward)
        if reward_components:
            self.component_analyzer.update(reward_components)
        if kl_divergence is not None:
            self.dynamics_monitor.update_kl(kl_divergence)
        if gradient_norm is not None:
            self.dynamics_monitor.update_gradient_norm(gradient_norm)
        if loss is not None:
            self.dynamics_monitor.update_loss(loss)

        # Reward anomaly detection
        if self.reward_stats.detect_anomaly(total_reward, self.config.reward_zscore_threshold):
            detections.append(DetectionResult(
                is_anomalous=True,
                severity='medium',
                detection_type='reward_anomaly',
                message=f'Reward anomaly: {total_reward:.2f} (mean: {self.reward_stats.get_mean():.2f}, std: {self.reward_stats.get_std():.2f})',
                metrics={'reward': total_reward, 'mean': self.reward_stats.get_mean(), 'std': self.reward_stats.get_std()}
            ))

        # Distribution shift detection
        shifted, shift_score = self.reward_stats.detect_distribution_shift()
        if shifted:
            detections.append(DetectionResult(
                is_anomalous=True,
                severity='medium',
                detection_type='distribution_shift',
                message=f'Reward distribution shift detected: shift score {shift_score:.2f}',
                metrics={'shift_score': shift_score, 'recent_mean': self.reward_stats.get_recent_mean(), 'historical_mean': self.reward_stats.get_mean()}
            ))

        # Response analysis
        length_issue = self.response_analyzer.analyze_length(response)
        if length_issue:
            detections.append(length_issue)

        repetition_issue = self.response_analyzer.analyze_repetition(response)
        if repetition_issue:
            detections.append(repetition_issue)

        diversity_issue = self.response_analyzer.analyze_diversity(response)
        if diversity_issue:
            detections.append(diversity_issue)

        # Format gaming detection
        if reward_components:
            format_reward = reward_components.get('format', 0)
            quality_reward = reward_components.get('accuracy', 0) + reward_components.get('numbers', 0)
            format_gaming = self.response_analyzer.analyze_format_gaming(response, format_reward, quality_reward)
            if format_gaming:
                detections.append(format_gaming)

        # Component exploitation
        component_exploit = self.component_analyzer.detect_exploitation()
        if component_exploit:
            detections.append(component_exploit)

        # Training dynamics
        if kl_divergence is not None:
            kl_issue = self.dynamics_monitor.check_kl_divergence(kl_divergence)
            if kl_issue:
                detections.append(kl_issue)

        if gradient_norm is not None:
            grad_issue = self.dynamics_monitor.check_gradient_norm(gradient_norm)
            if grad_issue:
                detections.append(grad_issue)

        loss_plateau = self.dynamics_monitor.check_loss_plateau()
        if loss_plateau:
            detections.append(loss_plateau)

        # Update detection statistics
        for detection in detections:
            self.total_detections += 1
            self.detections_by_type[detection.detection_type] += 1
            self.detections_by_severity[detection.severity] += 1

        return detections

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics for logging."""
        return {
            'total_detections': self.total_detections,
            'detections_by_type': dict(self.detections_by_type),
            'detections_by_severity': dict(self.detections_by_severity),
            'reward_mean': self.reward_stats.get_mean(),
            'reward_std': self.reward_stats.get_std(),
            'recent_reward_mean': self.reward_stats.get_recent_mean(),
        }

    def log_detections(self, step: int, detections: List[DetectionResult]):
        """Log detections with appropriate severity."""
        for detection in detections:
            log_msg = f'[Step {step}] {detection.severity.upper()}: {detection.message}'

            if detection.severity == 'critical':
                logger.critical(log_msg)
            elif detection.severity == 'high':
                logger.error(log_msg)
            elif detection.severity == 'medium':
                logger.warning(log_msg)
            else:
                logger.info(log_msg)
