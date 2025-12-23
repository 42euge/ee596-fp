"""
Unit tests for reward monitoring system.
"""

import unittest
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import directly from module files to avoid torch dependency
import importlib.util

def import_module_from_file(module_name, file_path):
    """Import a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import reward_monitoring directly
reward_monitoring_path = os.path.join(project_root, 'src', 'reward_monitoring.py')
reward_monitoring = import_module_from_file('reward_monitoring', reward_monitoring_path)

RewardHackDetector = reward_monitoring.RewardHackDetector
DetectionConfig = reward_monitoring.DetectionConfig
RewardStatistics = reward_monitoring.RewardStatistics
ResponseAnalyzer = reward_monitoring.ResponseAnalyzer
RewardComponentAnalyzer = reward_monitoring.RewardComponentAnalyzer
TrainingDynamicsMonitor = reward_monitoring.TrainingDynamicsMonitor


class TestRewardStatistics(unittest.TestCase):
    """Test reward statistics tracking."""

    def test_basic_statistics(self):
        """Test basic mean and std calculation."""
        stats = RewardStatistics(window_size=100)

        # Add some rewards
        for i in range(10):
            stats.update(float(i))

        self.assertAlmostEqual(stats.get_mean(), 4.5, places=1)
        self.assertGreater(stats.get_std(), 0)

    def test_anomaly_detection(self):
        """Test z-score based anomaly detection."""
        stats = RewardStatistics(window_size=100)

        # Add normal values
        for _ in range(20):
            stats.update(5.0)

        # Normal value should not be anomalous
        self.assertFalse(stats.detect_anomaly(5.0, threshold=3.0))

        # Very different value should be anomalous
        self.assertTrue(stats.detect_anomaly(50.0, threshold=3.0))

    def test_distribution_shift(self):
        """Test distribution shift detection."""
        stats = RewardStatistics(window_size=100)

        # Add initial distribution
        for _ in range(100):
            stats.update(5.0)

        # Add new distribution (higher values)
        for _ in range(20):
            stats.update(15.0)

        shifted, score = stats.detect_distribution_shift()
        self.assertTrue(shifted)
        self.assertGreater(score, 0)


class TestResponseAnalyzer(unittest.TestCase):
    """Test response quality analysis."""

    def setUp(self):
        self.config = DetectionConfig()
        self.analyzer = ResponseAnalyzer(self.config)

    def test_length_too_short(self):
        """Test detection of too-short responses."""
        short_response = "Hi"
        result = self.analyzer.analyze_length(short_response)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_anomalous)
        self.assertEqual(result.detection_type, 'response_length')

    def test_length_too_long(self):
        """Test detection of too-long responses."""
        long_response = "x" * 3000
        result = self.analyzer.analyze_length(long_response)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_anomalous)

    def test_length_normal(self):
        """Test normal length responses."""
        normal_response = "This is a normal length response with adequate content."
        result = self.analyzer.analyze_length(normal_response)

        self.assertIsNone(result)

    def test_token_repetition(self):
        """Test detection of excessive token repetition."""
        repetitive = "the " * 100
        result = self.analyzer.analyze_repetition(repetitive)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_anomalous)
        self.assertEqual(result.detection_type, 'token_repetition')

    def test_ngram_repetition(self):
        """Test detection of n-gram repetition."""
        # Create response with repeated 3-grams
        repetitive = "solve the problem " * 20
        result = self.analyzer.analyze_repetition(repetitive)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_anomalous)

    def test_format_gaming(self):
        """Test detection of format gaming."""
        # High format reward but minimal reasoning
        response = "<reasoning>x</reasoning><answer>42</answer>"
        result = self.analyzer.analyze_format_gaming(
            response,
            format_reward=3.0,
            quality_reward=0.0
        )

        self.assertIsNotNone(result)
        self.assertTrue(result.is_anomalous)

    def test_no_format_gaming(self):
        """Test normal response doesn't trigger format gaming."""
        response = "<reasoning>Let me think through this carefully. First, I'll analyze the problem statement. Then I'll work through the solution step by step.</reasoning><answer>42</answer>"
        result = self.analyzer.analyze_format_gaming(
            response,
            format_reward=3.0,
            quality_reward=2.0
        )

        self.assertIsNone(result)

    def test_mode_collapse_detection(self):
        """Test detection of mode collapse."""
        # Submit same response many times
        same_response = "Same response every time"

        for _ in range(25):
            result = self.analyzer.analyze_diversity(same_response)

        # Should eventually detect mode collapse
        self.assertIsNotNone(result)
        self.assertEqual(result.detection_type, 'mode_collapse')


class TestRewardComponentAnalyzer(unittest.TestCase):
    """Test reward component analysis."""

    def setUp(self):
        self.config = DetectionConfig()
        self.analyzer = RewardComponentAnalyzer(self.config)

    def test_component_exploitation(self):
        """Test detection of single component exploitation."""
        # Add data where format component dominates
        for _ in range(30):
            self.analyzer.update({
                'format': 10.0,
                'accuracy': 0.5,
                'numbers': 0.0,
                'other': 0.0,
            })

        result = self.analyzer.detect_exploitation()

        self.assertIsNotNone(result)
        self.assertTrue(result.is_anomalous)
        self.assertEqual(result.detection_type, 'component_exploitation')

    def test_balanced_components(self):
        """Test that balanced components don't trigger detection."""
        # Add balanced data
        for _ in range(30):
            self.analyzer.update({
                'format': 3.0,
                'accuracy': 2.5,
                'numbers': 1.5,
                'other': 1.0,
            })

        result = self.analyzer.detect_exploitation()

        self.assertIsNone(result)


class TestTrainingDynamicsMonitor(unittest.TestCase):
    """Test training dynamics monitoring."""

    def setUp(self):
        self.config = DetectionConfig()
        self.monitor = TrainingDynamicsMonitor(self.config)

    def test_kl_too_low(self):
        """Test detection of insufficient KL divergence."""
        result = self.monitor.check_kl_divergence(0.0001)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_anomalous)
        self.assertIn('too low', result.message.lower())

    def test_kl_too_high(self):
        """Test detection of excessive KL divergence."""
        result = self.monitor.check_kl_divergence(10.0)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_anomalous)
        self.assertIn('too high', result.message.lower())

    def test_kl_normal(self):
        """Test normal KL divergence."""
        result = self.monitor.check_kl_divergence(0.1)

        self.assertIsNone(result)

    def test_gradient_exploding(self):
        """Test detection of exploding gradients."""
        result = self.monitor.check_gradient_norm(100.0)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_anomalous)
        self.assertEqual(result.severity, 'critical')

    def test_gradient_vanishing(self):
        """Test detection of vanishing gradients."""
        result = self.monitor.check_gradient_norm(1e-8)

        self.assertIsNotNone(result)
        self.assertTrue(result.is_anomalous)

    def test_gradient_normal(self):
        """Test normal gradient norms."""
        result = self.monitor.check_gradient_norm(1.5)

        self.assertIsNone(result)

    def test_loss_plateau(self):
        """Test detection of loss plateau."""
        # Add very similar losses
        for _ in range(50):
            self.monitor.update_loss(2.5)

        result = self.monitor.check_loss_plateau()

        self.assertIsNotNone(result)
        self.assertEqual(result.detection_type, 'loss_plateau')


class TestRewardHackDetector(unittest.TestCase):
    """Test main detector integration."""

    def setUp(self):
        self.config = DetectionConfig()
        self.detector = RewardHackDetector(self.config)

    def test_basic_analysis(self):
        """Test basic step analysis."""
        response = "<reasoning>Let me solve this step by step with detailed reasoning.</reasoning><answer>42</answer>"

        detections = self.detector.analyze_step(
            response=response,
            total_reward=5.0,
            reward_components={'format': 3.0, 'accuracy': 2.0},
            kl_divergence=0.1,
            gradient_norm=1.5,
            loss=2.3,
        )

        # Should not detect issues with normal response
        self.assertEqual(len(detections), 0)

    def test_multiple_issues(self):
        """Test detection of multiple issues simultaneously."""
        # Create problematic response
        response = "x"  # Too short

        detections = self.detector.analyze_step(
            response=response,
            total_reward=50.0,  # Anomalously high
            reward_components={'format': 0.0, 'accuracy': 0.0},
            kl_divergence=10.0,  # Too high
            gradient_norm=100.0,  # Exploding
        )

        # Should detect multiple issues
        self.assertGreater(len(detections), 0)

        # Check for specific detection types
        detection_types = {d.detection_type for d in detections}
        self.assertIn('response_length', detection_types)
        self.assertIn('kl_divergence', detection_types)
        self.assertIn('gradient_norm', detection_types)

    def test_format_gaming_detection(self):
        """Test end-to-end format gaming detection."""
        response = "<reasoning>x</reasoning><answer>42</answer>"

        detections = self.detector.analyze_step(
            response=response,
            total_reward=3.0,
            reward_components={'format': 3.0, 'accuracy': 0.0},
        )

        # Should detect format gaming
        format_gaming_detected = any(
            d.detection_type == 'format_gaming' for d in detections
        )
        self.assertTrue(format_gaming_detected)

    def test_summary_metrics(self):
        """Test summary metrics generation."""
        # Run a few steps
        for i in range(10):
            self.detector.analyze_step(
                response=f"Response {i}",
                total_reward=5.0,
                reward_components={'format': 3.0, 'accuracy': 2.0},
            )

        metrics = self.detector.get_summary_metrics()

        self.assertIn('total_detections', metrics)
        self.assertIn('reward_mean', metrics)
        self.assertIn('reward_std', metrics)

    def test_detection_statistics(self):
        """Test that detection statistics are properly tracked."""
        # Create response that will trigger detection
        response = "x"

        for _ in range(5):
            self.detector.analyze_step(
                response=response,
                total_reward=5.0,
                reward_components={'format': 3.0, 'accuracy': 2.0},
            )

        self.assertGreater(self.detector.total_detections, 0)
        self.assertGreater(len(self.detector.detections_by_type), 0)


class TestDetectionConfig(unittest.TestCase):
    """Test configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DetectionConfig()

        self.assertEqual(config.reward_zscore_threshold, 3.0)
        self.assertEqual(config.min_response_length, 10)
        self.assertEqual(config.max_response_length, 2048)
        self.assertEqual(config.ngram_size, 3)

    def test_custom_config(self):
        """Test custom configuration."""
        config = DetectionConfig(
            reward_zscore_threshold=4.0,
            min_response_length=20,
            max_ngram_repetition_ratio=0.4,
        )

        self.assertEqual(config.reward_zscore_threshold, 4.0)
        self.assertEqual(config.min_response_length, 20)
        self.assertEqual(config.max_ngram_repetition_ratio, 0.4)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        self.detector = RewardHackDetector()

    def test_empty_response(self):
        """Test handling of empty response."""
        detections = self.detector.analyze_step(
            response="",
            total_reward=0.0,
            reward_components={},
        )

        # Should detect short response
        self.assertGreater(len(detections), 0)

    def test_very_first_step(self):
        """Test that first step doesn't crash."""
        detections = self.detector.analyze_step(
            response="Normal response",
            total_reward=5.0,
        )

        # Should work without crashing
        self.assertIsInstance(detections, list)

    def test_none_values(self):
        """Test handling of None values."""
        detections = self.detector.analyze_step(
            response="Normal response",
            total_reward=5.0,
            reward_components=None,
            kl_divergence=None,
            gradient_norm=None,
            loss=None,
        )

        # Should work without crashing
        self.assertIsInstance(detections, list)

    def test_zero_variance(self):
        """Test handling of zero variance in rewards."""
        # Add identical rewards
        for _ in range(20):
            self.detector.analyze_step(
                response="Response",
                total_reward=5.0,
            )

        # Should not crash
        metrics = self.detector.get_summary_metrics()
        self.assertIsInstance(metrics, dict)


if __name__ == '__main__':
    unittest.main()
