"""
Tests for reward quality assessment system.
"""

import pytest
import numpy as np
from TunRex.src.tunrex.datasets.reward_quality import (
    RewardQualityAssessor,
    QualityMetrics,
    PathologyAlert,
    RewardStats,
)
from TunRex.src.tunrex.datasets.reward_monitor import (
    RewardQualityMonitor,
    InterventionConfig,
    create_default_monitor,
)


class TestRewardQualityAssessor:
    """Tests for RewardQualityAssessor."""

    def test_initialization(self):
        """Test basic initialization."""
        assessor = RewardQualityAssessor(window_size=500, min_samples_for_analysis=50)
        assert assessor.window_size == 500
        assert assessor.min_samples_for_analysis == 50
        assert len(assessor.reward_history) == 0

    def test_assess_batch_basic(self):
        """Test basic batch assessment."""
        assessor = RewardQualityAssessor()

        responses = [
            "<reasoning>This is my reasoning</reasoning><answer>42</answer>",
            "<reasoning>Another reasoning</reasoning><answer>100</answer>",
        ]
        rewards = {
            'format': [3.0, 3.0],
            'accuracy': [1.5, 0.0],
        }

        metrics, alerts = assessor.assess_batch(responses, rewards)

        assert isinstance(metrics, QualityMetrics)
        assert metrics.format_compliance_rate == 1.0
        assert metrics.tag_correctness_rate == 1.0
        assert isinstance(alerts, list)

    def test_format_compliance_detection(self):
        """Test format compliance metrics."""
        assessor = RewardQualityAssessor()

        # Mix of correct and incorrect formats
        responses = [
            "<reasoning>Good</reasoning><answer>42</answer>",  # Correct
            "<reasoning>Good</reasoning>",  # Missing answer
            "<answer>42</answer>",  # Missing reasoning
            "No tags at all",  # No tags
        ]
        rewards = {'format': [3.0, 0.0, 0.0, 0.0]}

        metrics, _ = assessor.assess_batch(responses, rewards)

        assert metrics.format_compliance_rate == 0.25  # 1 out of 4
        assert metrics.tag_correctness_rate >= 0.25  # At least 1 has reasoning tags

    def test_repetition_score_computation(self):
        """Test repetition score calculation."""
        # No repetition
        score1 = RewardQualityAssessor._compute_repetition_score(
            "This is a unique sentence with different words each time"
        )
        assert score1 < 0.3

        # High repetition
        score2 = RewardQualityAssessor._compute_repetition_score(
            "word word word word word word word word word"
        )
        assert score2 > 0.7

        # Empty string
        score3 = RewardQualityAssessor._compute_repetition_score("")
        assert score3 == 0.0

    def test_empty_response_detection(self):
        """Test detection of empty/degenerate responses."""
        assert RewardQualityAssessor._is_empty_response(
            "<reasoning></reasoning><answer></answer>"
        ) == True

        assert RewardQualityAssessor._is_empty_response(
            "<reasoning>   </reasoning><answer> </answer>"
        ) == True

        assert RewardQualityAssessor._is_empty_response(
            "<reasoning>Good reasoning</reasoning><answer>42</answer>"
        ) == False

        assert RewardQualityAssessor._is_empty_response(
            "No tags"
        ) == True

    def test_lexical_diversity(self):
        """Test lexical diversity computation."""
        # High diversity
        responses1 = [
            "unique words here",
            "different vocabulary altogether",
            "completely distinct terms",
        ]
        diversity1 = RewardQualityAssessor._compute_lexical_diversity(responses1)
        assert diversity1 > 0.8

        # Low diversity (repeated words)
        responses2 = [
            "same same same",
            "same same same",
            "same same same",
        ]
        diversity2 = RewardQualityAssessor._compute_lexical_diversity(responses2)
        assert diversity2 < 0.3

    def test_reward_saturation_detection(self):
        """Test detection of reward saturation."""
        assessor = RewardQualityAssessor(min_samples_for_analysis=10)

        # Create batch with saturated rewards (all at max)
        responses = ["<reasoning>test</reasoning><answer>1</answer>"] * 150
        rewards = {'format': [3.0] * 150}  # All at maximum

        # Need to run multiple times to build history
        for i in range(15):
            batch_responses = responses[i*10:(i+1)*10]
            batch_rewards = {'format': rewards['format'][i*10:(i+1)*10]}
            _, alerts = assessor.assess_batch(batch_responses, batch_rewards)

        # Should detect saturation
        saturation_alerts = [a for a in alerts if a.pathology_type == 'reward_saturation_high']
        assert len(saturation_alerts) > 0
        assert saturation_alerts[0].severity in ['medium', 'high']

    def test_variance_collapse_detection(self):
        """Test detection of variance collapse."""
        assessor = RewardQualityAssessor(min_samples_for_analysis=10)

        # All rewards very similar (collapsed variance)
        responses = ["<reasoning>test</reasoning><answer>1</answer>"] * 150
        # All rewards almost identical
        rewards = {'accuracy': [1.5 + np.random.normal(0, 0.001) for _ in range(150)]}

        for i in range(15):
            batch_responses = responses[i*10:(i+1)*10]
            batch_rewards = {'accuracy': rewards['accuracy'][i*10:(i+1)*10]}
            _, alerts = assessor.assess_batch(batch_responses, batch_rewards)

        # Should detect variance collapse
        collapse_alerts = [a for a in alerts if a.pathology_type == 'variance_collapse']
        assert len(collapse_alerts) > 0

    def test_format_gaming_detection(self):
        """Test detection of format gaming (correct format, bad content)."""
        assessor = RewardQualityAssessor()

        # Responses with correct format but minimal/bad content
        responses = [
            "<reasoning>idk</reasoning><answer>?</answer>",
            "<reasoning>...</reasoning><answer>n/a</answer>",
            "<reasoning>a</reasoning><answer>x</answer>",
        ] * 10

        rewards = {
            'format': [3.0] * 30,  # High format reward
            'accuracy': [0.0] * 30,  # But wrong answers
        }

        metrics, alerts = assessor.assess_batch(responses, rewards)

        # Should detect format gaming
        gaming_alerts = [a for a in alerts if a.pathology_type == 'format_gaming']
        assert len(gaming_alerts) > 0
        assert gaming_alerts[0].severity == 'critical'

    def test_diversity_collapse_detection(self):
        """Test detection of mode collapse (lack of diversity)."""
        assessor = RewardQualityAssessor(min_samples_for_analysis=50)

        # All responses identical
        same_response = "<reasoning>Always the same</reasoning><answer>42</answer>"
        responses = [same_response] * 200
        rewards = {'format': [3.0] * 200}

        for i in range(20):
            batch_responses = responses[i*10:(i+1)*10]
            batch_rewards = {'format': rewards['format'][i*10:(i+1)*10]}
            _, alerts = assessor.assess_batch(batch_responses, batch_rewards)

        # Should detect diversity collapse
        diversity_alerts = [a for a in alerts if a.pathology_type == 'diversity_collapse']
        assert len(diversity_alerts) > 0

    def test_degenerate_output_detection(self):
        """Test detection of degenerate outputs."""
        assessor = RewardQualityAssessor()

        # Many empty/degenerate responses
        responses = [
            "<reasoning></reasoning><answer></answer>",
            "<reasoning>   </reasoning><answer> </answer>",
            "broken response",
            "<reasoning>ok</reasoning><answer></answer>",
        ] * 5

        rewards = {'format': [0.0] * 20}

        _, alerts = assessor.assess_batch(responses, rewards)

        # Should detect degenerate outputs
        degen_alerts = [a for a in alerts if a.pathology_type == 'degenerate_outputs']
        assert len(degen_alerts) > 0

    def test_reward_spike_detection(self):
        """Test detection of unusual reward spikes."""
        assessor = RewardQualityAssessor(min_samples_for_analysis=50)

        # Build normal distribution first
        normal_responses = ["<reasoning>test</reasoning><answer>1</answer>"] * 100
        normal_rewards = {'accuracy': [1.5] * 100}

        for i in range(10):
            batch_responses = normal_responses[i*10:(i+1)*10]
            batch_rewards = {'accuracy': normal_rewards['accuracy'][i*10:(i+1)*10]}
            assessor.assess_batch(batch_responses, batch_rewards)

        # Now introduce spike
        spike_responses = ["<reasoning>test</reasoning><answer>1</answer>"] * 10
        spike_rewards = {'accuracy': [100.0] * 10}  # Unusual spike

        _, alerts = assessor.assess_batch(spike_responses, spike_rewards)

        # Should detect spike
        spike_alerts = [a for a in alerts if a.pathology_type == 'reward_spike']
        assert len(spike_alerts) > 0

    def test_get_summary_statistics(self):
        """Test summary statistics generation."""
        assessor = RewardQualityAssessor()

        responses = ["<reasoning>test</reasoning><answer>1</answer>"] * 50
        rewards = {'format': [3.0] * 50, 'accuracy': [1.5] * 50}

        assessor.assess_batch(responses, rewards)

        summary = assessor.get_summary_statistics()

        assert 'total_samples' in summary
        assert summary['total_samples'] == 50
        assert 'reward_statistics' in summary
        assert 'format' in summary['reward_statistics']
        assert 'accuracy' in summary['reward_statistics']


class TestRewardQualityMonitor:
    """Tests for RewardQualityMonitor."""

    def test_initialization(self):
        """Test monitor initialization."""
        assessor = RewardQualityAssessor()
        config = InterventionConfig(enable_interventions=False)
        monitor = RewardQualityMonitor(assessor, config)

        assert monitor.assessor == assessor
        assert monitor.config == config
        assert monitor.wandb_run is None

    def test_monitor_batch(self):
        """Test monitoring a batch."""
        monitor = create_default_monitor(enable_interventions=False)

        responses = [
            "<reasoning>Good reasoning</reasoning><answer>42</answer>",
            "<reasoning>More reasoning</reasoning><answer>100</answer>",
        ]
        rewards = {'format': [3.0, 3.0], 'accuracy': [1.5, 0.0]}

        metrics = monitor.monitor_batch(responses, rewards, step=1)

        assert isinstance(metrics, QualityMetrics)
        assert metrics.format_compliance_rate == 1.0

    def test_alert_handling(self):
        """Test alert handling and logging."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            alert_file = os.path.join(tmpdir, 'alerts.jsonl')
            metrics_file = os.path.join(tmpdir, 'metrics.jsonl')

            config = InterventionConfig(
                enable_interventions=True,
                log_to_console=False,
                log_to_file=True,
                send_to_wandb=False,
                alert_log_file=alert_file,
                metrics_log_file=metrics_file,
            )

            assessor = RewardQualityAssessor(min_samples_for_analysis=10)
            monitor = RewardQualityMonitor(assessor, config)

            # Generate format gaming scenario
            responses = [
                "<reasoning>x</reasoning><answer>y</answer>"
            ] * 50
            rewards = {'format': [3.0] * 50}

            monitor.monitor_batch(responses, rewards, step=1)

            # Check that files were created
            assert os.path.exists(metrics_file)

    def test_intervention_throttling(self):
        """Test that alert throttling works."""
        config = InterventionConfig(
            enable_interventions=True,
            log_to_console=False,
            alert_aggregation_window=10,
            max_alerts_per_window=5,
        )

        assessor = RewardQualityAssessor(min_samples_for_analysis=5)
        monitor = RewardQualityMonitor(assessor, config)

        # Generate many alerts rapidly
        for step in range(20):
            responses = ["<reasoning></reasoning><answer></answer>"] * 10
            rewards = {'format': [0.0] * 10}
            monitor.monitor_batch(responses, rewards, step=step)

        # Alert count should be throttled
        recent_steps = monitor.recent_alert_steps
        assert len(recent_steps) <= config.max_alerts_per_window * 2  # Some tolerance

    def test_get_monitoring_summary(self):
        """Test monitoring summary generation."""
        monitor = create_default_monitor(enable_interventions=False)

        responses = ["<reasoning>test</reasoning><answer>1</answer>"] * 20
        rewards = {'format': [3.0] * 20}

        monitor.monitor_batch(responses, rewards, step=1)

        summary = monitor.get_monitoring_summary()

        assert 'total_samples' in summary
        assert 'monitoring' in summary
        assert 'total_alerts_by_severity' in summary['monitoring']


class TestIntegration:
    """Integration tests for the complete system."""

    def test_end_to_end_quality_monitoring(self):
        """Test complete end-to-end quality monitoring workflow."""
        monitor = create_default_monitor(enable_interventions=True)

        # Simulate training batches
        for step in range(10):
            # Generate varied responses
            responses = []
            for i in range(10):
                if i % 3 == 0:
                    # Good response
                    responses.append(
                        f"<reasoning>Step {step}: Let's solve this problem carefully</reasoning>"
                        f"<answer>{42 + i}</answer>"
                    )
                elif i % 3 == 1:
                    # Format gaming
                    responses.append("<reasoning>x</reasoning><answer>y</answer>")
                else:
                    # Degenerate
                    responses.append("<reasoning></reasoning><answer></answer>")

            rewards = {
                'format': [3.0 if i % 3 != 2 else 0.0 for i in range(10)],
                'accuracy': [1.5 if i % 3 == 0 else 0.0 for i in range(10)],
            }

            metrics = monitor.monitor_batch(responses, rewards, step=step)

            # Verify metrics are computed
            assert metrics.format_compliance_rate >= 0.0
            assert metrics.format_compliance_rate <= 1.0

        # Get final summary
        summary = monitor.get_monitoring_summary()
        assert summary['total_samples'] == 100
        assert len(summary['monitoring']['total_alerts_by_severity']) > 0

    def test_custom_thresholds(self):
        """Test using custom detection thresholds."""
        custom_thresholds = {
            'format_gaming_threshold': 0.1,  # Very sensitive
            'repetition_threshold': 0.3,
        }

        assessor = RewardQualityAssessor(
            alert_thresholds=custom_thresholds
        )

        # Should trigger with lower threshold
        responses = [
            "<reasoning>x</reasoning><answer>y</answer>",
            "<reasoning>Good reasoning here</reasoning><answer>42</answer>",
        ] * 5

        rewards = {'format': [3.0] * 10}

        _, alerts = assessor.assess_batch(responses, rewards)

        # Should detect with custom threshold
        gaming_alerts = [a for a in alerts if a.pathology_type == 'format_gaming']
        assert len(gaming_alerts) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
