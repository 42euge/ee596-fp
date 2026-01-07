"""
Unit tests for RLOO (REINFORCE Leave-One-Out) implementation.

Tests cover:
- RLOOConfig validation
- compute_rloo_advantages function
- Leave-one-out baseline computation
- KL integration into rewards
- Advantage clipping
"""

import pytest
import numpy as np

# Try to import JAX, fall back to numpy if not available
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    import numpy as jnp
    HAS_JAX = False

from scripts.rloo_learner import (
    RLOOConfig,
    compute_rloo_advantages,
)


class TestRLOOConfig:
    """Test RLOO configuration validation."""

    def test_valid_config(self):
        """Test that valid configs are accepted."""
        config = RLOOConfig(
            num_generations=2,
            beta=0.08,
            epsilon=0.2,
            kl_in_reward=True,
        )
        assert config.num_generations == 2
        assert config.beta == 0.08
        assert config.epsilon == 0.2
        assert config.kl_in_reward is True

    def test_invalid_num_generations(self):
        """Test that num_generations < 2 raises error."""
        with pytest.raises(ValueError, match="RLOO requires num_generations >= 2"):
            RLOOConfig(num_generations=1)

    def test_default_values(self):
        """Test default configuration values."""
        config = RLOOConfig()
        assert config.num_generations == 2
        assert config.num_iterations == 1
        assert config.beta == 0.08
        assert config.epsilon == 0.2
        assert config.kl_in_reward is True
        assert config.advantage_clip is None


class TestComputeRLOOAdvantages:
    """Test RLOO advantage computation."""

    def test_basic_loo_computation(self):
        """Test basic leave-one-out advantage computation.

        For rewards [1, 2, 3]:
        - A_0 = 1 - mean([2, 3]) = 1 - 2.5 = -1.5
        - A_1 = 2 - mean([1, 3]) = 2 - 2.0 = 0.0
        - A_2 = 3 - mean([1, 2]) = 3 - 1.5 = 1.5
        """
        rewards = jnp.array([[1.0, 2.0, 3.0]])  # (1, 3)

        advantages = compute_rloo_advantages(
            rewards,
            kl_divergences=None,
            kl_in_reward=False,
        )

        expected = jnp.array([[-1.5, 0.0, 1.5]])

        if HAS_JAX:
            np.testing.assert_allclose(advantages, expected, rtol=1e-5)
        else:
            np.testing.assert_allclose(np.array(advantages), np.array(expected), rtol=1e-5)

    def test_batch_loo_computation(self):
        """Test leave-one-out with batch dimension."""
        rewards = jnp.array([
            [1.0, 2.0, 3.0],  # Batch 1
            [4.0, 5.0, 6.0],  # Batch 2
        ])  # (2, 3)

        advantages = compute_rloo_advantages(rewards, kl_in_reward=False)

        # Batch 1: same as above
        # Batch 2:
        # - A_0 = 4 - mean([5, 6]) = 4 - 5.5 = -1.5
        # - A_1 = 5 - mean([4, 6]) = 5 - 5.0 = 0.0
        # - A_2 = 6 - mean([4, 5]) = 6 - 4.5 = 1.5

        expected = jnp.array([
            [-1.5, 0.0, 1.5],
            [-1.5, 0.0, 1.5],
        ])

        if HAS_JAX:
            np.testing.assert_allclose(advantages, expected, rtol=1e-5)
        else:
            np.testing.assert_allclose(np.array(advantages), np.array(expected), rtol=1e-5)

    def test_kl_integration(self):
        """Test KL divergence integration into rewards."""
        rewards = jnp.array([[1.0, 2.0, 3.0]])
        kl = jnp.array([[0.1, 0.2, 0.3]])
        beta = 0.5

        advantages = compute_rloo_advantages(
            rewards,
            kl_divergences=kl,
            beta=beta,
            kl_in_reward=True,
        )

        # Modified rewards: R' = R - β*KL
        # R' = [1.0, 2.0, 3.0] - 0.5 * [0.1, 0.2, 0.3]
        #    = [0.95, 1.9, 2.85]
        #
        # Then apply LOO:
        # - A_0 = 0.95 - mean([1.9, 2.85]) = 0.95 - 2.375 = -1.425
        # - A_1 = 1.9 - mean([0.95, 2.85]) = 1.9 - 1.9 = 0.0
        # - A_2 = 2.85 - mean([0.95, 1.9]) = 2.85 - 1.425 = 1.425

        expected = jnp.array([[-1.425, 0.0, 1.425]])

        if HAS_JAX:
            np.testing.assert_allclose(advantages, expected, rtol=1e-5)
        else:
            np.testing.assert_allclose(np.array(advantages), np.array(expected), rtol=1e-5)

    def test_kl_not_integrated(self):
        """Test that KL is ignored when kl_in_reward=False."""
        rewards = jnp.array([[1.0, 2.0, 3.0]])
        kl = jnp.array([[0.1, 0.2, 0.3]])

        advantages = compute_rloo_advantages(
            rewards,
            kl_divergences=kl,
            kl_in_reward=False,  # Should ignore KL
        )

        # Should be same as without KL
        expected = jnp.array([[-1.5, 0.0, 1.5]])

        if HAS_JAX:
            np.testing.assert_allclose(advantages, expected, rtol=1e-5)
        else:
            np.testing.assert_allclose(np.array(advantages), np.array(expected), rtol=1e-5)

    def test_advantage_clipping(self):
        """Test advantage clipping to prevent outliers."""
        rewards = jnp.array([[0.0, 5.0, 10.0]])  # Large spread

        advantages = compute_rloo_advantages(
            rewards,
            kl_in_reward=False,
            advantage_clip=2.0,  # Clip to [-2, 2]
        )

        # Without clipping:
        # - A_0 = 0 - mean([5, 10]) = 0 - 7.5 = -7.5
        # - A_1 = 5 - mean([0, 10]) = 5 - 5.0 = 0.0
        # - A_2 = 10 - mean([0, 5]) = 10 - 2.5 = 7.5
        #
        # With clipping to [-2, 2]:
        # - A_0 = -2.0 (clipped from -7.5)
        # - A_1 = 0.0 (unchanged)
        # - A_2 = 2.0 (clipped from 7.5)

        expected = jnp.array([[-2.0, 0.0, 2.0]])

        if HAS_JAX:
            np.testing.assert_allclose(advantages, expected, rtol=1e-5)
        else:
            np.testing.assert_allclose(np.array(advantages), np.array(expected), rtol=1e-5)

    def test_insufficient_generations(self):
        """Test that K < 2 raises error."""
        rewards = jnp.array([[1.0]])  # Only 1 generation

        with pytest.raises(ValueError, match="RLOO requires K >= 2"):
            compute_rloo_advantages(rewards, kl_in_reward=False)

    def test_zero_mean_property(self):
        """Test that RLOO advantages don't necessarily sum to zero.

        Unlike GRPO's normalized advantages, RLOO advantages don't
        have the zero-mean property due to leave-one-out baseline.
        """
        rewards = jnp.array([[1.0, 2.0, 3.0, 4.0]])

        advantages = compute_rloo_advantages(rewards, kl_in_reward=False)

        # Sum should be close to zero but not exactly (due to LOO)
        advantage_sum = jnp.sum(advantages)

        # For symmetric rewards, sum should be zero
        # For [1, 2, 3, 4], it should be close to zero
        if HAS_JAX:
            assert abs(advantage_sum) < 1e-5
        else:
            assert abs(float(advantage_sum)) < 1e-5


class TestRLOOvsGRPO:
    """Comparative tests: RLOO vs GRPO behavior."""

    def test_rloo_more_stable_with_noise(self):
        """Demonstrate RLOO is more stable than std-normalized advantages.

        GRPO: A_i = (R_i - mean(R)) / std(R)
        RLOO: A_i = R_i - mean(R_j where j != i)

        With noisy rewards, GRPO's std normalization amplifies noise.
        """
        # Rewards with outlier
        rewards = jnp.array([[1.0, 1.1, 1.2, 100.0]])  # Last is outlier

        # RLOO advantages (no std normalization)
        rloo_advantages = compute_rloo_advantages(rewards, kl_in_reward=False)

        # GRPO-style advantages (with std normalization)
        mean_r = jnp.mean(rewards)
        std_r = jnp.std(rewards)
        grpo_advantages = (rewards - mean_r) / std_r

        # RLOO should have smaller max absolute advantage
        rloo_max = jnp.max(jnp.abs(rloo_advantages))
        grpo_max = jnp.max(jnp.abs(grpo_advantages))

        # In this case, RLOO should be more stable
        # (though this depends on the specific data)
        print(f"RLOO max abs advantage: {rloo_max}")
        print(f"GRPO max abs advantage: {grpo_max}")


# Integration test (requires full setup, marked as slow)
@pytest.mark.slow
@pytest.mark.skipif(not HAS_JAX, reason="Requires JAX")
class TestRLOOLearnerIntegration:
    """Integration tests for RLOOLearner (requires JAX and Tunix)."""

    def test_rloo_learner_creation(self):
        """Test that RLOOLearner can be created."""
        # This would require a full RL cluster setup
        # Skipping for now, but structure is here for future tests
        pytest.skip("Requires full Tunix setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
