"""
Test suite for the MicroLoanEnv Gymnasium environment.
Validates API compliance, reward calculation, and state space construction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import env as _env_registration  # noqa: F401
import gymnasium as gym
from gymnasium.utils.env_checker import check_env as gym_check_env

from env.microloan_env import MicroLoanEnv, DEFAULT_CONFIG
from env.client_generator import SyntheticClientGenerator


# -----------------------------------------------------------------------
# Environment creation helpers
# -----------------------------------------------------------------------

def make_env(**overrides):
    config = {**DEFAULT_CONFIG, **overrides}
    return MicroLoanEnv(config=config)


# -----------------------------------------------------------------------
# 1. Gymnasium API compliance
# -----------------------------------------------------------------------

class TestGymAPICompliance:
    """Ensure the environment follows the Gymnasium API contract."""

    def test_check_env_discrete(self):
        """Run gymnasium's built-in env checker (discrete actions)."""
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        gym_check_env(env.unwrapped, skip_render_check=True)

    def test_check_env_with_nlp(self):
        """Run checker with NLP features (larger obs space)."""
        env = make_env(use_nlp_features=True, sentiment_dim=32)
        gym_check_env(env.unwrapped, skip_render_check=True)

    def test_reset_returns_correct_shapes(self):
        env = make_env(use_nlp_features=True, sentiment_dim=32)
        obs, info = env.reset(seed=42)
        assert obs.shape == (7 + 32,), f"Expected (39,), got {obs.shape}"
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(5)
        assert obs.shape == (7,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_terminates(self):
        env = make_env(use_nlp_features=False, sentiment_dim=0, max_steps=10)
        env.reset(seed=42)
        for i in range(10):
            _, _, terminated, truncated, _ = env.step(0)
        assert truncated, "Episode should be truncated after max_steps"

    def test_gym_make_registration(self):
        """Verify the env is properly registered."""
        env = gym.make("MicroLoan-v0", config={"use_nlp_features": False, "sentiment_dim": 0})
        obs, _ = env.reset()
        assert obs.shape == (7,)
        env.close()


# -----------------------------------------------------------------------
# 2. Reward function
# -----------------------------------------------------------------------

class TestRewardFunction:
    """Validate the multi-objective reward calculation."""

    def test_successful_repayment_positive_reward(self):
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        reward = env._compute_reward(
            rate=0.15, default_prob=0.1, did_default=False,
            client={"language_group": 0}
        )
        assert reward > 0, f"Successful repayment should yield positive reward, got {reward}"

    def test_default_negative_reward(self):
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        reward = env._compute_reward(
            rate=0.25, default_prob=0.8, did_default=True,
            client={"language_group": 0}
        )
        assert reward < 0, f"Default should yield negative reward, got {reward}"

    def test_bounded_default_penalty(self):
        """Ensure the log-bounded penalty doesn't produce extreme values."""
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        reward_low = env._compute_reward(
            rate=0.25, default_prob=0.1, did_default=True,
            client={"language_group": 0}
        )
        reward_high = env._compute_reward(
            rate=0.25, default_prob=0.95, did_default=True,
            client={"language_group": 0}
        )
        # Both should be negative but bounded
        assert reward_low > -10, f"Reward too extreme: {reward_low}"
        assert reward_high > -10, f"Reward too extreme: {reward_high}"

    def test_accessibility_penalty_applied(self):
        """Low-resource language clients should get reduced client utility."""
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        reward_hrl = env._compute_reward(
            rate=0.15, default_prob=0.1, did_default=False,
            client={"language_group": 0}
        )
        reward_lrl = env._compute_reward(
            rate=0.15, default_prob=0.1, did_default=False,
            client={"language_group": 1}
        )
        assert reward_lrl < reward_hrl, "LRL client should have lower reward due to accessibility penalty"


# -----------------------------------------------------------------------
# 3. Default probability
# -----------------------------------------------------------------------

class TestDefaultProbability:
    """Validate the default probability computation."""

    def test_higher_rate_increases_default_prob(self):
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        client = {
            "credit_score": 0.7, "dti_ratio": 0.3,
            "income_stability": 0.6, "repayment_history": 0.8,
            "unemployment_indicator": 0.1, "sentiment_embedding": np.array([]),
        }
        p_low = env._compute_default_probability(client, rate=0.08)
        p_high = env._compute_default_probability(client, rate=0.28)
        assert p_high > p_low, "Higher rate should increase default probability"

    def test_default_prob_bounded(self):
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        client = {
            "credit_score": 0.01, "dti_ratio": 0.99,
            "income_stability": 0.01, "repayment_history": 0.01,
            "unemployment_indicator": 0.99, "sentiment_embedding": np.array([]),
        }
        p = env._compute_default_probability(client, rate=0.30)
        assert 0.01 <= p <= 0.95, f"Default prob should be bounded, got {p}"


# -----------------------------------------------------------------------
# 4. Client generator
# -----------------------------------------------------------------------

class TestClientGenerator:
    def test_synthetic_generator_output(self):
        gen = SyntheticClientGenerator(sentiment_dim=32, use_nlp_features=True)
        rng = np.random.default_rng(42)
        client = gen.generate(rng)
        assert 0 <= client["credit_score"] <= 1
        assert 0 <= client["dti_ratio"] <= 1
        assert client["sentiment_embedding"].shape == (32,)
        assert client["demographic_group"] in (0, 1)

    def test_batch_generation(self):
        gen = SyntheticClientGenerator(sentiment_dim=16, use_nlp_features=True)
        rng = np.random.default_rng(42)
        batch = gen.generate_batch(rng, 100)
        assert len(batch) == 100
        # Check minority ratio is roughly correct (±20%)
        minority_count = sum(c["demographic_group"] for c in batch)
        assert 10 <= minority_count <= 50


# -----------------------------------------------------------------------
# 5. Action decoding
# -----------------------------------------------------------------------

class TestActionDecoding:
    def test_discrete_action_range(self):
        env = make_env(use_nlp_features=False, sentiment_dim=0, num_rate_tiers=10)
        env.reset(seed=42)
        rate_min = env._decode_action(0)
        rate_max = env._decode_action(9)
        assert abs(rate_min - 0.05) < 0.001
        assert abs(rate_max - 0.30) < 0.001

    def test_action_clamping(self):
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        env.reset(seed=42)
        # Step with extreme actions — rate should still be within bounds
        _, _, _, _, info = env.step(0)
        assert info["offered_rate"] >= 0.05
        _, _, _, _, info = env.step(9)
        assert info["offered_rate"] <= 0.30


# -----------------------------------------------------------------------
# 6. Info dict for fairness tracking
# -----------------------------------------------------------------------

class TestInfoDict:
    def test_info_contains_fairness_fields(self):
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        _, info = env.reset(seed=42)
        assert "demographic_group" in info
        assert "language_group" in info

    def test_step_info_has_rate_and_default(self):
        env = make_env(use_nlp_features=False, sentiment_dim=0)
        env.reset(seed=42)
        _, _, _, _, info = env.step(5)
        assert "offered_rate" in info
        assert "default_prob" in info
        assert "did_default" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
