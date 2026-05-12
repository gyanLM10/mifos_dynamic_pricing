"""
MicroLoanEnv — A high-fidelity Gymnasium environment modelling the Markov Decision
Process for dynamic micro-loan interest rate pricing.

State Space (S):
    Composite vector of tabular financial features AND a dense NLP sentiment embedding.
    - Tabular (7 dims): credit_score, dti_ratio, income_stability, repayment_history,
                         loan_amount_norm, gdp_indicator, unemployment_indicator
    - Sentiment (768 dims or configurable): XLM-RoBERTa embedding of client text
    Total: 7 + sentiment_dim

Action Space (A):
    Discrete risk tiers (default 10 tiers mapping to interest rates 5%–30%) or
    continuous delta-based adjustment.

Reward Function (R):
    R = w1*(Interest Earned) - w2*(Default Risk) + w3*(Client Utility) - Opportunity Cost
    All components are normalized to [-1, 1] before weighting.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Optional


DEFAULT_CONFIG = {
    # Reward weights (post-normalization)
    "w1": 1.0,   # Interest earned weight
    "w2": 1.3,   # Default risk weight
    "w3": 0.65,  # Client utility weight

    # Action space
    "num_rate_tiers": 10,        # Discrete tiers
    "min_rate": 0.05,            # 5% APR floor
    "max_rate": 0.30,            # 30% APR ceiling
    "use_continuous_actions": False,

    # State space
    "sentiment_dim": 768,        # XLM-RoBERTa hidden size
    "use_nlp_features": True,    # Toggle NLP branch

    # Episode
    "max_steps": 52,             # 52-week episode

    # Economic shock probability per step
    "shock_probability": 0.05,
    "shock_severity": 0.3,       # multiplier on default prob increase

    # Opportunity cost penalty for rejections
    "opportunity_cost": 0.05,

    # Accessibility penalty for non-native language enforcement
    "accessibility_penalty": 0.02,
}


class MicroLoanEnv(gym.Env):
    """
    A Gymnasium environment for simulating micro-loan dynamic pricing.

    The agent acts as a loan officer setting interest rates for incoming clients.
    Each episode represents a sequence of client interactions over a simulated year.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
        client_generator=None,
    ):
        super().__init__()

        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.render_mode = render_mode


        self.tabular_dim = 7
        self.sentiment_dim = (
            self.config["sentiment_dim"] if self.config["use_nlp_features"] else 0
        )
        self.obs_dim = self.tabular_dim + self.sentiment_dim


        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        if self.config["use_continuous_actions"]:
            # Delta-based: agent outputs a rate delta in [-0.005, +0.005]
            self.action_space = spaces.Box(
                low=-0.005, high=0.005, shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Discrete(self.config["num_rate_tiers"])


        self._client_gen = client_generator  # Injected; falls back to internal


        self._current_client: dict[str, Any] = {}
        self._current_rate: float = 0.15  # starting mid-range rate
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._episode_defaults: int = 0
        self._episode_approvals: int = 0
        self._rng: np.random.Generator = np.random.default_rng()


    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._episode_defaults = 0
        self._episode_approvals = 0
        self._current_rate = 0.15

        self._current_client = self._generate_client()
        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1


        offered_rate = self._decode_action(action)


        offered_rate = np.clip(
            offered_rate, self.config["min_rate"], self.config["max_rate"]
        )


        default_prob = self._compute_default_probability(
            self._current_client, offered_rate
        )

        # Apply economic shock
        if self._rng.random() < self.config["shock_probability"]:
            default_prob = min(
                1.0, default_prob + self.config["shock_severity"] * default_prob
            )

        did_default = self._rng.random() < default_prob


        reward = self._compute_reward(
            offered_rate, default_prob, did_default, self._current_client
        )

        self._cumulative_reward += reward
        self._current_rate = offered_rate

        if did_default:
            self._episode_defaults += 1
        else:
            self._episode_approvals += 1


        terminated = False
        truncated = self._step_count >= self.config["max_steps"]

        self._current_client = self._generate_client()
        obs = self._build_observation()

        info = self._build_info()
        info.update({
            "offered_rate": offered_rate,
            "default_prob": default_prob,
            "did_default": did_default,
            "step_reward": reward,
        })

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())

    def _render_ansi(self) -> str:
        c = self._current_client
        return (
            f"Step {self._step_count}/{self.config['max_steps']} | "
            f"Rate: {self._current_rate:.2%} | "
            f"Client DTI: {c.get('dti_ratio', 0):.2f} | "
            f"Credit: {c.get('credit_score', 0):.2f} | "
            f"Defaults: {self._episode_defaults} | "
            f"Approvals: {self._episode_approvals} | "
            f"Cum. Reward: {self._cumulative_reward:.3f}"
        )


    def _generate_client(self) -> dict[str, Any]:
        """Generate a synthetic client profile. Uses injected generator if available."""
        if self._client_gen is not None:
            return self._client_gen.generate(self._rng)

        # Fallback: internal synthetic generation
        credit_score = float(self._rng.beta(5, 2))  # skew toward good credit
        dti_ratio = float(self._rng.beta(2, 5))      # skew toward low DTI
        income_stability = float(self._rng.beta(4, 2))
        repayment_history = float(self._rng.beta(6, 2))  # mostly good
        loan_amount_norm = float(self._rng.beta(2, 3))
        gdp_indicator = float(self._rng.normal(0.5, 0.1))
        unemployment_indicator = float(self._rng.beta(2, 8))

        # Demographic group for fairness tracking (0 = majority, 1 = minority)
        demographic_group = int(self._rng.random() < 0.3)

        # Language group (0 = dominant, 1 = low-resource)
        language_group = int(self._rng.random() < 0.25)

        # Synthetic sentiment embedding
        if self.config["use_nlp_features"]:
            # Simulate sentiment: higher distress for lower credit/income
            distress_level = 1.0 - (credit_score * 0.4 + income_stability * 0.3 +
                                     repayment_history * 0.3)
            sentiment_embedding = self._rng.normal(
                loc=distress_level * 0.5,
                scale=0.3,
                size=self.sentiment_dim,
            ).astype(np.float32)
        else:
            sentiment_embedding = np.array([], dtype=np.float32)

        return {
            "credit_score": credit_score,
            "dti_ratio": dti_ratio,
            "income_stability": income_stability,
            "repayment_history": repayment_history,
            "loan_amount_norm": loan_amount_norm,
            "gdp_indicator": np.clip(gdp_indicator, 0, 1),
            "unemployment_indicator": unemployment_indicator,
            "demographic_group": demographic_group,
            "language_group": language_group,
            "sentiment_embedding": sentiment_embedding,
        }

    def _build_observation(self) -> np.ndarray:
        """Build the composite observation vector from the current client."""
        c = self._current_client
        tabular = np.array([
            c["credit_score"],
            c["dti_ratio"],
            c["income_stability"],
            c["repayment_history"],
            c["loan_amount_norm"],
            c["gdp_indicator"],
            c["unemployment_indicator"],
        ], dtype=np.float32)

        if self.config["use_nlp_features"]:
            return np.concatenate([tabular, c["sentiment_embedding"]])
        return tabular

    def _decode_action(self, action) -> float:
        """Convert the agent's action into an interest rate."""
        if self.config["use_continuous_actions"]:
            delta = float(np.squeeze(action))
            return self._current_rate + delta
        else:
            # Map discrete tier to rate
            min_r = self.config["min_rate"]
            max_r = self.config["max_rate"]
            n_tiers = self.config["num_rate_tiers"]
            return min_r + (max_r - min_r) * (int(action) / max(n_tiers - 1, 1))

    def _compute_default_probability(
        self, client: dict[str, Any], rate: float
    ) -> float:
        """
        Compute the probability of default given the client profile and offered rate.

        Higher rates → higher default probability (affordability stress).
        Lower credit/income → higher baseline default risk.
        Sentiment distress → additional default risk.
        """
        # Baseline risk from financial features
        baseline = (
            0.3 * (1.0 - client["credit_score"])
            + 0.25 * client["dti_ratio"]
            + 0.15 * (1.0 - client["income_stability"])
            + 0.2 * (1.0 - client["repayment_history"])
            + 0.1 * client["unemployment_indicator"]
        )

        # Rate sensitivity: higher rate → more stress
        rate_norm = (rate - self.config["min_rate"]) / (
            self.config["max_rate"] - self.config["min_rate"]
        )
        rate_impact = 0.3 * rate_norm ** 1.5  # Non-linear: extreme rates hurt more

        # Sentiment impact (mean of embedding as a crude distress proxy)
        if self.config["use_nlp_features"] and len(client["sentiment_embedding"]) > 0:
            sentiment_distress = float(np.mean(client["sentiment_embedding"]))
            sentiment_impact = 0.1 * np.clip(sentiment_distress, 0, 1)
        else:
            sentiment_impact = 0.0

        p_default = np.clip(baseline + rate_impact + sentiment_impact, 0.01, 0.95)
        return float(p_default)

    def _compute_reward(
        self,
        rate: float,
        default_prob: float,
        did_default: bool,
        client: dict[str, Any],
    ) -> float:
        """
        Multi-objective reward with normalization.

        R = w1 * InterestEarned - w2 * DefaultRisk + w3 * ClientUtility - OpportunityCost

        All components normalized to [-1, 1].
        """
        w1 = self.config["w1"]
        w2 = self.config["w2"]
        w3 = self.config["w3"]


        max_yield = self.config["max_rate"]
        if not did_default:
            interest_earned = rate / max_yield  # [0, 1]
        else:
            # Partial recovery on default (assume 30% recovery)
            interest_earned = -0.7  # Loss normalized


        # Using log-bounded penalty to prevent catastrophic Q-value collapse
        if did_default:
            default_penalty = np.log1p(default_prob) / np.log(2)  # bounded [0, 1]
        else:
            default_penalty = 0.1 * default_prob  # Small ongoing risk awareness


        # Lower rates = better for client; penalize language barriers
        rate_norm = (rate - self.config["min_rate"]) / (
            self.config["max_rate"] - self.config["min_rate"]
        )
        client_utility = 1.0 - rate_norm  # Higher when rate is lower

        # Accessibility penalty for low-resource language clients
        if client.get("language_group", 0) == 1:
            client_utility -= self.config["accessibility_penalty"]

        client_utility = np.clip(client_utility, -1, 1)


        # Penalize overly conservative behavior (very low rates → low yield)
        opportunity_cost = self.config["opportunity_cost"] * (1.0 - rate_norm)


        reward = (
            w1 * interest_earned
            - w2 * default_penalty
            + w3 * client_utility
            - opportunity_cost
        )

        return float(reward)

    def _build_info(self) -> dict[str, Any]:
        """Build the info dict with metadata for fairness tracking."""
        return {
            "step": self._step_count,
            "demographic_group": self._current_client.get("demographic_group", 0),
            "language_group": self._current_client.get("language_group", 0),
            "cumulative_reward": self._cumulative_reward,
            "episode_defaults": self._episode_defaults,
            "episode_approvals": self._episode_approvals,
        }
