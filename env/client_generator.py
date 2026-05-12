"""
Client Generator — Synthetic client profile generation calibrated with
configurable distributions. Can be swapped with MifosCalibrator for
historically-accurate distributions.
"""

from __future__ import annotations

import numpy as np
from typing import Any


class SyntheticClientGenerator:
    """
    Generates synthetic client profiles with configurable demographic
    distributions for use in the MicroLoanEnv.
    """

    def __init__(
        self,
        sentiment_dim: int = 768,
        minority_ratio: float = 0.3,
        low_resource_lang_ratio: float = 0.25,
        use_nlp_features: bool = True,
    ):
        self.sentiment_dim = sentiment_dim
        self.minority_ratio = minority_ratio
        self.low_resource_lang_ratio = low_resource_lang_ratio
        self.use_nlp_features = use_nlp_features

    def generate(self, rng: np.random.Generator) -> dict[str, Any]:
        """Generate a single client profile."""
        credit_score = float(rng.beta(5, 2))
        dti_ratio = float(rng.beta(2, 5))
        income_stability = float(rng.beta(4, 2))
        repayment_history = float(rng.beta(6, 2))
        loan_amount_norm = float(rng.beta(2, 3))
        gdp_indicator = float(np.clip(rng.normal(0.5, 0.1), 0, 1))
        unemployment_indicator = float(rng.beta(2, 8))

        demographic_group = int(rng.random() < self.minority_ratio)
        language_group = int(rng.random() < self.low_resource_lang_ratio)

        # Generate sentiment embedding correlated with financial distress
        if self.use_nlp_features:
            distress_level = 1.0 - (
                credit_score * 0.4 + income_stability * 0.3 +
                repayment_history * 0.3
            )
            sentiment_embedding = rng.normal(
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
            "gdp_indicator": gdp_indicator,
            "unemployment_indicator": unemployment_indicator,
            "demographic_group": demographic_group,
            "language_group": language_group,
            "sentiment_embedding": sentiment_embedding,
        }

    def generate_batch(
        self, rng: np.random.Generator, n: int
    ) -> list[dict[str, Any]]:
        """Generate a batch of n client profiles."""
        return [self.generate(rng) for _ in range(n)]


class CalibratedClientGenerator:
    """
    Generates client profiles calibrated from historical Mifos X data.
    Uses fitted distribution parameters rather than hardcoded betas.
    """

    def __init__(
        self,
        distribution_params: dict[str, Any],
        sentiment_dim: int = 768,
        use_nlp_features: bool = True,
    ):
        """
        Parameters
        ----------
        distribution_params : dict
            Pre-fitted distribution parameters from MifosCalibrator.
            Expected keys: 'credit_score', 'dti_ratio', etc., each with
            'alpha' and 'beta' for beta distributions.
        """
        self.params = distribution_params
        self.sentiment_dim = sentiment_dim
        self.use_nlp_features = use_nlp_features

    def generate(self, rng: np.random.Generator) -> dict[str, Any]:
        """Generate a client from calibrated distributions."""
        def _sample_beta(key: str) -> float:
            p = self.params.get(key, {"alpha": 2, "beta": 2})
            return float(rng.beta(p["alpha"], p["beta"]))

        credit_score = _sample_beta("credit_score")
        dti_ratio = _sample_beta("dti_ratio")
        income_stability = _sample_beta("income_stability")
        repayment_history = _sample_beta("repayment_history")
        loan_amount_norm = _sample_beta("loan_amount_norm")

        gdp = self.params.get("gdp_indicator", {"mean": 0.5, "std": 0.1})
        gdp_indicator = float(np.clip(rng.normal(gdp["mean"], gdp["std"]), 0, 1))

        unemployment_indicator = _sample_beta("unemployment_indicator")

        demo = self.params.get("minority_ratio", 0.3)
        lang = self.params.get("low_resource_lang_ratio", 0.25)
        demographic_group = int(rng.random() < demo)
        language_group = int(rng.random() < lang)

        if self.use_nlp_features:
            distress_level = 1.0 - (
                credit_score * 0.4 + income_stability * 0.3 +
                repayment_history * 0.3
            )
            sentiment_embedding = rng.normal(
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
            "gdp_indicator": gdp_indicator,
            "unemployment_indicator": unemployment_indicator,
            "demographic_group": demographic_group,
            "language_group": language_group,
            "sentiment_embedding": sentiment_embedding,
        }
