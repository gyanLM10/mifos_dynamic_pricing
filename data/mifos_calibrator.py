"""
Mifos X Data Calibrator — Loads historical loan data and fits statistical
distributions to calibrate the MicroLoanEnv's client generator.

Supports:
  - CSV export ingestion from Mifos X
  - Beta distribution fitting via MLE for bounded features
  - K-S test for distribution fidelity validation
  - Export of fitted parameters for CalibratedClientGenerator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# Feature definitions: name → (normalization range, distribution family)
FEATURE_SPEC = {
    "credit_score":         {"min": 0, "max": 1,    "dist": "beta"},
    "dti_ratio":            {"min": 0, "max": 1,    "dist": "beta"},
    "income_stability":     {"min": 0, "max": 1,    "dist": "beta"},
    "repayment_history":    {"min": 0, "max": 1,    "dist": "beta"},
    "loan_amount_norm":     {"min": 0, "max": 1,    "dist": "beta"},
    "gdp_indicator":        {"min": 0, "max": 1,    "dist": "normal"},
    "unemployment_indicator": {"min": 0, "max": 1,  "dist": "beta"},
}

# Column mappings from common Mifos X CSV exports to our internal features
MIFOS_COLUMN_MAP = {
    # Mifos X column name → internal feature name
    "clientCreditScore": "credit_score",
    "debtToIncomeRatio": "dti_ratio",
    "incomeStabilityIndex": "income_stability",
    "repaymentSuccessRate": "repayment_history",
    "loanPrincipal": "loan_amount_norm",
    "localGdpIndex": "gdp_indicator",
    "localUnemploymentRate": "unemployment_indicator",
    "demographicGroup": "demographic_group",
    "languageCode": "language_group",
}


class MifosCalibrator:
    """
    Calibrates environment distributions from historical Mifos X data.
    """

    def __init__(self, column_map: dict[str, str] | None = None):
        self.column_map = column_map or MIFOS_COLUMN_MAP
        self.fitted_params: dict[str, Any] = {}
        self.raw_data: pd.DataFrame | None = None
        self.ks_results: dict[str, dict] = {}

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """Load and normalize a Mifos X CSV export."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")

        # Rename columns using mapping
        rename_map = {k: v for k, v in self.column_map.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # Normalize loan_amount to [0, 1] if raw
        if "loan_amount_norm" in df.columns:
            col = df["loan_amount_norm"]
            if col.max() > 1.0:
                df["loan_amount_norm"] = (col - col.min()) / (col.max() - col.min())

        self.raw_data = df
        return df

    def fit_distributions(self, df: pd.DataFrame | None = None) -> dict[str, Any]:
        """
        Fit statistical distributions to each feature column.

        Returns a dict of fitted parameters suitable for CalibratedClientGenerator.
        """
        if df is None:
            df = self.raw_data
        if df is None:
            raise ValueError("No data loaded. Call load_csv() first.")

        params: dict[str, Any] = {}

        for feature, spec in FEATURE_SPEC.items():
            if feature not in df.columns:
                logger.warning(f"Feature '{feature}' not found in data, using defaults")
                if spec["dist"] == "beta":
                    params[feature] = {"alpha": 2.0, "beta": 2.0}
                else:
                    params[feature] = {"mean": 0.5, "std": 0.1}
                continue

            values = df[feature].dropna().values
            values = np.clip(values, 0.001, 0.999)  # avoid boundary issues for beta

            if spec["dist"] == "beta":
                alpha, beta_param, _, _ = stats.beta.fit(values, floc=0, fscale=1)
                params[feature] = {"alpha": float(alpha), "beta": float(beta_param)}
                logger.info(
                    f"  {feature}: Beta(α={alpha:.3f}, β={beta_param:.3f})"
                )
            elif spec["dist"] == "normal":
                mean, std = float(np.mean(values)), float(np.std(values))
                params[feature] = {"mean": mean, "std": max(std, 0.01)}
                logger.info(f"  {feature}: Normal(μ={mean:.3f}, σ={std:.3f})")

        # Demographics
        if "demographic_group" in df.columns:
            params["minority_ratio"] = float(df["demographic_group"].mean())
        else:
            params["minority_ratio"] = 0.3

        if "language_group" in df.columns:
            params["low_resource_lang_ratio"] = float(df["language_group"].mean())
        else:
            params["low_resource_lang_ratio"] = 0.25

        self.fitted_params = params
        return params

    def validate_fit(
        self, df: pd.DataFrame | None = None, n_samples: int = 10000
    ) -> dict[str, dict]:
        """
        Run Kolmogorov-Smirnov tests comparing the fitted distributions
        against the historical data.

        Returns a dict of {feature: {statistic, p_value, passed}} for each feature.
        """
        if df is None:
            df = self.raw_data
        if not self.fitted_params:
            raise ValueError("Must call fit_distributions() first.")

        rng = np.random.default_rng(42)
        results = {}

        for feature, spec in FEATURE_SPEC.items():
            if feature not in df.columns:
                continue

            historical = df[feature].dropna().values
            p = self.fitted_params.get(feature, {})

            if spec["dist"] == "beta":
                synthetic = rng.beta(
                    p.get("alpha", 2), p.get("beta", 2), size=n_samples
                )
            else:
                synthetic = rng.normal(
                    p.get("mean", 0.5), p.get("std", 0.1), size=n_samples
                )

            ks_stat, p_value = stats.ks_2samp(historical, synthetic)
            results[feature] = {
                "statistic": float(ks_stat),
                "p_value": float(p_value),
                "passed": p_value > 0.05,  # Null hypothesis: same distribution
            }
            logger.info(
                f"  K-S {feature}: D={ks_stat:.4f}, p={p_value:.4f} "
                f"{'✓' if p_value > 0.05 else '✗'}"
            )

        self.ks_results = results
        return results

    def export_params(self, path: str | Path) -> None:
        """Export fitted parameters to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.fitted_params, f, indent=2)
        logger.info(f"Exported fitted parameters to {path}")

    def load_params(self, path: str | Path) -> dict[str, Any]:
        """Load previously fitted parameters from JSON."""
        path = Path(path)
        with open(path) as f:
            self.fitted_params = json.load(f)
        return self.fitted_params


def generate_sample_mifos_data(
    path: str | Path, n_clients: int = 500, seed: int = 42
) -> None:
    """
    Generate a sample CSV mimicking a Mifos X loan export for development.
    """
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "clientId": list(range(1, n_clients + 1)),
        "clientCreditScore": rng.beta(5, 2, n_clients),
        "debtToIncomeRatio": rng.beta(2, 5, n_clients),
        "incomeStabilityIndex": rng.beta(4, 2, n_clients),
        "repaymentSuccessRate": rng.beta(6, 2, n_clients),
        "loanPrincipal": rng.uniform(100, 10000, n_clients),
        "localGdpIndex": np.clip(rng.normal(0.5, 0.1, n_clients), 0, 1),
        "localUnemploymentRate": rng.beta(2, 8, n_clients),
        "demographicGroup": (rng.random(n_clients) < 0.3).astype(int),
        "languageCode": (rng.random(n_clients) < 0.25).astype(int),
        "interestRate": rng.uniform(0.05, 0.30, n_clients),
        "didDefault": (rng.random(n_clients) < 0.15).astype(int),
    }

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    logger.info(f"Generated sample Mifos X data: {n_clients} clients → {path}")
