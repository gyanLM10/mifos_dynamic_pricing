"""
Fairness Metrics — Computes Statistical Parity Difference (SPD) and
Disparate Impact (DI) ratio across demographic and language groups.

Also provides a Stable Baselines 3 callback for live fairness monitoring
during training.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


def statistical_parity_difference(
    decisions: np.ndarray,
    group_labels: np.ndarray,
    favorable_value: int = 1,
) -> float:
    """
    Compute Statistical Parity Difference (SPD).

    SPD = P(Ŷ = favorable | A = minority) - P(Ŷ = favorable | A = majority)

    Parameters
    ----------
    decisions : array-like of int
        Binary decisions (1 = favorable, e.g., loan approved at good rate)
    group_labels : array-like of int
        Group membership (0 = majority, 1 = minority)
    favorable_value : int
        The value in decisions that represents a favorable outcome.

    Returns
    -------
    float
        SPD value. 0.0 = perfect parity. Positive = minority favored.
    """
    decisions = np.asarray(decisions)
    group_labels = np.asarray(group_labels)

    minority_mask = group_labels == 1
    majority_mask = group_labels == 0

    if minority_mask.sum() == 0 or majority_mask.sum() == 0:
        return 0.0

    p_minority = (decisions[minority_mask] == favorable_value).mean()
    p_majority = (decisions[majority_mask] == favorable_value).mean()

    return float(p_minority - p_majority)


def disparate_impact_ratio(
    decisions: np.ndarray,
    group_labels: np.ndarray,
    favorable_value: int = 1,
) -> float:
    """
    Compute Disparate Impact (DI) Ratio.

    DI = P(Ŷ = favorable | A = minority) / P(Ŷ = favorable | A = majority)

    Target range: 0.8 ≤ DI ≤ 1.25

    Returns
    -------
    float
        DI ratio. 1.0 = perfect parity.
    """
    decisions = np.asarray(decisions)
    group_labels = np.asarray(group_labels)

    minority_mask = group_labels == 1
    majority_mask = group_labels == 0

    if minority_mask.sum() == 0 or majority_mask.sum() == 0:
        return 1.0

    p_minority = (decisions[minority_mask] == favorable_value).mean()
    p_majority = (decisions[majority_mask] == favorable_value).mean()

    if p_majority == 0:
        return float("inf") if p_minority > 0 else 1.0

    return float(p_minority / p_majority)


def compute_fairness_report(
    rates: np.ndarray,
    groups: np.ndarray,
    language_groups: np.ndarray | None = None,
    favorable_rate_threshold: float = 0.15,
) -> dict[str, Any]:
    """
    Compute a comprehensive fairness report.

    Parameters
    ----------
    rates : array of float
        Interest rates offered by the policy.
    groups : array of int
        Demographic group (0 = majority, 1 = minority).
    language_groups : array of int, optional
        Language group (0 = high-resource, 1 = low-resource).
    favorable_rate_threshold : float
        Rates below this are considered "favorable" for clients.
    """
    rates = np.asarray(rates)
    groups = np.asarray(groups)

    # Convert rates to favorable/unfavorable decisions
    favorable = (rates <= favorable_rate_threshold).astype(int)

    report: dict[str, Any] = {
        "demographic": {
            "spd": statistical_parity_difference(favorable, groups),
            "di_ratio": disparate_impact_ratio(favorable, groups),
            "avg_rate_majority": float(rates[groups == 0].mean()) if (groups == 0).any() else None,
            "avg_rate_minority": float(rates[groups == 1].mean()) if (groups == 1).any() else None,
            "n_majority": int((groups == 0).sum()),
            "n_minority": int((groups == 1).sum()),
        },
    }

    # Assess DI compliance
    di = report["demographic"]["di_ratio"]
    report["demographic"]["di_compliant"] = 0.8 <= di <= 1.25
    report["demographic"]["spd_compliant"] = abs(report["demographic"]["spd"]) <= 0.1

    # Language group fairness
    if language_groups is not None:
        language_groups = np.asarray(language_groups)
        lang_favorable = favorable.copy()
        report["language"] = {
            "spd": statistical_parity_difference(lang_favorable, language_groups),
            "di_ratio": disparate_impact_ratio(lang_favorable, language_groups),
            "avg_rate_hrl": float(rates[language_groups == 0].mean()) if (language_groups == 0).any() else None,
            "avg_rate_lrl": float(rates[language_groups == 1].mean()) if (language_groups == 1).any() else None,
        }
        lang_di = report["language"]["di_ratio"]
        report["language"]["di_compliant"] = 0.8 <= lang_di <= 1.25

    return report


try:
    from stable_baselines3.common.callbacks import BaseCallback as _BaseCallback
except ImportError:
    _BaseCallback = object  # type: ignore[assignment,misc]


class FairnessMonitorCallback(_BaseCallback):
    """
    Stable Baselines 3 callback that tracks fairness metrics during training.
    Logs SPD and DI ratio for both demographic and language groups.
    """

    def __init__(self, log_interval: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._rates: list[float] = []
        self._demo_groups: list[int] = []
        self._lang_groups: list[int] = []
        self._last_logged = -1

    def _on_step(self) -> bool:
        # Collect data from info dicts
        infos = self.locals.get("infos", [])
        for info in infos:
            if "offered_rate" in info:
                self._rates.append(info["offered_rate"])
                self._demo_groups.append(info.get("demographic_group", 0))
                self._lang_groups.append(info.get("language_group", 0))

        # Periodic reporting
        if (
            self.num_timesteps - self._last_logged >= self.log_interval
            and len(self._rates) >= 50
        ):
            self._last_logged = self.num_timesteps

            report = compute_fairness_report(
                np.array(self._rates[-1000:]),
                np.array(self._demo_groups[-1000:]),
                np.array(self._lang_groups[-1000:]),
            )

            demo = report["demographic"]
            self.logger.record("fairness/demo_spd", demo["spd"])
            self.logger.record("fairness/demo_di", demo["di_ratio"])
            self.logger.record("fairness/demo_compliant", float(demo["di_compliant"]))

            if "language" in report:
                lang = report["language"]
                self.logger.record("fairness/lang_spd", lang["spd"])
                self.logger.record("fairness/lang_di", lang["di_ratio"])

            if self.verbose > 0:
                logger.info(
                    f"Fairness @ {self.num_timesteps}: "
                    f"Demo SPD={demo['spd']:.4f}, DI={demo['di_ratio']:.4f} "
                    f"({'✓' if demo['di_compliant'] else '✗'})"
                )

        return True
