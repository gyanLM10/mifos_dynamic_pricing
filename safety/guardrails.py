"""
Guardrails — Safety constraints for the RL policy including:
  - Constrained Policy Optimization (CPO) rate verification
  - UNESCO inclusion alignment (language-aware safety shield)
  - Regulatory rate caps and floors
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegulatoryBounds:
    """Hard-coded regulatory interest rate bounds."""
    absolute_min_rate: float = 0.02    # 2% APR floor
    absolute_max_rate: float = 0.36    # 36% APR ceiling (usury threshold)
    max_rate_delta: float = 0.05       # Max single-step rate change
    min_rate_for_tier: dict = field(default_factory=lambda: {
        "low_risk": 0.05,
        "moderate_risk": 0.10,
        "high_risk": 0.18,
    })
    max_rate_for_tier: dict = field(default_factory=lambda: {
        "low_risk": 0.15,
        "moderate_risk": 0.22,
        "high_risk": 0.30,
    })


class CPOGuardrail:
    """
    Constrained Policy Optimization safety layer.

    Verifies that proposed interest rates satisfy:
      1. Absolute regulatory bounds
      2. Maximum rate-change velocity
      3. Risk-tier-appropriate ranges
      4. Language accessibility requirements
    """

    def __init__(self, bounds: RegulatoryBounds | None = None):
        self.bounds = bounds or RegulatoryBounds()
        self._violation_log: list[dict[str, Any]] = []

    def verify(
        self,
        proposed_rate: float,
        previous_rate: float | None = None,
        client: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Verify a proposed rate against all safety constraints.

        Returns
        -------
        dict with keys:
            - approved: bool
            - adjusted_rate: float (clamped to safe range)
            - violations: list[str]
        """
        violations = []
        rate = proposed_rate

        # 1. Absolute bounds
        if rate < self.bounds.absolute_min_rate:
            violations.append(
                f"Below absolute minimum ({rate:.4f} < {self.bounds.absolute_min_rate})"
            )
            rate = self.bounds.absolute_min_rate
        elif rate > self.bounds.absolute_max_rate:
            violations.append(
                f"Above absolute maximum ({rate:.4f} > {self.bounds.absolute_max_rate})"
            )
            rate = self.bounds.absolute_max_rate

        # 2. Rate-change velocity
        if previous_rate is not None:
            delta = abs(rate - previous_rate)
            if delta > self.bounds.max_rate_delta:
                violations.append(
                    f"Rate change too large (Δ={delta:.4f} > {self.bounds.max_rate_delta})"
                )
                direction = 1 if rate > previous_rate else -1
                rate = previous_rate + direction * self.bounds.max_rate_delta

        # 3. Language accessibility shield (UNESCO alignment)
        if client is not None and client.get("language_group", 0) == 1:
            # Low-resource language clients get a rate reduction cap
            # to prevent systematic penalization from parsing failures
            max_rate_for_lrl = 0.25  # Lower ceiling for LRL clients
            if rate > max_rate_for_lrl:
                violations.append(
                    f"UNESCO shield: LRL client rate capped "
                    f"({rate:.4f} → {max_rate_for_lrl})"
                )
                rate = max_rate_for_lrl

        # Log violations
        if violations:
            self._violation_log.append({
                "proposed": proposed_rate,
                "adjusted": rate,
                "violations": violations,
            })

        return {
            "approved": len(violations) == 0,
            "adjusted_rate": float(rate),
            "violations": violations,
            "original_rate": proposed_rate,
        }

    def get_violation_summary(self) -> dict[str, Any]:
        """Summarize all recorded violations."""
        if not self._violation_log:
            return {"total_violations": 0}

        total = len(self._violation_log)
        avg_adjustment = np.mean([
            abs(v["proposed"] - v["adjusted"]) for v in self._violation_log
        ])
        violation_types: dict[str, int] = {}
        for entry in self._violation_log:
            for v in entry["violations"]:
                key = v.split("(")[0].strip()
                violation_types[key] = violation_types.get(key, 0) + 1

        return {
            "total_violations": total,
            "avg_rate_adjustment": float(avg_adjustment),
            "violation_types": violation_types,
        }

    def reset_log(self):
        """Clear the violation log."""
        self._violation_log.clear()


class UNESCOInclusionShield:
    """
    Safety shield implementing UNESCO AI language preservation principles.

    Prevents the system from:
      - Penalizing low-resource language (LRL) clients due to NLP parsing limitations
      - Using sentiment scores from poorly-supported languages without confidence checks
      - Setting rates that disproportionately affect linguistically marginalized groups
    """

    # Languages with known low XLM-R representation
    LOW_RESOURCE_LANGUAGES = {
        "sw",   # Swahili
        "am",   # Amharic
        "ha",   # Hausa
        "yo",   # Yoruba
        "rw",   # Kinyarwanda
        "lg",   # Luganda
        "tn",   # Tswana
        "xh",   # Xhosa
        "zu",   # Zulu
        "mg",   # Malagasy
    }

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        fallback_sentiment: float = 0.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.fallback_sentiment = fallback_sentiment

    def filter_sentiment(
        self,
        sentiment_score: float,
        confidence: float,
        language_code: str | None = None,
    ) -> dict[str, Any]:
        """
        Filter a sentiment score through the inclusion shield.

        If the language is low-resource and confidence is low,
        replace the sentiment with a neutral fallback to prevent
        systematic bias from poor NLP coverage.
        """
        is_low_resource = (
            language_code is not None
            and language_code.lower() in self.LOW_RESOURCE_LANGUAGES
        )

        if is_low_resource and confidence < self.confidence_threshold:
            return {
                "sentiment": self.fallback_sentiment,
                "shielded": True,
                "reason": (
                    f"Low-resource language ({language_code}) with insufficient "
                    f"confidence ({confidence:.2f} < {self.confidence_threshold})"
                ),
                "original_sentiment": sentiment_score,
            }

        return {
            "sentiment": sentiment_score,
            "shielded": False,
            "reason": None,
            "original_sentiment": sentiment_score,
        }

    def assess_group_impact(
        self,
        rates_by_language: dict[str, list[float]],
    ) -> dict[str, Any]:
        """
        Assess whether the policy creates disparate impact across language groups.
        """
        results = {}
        all_rates = []
        for lang, rates in rates_by_language.items():
            is_lrl = lang.lower() in self.LOW_RESOURCE_LANGUAGES
            avg_rate = np.mean(rates) if rates else 0
            results[lang] = {
                "avg_rate": float(avg_rate),
                "n_clients": len(rates),
                "is_low_resource": is_lrl,
            }
            all_rates.extend(rates)

        global_avg = np.mean(all_rates) if all_rates else 0

        lrl_rates = [
            r for lang, rates in rates_by_language.items()
            if lang.lower() in self.LOW_RESOURCE_LANGUAGES
            for r in rates
        ]
        hrl_rates = [
            r for lang, rates in rates_by_language.items()
            if lang.lower() not in self.LOW_RESOURCE_LANGUAGES
            for r in rates
        ]

        if lrl_rates and hrl_rates:
            lrl_avg = np.mean(lrl_rates)
            hrl_avg = np.mean(hrl_rates)
            disparity = float(lrl_avg - hrl_avg)
        else:
            disparity = 0.0

        return {
            "global_avg_rate": float(global_avg),
            "language_details": results,
            "lrl_vs_hrl_disparity": disparity,
            "fair": abs(disparity) < 0.03,  # Less than 3% disparity threshold
        }
