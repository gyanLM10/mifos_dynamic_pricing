"""
Tests for the safety guardrails and fairness metrics modules.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety.guardrails import CPOGuardrail, RegulatoryBounds, UNESCOInclusionShield
from safety.fairness_metrics import (
    statistical_parity_difference,
    disparate_impact_ratio,
    compute_fairness_report,
)


class TestCPOGuardrail:
    def test_rate_within_bounds_approved(self):
        g = CPOGuardrail()
        result = g.verify(0.15)
        assert result["approved"]
        assert result["adjusted_rate"] == 0.15

    def test_rate_below_minimum_clamped(self):
        g = CPOGuardrail()
        result = g.verify(0.01)
        assert not result["approved"]
        assert result["adjusted_rate"] == 0.02

    def test_rate_above_maximum_clamped(self):
        g = CPOGuardrail()
        result = g.verify(0.50)
        assert not result["approved"]
        assert result["adjusted_rate"] == 0.36

    def test_rate_delta_velocity_check(self):
        g = CPOGuardrail()
        result = g.verify(0.25, previous_rate=0.10)
        assert not result["approved"]
        assert abs(result["adjusted_rate"] - 0.15) < 0.001

    def test_unesco_lrl_shield(self):
        g = CPOGuardrail()
        result = g.verify(0.28, client={"language_group": 1})
        assert not result["approved"]
        assert result["adjusted_rate"] == 0.25

    def test_violation_summary(self):
        g = CPOGuardrail()
        g.verify(0.50)
        g.verify(0.01)
        summary = g.get_violation_summary()
        assert summary["total_violations"] == 2


class TestUNESCOShield:
    def test_lrl_low_confidence_shielded(self):
        shield = UNESCOInclusionShield(confidence_threshold=0.5)
        result = shield.filter_sentiment(
            sentiment_score=-0.8, confidence=0.3, language_code="sw"
        )
        assert result["shielded"]
        assert result["sentiment"] == 0.0

    def test_hrl_not_shielded(self):
        shield = UNESCOInclusionShield()
        result = shield.filter_sentiment(
            sentiment_score=-0.8, confidence=0.3, language_code="en"
        )
        assert not result["shielded"]

    def test_lrl_high_confidence_not_shielded(self):
        shield = UNESCOInclusionShield(confidence_threshold=0.5)
        result = shield.filter_sentiment(
            sentiment_score=-0.5, confidence=0.8, language_code="sw"
        )
        assert not result["shielded"]


class TestFairnessMetrics:
    def test_perfect_parity_spd(self):
        decisions = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        spd = statistical_parity_difference(decisions, groups)
        assert abs(spd) < 0.01

    def test_perfect_parity_di(self):
        decisions = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        di = disparate_impact_ratio(decisions, groups)
        assert abs(di - 1.0) < 0.01

    def test_biased_decisions_detected(self):
        decisions = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        spd = statistical_parity_difference(decisions, groups)
        assert spd < -0.5  # Minority heavily disfavored

    def test_comprehensive_report(self):
        rng = np.random.default_rng(42)
        rates = rng.uniform(0.05, 0.25, size=200)
        groups = (rng.random(200) < 0.3).astype(int)
        lang = (rng.random(200) < 0.25).astype(int)
        report = compute_fairness_report(rates, groups, lang)
        assert "demographic" in report
        assert "language" in report
        assert "di_ratio" in report["demographic"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
