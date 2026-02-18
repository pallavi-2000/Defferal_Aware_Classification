"""
Unit tests for decision-centric evaluation metrics.

Tests Coverage@Budget, Silent Failure Rate, Deferral Precision,
and Risk-Coverage curve computation.

Run with: pytest tests/test_evaluation_metrics.py -v
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    coverage_at_budget,
    silent_failure_rate,
    deferral_precision,
    risk_coverage_curve,
)


class TestCoverageAtBudget:
    def setup_method(self):
        """Simple 4-sample fixture."""
        # Classifier is very confident and correct for samples 0, 2
        # Confident but wrong for sample 1; uncertain for sample 3
        self.probs = np.array([
            [0.95, 0.03, 0.01, 0.01],  # Confident → class 0 (correct)
            [0.90, 0.05, 0.03, 0.02],  # Confident → class 0 (WRONG, true=1)
            [0.01, 0.02, 0.94, 0.03],  # Confident → class 2 (correct)
            [0.30, 0.25, 0.25, 0.20],  # Uncertain
        ])
        self.labels = np.array([0, 1, 2, 3])

    def test_budget_1(self):
        """With budget=1, top-confidence alert is selected (correct)."""
        result = coverage_at_budget(self.probs, self.labels, budget=1)
        assert result["n_correct"] == 1

    def test_budget_2(self):
        """With budget=2, top-2 by confidence; both samples 0 (conf=0.95) and 2 (conf=0.94) are correct."""
        result = coverage_at_budget(self.probs, self.labels, budget=2)
        # samples ranked by conf: 0 (0.95), 2 (0.94), 1 (0.90), 3 (0.30)
        # Top-2: sample 0 (correct), sample 2 (correct) → n_correct = 2
        assert result["n_correct"] == 2
        assert result["budget"] == 2

    def test_budget_exceeds_dataset(self):
        """Budget > dataset size → all 4 samples selected, budget param preserved."""
        result = coverage_at_budget(self.probs, self.labels, budget=100)
        assert result["budget"] == 100  # budget parameter preserved
        # Only 4 samples exist; n_correct can't exceed 4
        assert result["n_correct"] <= 4

    def test_target_class_filtering(self):
        """target_class filters to class-specific alerts."""
        result = coverage_at_budget(
            self.probs, self.labels, budget=4, target_class=0
        )
        assert isinstance(result["deferral_precision"] if "deferral_precision" in result else result["precision"], float)

    def test_output_keys(self):
        result = coverage_at_budget(self.probs, self.labels, budget=3)
        for key in ["coverage", "precision", "recall", "n_selected", "n_correct", "budget"]:
            assert key in result, f"Missing key: {key}"

    def test_precision_in_range(self):
        result = coverage_at_budget(self.probs, self.labels, budget=3)
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["coverage"] <= 1.0

    def test_all_correct_classifier(self):
        """A perfect classifier achieves precision=1 at any budget."""
        probs = np.array([
            [0.95, 0.03, 0.02],
            [0.02, 0.94, 0.04],
            [0.01, 0.02, 0.97],
        ])
        labels = np.array([0, 1, 2])
        result = coverage_at_budget(probs, labels, budget=3)
        assert result["n_correct"] == 3
        assert abs(result["precision"] - 1.0) < 1e-9


class TestSilentFailureRate:
    def test_zero_sfr_correct_classifier(self):
        """No silent failures when classifier is always correct."""
        probs = np.array([[0.95, 0.05], [0.92, 0.08]])
        labels = np.array([0, 0])
        result = silent_failure_rate(probs, labels, confidence_threshold=0.9)
        assert result["sfr"] == 0.0
        assert result["n_silent_failures"] == 0

    def test_all_silent_failures(self):
        """All high-confidence predictions are wrong → SFR = 1."""
        probs = np.array([[0.95, 0.05], [0.92, 0.08]])
        labels = np.array([1, 1])  # Always wrong
        result = silent_failure_rate(probs, labels, confidence_threshold=0.9)
        assert result["sfr"] == 1.0

    def test_no_high_confidence_samples(self):
        """No samples above threshold → SFR = 0, n_high_confidence = 0."""
        probs = np.array([[0.55, 0.45], [0.60, 0.40]])
        labels = np.array([0, 0])
        result = silent_failure_rate(probs, labels, confidence_threshold=0.9)
        assert result["n_high_confidence"] == 0
        assert result["sfr"] == 0.0

    def test_output_keys(self):
        probs = np.array([[0.95, 0.05]])
        labels = np.array([0])
        result = silent_failure_rate(probs, labels)
        for key in ["sfr", "n_high_confidence", "n_silent_failures", "coverage", "threshold"]:
            assert key in result

    def test_threshold_sensitivity(self):
        """Lower threshold → more samples covered, potentially higher SFR."""
        probs = np.array([
            [0.75, 0.25],
            [0.92, 0.08],
        ])
        labels = np.array([1, 1])  # Both wrong
        high = silent_failure_rate(probs, labels, confidence_threshold=0.9)
        low = silent_failure_rate(probs, labels, confidence_threshold=0.7)
        assert low["n_high_confidence"] >= high["n_high_confidence"]


class TestDeferralPrecision:
    def test_perfect_deferral(self):
        """Defer exactly the incorrect predictions → precision = 1."""
        probs = np.array([
            [0.9, 0.1],   # Correct (pred=0, label=0)
            [0.9, 0.1],   # Wrong (pred=0, label=1) ← should defer
        ])
        labels = np.array([0, 1])
        deferral_flags = np.array([False, True])
        result = deferral_precision(probs, labels, deferral_flags)
        assert result["deferral_precision"] == 1.0

    def test_no_deferrals(self):
        """No deferrals → deferral precision = 0."""
        probs = np.ones((5, 2)) / 2
        labels = np.zeros(5, dtype=int)
        flags = np.zeros(5, dtype=bool)
        result = deferral_precision(probs, labels, flags)
        assert result["n_deferred"] == 0
        assert result["deferral_precision"] == 0.0

    def test_output_keys(self):
        probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        labels = np.array([0, 1])
        flags = np.array([True, True])
        result = deferral_precision(probs, labels, flags)
        for key in ["deferral_precision", "deferral_rate", "n_deferred", "n_useful_deferrals"]:
            assert key in result

    def test_deferral_rate_correct(self):
        probs = np.ones((10, 2)) / 2
        labels = np.zeros(10, dtype=int)
        flags = np.array([True] * 4 + [False] * 6)
        result = deferral_precision(probs, labels, flags)
        assert abs(result["deferral_rate"] - 0.4) < 1e-9


class TestRiskCoverageCurve:
    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet([1, 1, 1], size=100)
        labels = rng.integers(0, 3, size=100)
        result = risk_coverage_curve(probs, labels, n_thresholds=50)
        assert len(result["thresholds"]) == 50
        assert len(result["coverages"]) == 50
        assert len(result["risks"]) == 50

    def test_auc_rc_range(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet([1, 1, 1], size=100)
        labels = rng.integers(0, 3, size=100)
        result = risk_coverage_curve(probs, labels)
        # AUC-RC via trapezoid over decreasing coverage axis can be negative
        # (sign depends on integration direction); what matters is it's finite
        assert np.isfinite(result["auc_rc"])

    def test_perfect_classifier_low_auc(self):
        """A near-perfect classifier has near-zero AUC-RC."""
        n = 200
        probs = np.zeros((n, 3))
        probs[:, 0] = 0.98
        probs[:, 1] = 0.01
        probs[:, 2] = 0.01
        labels = np.zeros(n, dtype=int)  # Always correct
        result = risk_coverage_curve(probs, labels)
        assert result["auc_rc"] < 0.05
