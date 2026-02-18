"""
Unit tests for calibration metrics.

Tests cover:
    - ECE computation correctness on synthetic examples
    - Perfect calibration → ECE = 0
    - Maximum miscalibration → ECE ≈ expected value
    - MCE and Brier Score
    - ECE sensitivity across bin counts

Run with: pytest tests/test_calibration_metrics.py -v
"""

import numpy as np
import pytest

from src.calibration.metrics import (
    compute_ece,
    compute_mce,
    brier_score,
    ece_sensitivity_check,
)


class TestComputeECE:
    def test_perfect_calibration(self):
        """A perfectly calibrated classifier has ECE = 0."""
        rng = np.random.default_rng(42)
        n = 1000
        # Construct perfect calibration: confidence = accuracy
        confidences = rng.uniform(0, 1, n)
        correct = rng.binomial(1, confidences)
        # Build 2-class probs: confidence for correct class
        probs = np.zeros((n, 2))
        probs[:, 0] = confidences
        probs[:, 1] = 1 - confidences
        # If correct=1, predicted class (argmax) is 0; label is 0 for correct
        labels = 1 - correct  # label=0 if correct, label=1 if wrong
        ece = compute_ece(probs, labels, n_bins=10)
        # Perfect calibration → ECE should be low (not exactly 0 due to sampling)
        assert ece < 0.05, f"Expected ECE < 0.05 for near-perfect calibration, got {ece:.4f}"

    def test_worst_case_miscalibration(self):
        """Always-confident-and-wrong classifier has high ECE."""
        n = 200
        # Classifier always predicts class 0 with confidence 0.99, always wrong
        probs = np.zeros((n, 3))
        probs[:, 0] = 0.99
        probs[:, 1] = 0.005
        probs[:, 2] = 0.005
        labels = np.ones(n, dtype=int)  # True class is always 1
        ece = compute_ece(probs, labels, n_bins=10)
        # ECE should be close to 0.99 (confidence ~ 0.99, accuracy ~ 0.0)
        assert ece > 0.8, f"Expected ECE > 0.8 for worst-case, got {ece:.4f}"

    def test_shape_1d(self):
        """1D probability input (binary case) is handled correctly."""
        probs = np.array([0.1, 0.4, 0.6, 0.9, 0.8, 0.3])
        labels = np.array([0, 0, 1, 1, 1, 0])
        ece = compute_ece(probs, labels, n_bins=5)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_shape_2d(self):
        """2D probability input (multiclass) is handled correctly."""
        rng = np.random.default_rng(0)
        probs = rng.dirichlet([2, 1, 1], size=100)
        labels = rng.integers(0, 3, size=100)
        ece = compute_ece(probs, labels, n_bins=10)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0

    def test_output_range(self):
        """ECE is always in [0, 1]."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            probs = rng.dirichlet([1, 1, 1, 1], size=200)
            labels = rng.integers(0, 4, size=200)
            ece = compute_ece(probs, labels)
            assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"

    def test_invalid_probs_shape(self):
        """3D input raises ValueError."""
        probs = np.ones((10, 4, 2))
        labels = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="1D or 2D"):
            compute_ece(probs, labels)

    def test_length_mismatch(self):
        """Mismatched probs/labels raises ValueError."""
        probs = np.ones((10, 3)) / 3
        labels = np.zeros(5, dtype=int)
        with pytest.raises(ValueError, match="same length"):
            compute_ece(probs, labels)

    def test_uniform_count_strategy(self):
        """Uniform-count binning produces valid ECE."""
        rng = np.random.default_rng(99)
        probs = rng.dirichlet([1, 1, 1], size=150)
        labels = rng.integers(0, 3, size=150)
        ece = compute_ece(probs, labels, n_bins=10, strategy="uniform_count")
        assert 0.0 <= ece <= 1.0

    def test_invalid_strategy(self):
        """Unknown binning strategy raises ValueError."""
        probs = np.ones((10, 2)) / 2
        labels = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_ece(probs, labels, strategy="bad_strategy")

    def test_single_sample_bin(self):
        """Single sample in a bin does not cause division by zero."""
        probs = np.array([[0.95, 0.05], [0.55, 0.45]])
        labels = np.array([0, 1])
        ece = compute_ece(probs, labels, n_bins=15)
        assert np.isfinite(ece)

    @pytest.mark.parametrize("n_bins", [5, 10, 15, 20, 25])
    def test_multiple_bin_counts(self, n_bins):
        """ECE computation is stable across different bin counts."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet([2, 1, 1], size=300)
        labels = rng.integers(0, 3, size=300)
        ece = compute_ece(probs, labels, n_bins=n_bins)
        assert 0.0 <= ece <= 1.0


class TestComputeMCE:
    def test_returns_float(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet([1, 1, 1], size=100)
        labels = rng.integers(0, 3, size=100)
        mce = compute_mce(probs, labels)
        assert isinstance(mce, float)
        assert 0.0 <= mce <= 1.0

    def test_mce_geq_ece(self):
        """MCE (max) should be >= ECE (weighted mean) always."""
        rng = np.random.default_rng(5)
        probs = rng.dirichlet([3, 1, 1], size=200)
        labels = rng.integers(0, 3, size=200)
        ece = compute_ece(probs, labels)
        mce = compute_mce(probs, labels)
        assert mce >= ece - 1e-9, f"MCE ({mce:.4f}) should be >= ECE ({ece:.4f})"


class TestBrierScore:
    def test_perfect_predictions(self):
        """Perfect predictions → Brier score = 0."""
        n = 50
        probs = np.zeros((n, 3))
        labels = np.zeros(n, dtype=int)
        probs[:, 0] = 1.0  # Always predict class 0 with certainty
        bs = brier_score(probs, labels)
        assert abs(bs) < 1e-9, f"Expected 0 for perfect predictions, got {bs}"

    def test_uniform_predictions(self):
        """Uniform (1/K) predictions have known Brier score."""
        n, k = 100, 4
        probs = np.full((n, k), 1.0 / k)
        labels = np.zeros(n, dtype=int)
        bs = brier_score(probs, labels)
        # Each sample: (1 - 1/4)^2 + 3 * (0 - 1/4)^2 = 9/16 + 3/16 = 0.75
        assert abs(bs - 0.75) < 1e-6, f"Expected 0.75 for uniform preds, got {bs}"

    def test_output_range(self):
        """Brier score in [0, 2] for multiclass."""
        rng = np.random.default_rng(0)
        probs = rng.dirichlet([1, 1, 1], size=100)
        labels = rng.integers(0, 3, size=100)
        bs = brier_score(probs, labels)
        assert 0.0 <= bs <= 2.0


class TestECESensitivityCheck:
    def test_returns_dict(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet([1, 1, 1], size=150)
        labels = rng.integers(0, 3, size=150)
        result = ece_sensitivity_check(probs, labels, bin_counts=(10, 15, 20))
        assert set(result.keys()) == {10, 15, 20}
        for v in result.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_qualitative_consistency(self):
        """Overconfident classifier has ECE > 0.1 across all bin counts."""
        n = 300
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.95
        probs[:, 1] = 0.05
        labels = np.ones(n, dtype=int)  # Always wrong
        result = ece_sensitivity_check(probs, labels)
        for n_bins, ece in result.items():
            assert ece > 0.5, (
                f"Expected high ECE for overconfident classifier at n_bins={n_bins}, got {ece:.4f}"
            )
