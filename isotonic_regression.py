"""
Isotonic regression calibration.

Non-parametric post-hoc calibration method. More flexible than temperature
scaling but prone to overfitting on small samples. Applied per-class
(one-vs-rest) for multiclass calibration.

References
----------
Zadrozny, B. & Elkan, C. (2002). Transforming classifier scores into accurate
    multiclass probability estimates. KDD.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from src.calibration.metrics import compute_ece


class IsotonicCalibrator:
    """
    Per-class isotonic regression calibrator for multiclass classifiers.

    Fits one IsotonicRegression per class in a one-vs-rest fashion,
    then renormalises the outputs to sum to 1.

    Parameters
    ----------
    out_of_bounds : str
        How to handle predictions outside [0, 1]. Default 'clip'.

    Attributes
    ----------
    calibrators_ : list of IsotonicRegression
        Fitted calibrators, one per class.
    n_classes_ : int
        Number of classes.
    """

    def __init__(self, out_of_bounds: str = "clip") -> None:
        self.out_of_bounds = out_of_bounds
        self.calibrators_: list[IsotonicRegression] | None = None
        self.n_classes_: int | None = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        """
        Fit calibrators on validation data.

        Parameters
        ----------
        probs : np.ndarray, shape (n_samples, n_classes)
        labels : np.ndarray, shape (n_samples,) — integer class labels

        Returns
        -------
        self
        """
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=int)
        n_classes = probs.shape[1]
        self.n_classes_ = n_classes

        self.calibrators_ = []
        for k in range(n_classes):
            binary_labels = (labels == k).astype(float)
            ir = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                increasing=True,
                out_of_bounds=self.out_of_bounds,
            )
            ir.fit(probs[:, k], binary_labels)
            self.calibrators_.append(ir)

        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration and renormalise.

        Parameters
        ----------
        probs : np.ndarray, shape (n_samples, n_classes)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes) — calibrated and normalised
        """
        if self.calibrators_ is None:
            raise RuntimeError("Call fit() before calibrate().")

        probs = np.asarray(probs, dtype=float)
        calibrated = np.column_stack([
            ir.predict(probs[:, k])
            for k, ir in enumerate(self.calibrators_)
        ])

        # Renormalise to ensure valid probability distribution
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return calibrated / row_sums

    def summary(self, probs: np.ndarray, labels: np.ndarray) -> dict:
        """Return ECE improvement summary."""
        if self.calibrators_ is None:
            raise RuntimeError("Call fit() before summary().")
        calibrated = self.calibrate(probs)
        return {
            "method": "isotonic_regression",
            "ece_before": compute_ece(probs, labels),
            "ece_after": compute_ece(calibrated, labels),
        }
