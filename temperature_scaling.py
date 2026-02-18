"""
Temperature scaling for post-hoc classifier calibration.

Temperature scaling is a single-parameter post-hoc calibration method
that divides the pre-softmax logits by a learned scalar T:

    p̂_k = exp(z_k / T) / Σ_j exp(z_j / T)

T > 1 softens the distribution (reduces overconfidence).
T < 1 sharpens the distribution.
T = 1 leaves probabilities unchanged.

This is the recommended first-pass calibration method (Guo et al. 2017)
as it preserves accuracy while improving calibration.

References
----------
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    On calibration of modern neural networks. ICML.
    https://arxiv.org/abs/1706.04599
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from src.calibration.metrics import compute_ece


class TemperatureScaling:
    """
    Post-hoc temperature scaling calibrator.

    Fits a single temperature parameter T by minimising negative log-likelihood
    (NLL) on a held-out validation set. Works on softmax probabilities directly
    (by converting back to log-space) or on logits if provided.

    Parameters
    ----------
    T_init : float, optional
        Initial temperature guess. Default 1.5 (mild softening).
    T_bounds : tuple[float, float]
        Search bounds for T. Default (0.1, 10.0).

    Attributes
    ----------
    T_ : float
        Fitted temperature after calling fit().
    nll_before_ : float
        Negative log-likelihood before calibration.
    nll_after_ : float
        Negative log-likelihood after calibration.

    Examples
    --------
    >>> ts = TemperatureScaling()
    >>> ts.fit(val_probs, val_labels)
    >>> calibrated = ts.calibrate(test_probs)
    >>> print(f"T = {ts.T_:.3f}")
    """

    def __init__(
        self,
        T_init: float = 1.5,
        T_bounds: tuple[float, float] = (0.1, 10.0),
    ) -> None:
        self.T_init = T_init
        self.T_bounds = T_bounds
        self.T_: float | None = None
        self.nll_before_: float | None = None
        self.nll_after_: float | None = None

    def _probs_to_logits(self, probs: np.ndarray) -> np.ndarray:
        """Convert softmax probabilities to log-space (pseudo-logits)."""
        probs = np.clip(probs, 1e-10, 1.0)
        return np.log(probs)

    def _nll(self, T: float, logits: np.ndarray, labels: np.ndarray) -> float:
        """Negative log-likelihood for temperature T."""
        scaled = logits / T
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(labels)), labels].mean()

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        """
        Fit temperature T on validation data.

        Parameters
        ----------
        probs : np.ndarray, shape (n_samples, n_classes)
            Softmax probabilities from the original classifier.
        labels : np.ndarray, shape (n_samples,)
            Integer ground-truth class labels.

        Returns
        -------
        self
        """
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=int)

        logits = self._probs_to_logits(probs)
        self.nll_before_ = self._nll(1.0, logits, labels)

        result = minimize_scalar(
            self._nll,
            args=(logits, labels),
            bounds=self.T_bounds,
            method="bounded",
        )
        self.T_ = float(result.x)
        self.nll_after_ = float(result.fun)

        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to new predictions.

        Parameters
        ----------
        probs : np.ndarray, shape (n_samples, n_classes)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Temperature-scaled probabilities.
        """
        if self.T_ is None:
            raise RuntimeError("Call fit() before calibrate().")

        probs = np.asarray(probs, dtype=float)
        logits = self._probs_to_logits(probs)
        return softmax(logits / self.T_, axis=1)

    def fit_calibrate(
        self, probs: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Convenience: fit on the same data and return calibrated probs."""
        self.fit(probs, labels)
        return self.calibrate(probs)

    def summary(self, probs: np.ndarray, labels: np.ndarray) -> dict:
        """
        Report calibration improvement summary.

        Returns
        -------
        dict with T, ECE before/after, NLL before/after
        """
        if self.T_ is None:
            raise RuntimeError("Call fit() before summary().")

        calibrated = self.calibrate(probs)
        return {
            "temperature": self.T_,
            "ece_before": compute_ece(probs, labels),
            "ece_after": compute_ece(calibrated, labels),
            "nll_before": self.nll_before_,
            "nll_after": self.nll_after_,
        }
