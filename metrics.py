"""
Calibration metrics for transient classifiers.

Implements Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
reliability diagrams, and Brier Score following:

    Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
    Nixon et al. (2019). Measuring Calibration in Deep Learning. CVPR Workshops.

Usage
-----
    from src.calibration.metrics import compute_ece, reliability_diagram
    ece = compute_ece(probs, labels, n_bins=15)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform_width",
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE is defined as the weighted mean absolute difference between
    predicted confidence and empirical accuracy across confidence bins:

        ECE = Σ_b (|B_b| / n) * |acc(B_b) - conf(B_b)|

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_classes) or (n_samples,)
        Predicted class probabilities. If 2D, the max confidence
        (argmax probability) is used.
    labels : np.ndarray, shape (n_samples,)
        Integer ground-truth class labels.
    n_bins : int, default 15
        Number of confidence bins. Sensitivity should be checked across
        10, 15, and 20 bins (Nixon et al. 2019 recommendation).
    strategy : str, one of {"uniform_width", "uniform_count"}
        Binning strategy. "uniform_width" uses equal-width bins [0, 1/n, ..., 1].
        "uniform_count" uses equal-count (quantile) bins.

    Returns
    -------
    float
        ECE value in [0, 1]. Lower is better. Values > 0.1 indicate
        substantial miscalibration.

    Notes
    -----
    For multi-class outputs, we adopt the confidence-accuracy formulation
    (top-label ECE), which measures calibration of the predicted class.
    Class-conditional ECE (per-class reliability) can be computed by
    calling this function on each class's binary predictions separately.

    References
    ----------
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
        On calibration of modern neural networks. ICML.
        https://arxiv.org/abs/1706.04599

    Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G., & Tran, D. (2019).
        Measuring calibration in deep learning. CVPR Workshops.
        https://arxiv.org/abs/1904.01685
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    elif probs.ndim == 1:
        confidences = probs
        predictions = (probs >= 0.5).astype(int)
    else:
        raise ValueError(f"probs must be 1D or 2D, got shape {probs.shape}")

    if len(confidences) != len(labels):
        raise ValueError(
            f"probs and labels must have same length, got "
            f"{len(confidences)} and {len(labels)}"
        )

    accuracies = (predictions == labels).astype(float)

    if strategy == "uniform_width":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "uniform_count":
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(confidences, percentiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'uniform_width' or 'uniform_count'.")

    ece = 0.0
    n_samples = len(confidences)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        # Include right endpoint only in last bin to cover confidence = 1.0
        if i < n_bins - 1:
            mask = (confidences >= low) & (confidences < high)
        else:
            mask = (confidences >= low) & (confidences <= high)

        if mask.sum() == 0:
            continue

        bin_accuracy = accuracies[mask].mean()
        bin_confidence = confidences[mask].mean()
        bin_weight = mask.sum() / n_samples

        ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_mce(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE = max_b |acc(B_b) - conf(B_b)|

    More sensitive to worst-case miscalibration than ECE.
    Relevant for high-confidence deferral decisions where individual
    bin errors matter more than the weighted average.

    Parameters
    ----------
    probs, labels, n_bins : same as compute_ece

    Returns
    -------
    float
        MCE value in [0, 1].
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(int)

    accuracies = (predictions == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    mce = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= low) & (confidences <= high)
        if mask.sum() == 0:
            continue
        gap = abs(accuracies[mask].mean() - confidences[mask].mean())
        mce = max(mce, gap)

    return float(mce)


def brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
    n_classes: Optional[int] = None,
) -> float:
    """
    Compute the multiclass Brier Score.

    BS = (1/n) Σ_i Σ_k (p_ik - y_ik)^2

    where y_ik is the one-hot encoding of the true label.
    Decomposed into: BS = Reliability - Resolution + Uncertainty.

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_classes)
    labels : np.ndarray, shape (n_samples,) — integer class labels
    n_classes : int, optional. Inferred from probs if not provided.

    Returns
    -------
    float
        Brier score in [0, 2] for multiclass (lower is better).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if n_classes is None:
        n_classes = probs.shape[1] if probs.ndim == 2 else 2

    one_hot = np.zeros((len(labels), n_classes))
    one_hot[np.arange(len(labels)), labels] = 1.0

    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    class_name: Optional[str] = None,
    save_path: Optional[str] = None,
) -> dict:
    """
    Compute and optionally plot a reliability diagram.

    Returns bin-level statistics as a dictionary for programmatic use,
    and renders the diagram if a save path is provided.

    Parameters
    ----------
    probs : np.ndarray, 1D (binary) or 2D (multiclass, uses top-1 confidence)
    labels : np.ndarray, integer ground-truth labels
    n_bins : int, number of equal-width confidence bins
    title : str, plot title
    class_name : str, optional class label for per-class diagrams
    save_path : str, optional path to save the figure (PNG/PDF)

    Returns
    -------
    dict with keys:
        bin_midpoints, bin_accuracies, bin_confidences, bin_counts, ece
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    if probs.ndim == 2:
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(int)

    accuracies = (predictions == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (confidences >= low) & (confidences < high)
        else:
            mask = (confidences >= low) & (confidences <= high)
        if mask.sum() > 0:
            bin_accuracies[i] = accuracies[mask].mean()
            bin_confidences[i] = confidences[mask].mean()
            bin_counts[i] = mask.sum()

    ece = compute_ece(probs, labels, n_bins=n_bins)

    if save_path is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: reliability diagram
        ax = axes[0]
        bar_positions = bin_midpoints[bin_counts > 0]
        bar_heights = bin_accuracies[bin_counts > 0]
        bar_width = 1.0 / n_bins

        ax.bar(bar_positions, bar_heights, width=bar_width * 0.9,
               alpha=0.7, color="#4878CF", label="Accuracy", align="center")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
        ax.fill_between(
            bar_positions,
            bar_positions,
            bar_heights,
            alpha=0.2,
            color="red",
            label=f"Gap (ECE={ece:.3f})",
        )
        ax.set_xlabel("Confidence", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(title + (f" — {class_name}" if class_name else ""), fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Right: confidence histogram
        ax2 = axes[1]
        ax2.bar(
            bin_midpoints[bin_counts > 0],
            bin_counts[bin_counts > 0],
            width=bar_width * 0.9,
            alpha=0.7,
            color="#6ACC65",
            align="center",
        )
        ax2.set_xlabel("Confidence", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Confidence Distribution", fontsize=13)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return {
        "bin_midpoints": bin_midpoints,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "ece": ece,
    }


def ece_sensitivity_check(
    probs: np.ndarray,
    labels: np.ndarray,
    bin_counts: tuple[int, ...] = (10, 15, 20),
) -> dict[int, float]:
    """
    Compute ECE across multiple bin counts to assess sensitivity.

    Following Nixon et al. (2019), ECE estimates can vary with bin choice.
    Qualitative findings should be robust across bin counts.

    Parameters
    ----------
    probs, labels : as in compute_ece
    bin_counts : tuple of int, bin counts to evaluate

    Returns
    -------
    dict mapping n_bins → ECE value
    """
    return {n: compute_ece(probs, labels, n_bins=n) for n in bin_counts}
