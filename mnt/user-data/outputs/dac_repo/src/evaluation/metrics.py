"""
Decision-centric evaluation metrics for deferral-aware transient classification.

These metrics operationalise the core resource allocation problem:
given a fixed spectroscopic budget B, how effectively can the system
prioritise which alerts to follow up?

Moving beyond accuracy-centric evaluation to metrics that reflect
the actual costs of classification decisions in time-domain astronomy.

Core insight
------------
Standard ML metrics (accuracy, F1, AUC-ROC) treat all errors equally.
In transient astronomy, errors have highly asymmetric costs:

    - Silent failure: classifier is confident AND wrong
      → spectroscope the wrong object → waste precious telescope time
      → MISS the real event (e.g., kilonova early phase)

    - Unnecessary deferral: classifier correctly flags uncertainty
      → human astronomer reviews → correct decision made
      → costs human time but preserves scientific value

References
----------
Geifman, Y. & El-Yaniv, R. (2017). Selective prediction under unknown
    test distribution. JMLR 18(1), 9988–10027.

El-Yaniv, R. & Wiener, Y. (2010). On the foundations of noise-free
    selective classification. JMLR 11, 1605–1641.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def coverage_at_budget(
    probs: np.ndarray,
    labels: np.ndarray,
    budget: int,
    target_class: Optional[int] = None,
) -> dict:
    """
    Coverage@Budget: classification performance within spectroscopic budget.

    Simulates the operational scenario: given B spectroscopic slots per night,
    rank all alerts by classifier confidence and allocate the top-B for follow-up.
    Computes what fraction of those B observations yield correct classifications.

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_classes)
        Classifier probabilities for each alert.
    labels : np.ndarray, shape (n_samples,)
        True integer class labels.
    budget : int
        Maximum number of spectroscopic observations (B).
        Typical values: 100 (nightly), 500 (weekly), 2000 (monthly).
    target_class : int, optional
        If specified, measure coverage only for this class (e.g., SNe Ia = 0).
        Useful for science-case-specific evaluation.

    Returns
    -------
    dict with keys:
        coverage : float — fraction of budget used for correct classifications
        precision : float — accuracy within selected alerts
        recall : float — fraction of total positive class captured
        n_selected : int — alerts within budget
        n_correct : int — correct classifications within budget

    Notes
    -----
    Alerts are ranked by top-1 confidence (max softmax probability).
    In operational deployment, confidence threshold rather than fixed
    budget may be more appropriate; both formulations are evaluated.

    Example
    -------
    "With a budget of 100 spectra, how many SNe Ia can we confirm?"
    >>> result = coverage_at_budget(probs, labels, budget=100, target_class=0)
    >>> print(f"Confirmed {result['n_correct']} SNe Ia with 100 spectra")
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    original_budget = budget
    if budget > len(labels):
        budget = len(labels)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    # Rank by descending confidence
    ranked_idx = np.argsort(confidences)[::-1]
    selected_idx = ranked_idx[:budget]

    selected_labels = labels[selected_idx]
    selected_predictions = predictions[selected_idx]

    if target_class is not None:
        # Filter to cases where prediction is the target class
        target_mask = selected_predictions == target_class
        selected_labels = selected_labels[target_mask]
        selected_predictions = selected_predictions[target_mask]

        total_positive = (labels == target_class).sum()
        n_correct = (selected_labels == target_class).sum()
        recall = n_correct / total_positive if total_positive > 0 else 0.0
    else:
        n_correct = (selected_labels == selected_predictions).sum()
        recall = n_correct / len(labels)

    n_selected = len(selected_labels) if target_class is not None else budget
    precision = n_correct / n_selected if n_selected > 0 else 0.0
    coverage = n_correct / budget

    return {
        "coverage": float(coverage),
        "precision": float(precision),
        "recall": float(recall),
        "n_selected": n_selected,
        "n_correct": int(n_correct),
        "budget": original_budget,
        "target_class": target_class,
    }


def silent_failure_rate(
    probs: np.ndarray,
    labels: np.ndarray,
    confidence_threshold: float = 0.9,
) -> dict:
    """
    Silent Failure Rate: P(confident AND wrong).

    The most dangerous failure mode for automated follow-up:
    the classifier is highly confident but incorrect, leading to:
        - Wasted spectra on the wrong object class
        - Missing the real scientific event (e.g., early-phase kilonova)

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_classes)
    labels : np.ndarray, shape (n_samples,)
    confidence_threshold : float
        Confidence above which predictions are considered "certain".
        Default 0.9 (typical high-confidence operating point).

    Returns
    -------
    dict with keys:
        sfr : float — fraction of high-confidence predictions that are wrong
        n_high_confidence : int
        n_silent_failures : int
        coverage : float — fraction of dataset covered at this threshold

    Notes
    -----
    Target: SFR < 1% at the operating confidence threshold.
    ALeRCE baseline audit finds substantial SFR at threshold=0.9,
    motivating calibration correction and deferral frameworks.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)

    high_conf_mask = confidences >= confidence_threshold
    n_high_confidence = high_conf_mask.sum()

    if n_high_confidence == 0:
        return {
            "sfr": 0.0,
            "n_high_confidence": 0,
            "n_silent_failures": 0,
            "coverage": 0.0,
            "threshold": confidence_threshold,
        }

    high_conf_correct = (predictions[high_conf_mask] == labels[high_conf_mask]).sum()
    n_silent_failures = n_high_confidence - high_conf_correct

    return {
        "sfr": float(n_silent_failures / n_high_confidence),
        "n_high_confidence": int(n_high_confidence),
        "n_silent_failures": int(n_silent_failures),
        "coverage": float(n_high_confidence / len(labels)),
        "threshold": confidence_threshold,
    }


def deferral_precision(
    probs: np.ndarray,
    labels: np.ndarray,
    deferral_flags: np.ndarray,
) -> dict:
    """
    Deferral Precision: P(spectroscopy yields value | deferred).

    Measures the quality of the deferral decision: are the objects
    flagged for spectroscopic follow-up actually the ones where
    a spectrum would be informative?

    "Informative" is operationalised as: the classifier's predicted
    class differs from ground truth (spectroscopy would correct the error),
    OR the classifier's confidence is below the calibration-adjusted
    threshold (uncertainty is genuine).

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_classes)
    labels : np.ndarray, shape (n_samples,)
    deferral_flags : np.ndarray, shape (n_samples,), dtype bool
        Boolean array where True = "defer to spectroscopy".

    Returns
    -------
    dict with keys:
        deferral_precision : float — P(useful | deferred)
        deferral_rate : float — fraction of alerts deferred
        n_deferred : int
        n_useful_deferrals : int
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    deferral_flags = np.asarray(deferral_flags, dtype=bool)

    n_deferred = deferral_flags.sum()
    if n_deferred == 0:
        return {
            "deferral_precision": 0.0,
            "deferral_rate": 0.0,
            "n_deferred": 0,
            "n_useful_deferrals": 0,
        }

    predictions = probs.argmax(axis=1)
    deferred_incorrect = (predictions[deferral_flags] != labels[deferral_flags]).sum()

    return {
        "deferral_precision": float(deferred_incorrect / n_deferred),
        "deferral_rate": float(n_deferred / len(labels)),
        "n_deferred": int(n_deferred),
        "n_useful_deferrals": int(deferred_incorrect),
    }


def risk_coverage_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 100,
) -> dict:
    """
    Compute the Risk-Coverage curve.

    The RC curve traces (coverage, risk) pairs as the confidence threshold
    varies from 1.0 (no predictions) to 0.0 (all predictions):
        - Coverage: fraction of dataset where prediction is made (not deferred)
        - Risk: error rate on that covered subset

    The AUC-RC summarises the curve: lower is better (low risk at high coverage).

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_classes)
    labels : np.ndarray, shape (n_samples,)
    n_thresholds : int
        Resolution of the threshold sweep.

    Returns
    -------
    dict with keys:
        thresholds, coverages, risks, auc_rc
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(float)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    coverages = []
    risks = []

    for threshold in thresholds:
        mask = confidences >= threshold
        coverage = mask.mean()
        if mask.sum() == 0:
            risk = 0.0
        else:
            risk = 1.0 - correct[mask].mean()
        coverages.append(coverage)
        risks.append(risk)

    coverages = np.array(coverages)
    risks = np.array(risks)

    # AUC-RC via trapezoidal integration (lower is better)
    # np.trapz renamed to np.trapezoid in NumPy 2.0
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    auc_rc = float(trapz_fn(risks, coverages))

    return {
        "thresholds": thresholds,
        "coverages": coverages,
        "risks": risks,
        "auc_rc": auc_rc,
    }
