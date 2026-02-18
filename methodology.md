# Methodology Notes: Baseline Calibration Audit (v0)

*Internal documentation of methodological decisions, assumptions, and known limitations.*

---

## Overview

This document records the methodological decisions and acknowledged limitations
of the Phase 1 baseline calibration audit. Transparent documentation of limitations
is essential for scientific integrity and should be addressed in Phase 1 publications.

---

## Dataset Construction

### Source

Classifier outputs were retrieved from the **ALeRCE API** (Sánchez-Sáez et al. 2021)
for ZTF transients with spectroscopic classifications recorded in the
**Transient Name Server (TNS)**.

### Cross-matching Procedure

1. Retrieved ALeRCE transient classifier outputs for ZTF objects classified by the
   `lc_classifier_transient` module (Carrasco-Davis et al. 2021).
2. Performed cone search (radius = 2 arcsec) against TNS to identify spectroscopically
   confirmed objects.
3. Filtered to objects where TNS classification maps to one of four audit classes:
   **SN Ia, SN II, TDE, AGN**.
4. Final sample: **815 objects** after removing ambiguous, unclassified, or
   multi-epoch classification conflicts.

### Class Distribution

| Class | Count | Fraction |
|---|---|---|
| SN Ia | 312 | 38.3% |
| SN II | 247 | 30.3% |
| AGN | 163 | 20.0% |
| TDE | 93 | 11.4% |

Note: Class imbalance reflects both intrinsic rates and spectroscopic follow-up priorities.

---

## Known Limitations and Selection Biases

### 1. Spectroscopic Selection Bias (Critical)

**The spectroscopic sample is NOT a random sample of ZTF transients.**

Objects with TNS spectroscopic classifications are systematically biased toward:
- Brighter events (easier to obtain spectra)
- Scientifically "interesting" events (human astronomers selected them)
- Events near existing spectroscopic campaigns
- Objects in well-studied fields

**Consequence:** ECE estimates from this sample may not generalise to the full
alert stream. The miscalibration we observe (ECE = 0.297) is measured on a
*non-representative* subset of the population the classifier will encounter.

**Mitigation:** Phase 1 paper will report ECE for each class separately, discuss
selection bias explicitly, and compare with PLAsTiCC simulation results where
ground truth is unbiased. The audit is framed as establishing a *lower bound*
on miscalibration concern, not an unbiased estimate.

### 2. Temporal Dependency

ZTF alerts are time-series objects; the same astrophysical transient appears
in multiple alerts as it evolves. Our calibration analysis treats each
*object* (not alert) as an independent sample using the final/peak-brightness
classifier output. This prevents data leakage but means the classifier has seen
more photometric data than at the time of initial classification.

**Consequence:** Calibration performance at early classification (first 3-5 epochs)
may be worse than reported. Early-time calibration will be evaluated separately.

### 3. Classifier Version Pinning

ALeRCE periodically updates its classification models. Results are pinned to
the model version active during the data collection window. Model updates may
change calibration characteristics.

**Mitigation:** Model version and data collection date documented in
`results/baseline_audit_v0/metadata.json`.

### 4. ECE Binning Sensitivity

ECE estimates depend on the number of bins chosen. We report ECE across
three binning choices (10, 15, 20 bins) to demonstrate robustness of
qualitative findings. The headline ECE = 0.297 uses 15-bin equal-width
binning, consistent with Guo et al. (2017) and Nixon et al. (2019) practice.

### 5. Class Mapping Simplification

ALeRCE's full transient taxonomy includes 8 classes (SN Ia, SN II, SN IIn,
SN Ibc, TDE, AGN, CV/Nova, SLSN). We collapsed this to 4 classes for the
baseline audit to maximise spectroscopic label availability. This may obscure
intra-class miscalibration (e.g., SN Ia vs SN Ibc confusion).

---

## ECE Computation Details

Following Guo et al. (2017), ECE is computed using **top-label confidence**
(confidence = softmax probability of the predicted class).

For multiclass calibration, we additionally report **class-conditional ECE**:
treating each class as a binary one-vs-rest problem. This reveals which
specific classes are most miscalibrated and is more actionable for
targeted calibration methods.

### Sensitivity Check Results

| n_bins | ECE (ALeRCE, all classes) | Qualitative finding |
|---|---|---|
| 10 | 0.281 | Substantial miscalibration |
| 15 | 0.297 | Substantial miscalibration |
| 20 | 0.304 | Substantial miscalibration |

Qualitative finding (substantial overconfidence at high confidence levels)
is robust across all tested bin counts.

---

## Post-Hoc Calibration Methods

### Temperature Scaling

Single parameter T fitted by minimising NLL on a 20% held-out validation split.
Full dataset (n=815) split: 652 fit, 163 held-out.

- Optimisation: `scipy.optimize.minimize_scalar` with bounded search T ∈ [0.1, 10.0]
- Fitted temperature: T ≈ 2.3 (substantial softening)
- Interpretation: ALeRCE classifier is significantly overconfident; T > 2 indicates
  the raw confidence scores need substantial reduction.

### Isotonic Regression

Non-parametric calibration fitted on same validation split as temperature scaling.
Implemented via `sklearn.isotonic.IsotonicRegression` with `out_of_bounds='clip'`.

### Platt Scaling

Logistic regression fitted on softmax probabilities (per-class, one-vs-rest).
More flexible than temperature scaling but prone to overfitting on small samples.

### Comparison

Isotonic regression achieves the lowest post-calibration ECE (≈ 0.11) but
is more susceptible to overfitting than temperature scaling. For Phase 1
publication, we will evaluate on a fully held-out test set to prevent
over-optimistic calibration improvement estimates.

---

## What These Results Do and Do Not Show

**Do show:**
- ALeRCE's transient classifier outputs are substantially miscalibrated on
  spectroscopically confirmed ZTF transients
- Post-hoc calibration can improve ECE meaningfully
- There is a genuine research gap: no broker publishes calibration metrics

**Do NOT show:**
- The magnitude of miscalibration in the full, unbiased ZTF alert stream
- That miscalibration *caused* incorrect follow-up decisions (counterfactual)
- That our post-hoc methods generalise across classifier updates
- Publication-ready results (this is a v0 baseline audit)

---

## References

Sánchez-Sáez et al. (2021). Alert Classification for the ALeRCE Broker System.
    AJ 161, 141. https://arxiv.org/abs/2008.03312

Carrasco-Davis et al. (2021). Alert Classification for the ALeRCE Broker System:
    The Light Curve Classifier. AJ 162, 231. https://arxiv.org/abs/2008.03311

Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
    https://arxiv.org/abs/1706.04599

Nixon et al. (2019). Measuring Calibration in Deep Learning. CVPR Workshops.
    https://arxiv.org/abs/1904.01685
