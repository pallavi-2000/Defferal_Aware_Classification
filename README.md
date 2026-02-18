# Deferral-Aware Transient Classification for Spectroscopic Follow-up Prioritization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Pallavi Sati** |Project Repository  

---

## The Problem

The Vera C. Rubin Observatory's LSST will generate approximately **10 million transient alerts per night**. Global spectroscopic follow-up capacity is fixed at **~100,000 spectra per year** — a mismatch of **10,000:1**.

Current alert brokers classify transients but provide predictions *without decision-making capability*. They answer **"What is this?"** but not **"Should we look closer?"**

This project develops a **deferral-aware classification framework** that learns *when* to defer decisions to human astronomers and spectroscopic confirmation.

---

## Repository Structure

```
.
├── notebooks/
│   └── 02_alerce_calibration_audit.ipynb   ← Full Phase 1 analysis with live outputs
├── src/
│   ├── data/               # ALeRCE API client, ZTF BTS ingestion
│   ├── calibration/        # ECE, reliability diagrams, temperature scaling, isotonic regression
│   ├── deferral/           # Learning-to-defer framework (Phase 3, planned)
│   └── evaluation/         # Coverage@Budget, Silent Failure Rate, AUC-RC
├── tests/                  # 41 unit tests, all passing
├── results/
│   └── baseline_audit_v0/
│       ├── figures/        # Reliability diagram, confidence analysis
│       └── metadata.json
└── docs/
    ├── methodology.md      # Methodological decisions and known limitations
    └── research_gaps.md    # Evidence-based audit of existing broker capabilities
```

---

## Phase 1 Results: Baseline Calibration Audit

> **Framing:** Phase 1 baseline audit — exploratory results demonstrating research execution capability. See [`docs/methodology.md`](docs/methodology.md) for full discussion of limitations and selection biases. The notebook shows the complete methodology with all outputs.

### Dataset

| Property | Value |
|---|---|
| Source | ZTF Bright Transient Survey (BTS) × ALeRCE LC Classifier |
| Ground truth | Spectroscopic types from ZTF BTS catalogue |
| Sample size | **815 spectroscopically confirmed transients** |
| Original pool | 10,181 ZTF BTS objects (70.1% spectroscopically typed) |
| Class distribution | SNIa (504), SNII (217), SNIbc (64), SLSN (15), TDE (15) |
| Collection date | January 2026 |
| Known selection bias | BTS sample is brightness-biased; see methodology notes |

### Key Finding: ALeRCE is Significantly Underconfident

| Metric | Value |
|---|---|
| Overall accuracy | **0.750** (611 / 815 correct) |
| ECE | **0.2968** — significantly miscalibrated |
| Mean confidence | 0.453 |
| Mean accuracy | 0.742 |
| Calibration gap | **−0.240** (accuracy consistently exceeds confidence) |
| Direction | **Underconfident** |
| Confidence range | [0.144, 0.880] |

The classifier is *underconfident* across all confidence bins: ALeRCE outputs lower probability scores than its accuracy warrants. This is the inverse of typical deep learning overconfidence (Guo et al. 2017) and has a distinct operational consequence — the system suppresses its own signal, causing unnecessary deferrals and missed opportunities to confidently confirm rare transients.

### Reliability Diagram

![Reliability Diagram](results/baseline_audit_v0/figures/reliability_diagram.png)

*The red line lies above the perfect-calibration diagonal (dashed) across all bins — systematic underconfidence. Error bars show 95% binomial confidence intervals.*

**Per-bin breakdown:**

| Confidence Bin | Mean Confidence | Observed Accuracy | Gap | N |
|---|---|---|---|---|
| 0.1–0.2 | 0.176 | 0.235 | −0.059 | 17 |
| 0.2–0.3 | 0.263 | 0.500 | −0.237 | 50 |
| 0.3–0.4 | 0.354 | 0.629 | −0.275 | 202 |
| 0.4–0.5 | 0.451 | 0.759 | −0.309 | 266 |
| 0.5–0.6 | 0.546 | 0.892 | **−0.346** | 194 |
| 0.6–0.7 | 0.637 | 0.924 | −0.287 | 79 |
| 0.7–0.8 | 0.738 | 1.000 | −0.262 | 5 |
| 0.8–0.9 | 0.857 | 1.000 | −0.143 | 2 |

The gap peaks at the 0.5–0.6 bin (gap = −0.346): predictions with ~55% confidence are correct ~89% of the time. Crucially, the highest-confidence predictions in this dataset (0.7–0.9) are 100% accurate — yet the classifier never outputs confidence > 0.88, severely limiting its usability for automated allocation.

### Confidence Distribution

![Confidence Analysis](results/baseline_audit_v0/figures/confidence_analysis.png)

### Per-Class Calibration

| Class | N | Accuracy | ECE | Status |
|---|---|---|---|---|
| SNIa | 504 | 0.843 | 0.3882 | Miscalibrated |
| SNII | 217 | 0.544 | 0.1166 | Miscalibrated |
| SNIbc | 64 | 0.875 | 0.3814 | Miscalibrated |

SNIa and SNIbc show the largest ECE, both strongly underconfident relative to their actual accuracy. SNII's lower ECE reflects genuine photometric ambiguity between stripped-envelope subtypes.

### Silent Failure Analysis

| Confidence Threshold | High-Conf Predictions | Silent Failures | SFR | Coverage |
|---|---|---|---|---|
| ≥ 0.50 | 282 (34.6%) | 28 | 3.4% | 34.6% |
| ≥ 0.60 | 88 (10.8%) | 6 | 0.7% | 10.8% |
| ≥ 0.70 | 7 (0.9%) | 0 | **0.0%** | 0.9% |
| ≥ 0.80 | 2 (0.2%) | 0 | **0.0%** | 0.2% |

Silent failures are eliminated at threshold ≥ 0.70, but coverage collapses to <1%. This is the core deferral design tension: **the threshold that eliminates dangerous errors also eliminates nearly all automated decisions.** A learned deferral policy is needed to find the Pareto-optimal operating point.

---

## Interpreting the Underconfidence Finding

Underconfidence is equally problematic as overconfidence for automated follow-up, but requires different remediation:

| Problem | Failure Mode | Operational Consequence |
|---|---|---|
| **Overconfidence** (confidence > accuracy) | Confident wrong predictions | Silent failures, wasted spectra |
| **Underconfidence** (accuracy > confidence) | Doubting correct predictions | Unnecessary deferrals, missed rare events |

A kilonova correctly classified at 55% confidence would be unnecessarily deferred under any reasonable threshold policy — even though the classifier is right ~89% of the time at that confidence level. Post-hoc calibration (Phase 1) and learned deferral with asymmetric costs (Phase 3) are both motivated by this finding.

---

## Project Status

| Phase | Status | Description |
|---|---|---|
| **1a** | ✅ Complete | Baseline ECE audit — 815 ZTF BTS transients |
| **1b** | 🔄 In progress | Post-hoc calibration: temperature scaling, isotonic regression |
| **2** | 📋 Planned | Deep ensemble uncertainty quantification (PLAsTiCC / ELAsTiCC) |
| **3** | 📋 Planned | Learning-to-defer framework (Madras et al. 2018; Mozannar & Sontag 2020) |
| **4** | 📋 Planned | Spectroscopic utility prediction |
| **5** | 📋 Planned | Integration with ALeRCE / Fink, LSST deployment |

---

## Novel Contributions

1. **First calibration benchmark of a production transient broker** — ECE measurement, reliability diagrams, and per-class analysis on 815 spectroscopically confirmed ZTF transients. Underconfidence identified as the dominant failure mode.

2. **First learning-to-defer framework for time-domain astronomy** — Learned policy P(defer | features) following [Madras et al. (2018)](https://arxiv.org/abs/1802.09010) and [Mozannar & Sontag (2020)](https://arxiv.org/abs/2006.01862).

3. **Decision-centric evaluation metrics** — Coverage@Budget, Silent Failure Rate, and Deferral Precision aligned to the actual operational resource allocation problem.

4. **Spectroscopic utility prediction** — Value-of-information model for proactive follow-up allocation.

---

## Evaluation Metrics

| Metric | Definition | Target |
|---|---|---|
| ECE | Expected Calibration Error: weighted mean \|confidence − accuracy\| | < 5% |
| Coverage@Budget | Correct classifications within fixed spectroscopic budget B | Maximize |
| Silent Failure Rate | P(confident AND wrong) | < 1% |
| Deferral Precision | P(spectroscopy informative \| deferred) | > 80% |
| AUC-RC | Area under Risk-Coverage curve | Maximize |

---

## Installation

```bash
git clone https://github.com/pallavi-2000/Defferal_Aware_Classification.git
cd Defferal_Aware_Classification
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Open the calibration audit notebook
jupyter notebook notebooks/02_alerce_calibration_audit.ipynb
```

---

## Key References

- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Nixon et al. (2019). *Measuring Calibration in Deep Learning.* CVPR Workshops. [arXiv:1904.01685](https://arxiv.org/abs/1904.01685)
- Madras et al. (2018). *Predict Responsibly.* NeurIPS. [arXiv:1802.09010](https://arxiv.org/abs/1802.09010)
- Mozannar & Sontag (2020). *Consistent Estimators for L2D.* ICML. [arXiv:2006.01862](https://arxiv.org/abs/2006.01862)
- Sánchez-Sáez et al. (2021). *ALeRCE Alert Classification.* AJ 161, 141. [arXiv:2008.03312](https://arxiv.org/abs/2008.03312)
- Carrasco-Davis et al. (2021). *ALeRCE Light Curve Classifier.* AJ 162, 231. [arXiv:2008.03311](https://arxiv.org/abs/2008.03311)
- Fremling et al. (2020). *ZTF Bright Transient Survey I.* ApJ 895, 32. [arXiv:1910.12973](https://arxiv.org/abs/1910.12973)

---

## License

MIT — see [LICENSE](LICENSE).
