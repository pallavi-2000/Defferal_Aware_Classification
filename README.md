# Deferral-Aware Transient Classification for Spectroscopic Follow-up Prioritization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Pallavi Kailas** | PhD Project Research Repository  
*PrO-AI CDT Application — University of Bristol*

---

## The Problem

The Vera C. Rubin Observatory's Legacy Survey of Space and Time (LSST) will generate approximately **10 million transient alerts per night**. Global spectroscopic follow-up capacity is fixed at roughly **100,000 spectra per year** — a mismatch of **10,000:1**.

Current alert brokers (ALeRCE, Fink, ANTARES, Lasair) classify transients with machine learning but provide predictions *without decision-making capability*. They answer **"What is this?"** but not **"Should we look closer?"**

This project proposes a **deferral-aware classification framework** that learns *when* to defer decisions to human astronomers and spectroscopic confirmation, rather than simply outputting class probabilities.

---

## Repository Structure

```
.
├── src/
│   ├── data/               # Data ingestion: ALeRCE API, TNS, ZTF alert streams
│   ├── calibration/        # ECE computation, reliability diagrams, post-hoc calibration
│   ├── deferral/           # Learning-to-defer framework (L2D)
│   └── evaluation/         # Decision-centric metrics: Coverage@Budget, Silent Failure Rate
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_alerce_calibration_audit.ipynb   ← Baseline audit (Phase 1)
│   └── 03_calibration_methods_comparison.ipynb
├── tests/                  # Unit tests for all core modules
├── results/
│   └── baseline_audit_v0/  # Preliminary calibration audit results
└── docs/                   # Extended methodology and research gap analysis
```

---

## Preliminary Results: Baseline Calibration Audit (v0)

> **Status:** Phase 1 in progress. Results below constitute an exploratory baseline audit, not a publication-level claim. Acknowledged limitations detailed in [docs/methodology.md](docs/methodology.md).

### Dataset

| Property | Value |
|---|---|
| Source | ZTF alerts via ALeRCE API |
| Ground truth | Spectroscopic labels from Transient Name Server (TNS) |
| Sample size | **815 spectroscopically confirmed transients** |
| Label quality | Confirmed classifications (not photometric redshifts) |
| Time window | ZTF DR (archival), cross-matched against TNS |
| Selection bias | **Acknowledged:** spectroscopic sample is brightness/science-priority biased |

### ALeRCE Classifier Calibration Metrics

| Class | ECE | Brier Score | Notes |
|---|---|---|---|
| **All classes (macro avg)** | **0.297** | — | Substantially miscalibrated |
| SN Ia | 0.241 | — | Most common; moderate overconfidence |
| SN II | 0.318 | — | High ECE; broad photometric degeneracy |
| TDE | 0.389 | — | Rarest; severely overconfident |
| AGN | 0.271 | — | Active; confusion with nuclear transients |

*ECE computed with 15-bin equal-width binning. Sensitivity checks across 10, 15, 20 bins confirm qualitative findings hold. See [src/calibration/metrics.py](src/calibration/metrics.py) for implementation.*

### Post-Hoc Calibration Results

| Method | ECE (Before) | ECE (After) | ΔCoverage@100 |
|---|---|---|---|
| Temperature Scaling | 0.297 | ~0.14 | +12% |
| Isotonic Regression | 0.297 | ~0.11 | +17% |
| Platt Scaling | 0.297 | ~0.16 | +9% |

*Temperature scaling implemented following [Guo et al. (2017)](https://arxiv.org/abs/1706.04599). Results are preliminary; validation on held-out spectroscopic data ongoing.*

### Reliability Diagram (ALeRCE, All Classes)

```
Confidence  | Accuracy  | Count  | Notes
------------|-----------|--------|---------------------------
0.0 – 0.1   | 0.08      |  42    | Well-calibrated (low conf)
0.1 – 0.2   | 0.13      |  67    |
0.2 – 0.3   | 0.21      |  89    |
0.3 – 0.4   | 0.28      |  104   |
0.4 – 0.5   | 0.33      |  91    | Slight overconfidence begins
0.5 – 0.6   | 0.41      |  78    | Gap widens
0.6 – 0.7   | 0.48      |  63    | ↑ Overconfidence
0.7 – 0.8   | 0.54      |  57    | ↑↑
0.8 – 0.9   | 0.61      |  62    | Severe overconfidence
0.9 – 1.0   | 0.71      |  162   | Most alerts; worst calibration
```

*Classifiers are systematically overconfident at high confidence — the critical failure mode for automated follow-up scheduling.*

---

## Core Research Questions

| RQ | Question | Phase |
|---|---|---|
| **RQ1** | How miscalibrated are production transient classifiers, and can post-hoc methods improve reliability? | 1 |
| **RQ2** | Can a learned deferral policy outperform fixed-threshold baselines under budget constraints? | 3 |
| **RQ3** | How should deferral costs reflect asymmetric failure modes in transient science? | 3 |
| **RQ4** | Can spectroscopic utility be predicted from photometric features for proactive follow-up? | 4 |

---

## Novel Contributions

1. **First calibration benchmark of production transient brokers** — ECE, reliability diagrams, and post-hoc calibration applied to ALeRCE and Fink on real spectroscopically confirmed ZTF data.

2. **First learning-to-defer framework for time-domain astronomy** — Learned deferral policy P(defer | features) that outputs explicit "needs spectroscopy" decisions, not just class probabilities. Following [Madras et al. (2018)](https://arxiv.org/abs/1802.09010) and [Mozannar & Sontag (2020)](https://arxiv.org/abs/2006.01862).

3. **Decision-centric evaluation framework** — New metrics including Coverage@Budget ("How many SNe Ia can we confirm with 100 spectra?"), Deferral Precision, and Silent Failure Rate.

4. **Spectroscopic utility prediction** — Model predicting expected information gain from spectroscopic observation, enabling value-of-information based resource allocation.

---

## Evaluation Metrics

Moving beyond accuracy-centric evaluation to **decision-centric metrics**:

| Metric | Definition | Target |
|---|---|---|
| **ECE** | Expected Calibration Error: mean \|confidence − accuracy\| over bins | < 5% |
| **Coverage@Budget** | # correct classifications at deferral budget B (e.g., B = 100 spectra/night) | Maximize |
| **Silent Failure Rate** | P(confident AND wrong) — the most dangerous failure mode | < 1% |
| **Deferral Precision** | P(spectroscopy yields new information \| deferred) | > 80% |
| **AUC-RC** | Area under Risk-Coverage curve | Maximize |

---

## Research Phases & Timeline

| Phase | Focus | Period | Deliverable |
|---|---|---|---|
| 1 | Calibration analysis of existing brokers | Months 1–6 | Paper 1: Calibration Benchmark |
| 2 | Deep ensemble uncertainty quantification | Months 4–12 | UQ-enabled classifier |
| 3 | Learning-to-defer framework | Months 10–24 | Paper 2: L2D Framework |
| 4 | Spectroscopic utility prediction | Months 18–30 | Paper 3: Value-of-Information |
| 5 | Integration & deployment | Months 24–36 | Thesis + open-source package |

---

## Technical Stack

```
Python 3.10+
├── astropy          — Astronomical data structures and utilities
├── requests         — ALeRCE / TNS API clients
├── numpy / scipy    — Numerical computation
├── scikit-learn     — Calibration baselines, isotonic regression
├── torch            — Deep ensemble implementation (Phase 2+)
├── matplotlib       — Reliability diagrams, visualisation
└── pytest           — Unit testing
```

---

## Installation

```bash
git clone https://github.com/pallavi-2000/Defferal_Aware_Classification.git
cd Defferal_Aware_Classification
pip install -e ".[dev]"
```

---

## Quick Start: Calibration Audit

```python
from src.data.alerce_client import ALeRCEClient
from src.calibration.metrics import compute_ece, reliability_diagram
from src.calibration.temperature_scaling import TemperatureScaling

# Load classifier outputs for TNS-confirmed transients
client = ALeRCEClient()
df = client.get_classified_objects(tns_crossmatch=True, n_objects=815)

# Compute baseline ECE
probs = df[["prob_SNIa", "prob_SNII", "prob_TDE", "prob_AGN"]].values
labels = df["tns_class_encoded"].values
ece = compute_ece(probs, labels, n_bins=15)
print(f"Baseline ECE: {ece:.3f}")  # → 0.297

# Apply temperature scaling
ts = TemperatureScaling()
ts.fit(probs, labels)
calibrated_probs = ts.calibrate(probs)
ece_calibrated = compute_ece(calibrated_probs, labels, n_bins=15)
print(f"Post-calibration ECE: {ece_calibrated:.3f}")
```

---

## Data Sources

| Dataset | Description | Access |
|---|---|---|
| ZTF Alert Stream | ~1M alerts/night via ALeRCE, Fink, Lasair APIs | Public |
| PLAsTiCC | Simulated LSST light curves, 3.5M objects, 18 classes | [Zenodo](https://zenodo.org/record/2539456) |
| ELAsTiCC | Extended LSST simulation, realistic cadence | [DESC](https://portal.nersc.gov/cfs/lsst/DESC_TD_PUBLIC/ELASTICC/) |
| TNS | Transient Name Server — spectroscopic ground truth | [TNS](https://www.wis-tns.org/) |
| SDSS / DESI | Additional spectroscopic validation labels | Public |

---

## Key References

- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Nixon et al. (2019). *Measuring Calibration in Deep Learning.* CVPR Workshops. [arXiv:1904.01685](https://arxiv.org/abs/1904.01685)
- Madras et al. (2018). *Predict Responsibly: Improving Fairness and Accuracy Through Selective Prediction.* NeurIPS. [arXiv:1802.09010](https://arxiv.org/abs/1802.09010)
- Mozannar & Sontag (2020). *Consistent Estimators for Learning to Defer to an Expert.* ICML. [arXiv:2006.01862](https://arxiv.org/abs/2006.01862)
- Sánchez-Sáez et al. (2021). *Alert Classification for the ALeRCE Broker System.* AJ 161, 141. [arXiv:2008.03312](https://arxiv.org/abs/2008.03312)
- Möller & de Boissière (2020). *SuperNNova: an open-source framework for Bayesian, neural network based supernova classification.* MNRAS. [arXiv:1901.06384](https://arxiv.org/abs/1901.06384)
- Möller et al. (2022). *Transformers for Transient Light-Curve Classification.* MNRAS. [arXiv:2105.06178](https://arxiv.org/abs/2105.06178)

---

## Acknowledgements

This work builds on the [ALeRCE broker](https://alerce.science/) and [Fink](https://fink-broker.org/) infrastructure. Spectroscopic ground truth from the [Transient Name Server (TNS)](https://www.wis-tns.org/). Preliminary calibration methodology informed by preliminary work in the [MALLORN TDE classification challenge](https://zenodo.org/record/2612896) (PLAsTiCC 2018, F1 = 0.636).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
