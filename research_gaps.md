# Research Gap Analysis: Calibration and Deferral in Transient Astronomy

*Evidence-based assessment of the gap between current broker capabilities and research needs.*

---

## The Core Gap

Alert brokers classify transients. They do not tell observers *whether to act* on those classifications.

The distinction matters enormously at the scale of LSST (10⁷ alerts/night vs 10⁵ spectra/year).

---

## Systematic Gap Audit

The following table documents what was checked, where, and what was found.

### ALeRCE

| Capability | Documentation Checked | Finding |
|---|---|---|
| Multi-class classification | Sánchez-Sáez et al. (2021), AJ 161, 141 | ✓ Well-documented (8 classes) |
| Classifier architecture | Carrasco-Davis et al. (2021), AJ 162, 231 | ✓ BHRF + Balanced RF |
| Uncertainty quantification | Both papers + ALeRCE API docs | ✗ Not discussed |
| Calibration metrics (ECE, reliability) | Both papers, GitHub, API schema | ✗ Absent |
| Confidence threshold recommendations | User docs, tutorials | ✗ None published |
| Deferral framework | All ALeRCE publications | ✗ Not present |

**API output schema** includes `probabilities` for each class, but no confidence
interval, calibration flag, or deferral recommendation. Users receive raw softmax
probabilities with no guidance on how to threshold or trust them.

### Fink

| Capability | Documentation Checked | Finding |
|---|---|---|
| Multi-class classification | Möller et al. (2021), MNRAS | ✓ SuperNNova + ParSNIP |
| Active learning for training | Leoni et al. (2022) | ✓ Active in production |
| "Classification uncertainties" | Möller et al. (2021) | ⚠️ Mentioned but not evaluated |
| ECE / reliability diagrams | All Fink publications | ✗ Not published |
| Deferral mechanism | All Fink publications | ✗ Not present |

**Note on Fink's uncertainty claim:** Möller et al. (2021) refer to "classification
uncertainties" from SuperNNova's Bayesian framework, but no paper evaluates these
uncertainties using calibration metrics (ECE, reliability diagrams). Bayesian
uncertainty and calibration are related but distinct: a model can produce
uncertainty estimates that are themselves poorly calibrated.

### ANTARES

| Capability | Documentation Checked | Finding |
|---|---|---|
| Alert stream filtering | Narayan et al. (2018), PASP | ✓ Production |
| Machine learning classification | Matheson et al. (2021) | ✓ Active |
| Calibration analysis | All ANTARES publications | ✗ Absent |
| Deferral | All ANTARES publications | ✗ Absent |

### Lasair

| Capability | Documentation Checked | Finding |
|---|---|---|
| ZTF alert stream | Smith et al. (2019) | ✓ Production |
| Sherlock contextual classification | Smith et al. (2019) | ✓ Rule-based |
| ML-based classification | — | Primarily rule-based; limited ML |
| Calibration / deferral | — | ✗ Not applicable (rule-based) |

---

## The Machine Learning Calibration Literature

Calibration is a mature subfield of ML, yet it has not been applied to
transient astronomy brokers. Key methods that are available but unused:

| Method | Reference | Applied to Astronomy? |
|---|---|---|
| Temperature scaling | Guo et al. (2017) ICML | ✗ No |
| Platt scaling | Platt (1999) | ✗ No |
| Isotonic regression | Zadrozny & Elkan (2002) | ✗ No |
| Beta calibration | Kull et al. (2017) AISTATS | ✗ No |
| Dirichlet calibration | Kull et al. (2019) NeurIPS | ✗ No |
| Histogram binning | Zadrozny & Elkan (2001) KDD | ✗ No |
| Ensemble calibration | Lakshminarayanan et al. (2017) | ✗ No |

---

## Learning-to-Defer: Entirely Absent from Astronomy

Learning-to-defer (L2D) is a framework for training models to explicitly
decide whether to make a prediction or defer to a human expert. It is
an active research area in ML but has zero published applications in
time-domain astronomy.

| L2D Approach | Reference | Applied to Astronomy? |
|---|---|---|
| Selective prediction | Geifman & El-Yaniv (2017) | ✗ No |
| Learning to reject | Cortes et al. (2016) | ✗ No |
| Joint prediction + deferral | Madras et al. (2018) NeurIPS | ✗ No |
| Consistent L2D estimators | Mozannar & Sontag (2020) ICML | ✗ No |
| Cost-sensitive deferral | Verma & Nalisnick (2022) | ✗ No |

The astronomy use case is a near-perfect fit for L2D:
- Human experts (astronomers) with genuine skill
- Expensive human action (spectroscopy) with opportunity cost
- Asymmetric error costs (missing a kilonova >> missing an SN Ia)
- Hard budget constraints (10⁵ spectra/year)

---

## Why Now?

The research gap has existed for years but is becoming **critical** with LSST:

| Factor | ZTF (current) | LSST (2025+) |
|---|---|---|
| Alerts per night | ~1 million | ~10 million |
| Spectroscopic capacity | ~100,000/year | ~100,000/year (fixed) |
| Mismatch ratio | 3,650:1 | 36,500:1 |
| Automated decision requirement | Optional | Mandatory |

At LSST scale, every spectroscopic allocation decision must be defensible.
A framework that can say "we deferred this alert because confidence = 0.73,
below the threshold where our model is reliable" is far more useful than
one that silently makes a wrong call.

---

## References

Carrasco-Davis et al. (2021). ALeRCE Light Curve Classifier. AJ 162, 231.
    https://arxiv.org/abs/2008.03311

Geifman, Y. & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks.
    NeurIPS. https://arxiv.org/abs/1705.08500

Guo et al. (2017). On Calibration of Modern Neural Networks. ICML.
    https://arxiv.org/abs/1706.04599

Madras et al. (2018). Predict Responsibly. NeurIPS. https://arxiv.org/abs/1802.09010

Matheson et al. (2021). The ANTARES Astronomical Alert Broker. AJ 161, 107.

Möller et al. (2021). SuperNNova: Bayesian supernova classification.
    MNRAS. https://arxiv.org/abs/1901.06384

Mozannar & Sontag (2020). Consistent Estimators for L2D. ICML.
    https://arxiv.org/abs/2006.01862

Narayan et al. (2018). Machine Learning-based Brokers. ApJS 236, 9.

Nixon et al. (2019). Measuring Calibration in Deep Learning. CVPR Workshops.
    https://arxiv.org/abs/1904.01685

Sánchez-Sáez et al. (2021). ALeRCE Stamp Classifier. AJ 161, 141.
    https://arxiv.org/abs/2008.03312

Smith et al. (2019). Lasair: The Alert Broker for LSST:UK. RNAAS 3, 26.
