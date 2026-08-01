# Competitive landscape

Scanned 2026-07-26. Sources at the bottom.

## The uncomfortable finding

**bioLeak** is an R package on CRAN, published April 2026, that does most of
what this prototype does — for free, with a wider feature set, and earlier.

| bioLeak | NerveML equivalent |
|---|---|
| leakage-aware split construction | `evaluate_grouped()` |
| train-fold-only preprocessing | pipeline-wrapped scaler |
| cross-validated fitting + nested tuning | ours has no nested tuning |
| post hoc leakage audits | risk rules |
| HTML reporting | JSON + Streamlit |
| **inflation summaries** | **our generalization gap** |
| simulated controlled leakage mechanisms | **our synthetic datasets** |

Its grouping modes are broader than ours: `subject_grouped`, `batch_blocked`,
`study_loocv`, `time_series`. Its task coverage is broader too: binary,
multiclass, regression, survival.

Read its paper and source before doing further work on the validity half. The
`batch_blocked` and `time_series` modes cover dependency structures this
prototype does not handle at all.

Also in this space: `leakr` (R, leakage detection in ML workflows) and
`LeakageDetector` (PyCharm plugin, static analysis of pipeline code).

## Subject-wise CV is becoming free infrastructure

**EEGDash**, an open-source platform for ML on public neurophysiological data,
ships subject-wise cross-validation (GroupKFold, LeaveOneGroupOut) with
dedicated leakage-and-evaluation documentation.

In EEG specifically, "did you hold out subjects?" is on its way to being a
default, not a differentiator.

## General AI audit is funded and crowded

- **Credo AI** — enterprise AI governance; board-ready audit reports and risk
  dashboards. Ranked #6 in Applied AI on Fast Company's Most Innovative
  Companies 2026.
- **Holistic AI** — positions as an algorithmic auditing platform providing
  "deep empirical validation required for deploying high-risk AI systems."
  Named a Representative Vendor in Gartner's first Market Guide for Guardian
  Agents, March 2026.
- **BABL AI** — independent third-party audit firm; EU AI Act preparation,
  ISO 42001, NYC Local Law 144.

Holistic AI's positioning overlaps ours. They lack neural-domain depth, but
adding a vertical is easier for them than building a brand is for us.

## The privacy half is academically active and commercially empty

Membership inference, attribute inference, identifiability scores and
synthetic-data privacy auditing are well developed in the literature.
Synthetic-data vendors expose re-identification risk metrics as a feature.

No vendor was found selling "upload your dataset, we run the attacks and tell
you what can be inferred" as a product. This is the genuine gap.

## Potential partners, not competitors

- **Fraunhofer IDMT — "EEG without Brainprint"** — removing the identity
  fingerprint from EEG.
- **NEMO project** — EEG anonymisation methods, EUSIPCO 2026.

Both build remediation. Remediation has to be shown to work, and showing it
requires exactly the identity-inference probe in spec 6.2. Fraunhofer is also
institutional credibility, which addresses the weakest point in the pitch.

## What this means for positioning

Do not lead with leakage detection. That competes with a free CRAN package and
loses.

Lead with the pairing no one else offers: **one scan that answers both the
validity question and the inference-privacy question, for neural data
specifically** — the two questions that two different regulations are about to
make mandatory. See `regulatory-landscape.md`.

## Sources

- bioLeak — https://arxiv.org/abs/2604.10965 · https://cran.r-project.org/web/packages/bioLeak/index.html
- EEGDash — https://arxiv.org/html/2606.16041v1
- LeakageDetector — https://arxiv.org/pdf/2503.14723
- leakr — https://github.com/cherylisabella/leakr
- Credo AI — https://www.credo.ai/product
- BABL AI — https://aicompliancevendors.com/vendors/babl-ai
- Fraunhofer IDMT, EEG without Brainprint — https://www.idmt.fraunhofer.de/en/Press_and_Media/press_releases/2026/EEG-without-Brainprint.html
