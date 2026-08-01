# NerveML Audit Report — Demo V1

**Neural-model auditing on a public EEG/BCI dataset.**
Prepared for the Norve / NerveML team. Reproducible end-to-end from a clean checkout.

- Dataset: PhysioNet **EEG Motor Movement/Imagery** (`eegmmidb`), 20 subjects, 900 trials, 64 channels
- Decoder / probe: **logistic regression** over **log band power** (Welch PSD; δ/θ/α/β/γ per channel → 320 features)
- Task label: imagined **left vs right fist** (binary, self-reported protocol label)
- Scan config: 5 folds, 100 within-subject permutations, subjects as the unit of independence
- Toolchain: `nerveml` prototype, 737 tests green, Python 3.12 venv
- Artifacts: `outputs/audit_report.json`, `outputs/fold_metrics.csv`, six evidence figures, `outputs/risk_report.pdf`

Reproduce:
```bash
python -c "from nerveml.eegbci import load_eegbci; load_eegbci(20, cache_path='sample_data/eegbci_20.csv')"
nerveml --dataset sample_data/eegbci_20.csv --model logistic_regression
# optimised CSP decoder vs the band-power baseline (§2c)
nerveml-decode --subjects 20 --components 8 --baseline-csv sample_data/eegbci_20.csv
```

---

## 1. Executive summary

We ran the full NerveML audit on a real public EEG dataset with a baseline decoder, under two
validation protocols (trial-wise vs subject-wise), plus a re-identification probe and a
physical-vs-neural confound test. Two findings, one asymmetric conclusion:

1. **The task label barely transfers to a new person.** Subject-held-out ROC-AUC is **0.555**,
   95% CI **[0.522, 0.607]**, permutation *p* = **0.099**. The interval straddles the 0.55
   decision threshold, so the scan **declines to issue a verdict** and estimates that settling it
   would take **~1,442 subjects** (we have 20). An *optimised* CSP decoder lifts this only to
   **0.620** (§2c) — still nowhere near the identity score below, so weak task decoding is not a
   decoder artefact.

2. **The same records identify the individual almost perfectly.** A held-out recording can be
   linked to the person who produced it with **0.987** accuracy — **19.7× chance** for 20
   identities — mean per-identity recall 95% CI **[0.977, 0.994]**.

The audit's value is not either number alone; it is that **the same features, the same model
family, one scan** produce them side by side. What the data will not reliably tell you is what
someone was doing. What it will tell you, almost perfectly, is **who they are**. That asymmetry —
measured, not argued — is the product thesis.

---

## 2. Trial-wise vs subject-wise split (the leakage test)

The core mechanism NerveML exists to catch: a headline number produced by a **trial-random**
split can come entirely from the model *recognising the person*, not *decoding the label*. When
you re-run the identical features and model under a **subject-held-out** split, the score
collapses. The gap between the two protocols — and, from a single protocol, the gap between the
**pooled** score and the **within-subject** score — quantifies that leakage.

### 2a. Synthetic ground-truth scenarios (where the correct answer is known)

Synthetic generators are used first *because* the true amount of signal is known, so we can assert
the pipeline reaches the **correct** conclusion, not merely a plausible one.

| scenario | trial-random | within-subject | subject-held-out | gap | perm. *p* | verdict |
|---|---|---|---|---|---|---|
| `focus_tracker` (leaky product) | 0.904 | **0.498** | 0.533 | **+0.371** | 0.347 | high generalization warning |
| `calibrated_detector` (rebuilt) | 1.000 | 1.000 | 1.000 | 0.000 | 0.010 | no warning / elevated |
| `subject_leakage` (identity=label) | 0.761 | **0.488** | 0.516 | +0.245 | 0.119 | low / inconclusive |
| `true_signal` (shared signal) | 0.902 | **0.904** | 0.892 | +0.010 | 0.010 | elevated (real signal) |

Reading: where signal is real (`true_signal`, `calibrated_detector`) the three columns agree.
Where the score was memorisation (`focus_tracker`, `subject_leakage`) the pooled/trial-random
number stays high while the within-person number falls to chance. `focus_tracker` is the failure
mode this product catches — **0.904 in a deck, at chance inside every single participant.**

### 2b. Real EEG (PhysioNet motor imagery)

| metric | value | 95% CI |
|---|---|---|
| trial-random balanced acc | 0.528 (AUC 0.564) | — |
| within-subject (same preds) | 0.542 | — |
| **subject-held-out** balanced acc | 0.524 (AUC 0.555) | AUC [0.522, 0.607] |
| generalization gap | **0.004** | [-0.028, 0.035] |
| permutation *p* | 0.099 | null mean 0.500, sd 0.018 |
| worst held-out subject | S008 @ 0.379 | — |

Here trial-wise and subject-wise **agree** — but at ~chance. These crude band-power features do
not decode motor imagery well (established CSP pipelines do far better — see §2c). Crucially, this
is the *same* feature table that identifies people at 0.987 below, so weak features cannot explain
the identity result. The scan reports the interval and **refuses a verdict** on decodability:
`sensitive_inference_inconclusive_at_this_sample_size`.

### 2c. Model optimization — the CSP decoder (`nerveml-decode`)

The obvious objection to §2b is that the decoder is too weak: band power over a whole trial throws
away the spatial covariance motor imagery lives in. So we added the standard optimised decoder —
**Common Spatial Patterns (CSP)**: supervised spatial filters, fitted **strictly inside each CV
fold** (leakage red-line intact), on mu+beta (8–30 Hz) band-passed raw epochs, feeding LDA.

| decoder | subject-held-out AUC |
|---|---|
| band-power + logistic (baseline) | 0.555 |
| CSP, broadband (no band-pass) | 0.466 |
| **CSP + LDA, mu+beta band-pass** | **0.611 (6 filters) / 0.620 (8 filters)** |
| filterbank CSP (mu+beta sub-bands) | 0.566 / 0.535 |

Two honest findings the audit does not hide:

- **The band-pass is the decoder, not a knob.** Broadband CSP scored *below* baseline (0.466);
  low-frequency drift dominated the variance CSP maximises. The mu+beta filter is what makes it
  work — a standard, necessary step, kept as a first-class part of the pipeline.
- **More complex is not better here.** Filterbank CSP (five sub-bands) *underperformed* single-band
  CSP — the extra features overfit at 20 subjects. So the reported best is single-band CSP at
  **0.620**, not the fancier variant. (Reported, not buried: the same anti-overclaim discipline the
  product enforces on customers.)

**What this changes for the thesis:** the optimised decoder lifts task decoding above chance and
above baseline, and it *still* lands at 0.62 — far below the 0.987 re-identification score on the
identical recordings. Weak task decoding is not an artefact of a weak decoder; it survives a good
one. The asymmetry is now measured against a *proper* decoder, which makes it harder to argue with,
not easier.

Reproduce:
```bash
nerveml-decode --subjects 20 --components 8 --baseline-csv sample_data/eegbci_20.csv
nerveml-decode --subjects 20 --method fbcsp        # the filterbank variant, for comparison
```

---

## 3. Re-identification (subject leakage, made explicit)

Attack model: linkage against a labelled reference set — the adversary already holds identified
recordings and attaches held-out records to them. Reference and held-out recordings are different
records, so what survives belongs to the **person**, not to one recording.

| quantity | value |
|---|---|
| held-out record → identity accuracy | **0.987** |
| lift over chance (20 identities, chance 0.05) | **19.7×** |
| mean per-identity recall (95% CI) | [0.977, 0.994] |
| most identifiable | S001 @ 1.000 recall |
| weakest identity recall | S003 @ 0.911 |

Removing names and IDs does **not** remove this. Under California's standard the relevant question
is whether data can be linked to a person — and here it can, from the signal alone. Flag raised:
`re_identifiable`.

---

## 4. Artifact / confound proxy test (physical vs neural fingerprint)

A re-identification number is easy to get for the wrong reason. We split the feature set to ask
whether identity rides on **amplitude** (which skull thickness, electrode impedance and cap fit
all move — a *physical* confound, essentially an artifact-of-acquisition proxy) or on **spectral
shape** with amplitude divided out (closer to a *neural* property).

| attacked with | features | accuracy | lift |
|---|---|---|---|
| everything | 320 | **0.987** | 19.7× |
| amplitude only (per-channel power) | 64 | **0.930** | 18.6× |
| spectral shape only (amplitude normalised out) | 256 | **0.937** | 18.7× |

**Conclusion: `carried_by = both`.** Amplitude alone gets almost the whole way, so the fingerprint
is *substantially physical* — calling it a purely *neural* fingerprint would overclaim. But shape
alone gets there too, so it is not purely an acquisition artifact either. The audit **narrows the
scientific claim** rather than leaving it to a reviewer.

For the **privacy** conclusion this changes nothing: two independent routes to identification is a
*stronger* finding than one. It is only the *scientific* framing ("neural fingerprint") that has
to be qualified — which is exactly the kind of overclaim an auditor should catch.

---

## 5. What this scan does and does not support

**Permitted claim (verbatim from the report):** *"This sample cannot decide whether the
self-reported label is decodable across held-out subjects (subject-held-out ROC-AUC 0.55, 95% CI
0.52 to 0.61, permutation p 0.099). The interval spans the threshold, so a different sample of the
same size could land on either side."*

**Not supported by this scan:**
- reads a person's true emotion or mental state
- certifies the dataset is anonymous or safe to release
- proves the dataset cannot be used to identify a participant
- the ranked features are causal neural correlates of the target

**Control that bounds the identity claim (from the spec/README run):** holding out whole *runs* so
reference and held-out records never share a recording costs identity accuracy only 0.997 → 0.987,
weakest recall 0.911 — so it is not a fingerprint of one recording's drift. It is **not** yet
tested across days, cap replacements, or devices; that is a V1 item.

---

## 6. Evidence figures (in `outputs/`)

| file | shows |
|---|---|
| `scorecard.png` | headline result cards |
| `fold_scores.png` | per-fold scores vs a chance reference |
| `null_distribution.png` | permutation null with the observed score marked |
| `per_group.png` | per-subject scores diverging from chance (names the worst) |
| `identity.png` | re-identification recall per identity |
| `top_features.png` | ranked features driving the probe |

---

## 7. Method notes, limits, and the V1 backlog

- **Unit of independence is the subject.** Confidence intervals resample subjects, never trials —
  trials within a person are not independent. A verdict fires only when the *whole* 95% interval
  clears the threshold; otherwise the scan says the sample cannot decide and estimates the *n*
  that would.
- **The verdict depends on the probe.** Logistic regression and random forest give different gaps
  on the same data; the report records `model_kind`. For live demos use logistic regression
  (~4 s vs ~140 s for the forest, dominated by the permutation null).
- **Feature associations are model-dependent and non-causal.**
- **This is a prototype**, not a clinical/legal/regulatory certification. Every risk flag carries
  the rule that produced it, so it can be argued with.
- **Decoder optimization (added since first draft):** raw-EEG CSP intake now exists —
  `nerveml/csp.py` (`CSP`, mu+beta `BandpassFilter`, `FilterbankCSP`, `decode_benchmark`) and the
  `nerveml-decode` CLI, with the numbers in §2c. Every CSP fit is in-fold.
- **Architecture (added since first draft):** feature-table decoders are a `MODEL_REGISTRY`
  (declarative `@register_model`); the optional probes are a `PROBE_REGISTRY` in `scan.py`
  (identity / composition / artifact / secondary), so adding a probe or model is one declaration.
- **V1 backlog still open:** cross-day / cross-cap / cross-device identity controls, probability
  calibration, input-bytes hashing for report provenance, per-band tmin/tmax tuning for a further
  decoding lift, and DEAP/SEED once their data agreements clear.

---

## 8. How to reproduce every number here

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e . && pip install -r requirements.txt
pytest -q                                             # 737 tests
# synthetic ground-truth contrast
nerveml --dataset focus_tracker      --model logistic_regression
nerveml --dataset calibrated_detector --model logistic_regression
# real EEG flagship (first run downloads ~200 MB)
python -c "from nerveml.eegbci import load_eegbci; load_eegbci(20, cache_path='sample_data/eegbci_20.csv')"
nerveml --dataset sample_data/eegbci_20.csv --model logistic_regression
```
