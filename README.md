# NerveML

Neural data privacy and validation scanner — prototype.

Before neural data is shared, sold, or used to train AI, NerveML tests what
sensitive information can be inferred from it and whether apparent model
performance survives leakage-safe validation.

Implements the P0 acceptance criteria in
`NerveML_Prototype_Product_and_Engineering_Spec.pdf`.

## Quickstart

```bash
pip install -e .
pytest                              # 285 tests
nerveml                             # command-line scan
streamlit run nerveml/app.py        # browser interface
```

## What one scan does

1. Integrity-checks the dataset and stops or warns before measuring anything.
2. Trains a baseline classifier for the binary target.
3. Evaluates it twice over identical features and an identical model family —
   once with a trial-random split, once with subjects held out.
4. Reports the gap between them.
5. Rescores the same trial-random predictions inside each unit separately.
   Pooled, a model earns credit for telling people apart; scored within a
   person, only label discrimination survives. When the two diverge, the
   headline number was never about the label.
6. Reruns the entire grouped pipeline against labels shuffled *within* each
   subject, to find where chance actually sits.
7. Scores every held-out unit separately and names the worst one, so a mean
   cannot hide a participant the model fails completely.
8. Turns the metrics into transparent risk flags, a bounded claim, an explicit
   list of claims the scan does **not** support, and recommended next tests.
9. Writes `outputs/audit_report.json`, `outputs/fold_metrics.csv`, and four
   evidence figures: per-fold scores against a chance reference, the null
   distribution with the observed score marked, per-unit scores diverging from
   chance, and the ranked features.

## The demonstration

A wearable company's attention classifier, forty participants, and the number
their training script produces by default:

```bash
nerveml --dataset focus_tracker      # what the team is about to show a partner
nerveml --dataset calibrated_detector  # the same team's rebuilt version
```

| | trial-random | within-subject | subject-held-out | gap | verdict |
|---|---|---|---|---|---|
| `focus_tracker` | 0.904 | **0.498** | 0.533 | +0.371 | high generalization warning |
| `calibrated_detector` | 0.902 | 0.901 | 0.895 | +0.007 | no warning |

The first is the failure this product exists to catch: a number good enough to
put in a deck, from a model that is at chance inside every single participant.
The second has to pass, or a warning from the first means nothing.

The mechanism, isolated:

```bash
nerveml --dataset subject_leakage    # labels are a property of the person
nerveml --dataset true_signal        # labels follow a signal everyone shares
```

| | trial-random | within-subject | subject-held-out | gap | perm. p | verdict |
|---|---|---|---|---|---|---|
| `subject_leakage` | 0.761 | **0.488** | 0.516 | 0.245 | 0.119 | low / inconclusive |
| `true_signal`     | 0.902 | **0.904** | 0.892 | 0.010 | 0.010 | elevated |

Two independent readings of the same discrepancy.

The **gap** needs both protocols run and subtracted. The **within-subject**
column needs only one: it rescores the trial-random predictions inside each
person. Where the signal is real the two columns agree; where the score was
memorisation, the pooled number stays high while the within-person number
falls to chance. That second reading applies to a number a customer already
has, without rerunning anything.

## The result on real data

PhysioNet EEG Motor Movement/Imagery, 20 subjects, 900 trials, 64 channels
reduced to log band power over five bands. Imagined left versus right fist.

| question | answer |
|---|---|
| does the task label transfer to a new person? | held-out ROC-AUC **0.555**, 95% CI 0.521–0.606, permutation p 0.079 |
| can a held-out record be linked to its producer? | **0.987** accuracy, 19.7x chance, mean per-identity recall 95% CI 0.977–0.994 |

The scan declines to give the first question a verdict. Its interval straddles
the 0.55 threshold, and the estimate sits almost exactly on it, so a different
twenty people could land on either side. Rather than pick one, the report says
the sample cannot decide and that settling it would take roughly 1,400
subjects. The second question needs no such hedge — its interval is nowhere
near anything.

Two controls, because a re-identification number is easy to get for the wrong
reason.

**Does it survive more identities?** Going from 10 subjects to 20 halves the
chance baseline and doubles the number of wrong answers available. Accuracy
moved 0.998 to 0.997. It is not winning by having few people to choose between.

**Does it survive a change of recording?** Holding out whole runs — so the
reference records and the held-out records never come from the same recording —
costs 0.997 to 0.987. All twenty identities stay above 0.9 recall, the weakest
at 0.911. It is not a fingerprint of one recording's drift.

**Is it neural, or is it anatomy?** Splitting the features answers this, and the
answer narrows the claim:

| attacked with | accuracy |
|---|---|
| everything (320 features) | 0.987 |
| per-channel amplitude only (64) | **0.931** |
| spectral shape, amplitude divided out | **0.937** |

Amplitude alone — the part that skull thickness, electrode impedance and cap
fit move — gets almost the whole way there. So this is substantially a physical
fingerprint, and calling it a *neural* fingerprint would be overclaiming. But
shape alone gets there too, so it is not purely physical either.

For the privacy question this changes nothing: California's standard is whether
the data can be linked to a person, not whether the link runs through cortex.
Two independent routes to identification is a stronger privacy finding than
one. It is the *scientific* claim that has to be narrowed, and the scan narrows
it rather than leaving it to a reviewer.

The same features, the same model family, one scan. What the data will not tell
you is what someone was thinking. What it will tell you, almost perfectly, is
who they are.

That asymmetry is the company's thesis, measured rather than argued, and the
within-experiment comparison controls for the obvious objection: these features
may well be too crude to decode motor imagery — established pipelines use
common spatial patterns and do considerably better — but they are the *same*
features that identify people at 0.998. Weak features cannot explain the gap
between the two answers.

Reproduce it:

```bash
python -c "from nerveml.eegbci import load_eegbci; load_eegbci(20, cache_path='sample_data/eegbci_20.csv')"
nerveml --dataset sample_data/eegbci_20.csv --model logistic_regression
```

First run downloads about 10 MB per subject. No application or agreement is
required, which is why this dataset and not DEAP or SEED.

### What this result does not say

The runs held out above are consecutive runs of a single session: the cap is
never removed and replaced. So the control rules out run-specific drift and
temporal adjacency, but not a fingerprint of that particular electrode
placement. Linking across days, across cap placements or across devices is
harder and is not measured here.

The scan reports which attack it ran. Given a `session_id` column it holds
whole recordings out; without one it says so in the attack model text rather
than quietly running the easier test.

## Why synthetic data first

Real EEG cannot tell you whether a leakage detector works, because you do not
know how much genuine signal the data held to begin with. These two generators
do, so the test suite can assert the pipeline reaches the *correct* conclusion
rather than merely a plausible one. `tests/test_validation.py` is where that
claim lives.

DEAP and SEED both require a signed agreement and manual approval, so they are
V1 inputs, not prototype blockers.

## Bringing your own data

Reduce the dataset to one row per trial and load it directly:

| column | meaning |
|---|---|
| `subject_id` | independent person or patient identifier |
| `trial_id` | trial or recording unit identifier |
| `target_label` | binary sensitive attribute under test |
| anything else | feature columns, inferred automatically |

```bash
nerveml --dataset path/to/features.csv
```

Any column can be the independent unit that is held out whole, so the same
pipeline answers "does this survive a new session / site / device / stimulus?"
without changes:

```bash
nerveml --dataset path/to/features.csv --group-column session_id
```

Raw EEG is not yet supported. The missing piece is a `features.py` that turns
epochs into log band power per channel (Welch PSD, delta/theta/alpha/beta/gamma);
everything downstream of the feature table already works.

## Layout

```
nerveml/
  synth.py        two synthetic datasets with a known correct answer
  loaders.py      CSV intake and the integrity scan (spec 10.1)
  models.py       baselines, each carrying its own preprocessing
  validation.py   trial-random vs subject-grouped evaluation (spec 10.4-10.6)
  permutation.py  within-subject label permutation null (spec 10.7)
  interpret.py    feature ranking (spec 10.8)
  risk_rules.py   the heuristics table (spec 11.2)
  scan.py         orchestration and report assembly (spec 22)
  identity.py     re-identification probe (spec 6.2)
  features.py     log band power per channel, via Welch (spec 10.2)
  eegbci.py       PhysioNet Motor Movement/Imagery loader
  confounds.py    is the fingerprint physical or spectral? (spec 6.2, 18)
  pdf.py          the risk report as a document (spec 12)
  fonts/          IBM Plex, vendored and embedded (SIL OFL 1.1, see OFL.txt)
  scenarios.py    demonstration datasets shaped like a customer's
  uncertainty.py  bootstrap intervals over the unit of independence
  plots.py        the six evidence figures (spec 13.5)
  cli.py          terminal entry point
  app.py          Streamlit interface (spec 13)
tests/            285 tests
```

## Runtime

The permutation null reruns the entire grouped pipeline once per permutation,
which is the cost of a null that actually reflects this pipeline. On 20
subjects x 40 trials, 5 folds, 100 permutations:

| model | time |
|---|---|
| `logistic_regression` | ~4 s |
| `random_forest` | ~140 s |

Use logistic regression for live demos. Permutations are spread over threads
while each forest is fitted single-threaded, because parallelising at the
coarser level beats letting the two levels contend; the thread count never
changes the result, which `tests/test_permutation.py` asserts.

Process-based parallelism is deliberately not used: joblib's loky backend
encodes temp paths as ASCII and crashes on systems whose user directory is not
ASCII.

## Status and limits

Prototype. Not a clinical, legal, or regulatory certification system.

- The risk thresholds are prototype UX heuristics, not standards. Every flag
  carries the rule that produced it so it can be argued with.
- A verdict is stated only when the whole 95% interval sits on one side of the
  threshold. When it straddles, the report says the sample cannot decide and
  estimates what would. Intervals resample subjects, never trials, because
  trials inside one person are not independent of each other.
- A scan is evidence under one stated attack model. It cannot certify that a
  dataset is private or safe to release.
- Feature associations are model-dependent and non-causal.
- Balanced accuracy is not reported for a unit held out with only one class
  present, because against one class it collapses to that class's recall and
  does not compare with the other units.
- The target is a self-reported label. The scan tests whether that label is
  statistically inferable — not whether anyone's actual mental state was read.

## Not built yet

Later from the spec: artifact-only session/site/device/stimulus held-out protocols, PDF export,
probability calibration, and support for a customer-supplied trained model.

The Streamlit run screen shows a single indicator rather than per-stage
progress, because the scan does not yet report stage boundaries and a progress
bar that does not track real stages is a placeholder.

## License

Proprietary. All rights reserved. See [LICENSE](LICENSE).

The report embeds IBM Plex (Sans, Serif, Mono), copyright IBM Corp., licensed
under the SIL Open Font License 1.1. The licence travels with the fonts in
`nerveml/fonts/OFL.txt`. Embedding rather than relying on the PDF base-14 faces
means the document renders identically wherever it is opened.
