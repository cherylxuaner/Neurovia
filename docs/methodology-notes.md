# Methodology notes

Working notes from a review of the validation method. Measured findings, two
retracted claims, and a proposed direction that has not been implemented.

Everything below was measured on the synthetic datasets at 20 subjects x 40
trials x 16 features, seed 0, random forest probe.

## Finding 1 — the pooled score hides what the model actually learned

Scoring every test trial together, versus scoring each subject's trials
separately and averaging:

| dataset | protocol | pooled | within-subject | within sd |
|---|---|---|---|---|
| `subject_leakage` | trial-random | 0.844 | **0.496** | 0.006 |
| `subject_leakage` | subject-held-out | 0.440 | 0.485 | 0.062 |
| `true_signal` | trial-random | 0.845 | **0.845** | 0.050 |
| `true_signal` | subject-held-out | 0.836 | 0.836 | 0.059 |

Where the signal is real, pooled equals within-subject. Where the score is pure
memorisation, pooled reaches 0.844 while within-subject sits at chance with a
standard deviation of 0.006 — every subject, without exception.

The 0.844 was never discrimination between labels. It was discrimination
between people, and a pooled average cannot tell the two apart. This is
Simpson's paradox in the metric.

**Why this matters more than the gap.** The gap needs two protocols run and
subtracted. This decomposition works on a single protocol, so it can be applied
to a number a customer already has, without rerunning anything.

## Finding 2 — the gap understates memorisation

Training one model once, then varying only the test set:

| dataset | test on seen subjects | test on unseen subjects | difference |
|---|---|---|---|
| `subject_leakage` | 0.845 | 0.351 | **+0.494** |
| `true_signal` | 0.827 | 0.817 | +0.010 |

The reported gap for `subject_leakage` is +0.404, against a true memorisation
effect of +0.494 — an understatement of about 19%. The two protocols differ in
more than whether test subjects were seen: training subject counts differ (20
vs 16) and the test distribution shifts. Both contributions land in the gap.

## Finding 3 — bootstrap intervals are effectively free

Subjects are the independent unit, and per-subject scores are already computed,
so a confidence interval needs no refitting: 10,000 resamples of the per-subject
scores took 0.004 s.

| quantity | estimate | 95% CI |
|---|---|---|
| subject-held-out balanced accuracy | 0.485 | [0.457, 0.512] |

## Retracted claim — StratifiedGroupKFold

Asserted without measuring, then contradicted by measurement:

| splitter | positive-class share per fold | range |
|---|---|---|
| `GroupKFold` (current) | 0.50, 0.50, 0.50, 0.50, 0.50 | **0.000** |
| `StratifiedGroupKFold` | 0.33, 0.33, 0.68, 0.50, 0.68 | 0.350 |

The current splitter produces perfectly balanced folds here, because subject
label rates alternate and round-robin assignment happens to split them evenly.
Revisit only on real data with unequal group sizes, and measure first.

## Retracted claim — bootstrap cost

Assumed to require refitting the pipeline thousands of times. It does not; see
Finding 3.

## Proposed direction (not implemented, not approved)

Promote within-subject discrimination to a first-class metric reported beside
the pooled score, give both bootstrap intervals over subjects, and make the
risk verdicts interval-aware: a verdict fires only when the whole interval
clears its threshold, otherwise the report says the sample cannot decide and
states roughly how many subjects would.

Deferred: repeated cross-validation, a paired interval on the gap itself,
multiple-comparison handling once several attributes are scanned in one run.

## Open problems not addressed by the above

- The verdict depends on the probe. The same data gives gap +0.245 under
  logistic regression and +0.404 under a random forest. The report records
  `model_kind`, but the risk level is named as a property of the data.
- The synthetic generators use independent Gaussian noise, equal trial counts,
  no outliers, no missing channels and no drift. The pipeline has never been
  exercised against a realistically ugly dataset.
- No hash of the input data is recorded, so two reports claiming the same
  dataset cannot be shown to concern the same bytes.
- No real neural data has been used at any point.
