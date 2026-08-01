"""Synthetic multi-subject datasets with a known correct answer.

Real EEG never tells you whether a leakage detector works, because you do not
know how much genuine signal the data contained to begin with. These generators
do: one dataset where the label is knowable only by recognising the person, one
where it is knowable from a signal every person shares.
"""

import numpy as np
import pandas as pd

FEATURE_PREFIX = "f"

MODES = ("subject_leakage", "true_signal")

# Feature values stay in arbitrary units; the pipeline standardises them anyway.
_SUBJECT_OFFSET_SCALE = {"subject_leakage": 3.0, "true_signal": 0.2}

# In subject_leakage, subjects alternate between strongly positive and strongly
# negative label rates, which keeps the dataset globally balanced.
_LEAKY_LABEL_RATES = (0.85, 0.15)


def feature_columns(df):
    """Feature column names, in order, for a NerveML dataframe."""
    return [
        c
        for c in df.columns
        if c.startswith(FEATURE_PREFIX) and c[len(FEATURE_PREFIX) :].isdigit()
    ]


def make_synthetic(mode, n_subjects=20, n_trials=40, n_features=16, seed=0):
    """Build a synthetic dataset in NerveML's data contract.

    Returns a dataframe with subject_id, trial_id, target_label and n_features
    feature columns named f00, f01, ...
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    rng = np.random.default_rng(seed)
    offsets = rng.normal(0, _SUBJECT_OFFSET_SCALE[mode], size=(n_subjects, n_features))
    direction = rng.normal(0, 1, size=n_features)
    direction /= np.linalg.norm(direction)

    frames = []
    for s in range(n_subjects):
        noise = rng.normal(0, 1, size=(n_trials, n_features))

        if mode == "subject_leakage":
            # The label is a property of the person, not of the signal.
            rate = _LEAKY_LABEL_RATES[s % len(_LEAKY_LABEL_RATES)]
            n_high = round(rate * n_trials)
            labels = np.array([1] * n_high + [0] * (n_trials - n_high))
            rng.shuffle(labels)
        else:
            # The label follows a direction in feature space that every subject
            # shares. Splitting at each subject's own median keeps label rates
            # identical across people, so subject identity predicts nothing.
            score = noise @ direction
            labels = (score > np.median(score)).astype(int)

        features = offsets[s] + noise
        subject_id = f"s{s:02d}"
        frame = pd.DataFrame(
            features, columns=[f"{FEATURE_PREFIX}{i:02d}" for i in range(n_features)]
        )
        frame.insert(0, "subject_id", subject_id)
        frame.insert(1, "trial_id", [f"{subject_id}_t{t:03d}" for t in range(n_trials)])
        frame.insert(2, "target_label", labels)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
