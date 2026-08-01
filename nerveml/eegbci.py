"""PhysioNet EEG Motor Movement/Imagery database (109 subjects).

Chosen because it downloads without an application: DEAP and SEED both require
a signed agreement and manual approval, which no deadline survives.

The task is imagined left versus right fist. Motor imagery is famously
subject-dependent, so it exercises the validity half honestly, and 109
identities make the re-identification probe meaningful in a way that a
twenty-person emotion set never could.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nerveml.features import band_power_features, feature_names

# Runs 4, 8 and 12 are imagined left fist (T1) versus imagined right fist (T2).
IMAGERY_RUNS = (4, 8, 12)
LABELS = {"T1": 0, "T2": 1}

# The four motor tasks in the PhysioNet EEGMMI protocol, each recorded over three
# runs. Every task annotates its two conditions T1/T2, so the same {T1:0, T2:1}
# mapping gives a valid binary label throughout — only the meaning changes. This
# lets one audit run across executed vs imagined movement and across fists vs
# feet, testing whether re-identification survives a change of task the way it
# survives a change of recording.
TASKS = {
    "imagined_fists": (4, 8, 12),        # imagine left fist vs right fist
    "imagined_hands_feet": (6, 10, 14),  # imagine both fists vs both feet
    "executed_fists": (3, 7, 11),        # actually move left fist vs right fist
    "executed_hands_feet": (5, 9, 13),   # actually move both fists vs both feet
}

# Human-readable label meaning per task, for the report's target description.
TASK_LABELS = {
    "imagined_fists": "imagined left vs right fist",
    "imagined_hands_feet": "imagined both fists vs both feet",
    "executed_fists": "executed left vs right fist",
    "executed_hands_feet": "executed both fists vs both feet",
}


def task_runs(task):
    """Resolve a task name to its three run numbers."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {tuple(TASKS)}")
    return TASKS[task]

# These recordings differ from the rest in sampling rate or annotation scheme.
# Pooling them silently corrupts a shared feature table.
EXCLUDED_SUBJECTS = (88, 89, 92, 100)

N_SUBJECTS_TOTAL = 109

# Seconds relative to cue onset. Skipping the first half second avoids the
# evoked response to the cue itself, which is not motor imagery.
TMIN, TMAX = 0.5, 3.5


def usable_subjects(n_subjects):
    """The first n_subjects PhysioNet ids, skipping the damaged recordings."""
    available = [
        s for s in range(1, N_SUBJECTS_TOTAL + 1) if s not in EXCLUDED_SUBJECTS
    ]
    if n_subjects > len(available):
        raise ValueError(
            f"only {len(available)} usable subjects exist, asked for {n_subjects}"
        )
    return available[:n_subjects]


def epochs_to_frame(
    epochs, labels, subject_id, sampling_rate, channel_names, session=0
):
    """One run of one subject, as rows in NerveML's data contract.

    session records which run the epochs came from. Runs are separate
    recordings, so holding one out is what lets the re-identification probe
    tell a person's fingerprint apart from their electrode placement.
    """
    features = band_power_features(epochs, sampling_rate, channel_names)
    frame = pd.DataFrame(features, columns=feature_names(channel_names))
    frame.insert(0, "subject_id", subject_id)
    frame.insert(1, "session_id", session)
    frame.insert(
        2,
        "trial_id",
        [f"{subject_id}_r{session:02d}_t{i:03d}" for i in range(len(frame))],
    )
    frame.insert(3, "target_label", np.asarray(labels, dtype=int))
    return frame


def _epoched_run(subject, run, verbose=False):
    """Read and epoch one run. Returns (data, labels, sfreq, ch_names) or None.

    data is (n_epochs, n_channels, n_samples) — the raw epoch tensor, before any
    feature reduction. Both the band-power feature loader and the raw-epoch
    loader (for CSP) build on this, so the MNE reading rules live in one place.

    Runs are read separately rather than concatenated, so each epoch keeps the
    identity of the recording it came from.
    """
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf

    path = eegbci.load_data(subject, [run], update_path=True, verbose=verbose)[0]
    raw = read_raw_edf(path, preload=True, verbose=verbose)
    eegbci.standardize(raw)
    raw.rename_channels(lambda name: name.strip("."))

    events, event_id = mne.events_from_annotations(raw, verbose=verbose)
    wanted = {name: event_id[name] for name in LABELS if name in event_id}
    if len(wanted) < 2:
        return None

    epochs = mne.Epochs(
        raw,
        events,
        wanted,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=verbose,
    )
    if len(epochs) == 0:
        return None

    inverse = {code: name for name, code in wanted.items()}
    labels = [LABELS[inverse[code]] for code in epochs.events[:, -1]]
    return epochs.get_data(copy=True), labels, raw.info["sfreq"], epochs.ch_names


def _load_one_run(subject, run, verbose=False):
    """Epoch a single run into the feature table. None if unusable."""
    result = _epoched_run(subject, run, verbose=verbose)
    if result is None:
        return None
    data, labels, sfreq, ch_names = result
    return epochs_to_frame(
        data, labels, f"S{subject:03d}", sfreq, ch_names, session=run
    )


def load_eegbci(n_subjects=20, runs=IMAGERY_RUNS, cache_path=None, verbose=False,
                task=None):
    """Download, epoch and featurise n_subjects. Caches the feature table.

    Downloads roughly 10 MB per subject per run on first use; afterwards MNE
    serves from its own cache and this function serves from cache_path. Pass a
    task name (see TASKS) to select a different motor task; it overrides runs.
    """
    if task is not None:
        runs = task_runs(task)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return pd.read_csv(cache_path)

    frames = []
    for subject in usable_subjects(n_subjects):
        for run in runs:
            frame = _load_one_run(subject, run, verbose=verbose)
            if frame is not None:
                frames.append(frame)

    if not frames:
        raise ValueError("no usable recordings were produced")

    table = pd.concat(frames, ignore_index=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache_path, index=False)
    return table


@dataclass
class EegbciEpochs:
    """Raw epochs for the CSP decoder, before feature reduction.

    X is (n_trials, n_channels, n_samples). y is the binary imagined-fist label.
    subjects and sessions label every trial for held-out cross-validation, the
    same way the feature table's subject_id / session_id columns do.
    """

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sessions: np.ndarray
    sampling_rate: float
    channel_names: list


def load_eegbci_epochs(n_subjects=20, runs=IMAGERY_RUNS, verbose=False, task=None):
    """Download and epoch n_subjects, returning the raw epoch tensor.

    Unlike load_eegbci, this keeps the (n_channels, n_samples) time series each
    trial rather than reducing it to band power, because CSP learns from the
    spatial covariance the reduction discards. Runs whose epoch shape does not
    match the first usable run are skipped rather than silently reshaped, so the
    stacked tensor is never a lie about what was recorded. Pass a task name (see
    TASKS) to select a different motor task; it overrides runs.
    """
    if task is not None:
        runs = task_runs(task)
    blocks, ys, subs, sess = [], [], [], []
    sfreq = None
    channel_names = None
    ref_shape = None

    for subject in usable_subjects(n_subjects):
        for run in runs:
            result = _epoched_run(subject, run, verbose=verbose)
            if result is None:
                continue
            data, labels, run_sfreq, ch_names = result
            trial_shape = data.shape[1:]
            if ref_shape is None:
                ref_shape = trial_shape
                sfreq = run_sfreq
                channel_names = list(ch_names)
            elif trial_shape != ref_shape:
                # A recording with a different channel count or length would
                # corrupt a stacked tensor; skip it, as load_eegbci excludes the
                # damaged subjects for the same reason.
                continue
            blocks.append(np.asarray(data, dtype=float))
            ys.extend(int(v) for v in labels)
            subs.extend([f"S{subject:03d}"] * len(labels))
            sess.extend([run] * len(labels))

    if not blocks:
        raise ValueError("no usable recordings were produced")

    return EegbciEpochs(
        X=np.concatenate(blocks, axis=0),
        y=np.asarray(ys, dtype=int),
        subjects=np.asarray(subs),
        sessions=np.asarray(sess),
        sampling_rate=sfreq,
        channel_names=channel_names,
    )
