"""Load MOABB motor-imagery datasets into NerveML's feature-table contract.

MOABB (Mother of All BCI Benchmarks) serves many public EEG datasets
programmatically, several without a signed data agreement, so it is the route to
running the audit *across datasets* rather than only across the tasks of one.
Each dataset's epochs are reduced to the same log-band-power features as
``eegbci.py``, so the unchanged scan runs on them and the re-identification
finding can be checked for cross-dataset generalisation.

moabb is an optional dependency, imported lazily: importing this module never
requires it, only calling the loader does.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from nerveml.features import band_power_features, feature_names

# Small, agreement-free motor-imagery datasets that expose a clean binary
# left-vs-right-hand contrast through MOABB's LeftRightImagery paradigm.
KNOWN_DATASETS = {
    "BNCI2014_001": 9,   # BCI Competition IV 2a, 9 subjects, 22 EEG channels
    "BNCI2014_004": 9,   # BCI Competition IV 2b, 9 subjects, 3 bipolar channels
    "Zhou2016": 4,       # 4 subjects
    "Cho2017": 52,       # 52 subjects — large identity set for re-identification
}


def load_moabb_dataset(dataset="BNCI2014_001", n_subjects=None, cache_path=None,
                       verbose=False):
    """Featurise a MOABB left-vs-right motor-imagery dataset.

    Returns a DataFrame in NerveML's contract: subject_id, session_id, trial_id,
    target_label (0 = left hand, 1 = right hand), and one log-band-power column
    per channel-band pair. Caches to cache_path so a scan does not re-download.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return pd.read_csv(cache_path)

    import moabb.datasets as datasets
    from moabb.paradigms import LeftRightImagery

    if not hasattr(datasets, dataset):
        raise ValueError(f"unknown MOABB dataset {dataset!r}")
    ds = getattr(datasets, dataset)()
    subjects = ds.subject_list
    if n_subjects is not None:
        subjects = subjects[:n_subjects]

    # return_epochs gives an MNE Epochs object, which carries the sampling rate
    # and channel names band_power_features needs — meta alone does not.
    paradigm = LeftRightImagery()
    epochs, labels, meta = paradigm.get_data(
        dataset=ds, subjects=subjects, return_epochs=True
    )

    data = epochs.get_data(copy=True)  # (n_trials, n_channels, n_samples)
    sfreq = epochs.info["sfreq"]
    ch_names = epochs.ch_names

    frame = pd.DataFrame(
        band_power_features(data, sfreq, ch_names),
        columns=feature_names(ch_names),
    )
    # LeftRightImagery labels are the strings 'left_hand' / 'right_hand'.
    labels = np.asarray(labels)
    frame.insert(0, "subject_id", [f"S{int(s):03d}" for s in meta["subject"].to_numpy()])
    frame.insert(1, "session_id", meta["session"].to_numpy())
    frame.insert(
        2, "trial_id",
        [f"{dataset}_{i:05d}" for i in range(len(frame))],
    )
    frame.insert(3, "target_label", (labels == "right_hand").astype(int))

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
    return frame
