"""PhysioNet EEG Motor Movement/Imagery loader.

Network access is needed only by the download itself, which is marked and
skipped by default. Everything that turns epochs into NerveML's data contract
is pure and tested here without touching the network.
"""

import numpy as np
import pytest

from nerveml.eegbci import (
    EXCLUDED_SUBJECTS,
    IMAGERY_RUNS,
    LABELS,
    TASK_LABELS,
    TASKS,
    epochs_to_frame,
    task_runs,
    usable_subjects,
)
from nerveml.loaders import validate_dataset
from nerveml.synth import feature_columns


def fake_epochs(n_epochs=8, n_channels=4, n_times=320):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n_epochs, n_channels, n_times))


def test_damaged_subjects_are_excluded():
    # These recordings differ in sampling rate or annotation scheme and would
    # silently contaminate a pooled feature table.
    assert EXCLUDED_SUBJECTS
    chosen = usable_subjects(n_subjects=10)

    assert not set(chosen) & set(EXCLUDED_SUBJECTS)
    assert len(chosen) == 10
    assert chosen == sorted(chosen)


def test_usable_subjects_skips_over_the_excluded_ones():
    everyone = usable_subjects(n_subjects=109 - len(EXCLUDED_SUBJECTS))

    assert set(everyone).isdisjoint(EXCLUDED_SUBJECTS)


def test_asking_for_more_subjects_than_exist_is_rejected():
    with pytest.raises(ValueError, match="only"):
        usable_subjects(n_subjects=200)


def test_frame_follows_the_data_contract():
    epochs = fake_epochs(n_epochs=6)
    labels = np.array([0, 1, 0, 1, 0, 1])

    frame = epochs_to_frame(epochs, labels, "S007", 160.0, ["C3", "C4", "Cz", "Pz"])

    assert list(frame.columns[:4]) == [
        "subject_id",
        "session_id",
        "trial_id",
        "target_label",
    ]
    assert (frame["subject_id"] == "S007").all()
    assert frame["trial_id"].nunique() == 6
    assert set(frame["target_label"]) == {0, 1}


def test_frame_carries_one_feature_per_channel_band():
    epochs = fake_epochs(n_channels=4)

    frame = epochs_to_frame(
        epochs, np.zeros(8, dtype=int), "S001", 160.0, ["C3", "C4", "Cz", "Pz"]
    )

    assert len(feature_columns(frame)) == 0  # not the synthetic f00 naming
    assert "C3_alpha" in frame.columns
    assert len([c for c in frame.columns if c.startswith("C3_")]) == 5


def test_trial_ids_are_unique_across_subjects():
    epochs = fake_epochs(n_epochs=4)
    labels = np.zeros(4, dtype=int)

    a = epochs_to_frame(epochs, labels, "S001", 160.0, ["a", "b", "c", "d"])
    b = epochs_to_frame(epochs, labels, "S002", 160.0, ["a", "b", "c", "d"])

    assert set(a["trial_id"]).isdisjoint(b["trial_id"])


def test_a_frame_passes_the_integrity_scan():
    import pandas as pd

    frames = []
    for index in range(6):
        rng = np.random.default_rng(index)
        epochs = rng.normal(size=(10, 4, 320))
        labels = np.tile([0, 1], 5)
        frames.append(
            epochs_to_frame(epochs, labels, f"S{index:03d}", 160.0, ["a", "b", "c", "d"])
        )
    df = pd.concat(frames, ignore_index=True)
    feats = [c for c in df.columns if "_" in c and c != "subject_id"]
    feats = [c for c in feats if c not in ("trial_id", "target_label")]

    summary = validate_dataset(df, feats, n_splits=5)

    assert summary.n_subjects == 6
    assert summary.n_trials == 60


def test_frame_records_which_run_each_epoch_came_from():
    epochs = fake_epochs(n_epochs=4)

    frame = epochs_to_frame(
        epochs, np.zeros(4, dtype=int), "S001", 160.0, ["a", "b", "c", "d"], session=8
    )

    # Without this the cross-recording attack has nothing to hold out.
    assert (frame["session_id"] == 8).all()
    assert list(frame.columns[:4]) == [
        "subject_id",
        "session_id",
        "trial_id",
        "target_label",
    ]


def test_trial_ids_stay_unique_across_runs_of_one_subject():
    epochs = fake_epochs(n_epochs=4)
    labels = np.zeros(4, dtype=int)

    a = epochs_to_frame(epochs, labels, "S001", 160.0, list("abcd"), session=4)
    b = epochs_to_frame(epochs, labels, "S001", 160.0, list("abcd"), session=8)

    assert set(a["trial_id"]).isdisjoint(b["trial_id"])


def test_label_map_covers_both_imagery_classes():
    assert set(LABELS.values()) == {0, 1}
    assert IMAGERY_RUNS


# --- the four-task selection ----------------------------------------------

def test_tasks_and_labels_cover_the_same_names():
    assert set(TASKS) == set(TASK_LABELS)
    # The four PhysioNet motor tasks, imagined and executed, fists and feet.
    assert set(TASKS) == {
        "imagined_fists", "imagined_hands_feet",
        "executed_fists", "executed_hands_feet",
    }


def test_task_runs_are_three_distinct_runs_each():
    seen = set()
    for name in TASKS:
        runs = task_runs(name)
        assert len(runs) == 3
        assert not (set(runs) & seen), "run sets must not overlap across tasks"
        seen |= set(runs)
    # imagined_fists stays the historical default run set.
    assert task_runs("imagined_fists") == IMAGERY_RUNS


def test_task_runs_rejects_unknown_task():
    with pytest.raises(ValueError, match="unknown task"):
        task_runs("telepathy")
