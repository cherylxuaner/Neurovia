"""Tests for stimulus_id and device_id held-out axes in multi-axis validation.

The CANDIDATE_GROUPING_AXES list in scan.py includes stimulus_id and device_id,
but prior test fixtures only exercised session_id and site_id.  These tests
confirm that the auto-detection and evaluation paths work for every axis.
"""

import json

import numpy as np
import pytest

from nerveml.scan import CANDIDATE_GROUPING_AXES, _detect_extra_axes, run_scan
from nerveml.synth import make_synthetic

FAST = dict(model_kind="logistic_regression", n_permutations=20, n_splits=4)


def _make_csv(tmp_path, axes, filename="multi.csv"):
    """Feature CSV with subject_id as primary group + any named extra axes.

    Each axis listed in *axes* is assigned one of 4 integer values drawn
    uniformly at random, giving >=4 unique values so _detect_extra_axes
    will include them.
    """
    rng = np.random.default_rng(7)
    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    for col in axes:
        df[col] = rng.integers(0, 4, size=len(df)).astype(int)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# _detect_extra_axes — stimulus_id and device_id detection
# ---------------------------------------------------------------------------

def test_detect_extra_axes_finds_stimulus_id():
    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    df["stimulus_id"] = list(range(5)) * (len(df) // 5)

    extra = _detect_extra_axes(df, primary_group_column="subject_id", n_splits=4)

    assert "stimulus_id" in extra


def test_detect_extra_axes_finds_device_id():
    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    df["device_id"] = list(range(4)) * (len(df) // 4)

    extra = _detect_extra_axes(df, primary_group_column="subject_id", n_splits=4)

    assert "device_id" in extra


def test_detect_extra_axes_finds_all_four_candidate_axes():
    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    for col in CANDIDATE_GROUPING_AXES:
        df[col] = list(range(4)) * (len(df) // 4)

    extra = _detect_extra_axes(df, primary_group_column="subject_id", n_splits=4)

    assert "session_id" in extra
    assert "site_id" in extra
    assert "device_id" in extra
    assert "stimulus_id" in extra
    assert "subject_id" not in extra


def test_detect_extra_axes_stimulus_excluded_when_too_few_unique():
    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    # Only 3 unique values < n_splits=4 → should be excluded.
    df["stimulus_id"] = (np.arange(len(df)) % 3).astype(int)

    extra = _detect_extra_axes(df, primary_group_column="subject_id", n_splits=4)

    assert "stimulus_id" not in extra


def test_detect_extra_axes_device_excluded_when_primary():
    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    df["device_id"] = list(range(4)) * (len(df) // 4)

    extra = _detect_extra_axes(df, primary_group_column="device_id", n_splits=4)

    assert "device_id" not in extra


# ---------------------------------------------------------------------------
# run_scan end-to-end — stimulus_id axis produces correct multi_axis entry
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def stimulus_report(tmp_path_factory):
    path = _make_csv(tmp_path_factory.mktemp("stim"), ["stimulus_id"])
    return run_scan(dataset=str(path), seed=0, **FAST)


def test_stimulus_axis_appears_in_multi_axis_validation(stimulus_report):
    axes = [e["axis"] for e in stimulus_report["multi_axis_validation"]]
    assert "stimulus_id" in axes


def test_stimulus_axis_scheme_is_correctly_named(stimulus_report):
    entry = next(
        e for e in stimulus_report["multi_axis_validation"] if e["axis"] == "stimulus_id"
    )
    assert entry["scheme"] == "stimulus_id_grouped"


def test_stimulus_axis_entry_has_required_keys(stimulus_report):
    entry = next(
        e for e in stimulus_report["multi_axis_validation"] if e["axis"] == "stimulus_id"
    )
    assert set(entry) >= {
        "axis",
        "n_units",
        "scheme",
        "grouped_balanced_accuracy",
        "grouped_roc_auc",
        "grouped_balanced_accuracy_std",
        "generalization_gap",
        "worst_group",
        "per_group",
        "confidence_intervals",
    }


def test_stimulus_axis_scores_are_bounded(stimulus_report):
    entry = next(
        e for e in stimulus_report["multi_axis_validation"] if e["axis"] == "stimulus_id"
    )
    assert 0.0 <= entry["grouped_balanced_accuracy"] <= 1.0
    assert 0.0 <= entry["grouped_roc_auc"] <= 1.0
    assert entry["grouped_balanced_accuracy_std"] >= 0.0
    assert -1.0 <= entry["generalization_gap"] <= 1.0


def test_stimulus_axis_n_units_matches_unique_values(stimulus_report):
    entry = next(
        e for e in stimulus_report["multi_axis_validation"] if e["axis"] == "stimulus_id"
    )
    assert entry["n_units"] == 4


def test_stimulus_axis_is_json_serialisable(stimulus_report):
    json.dumps(stimulus_report["multi_axis_validation"])


# ---------------------------------------------------------------------------
# run_scan end-to-end — device_id axis
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device_report(tmp_path_factory):
    path = _make_csv(tmp_path_factory.mktemp("dev"), ["device_id"])
    return run_scan(dataset=str(path), seed=0, **FAST)


def test_device_axis_appears_in_multi_axis_validation(device_report):
    axes = [e["axis"] for e in device_report["multi_axis_validation"]]
    assert "device_id" in axes


def test_device_axis_scheme_is_correctly_named(device_report):
    entry = next(
        e for e in device_report["multi_axis_validation"] if e["axis"] == "device_id"
    )
    assert entry["scheme"] == "device_id_grouped"


def test_device_axis_scores_are_bounded(device_report):
    entry = next(
        e for e in device_report["multi_axis_validation"] if e["axis"] == "device_id"
    )
    assert 0.0 <= entry["grouped_balanced_accuracy"] <= 1.0
    assert 0.0 <= entry["grouped_roc_auc"] <= 1.0
    assert -1.0 <= entry["generalization_gap"] <= 1.0


# ---------------------------------------------------------------------------
# run_scan — all four axes simultaneously
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_axes_report(tmp_path_factory):
    path = _make_csv(
        tmp_path_factory.mktemp("all"),
        ["session_id", "site_id", "device_id", "stimulus_id"],
        filename="all_axes.csv",
    )
    return run_scan(dataset=str(path), seed=0, **FAST)


def test_all_four_axes_appear_in_multi_axis_validation(all_axes_report):
    axes = {e["axis"] for e in all_axes_report["multi_axis_validation"]}
    assert axes == {"session_id", "site_id", "device_id", "stimulus_id"}


def test_all_axes_report_is_json_serialisable(all_axes_report):
    json.dumps(all_axes_report["multi_axis_validation"])
