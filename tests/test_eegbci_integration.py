"""End-to-end integration test: PhysioNet EEG Motor Imagery → NerveML scan.

Marked `slow` so the CI fast path skips it.  Run explicitly with:

    pytest -m slow tests/test_eegbci_integration.py -v

The test downloads 2 subjects (~20 MB) via MNE's PhysioNet mirror, builds a
feature table, writes it to a temp CSV, and runs the full NerveML scan.  It
verifies that:

  * the feature table satisfies the data contract (validate_dataset passes)
  * run_scan returns a well-formed report with all required top-level keys
  * every numeric metric is a real float (no NaN / None from a missing stage)
  * the permitted_claim text exists and does NOT contain unhedged overclaims

This is the only test that touches the network; all other eegbci tests use
synthetic arrays.
"""

import json
import math
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from nerveml.eegbci import load_eegbci
from nerveml.loaders import validate_dataset
from nerveml.scan import run_scan

# Two subjects, one run each — enough to smoke-test the pipeline cheaply.
N_SUBJECTS = 2

# Only one run so the download is quick (~10 MB total).
RUNS = (4,)

# With 2 subjects we can only hold one out per fold.
N_SPLITS = 2


@pytest.mark.slow
def test_load_eegbci_produces_valid_dataset(tmp_path):
    """load_eegbci output satisfies the NerveML data contract."""
    df = load_eegbci(n_subjects=N_SUBJECTS, runs=RUNS)

    assert isinstance(df, pd.DataFrame), "expected a DataFrame"
    assert len(df) > 0, "no rows produced"

    # Required metadata columns.
    for col in ("subject_id", "session_id", "trial_id", "target_label"):
        assert col in df.columns, f"missing required column: {col}"

    # Both motor imagery classes must appear.
    assert set(df["target_label"].unique()) == {
        0,
        1,
    }, "expected both imagery classes (0 and 1)"

    # Subjects are distinct.
    assert df["subject_id"].nunique() == N_SUBJECTS

    # Feature columns follow the <channel>_<band> convention.
    feat_cols = [c for c in df.columns if c not in ("subject_id", "session_id", "trial_id", "target_label")]
    assert len(feat_cols) > 0, "no feature columns found"
    assert all("_" in c for c in feat_cols), "expected <channel>_<band> naming"

    # validate_dataset must not raise.
    summary = validate_dataset(df, feat_cols, n_splits=N_SPLITS)
    assert summary.n_subjects == N_SUBJECTS
    assert summary.n_trials == len(df)


@pytest.mark.slow
def test_run_scan_on_real_eegbci_data(tmp_path):
    """Full run_scan pipeline on real PhysioNet data produces a valid report."""
    # Build and cache the feature table.
    cache_csv = tmp_path / "eegbci_2sub.csv"
    df = load_eegbci(n_subjects=N_SUBJECTS, runs=RUNS, cache_path=str(cache_csv))
    df.to_csv(cache_csv, index=False)  # ensure file exists for run_scan's path dispatch

    report = run_scan(
        dataset=str(cache_csv),
        n_splits=N_SPLITS,
        n_permutations=50,   # small — this is a smoke test, not a power study
        top_k=5,
        n_jobs=1,
    )

    # --- top-level structure ---
    required_top = {"schema_version", "dataset", "evaluation", "permutation_test",
                    "permitted_claim", "unsupported_claims", "recommendations"}
    missing = required_top - set(report.keys())
    assert not missing, f"report missing top-level keys: {missing}"

    # --- schema version ---
    assert report["schema_version"] == "0.2.0"

    # --- dataset sub-dict ---
    ds = report["dataset"]
    assert ds["modality"] == "feature_table"
    assert ds["n_subjects"] == N_SUBJECTS
    assert ds["n_trials"] == len(df)
    assert ds["n_features"] > 0

    # --- evaluation sub-dict: all core metrics must be finite floats in [0,1] ---
    ev = report["evaluation"]
    for field in ("grouped_balanced_accuracy", "grouped_roc_auc",
                  "trial_random_balanced_accuracy", "generalization_gap"):
        val = ev[field]
        assert isinstance(val, (int, float)), f"evaluation.{field} is not numeric: {val!r}"
        assert math.isfinite(val), f"evaluation.{field} is not finite: {val}"

    assert 0.0 <= ev["grouped_balanced_accuracy"] <= 1.0
    assert 0.0 <= ev["trial_random_balanced_accuracy"] <= 1.0

    # --- permutation_test sub-dict ---
    pt = report["permutation_test"]
    p_value = pt["p_value"]
    assert isinstance(p_value, (int, float)) and math.isfinite(p_value)
    assert 0.0 <= p_value <= 1.0

    # --- permitted_claim must be non-empty and must not overclaim ---
    claim = report["permitted_claim"]
    assert isinstance(claim, str) and len(claim) > 20, "permitted_claim too short"
    for word in ("proven", "certain", "definitively", "guarantees"):
        assert word not in claim.lower(), f"overclaim word '{word}' found in permitted_claim"

    # --- report must be JSON-serialisable ---
    reloaded = json.loads(json.dumps(report))
    assert reloaded["schema_version"] == report["schema_version"]


@pytest.mark.slow
def test_eegbci_cache_is_stable(tmp_path):
    """load_eegbci with cache_path returns a consistent DataFrame on re-read."""
    cache = tmp_path / "cache.csv"

    df1 = load_eegbci(n_subjects=N_SUBJECTS, runs=RUNS, cache_path=str(cache))
    assert cache.exists(), "cache file not created after first call"

    df2 = load_eegbci(n_subjects=N_SUBJECTS, runs=RUNS, cache_path=str(cache))

    assert len(df1) == len(df2), "cached vs live row count mismatch"
    assert list(df1.columns) == list(df2.columns), "cached vs live column mismatch"
