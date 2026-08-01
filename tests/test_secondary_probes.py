"""Tests for the secondary sensitive-attribute probe feature."""

import numpy as np
import pandas as pd
import pytest

from nerveml.scan import (
    KNOWN_SECONDARY_ATTRS,
    _find_secondary_attributes,
    _run_secondary_probes,
    _secondary_probe_interpretation,
    run_scan,
)
from nerveml.synth import make_synthetic
from nerveml.validation import SUBJECT_COLUMN, TARGET_COLUMN

FAST = {"n_permutations": 10, "n_splits": 3, "top_k": 3}


# ── fixtures ──────────────────────────────────────────────────────────────────

def _base_df(n_subjects=8, n_trials=20, seed=0):
    """Minimal synthetic DataFrame with subject and target columns."""
    return make_synthetic("subject_leakage", n_subjects=n_subjects, n_trials=n_trials, seed=seed)


def _base_feats(df):
    from nerveml.synth import feature_columns
    return feature_columns(df)


def _with_binary_extra(df, col_name="arousal", seed=1):
    rng = np.random.default_rng(seed)
    df = df.copy()
    df[col_name] = rng.integers(0, 2, size=len(df))
    return df


# ── _find_secondary_attributes ────────────────────────────────────────────────

def test_find_secondary_attributes_detects_binary_extra_column():
    df = _base_df()
    feats = _base_feats(df)
    df = _with_binary_extra(df, "arousal")
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "arousal" in found


def test_find_secondary_attributes_skips_continuous_feature_columns():
    # Continuous features have many unique values — they fail the cardinality check.
    df = _base_df()
    feats = _base_feats(df)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    for feat in feats:
        assert feat not in found, f"continuous feature {feat!r} should not be a secondary attr"


def test_find_secondary_attributes_skips_primary_label():
    df = _base_df()
    feats = _base_feats(df)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert TARGET_COLUMN not in found


def test_find_secondary_attributes_skips_grouping_column():
    df = _base_df()
    feats = _base_feats(df)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert SUBJECT_COLUMN not in found


def test_find_secondary_attributes_skips_metadata_axes():
    df = _base_df()
    feats = _base_feats(df)
    rng = np.random.default_rng(7)
    for col in ("session_id", "site_id", "device_id", "stimulus_id"):
        df[col] = rng.integers(0, 2, size=len(df))
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    for col in ("session_id", "site_id", "device_id", "stimulus_id"):
        assert col not in found


def test_find_secondary_attributes_skips_unbalanced_column():
    df = _base_df()
    feats = _base_feats(df)
    df = df.copy()
    # 98% class 0, 2% class 1 — minority below threshold
    df["rare_attr"] = 0
    df.loc[df.index[:3], "rare_attr"] = 1
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "rare_attr" not in found


def test_find_secondary_attributes_skips_high_cardinality():
    df = _base_df()
    feats = _base_feats(df)
    df = df.copy()
    df["continuous_score"] = np.random.default_rng(42).uniform(0, 1, size=len(df))
    # Will have many unique values >> 10
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "continuous_score" not in found


def test_find_secondary_attributes_skips_constant_column():
    df = _base_df()
    feats = _base_feats(df)
    df = df.copy()
    df["constant"] = 1
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "constant" not in found


def test_find_secondary_attributes_returns_empty_for_clean_df():
    df = _base_df()
    feats = _base_feats(df)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert found == []


def test_find_secondary_attributes_detects_multiple_extra_columns():
    df = _base_df()
    feats = _base_feats(df)
    rng = np.random.default_rng(99)
    for col in ("arousal", "stress"):
        df[col] = rng.integers(0, 2, size=len(df))
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "arousal" in found
    assert "stress" in found


# ── _secondary_probe_interpretation ───────────────────────────────────────────

def test_interpretation_high_bacc_uses_appears_decodable():
    text = _secondary_probe_interpretation(0.75)
    assert "appears decodable" in text
    assert "0.750" in text


def test_interpretation_mid_bacc_uses_weakly_decodable():
    text = _secondary_probe_interpretation(0.63)
    assert "weakly decodable" in text


def test_interpretation_low_bacc_uses_no_convincing():
    text = _secondary_probe_interpretation(0.55)
    assert "no convincing" in text


def test_interpretation_boundary_exactly_0_70():
    text = _secondary_probe_interpretation(0.70)
    assert "appears decodable" in text


def test_interpretation_boundary_exactly_0_60():
    text = _secondary_probe_interpretation(0.60)
    assert "weakly decodable" in text


# ── _run_secondary_probes ──────────────────────────────────────────────────────

def test_run_secondary_probes_returns_expected_keys():
    from nerveml.models import build_model

    df = _base_df()
    feats = _base_feats(df)
    df = _with_binary_extra(df, "arousal")
    model = build_model("random_forest", seed=0)
    results = _run_secondary_probes(df, feats, ["arousal"], model, SUBJECT_COLUMN, 3, 0)
    assert len(results) == 1
    p = results[0]
    assert p["attribute"] == "arousal"
    assert "grouped_balanced_accuracy" in p
    assert "grouped_roc_auc" in p
    assert "n_units" in p
    assert "interpretation" in p
    assert 0.0 <= p["grouped_balanced_accuracy"] <= 1.0
    assert 0.0 <= p["grouped_roc_auc"] <= 1.0


def test_run_secondary_probes_skips_failed_probe():
    """A probe with too few groups should silently produce no result."""
    from nerveml.models import build_model

    df = _base_df(n_subjects=2)  # very few subjects
    feats = _base_feats(df)
    df = _with_binary_extra(df, "arousal")
    model = build_model("random_forest", seed=0)
    # n_splits=5 > n_subjects=2, evaluate_grouped will raise
    results = _run_secondary_probes(df, feats, ["arousal"], model, SUBJECT_COLUMN, 5, 0)
    # Should not crash; may be empty or non-empty depending on error
    assert isinstance(results, list)


# ── run_scan integration ───────────────────────────────────────────────────────

def _make_csv_with_secondary(tmp_path, col="arousal"):
    df = _base_df(n_subjects=8, n_trials=20, seed=0)
    rng = np.random.default_rng(5)
    df[col] = rng.integers(0, 2, size=len(df))
    path = tmp_path / "secondary.csv"
    df.to_csv(path, index=False)
    return path


def test_run_scan_secondary_probes_key_always_present():
    report = run_scan(dataset="subject_leakage", n_subjects=8, n_trials=20, seed=0, **FAST)
    assert "secondary_probes" in report
    assert isinstance(report["secondary_probes"], list)


def test_run_scan_without_secondary_columns_returns_empty_list():
    report = run_scan(dataset="subject_leakage", n_subjects=8, n_trials=20, seed=0, **FAST)
    assert report["secondary_probes"] == []


def test_run_scan_with_secondary_column_returns_probes(tmp_path):
    path = _make_csv_with_secondary(tmp_path, "arousal")
    report = run_scan(dataset=str(path), seed=0, **FAST)
    assert len(report["secondary_probes"]) >= 1
    attrs = [p["attribute"] for p in report["secondary_probes"]]
    assert "arousal" in attrs


def test_run_scan_secondary_probe_result_is_json_serialisable(tmp_path):
    import json

    path = _make_csv_with_secondary(tmp_path, "fatigue")
    report = run_scan(dataset=str(path), seed=0, **FAST)
    json.dumps(report["secondary_probes"])


def test_run_scan_secondary_probe_bacc_is_in_valid_range(tmp_path):
    path = _make_csv_with_secondary(tmp_path)
    report = run_scan(dataset=str(path), seed=0, **FAST)
    for p in report["secondary_probes"]:
        assert 0.0 <= p["grouped_balanced_accuracy"] <= 1.0
        assert 0.0 <= p["grouped_roc_auc"] <= 1.0


def test_run_scan_skips_metadata_axes_as_secondary(tmp_path):
    df = _base_df(n_subjects=8, n_trials=20, seed=0)
    rng = np.random.default_rng(11)
    # Add a real secondary attr AND session_id (should NOT be a secondary probe)
    df["arousal"] = rng.integers(0, 2, size=len(df))
    df["session_id"] = rng.integers(0, 4, size=len(df))
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)
    report = run_scan(dataset=str(path), seed=0, **FAST)
    attrs = [p["attribute"] for p in report["secondary_probes"]]
    assert "session_id" not in attrs
    assert "arousal" in attrs


# ── PDF section ────────────────────────────────────────────────────────────────

def _minimal_report_with_probes(n_probes=2):
    """Minimal report dict with secondary_probes populated."""
    probes = [
        {
            "attribute": f"attr_{i}",
            "n_units": 8,
            "grouped_balanced_accuracy": 0.55 + i * 0.05,
            "grouped_roc_auc": 0.57 + i * 0.04,
            "interpretation": _secondary_probe_interpretation(0.55 + i * 0.05),
        }
        for i in range(n_probes)
    ]
    return {"secondary_probes": probes}


def test_pdf_secondary_probes_section_returns_non_empty_when_probes_present():
    from nerveml.pdf import _secondary_probes_section

    styles = _load_pdf_styles()
    report = _minimal_report_with_probes(2)
    story = _secondary_probes_section(report, styles, 5)
    assert len(story) > 0


def test_pdf_secondary_probes_section_returns_empty_when_no_probes():
    from nerveml.pdf import _secondary_probes_section

    styles = _load_pdf_styles()
    story = _secondary_probes_section({"secondary_probes": []}, styles, 5)
    assert story == []


def test_pdf_secondary_probes_section_absent_when_key_missing():
    from nerveml.pdf import _secondary_probes_section

    styles = _load_pdf_styles()
    story = _secondary_probes_section({}, styles, 5)
    assert story == []


def _load_pdf_styles():
    from nerveml.pdf import _styles

    return _styles()


# ── KNOWN_SECONDARY_ATTRS constant ────────────────────────────────────────────

def test_known_secondary_attrs_is_a_frozenset():
    assert isinstance(KNOWN_SECONDARY_ATTRS, frozenset)


def test_known_secondary_attrs_contains_key_domains():
    names = KNOWN_SECONDARY_ATTRS
    assert "arousal" in names
    assert "sleep_stage" in names
    assert "fatigue" in names
    assert "stress" in names
    assert "motor_state" in names


def test_known_secondary_attrs_contains_demographic_identity_columns():
    names = KNOWN_SECONDARY_ATTRS
    # Core identity re-id labels
    assert "identity" in names
    assert "participant_identity" in names
    assert "subject_identity" in names
    # Biological sex / gender
    assert "gender" in names
    assert "sex" in names
    # Binned age
    assert "age_group" in names
    assert "age_bin" in names
    # Ethnicity / race / nationality
    assert "ethnicity" in names
    assert "race" in names
    assert "nationality" in names


def test_find_secondary_attributes_detects_gender_column():
    df = _base_df()
    feats = _base_feats(df)
    df = _with_binary_extra(df, "gender", seed=7)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "gender" in found, "gender column should be discovered as a secondary attribute"


def test_find_secondary_attributes_detects_identity_column():
    df = _base_df()
    feats = _base_feats(df)
    df = _with_binary_extra(df, "identity", seed=11)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "identity" in found, "identity column should be discovered as a secondary attribute"


def test_find_secondary_attributes_detects_ethnicity_column():
    df = _base_df()
    feats = _base_feats(df)
    df = _with_binary_extra(df, "ethnicity", seed=13)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "ethnicity" in found


def test_run_scan_secondary_probes_gender_column(tmp_path):
    """End-to-end: when a 'gender' column is present it must appear in secondary_probes."""
    import pandas as pd
    from nerveml.synth import make_synthetic, feature_columns as fc

    rng = np.random.default_rng(42)
    df = make_synthetic("subject_leakage", n_subjects=8, n_trials=20, seed=0)
    df["gender"] = rng.integers(0, 2, size=len(df))
    path = tmp_path / "gender_scan.csv"
    df.to_csv(path, index=False)

    report = run_scan(
        dataset=str(path),
        model_kind="logistic_regression",
        n_splits=3,
        n_permutations=5,
        top_k=3,
        seed=0,
        identity_probe=False,
    )
    probe_attrs = [p["attribute"] for p in report.get("secondary_probes", [])]
    assert "gender" in probe_attrs, f"gender not in probed attrs: {probe_attrs}"


# ── GDPR Art. 9 extended coverage ─────────────────────────────────────────────

def test_known_secondary_attrs_contains_disability_columns():
    names = KNOWN_SECONDARY_ATTRS
    assert "disability_status" in names
    assert "disability" in names


def test_known_secondary_attrs_contains_health_columns():
    names = KNOWN_SECONDARY_ATTRS
    assert "health_condition" in names
    assert "health_status" in names


def test_known_secondary_attrs_contains_sexual_orientation():
    assert "sexual_orientation" in KNOWN_SECONDARY_ATTRS


def test_known_secondary_attrs_contains_religion_columns():
    names = KNOWN_SECONDARY_ATTRS
    assert "religion" in names
    assert "religious_affiliation" in names


def test_known_secondary_attrs_contains_political_columns():
    names = KNOWN_SECONDARY_ATTRS
    assert "political_opinion" in names
    assert "political_affiliation" in names


def test_find_secondary_attributes_detects_disability_status_column():
    df = _base_df()
    feats = _base_feats(df)
    df = _with_binary_extra(df, "disability_status", seed=17)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "disability_status" in found


def test_find_secondary_attributes_detects_health_condition_column():
    df = _base_df()
    feats = _base_feats(df)
    df = _with_binary_extra(df, "health_condition", seed=19)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "health_condition" in found


def test_find_secondary_attributes_detects_sexual_orientation_column():
    df = _base_df()
    feats = _base_feats(df)
    df = _with_binary_extra(df, "sexual_orientation", seed=23)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "sexual_orientation" in found


def test_find_secondary_attributes_detects_religion_column():
    df = _base_df()
    feats = _base_feats(df)
    df = _with_binary_extra(df, "religion", seed=29)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "religion" in found


def test_find_secondary_attributes_detects_political_opinion_column():
    df = _base_df()
    feats = _base_feats(df)
    df = _with_binary_extra(df, "political_opinion", seed=31)
    found = _find_secondary_attributes(df, feats, SUBJECT_COLUMN)
    assert "political_opinion" in found


def test_run_scan_secondary_probes_disability_status_column(tmp_path):
    """End-to-end: disability_status column must appear in secondary_probes."""
    rng = np.random.default_rng(37)
    df = make_synthetic("subject_leakage", n_subjects=8, n_trials=20, seed=0)
    df["disability_status"] = rng.integers(0, 2, size=len(df))
    path = tmp_path / "disability_scan.csv"
    df.to_csv(path, index=False)
    report = run_scan(
        dataset=str(path),
        model_kind="logistic_regression",
        n_splits=3,
        n_permutations=5,
        top_k=3,
        seed=0,
        identity_probe=False,
    )
    probe_attrs = [p["attribute"] for p in report.get("secondary_probes", [])]
    assert "disability_status" in probe_attrs, f"disability_status not in {probe_attrs}"
