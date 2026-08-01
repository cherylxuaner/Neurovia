"""Spec section 10.1 - stop or warn before anything gets measured.

The distinction that matters: conditions that make the scan meaningless raise,
conditions that make it weaker warn. Silently scanning a dataset with no
grouping variable would produce a confident number with nothing behind it.
"""

import numpy as np
import pandas as pd
import pytest

from nerveml.loaders import (
    detect_dataset_format,
    load_deap_feature_csv,
    load_feature_csv,
    load_seed_feature_csv,
    validate_dataset,
)
from nerveml.synth import feature_columns, make_synthetic


@pytest.fixture
def clean():
    df = make_synthetic("subject_leakage", n_subjects=10, n_trials=20, n_features=8, seed=0)
    return df, feature_columns(df)


def warning_codes(summary):
    return [w["code"] for w in summary.warnings]


def test_summary_counts_the_dataset(clean):
    df, feats = clean

    summary = validate_dataset(df, feats)

    assert summary.n_subjects == 10
    assert summary.n_trials == 200
    assert summary.n_features == 8


def test_summary_reports_class_balance(clean):
    df, feats = clean

    summary = validate_dataset(df, feats)

    assert summary.class_balance == {0: 100, 1: 100}


def test_clean_dataset_produces_no_warnings(clean):
    df, feats = clean

    assert validate_dataset(df, feats).warnings == []


def test_missing_subject_column_stops_the_scan(clean):
    df, feats = clean

    with pytest.raises(ValueError, match="subject_id"):
        validate_dataset(df.drop(columns=["subject_id"]), feats)


def test_missing_target_column_stops_the_scan(clean):
    df, feats = clean

    with pytest.raises(ValueError, match="target_label"):
        validate_dataset(df.drop(columns=["target_label"]), feats)


def test_single_class_target_stops_the_scan(clean):
    df, feats = clean
    df = df.assign(target_label=1)

    with pytest.raises(ValueError, match="two classes"):
        validate_dataset(df, feats)


def test_non_binary_target_stops_the_scan(clean):
    df, feats = clean
    df = df.assign(target_label=range(len(df)))

    with pytest.raises(ValueError, match="binary"):
        validate_dataset(df, feats)


def test_single_subject_stops_the_scan(clean):
    df, feats = clean
    df = df.assign(subject_id="s00")

    with pytest.raises(ValueError, match="subject-held-out"):
        validate_dataset(df, feats)


def test_missing_feature_values_are_warned_about(clean):
    df, feats = clean
    df = df.copy()
    df.loc[0, feats[0]] = np.nan

    summary = validate_dataset(df, feats)

    assert "missing_values" in warning_codes(summary)
    assert summary.missing_values == 1


def test_duplicate_trial_ids_are_warned_about(clean):
    df, feats = clean
    df = df.copy()
    df.loc[1, "trial_id"] = df.loc[0, "trial_id"]

    summary = validate_dataset(df, feats)

    assert "duplicate_trial_ids" in warning_codes(summary)


def test_too_few_subjects_for_grouped_folds_is_warned_about(clean):
    df, feats = clean
    df = df[df["subject_id"].isin(["s00", "s01", "s02"])]

    summary = validate_dataset(df, feats, n_splits=5)

    assert "too_few_subjects" in warning_codes(summary)


def test_a_subject_with_only_one_class_is_warned_about(clean):
    df, feats = clean
    df = df.copy()
    df.loc[df["subject_id"] == "s00", "target_label"] = 1

    summary = validate_dataset(df, feats)

    assert "single_class_subjects" in warning_codes(summary)


def test_severe_class_imbalance_is_warned_about(clean):
    df, feats = clean
    df = df.copy()
    df.loc[df.index[:180], "target_label"] = 1

    summary = validate_dataset(df, feats)

    assert "class_imbalance" in warning_codes(summary)


def test_every_warning_explains_itself(clean):
    df, feats = clean
    df = df.copy()
    df.loc[0, feats[0]] = np.nan

    for warning in validate_dataset(df, feats).warnings:
        assert warning["message"]


def test_csv_round_trips_with_inferred_feature_columns(tmp_path, clean):
    df, feats = clean
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)

    loaded, loaded_feats = load_feature_csv(path)

    assert loaded_feats == feats
    assert len(loaded) == len(df)
    pd.testing.assert_frame_equal(loaded[feats], df[feats], check_exact=False)


def test_csv_loader_accepts_explicit_feature_columns(tmp_path, clean):
    df, _ = clean
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)

    _, loaded_feats = load_feature_csv(path, feature_columns=["f00", "f01"])

    assert loaded_feats == ["f00", "f01"]


def test_csv_loader_rejects_unknown_feature_columns(tmp_path, clean):
    df, _ = clean
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="not present"):
        load_feature_csv(path, feature_columns=["nope"])


# ── DEAP-style loader tests ───────────────────────────────────────────────────


def _make_deap_csv(tmp_path, n_subjects=6, n_trials=10, n_features=4, seed=0):
    """Minimal DEAP-style CSV: participant_id, valence, arousal, dominance,
    liking, and n_features feature columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for t in range(n_trials):
            row = {
                "participant_id": f"p{s:02d}",
                "valence": rng.uniform(1, 9),
                "arousal": rng.uniform(1, 9),
                "dominance": rng.uniform(1, 9),
                "liking": rng.uniform(1, 9),
            }
            for f in range(n_features):
                row[f"feat_{f:02d}"] = rng.standard_normal()
            rows.append(row)
    df = pd.DataFrame(rows)
    path = tmp_path / "deap_features.csv"
    df.to_csv(path, index=False)
    return path, df


def test_deap_loader_renames_participant_id_to_subject_id(tmp_path):
    path, _ = _make_deap_csv(tmp_path)
    df, _ = load_deap_feature_csv(path)
    assert "subject_id" in df.columns
    assert "participant_id" not in df.columns


def test_deap_loader_binarises_valence_at_threshold(tmp_path):
    path, raw = _make_deap_csv(tmp_path)
    df, _ = load_deap_feature_csv(path, target="valence", threshold=5.0)
    expected = (raw["valence"] > 5.0).astype(int).values
    assert (df["target_label"].values == expected).all()


def test_deap_loader_respects_custom_threshold(tmp_path):
    path, raw = _make_deap_csv(tmp_path)
    df, _ = load_deap_feature_csv(path, target="valence", threshold=7.0)
    expected = (raw["valence"] > 7.0).astype(int).values
    assert (df["target_label"].values == expected).all()


def test_deap_loader_accepts_arousal_target(tmp_path):
    path, raw = _make_deap_csv(tmp_path)
    df, _ = load_deap_feature_csv(path, target="arousal")
    expected = (raw["arousal"] > 5.0).astype(int).values
    assert (df["target_label"].values == expected).all()


def test_deap_loader_excludes_all_deap_label_columns_from_features(tmp_path):
    path, _ = _make_deap_csv(tmp_path)
    _, feats = load_deap_feature_csv(path)
    for label in ("valence", "arousal", "dominance", "liking"):
        assert label not in feats, f"{label} should not be a feature column"


def test_deap_loader_synthesises_trial_id_when_absent(tmp_path):
    path, _ = _make_deap_csv(tmp_path)
    df, _ = load_deap_feature_csv(path)
    assert "trial_id" in df.columns
    assert df["trial_id"].nunique() == len(df)


def test_deap_loader_preserves_existing_trial_id(tmp_path):
    path, raw = _make_deap_csv(tmp_path)
    raw_with_trial = pd.read_csv(path)
    raw_with_trial["trial_id"] = range(len(raw_with_trial))
    raw_with_trial.to_csv(path, index=False)
    df, _ = load_deap_feature_csv(path)
    assert list(df["trial_id"]) == list(range(len(raw_with_trial)))


def test_deap_loader_rejects_missing_subject_column(tmp_path):
    path, raw = _make_deap_csv(tmp_path)
    df = pd.read_csv(path).rename(columns={"participant_id": "unknown_col"})
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="no subject column"):
        load_deap_feature_csv(bad)


def test_deap_loader_rejects_missing_target_column(tmp_path):
    path, _ = _make_deap_csv(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        load_deap_feature_csv(path, target="nonexistent_rating")


def test_deap_loader_accepts_explicit_feature_columns(tmp_path):
    path, _ = _make_deap_csv(tmp_path, n_features=4)
    _, feats = load_deap_feature_csv(path, feature_columns=["feat_00", "feat_01"])
    assert feats == ["feat_00", "feat_01"]


def test_deap_loader_rejects_unknown_explicit_feature_columns(tmp_path):
    path, _ = _make_deap_csv(tmp_path)
    with pytest.raises(ValueError, match="not present"):
        load_deap_feature_csv(path, feature_columns=["does_not_exist"])


def test_deap_loader_output_passes_validate_dataset(tmp_path):
    path, _ = _make_deap_csv(tmp_path, n_subjects=6, n_trials=20)
    df, feats = load_deap_feature_csv(path)
    # Should not raise — the normalised frame satisfies all hard requirements.
    summary = validate_dataset(df, feats)
    assert summary.n_subjects == 6


# ── SEED-style loader tests ───────────────────────────────────────────────────


def _make_seed_csv(tmp_path, n_subjects=6, n_trials_per_class=8, n_features=5,
                   encoding="signed", seed=1):
    """SEED-style CSV: subject column, emotion label, feature columns.

    encoding="signed"  → labels are -1/0/+1 (SEED-I convention)
    encoding="unsigned" → labels are 0/1/2
    """
    rng = np.random.default_rng(seed)
    if encoding == "signed":
        classes = [-1, 0, 1]
    else:
        classes = [0, 1, 2]

    rows = []
    for s in range(n_subjects):
        for cls in classes:
            for _ in range(n_trials_per_class):
                row = {"subject": f"s{s:02d}", "emotion": cls}
                for f in range(n_features):
                    row[f"de_{f:02d}"] = rng.standard_normal()
                rows.append(row)
    df = pd.DataFrame(rows)
    path = tmp_path / "seed_features.csv"
    df.to_csv(path, index=False)
    return path, df


def test_seed_loader_renames_subject_column(tmp_path):
    path, _ = _make_seed_csv(tmp_path)
    df, _ = load_seed_feature_csv(path)
    assert "subject_id" in df.columns
    assert "subject" not in df.columns


def test_seed_loader_drops_neutral_and_maps_to_binary_signed(tmp_path):
    path, raw = _make_seed_csv(tmp_path, encoding="signed")
    df, _ = load_seed_feature_csv(path)
    # Neutral (0) rows are gone.
    assert 0 not in raw[raw["emotion"] == 0].index or len(df) < len(raw)
    assert set(df["target_label"].unique()) <= {0, 1}
    # Remaining rows == non-neutral rows in raw.
    assert len(df) == len(raw[raw["emotion"] != 0])


def test_seed_loader_drops_neutral_and_maps_to_binary_unsigned(tmp_path):
    path, raw = _make_seed_csv(tmp_path, encoding="unsigned")
    df, _ = load_seed_feature_csv(path)
    # Neutral class (1 in unsigned) rows are gone.
    assert len(df) == len(raw[raw["emotion"] != 1])
    assert set(df["target_label"].unique()) <= {0, 1}


def test_seed_loader_one_vs_rest_positive_class(tmp_path):
    path, raw = _make_seed_csv(tmp_path, encoding="signed")
    df, _ = load_seed_feature_csv(path, positive_class=1)
    # All rows kept; positive class = 1 maps to label 1.
    assert len(df) == len(raw)
    expected = (raw["emotion"] == 1).astype(int).values
    assert (df["target_label"].values == expected).all()


def test_seed_loader_binary_input_passes_through(tmp_path):
    """If the CSV already has only 2 emotion values, no rows are dropped."""
    rng = np.random.default_rng(2)
    rows = [{"subject": f"s{s}", "emotion": s % 2,
              **{f"de_{f}": rng.standard_normal() for f in range(4)}}
            for s in range(6) for _ in range(10)]
    df = pd.DataFrame(rows)
    path = tmp_path / "binary.csv"
    df.to_csv(path, index=False)
    out, _ = load_seed_feature_csv(path)
    assert len(out) == len(df)
    assert set(out["target_label"].unique()) == {0, 1}


def test_seed_loader_raises_on_four_classes(tmp_path):
    rng = np.random.default_rng(3)
    rows = [{"subject": f"s{s}", "emotion": s % 4,
              **{f"de_{f}": rng.standard_normal() for f in range(3)}}
            for s in range(8) for _ in range(5)]
    df = pd.DataFrame(rows)
    path = tmp_path / "four_class.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="3 unique"):
        load_seed_feature_csv(path)


def test_seed_loader_raises_when_drop_neutral_false_on_three_class(tmp_path):
    path, _ = _make_seed_csv(tmp_path)
    with pytest.raises(ValueError, match="3-class"):
        load_seed_feature_csv(path, drop_neutral=False)


def test_seed_loader_excludes_emotion_column_from_features(tmp_path):
    path, _ = _make_seed_csv(tmp_path)
    _, feats = load_seed_feature_csv(path)
    assert "emotion" not in feats
    assert "subject" not in feats


def test_seed_loader_synthesises_trial_id(tmp_path):
    path, _ = _make_seed_csv(tmp_path)
    df, _ = load_seed_feature_csv(path)
    assert "trial_id" in df.columns


def test_seed_loader_output_passes_validate_dataset(tmp_path):
    path, _ = _make_seed_csv(tmp_path, n_subjects=6, n_trials_per_class=10)
    df, feats = load_seed_feature_csv(path)
    summary = validate_dataset(df, feats)
    assert summary.n_subjects == 6


def test_seed_loader_raises_on_missing_subject_column(tmp_path):
    path, raw = _make_seed_csv(tmp_path)
    bad_df = pd.read_csv(path).rename(columns={"subject": "nobody"})
    bad = tmp_path / "bad.csv"
    bad_df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="no subject column"):
        load_seed_feature_csv(bad)


def test_seed_loader_raises_on_missing_label_column(tmp_path):
    path, raw = _make_seed_csv(tmp_path)
    bad_df = pd.read_csv(path).rename(columns={"emotion": "mood"})
    bad = tmp_path / "bad.csv"
    bad_df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="emotion label column not found"):
        load_seed_feature_csv(bad)


def test_seed_loader_raises_on_invalid_positive_class(tmp_path):
    path, _ = _make_seed_csv(tmp_path)
    with pytest.raises(ValueError, match="positive_class"):
        load_seed_feature_csv(path, positive_class=99)


# ── detect_dataset_format tests ───────────────────────────────────────────────


def test_detect_nerveml_format(tmp_path, clean):
    df, _ = clean
    path = tmp_path / "nerveml.csv"
    df.to_csv(path, index=False)
    result = detect_dataset_format(path)
    assert result["format"] == "nerveml"
    assert result["confidence"] == "high"


def test_detect_deap_format_high_confidence(tmp_path):
    path, _ = _make_deap_csv(tmp_path)
    result = detect_dataset_format(path)
    assert result["format"] == "deap"
    assert result["confidence"] == "high"
    assert "valence" in result["detected_targets"]


def test_detect_deap_format_low_confidence_two_ratings(tmp_path):
    """Only two DEAP rating columns → low-confidence DEAP detection."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "participant_id": [f"p{i}" for i in range(20)],
        "valence": rng.uniform(1, 9, 20),
        "arousal": rng.uniform(1, 9, 20),
        "feat_0": rng.standard_normal(20),
    })
    path = tmp_path / "two_ratings.csv"
    df.to_csv(path, index=False)
    result = detect_dataset_format(path)
    assert result["format"] == "deap"
    assert result["confidence"] == "low"


def test_detect_seed_format(tmp_path):
    path, _ = _make_seed_csv(tmp_path)
    result = detect_dataset_format(path)
    assert result["format"] == "seed"
    assert result["confidence"] == "high"


def test_detect_unknown_format(tmp_path):
    df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
    path = tmp_path / "mystery.csv"
    df.to_csv(path, index=False)
    result = detect_dataset_format(path)
    assert result["format"] == "unknown"
    assert "columns" in result


# ── Leakage-smell detection ───────────────────────────────────────────────────


def _make_clean_df(n_subjects=6, n_trials_each=10, seed=0):
    """Minimal valid dataset: numeric features, no leakage signals."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for t in range(n_trials_each):
            rows.append(
                {
                    "subject_id": f"s{s}",
                    "trial_id": f"s{s}_t{t}",
                    "target_label": int(rng.integers(0, 2)),
                    "feat_alpha": rng.standard_normal(),
                    "feat_beta": rng.standard_normal(),
                }
            )
    return pd.DataFrame(rows)


def test_clean_features_produce_no_leakage_warnings():
    df = _make_clean_df()
    feats = ["feat_alpha", "feat_beta"]
    codes = warning_codes(validate_dataset(df, feats))
    assert "label_name_in_features" not in codes
    assert "feature_label_high_correlation" not in codes


def test_label_name_in_features_warns_on_exact_keyword():
    df = _make_clean_df()
    df["valence"] = np.random.default_rng(1).standard_normal(len(df))
    feats = ["feat_alpha", "valence"]
    codes = warning_codes(validate_dataset(df, feats))
    assert "label_name_in_features" in codes


def test_label_name_in_features_warns_on_substring_keyword():
    df = _make_clean_df()
    df["norm_arousal"] = np.random.default_rng(2).standard_normal(len(df))
    feats = ["feat_alpha", "norm_arousal"]
    codes = warning_codes(validate_dataset(df, feats))
    assert "label_name_in_features" in codes


def test_label_name_in_features_warns_on_emotion_column():
    df = _make_clean_df()
    df["emotion_score"] = np.random.default_rng(3).standard_normal(len(df))
    feats = ["feat_alpha", "emotion_score"]
    codes = warning_codes(validate_dataset(df, feats))
    assert "label_name_in_features" in codes


def test_label_name_in_features_message_is_informative():
    df = _make_clean_df()
    df["valence"] = np.random.default_rng(4).standard_normal(len(df))
    feats = ["feat_alpha", "valence"]
    summary = validate_dataset(df, feats)
    w = next(w for w in summary.warnings if w["code"] == "label_name_in_features")
    assert "valence" in w["message"]
    assert "leakage" in w["message"].lower()
    assert "prototype heuristic" in w["message"].lower()


def test_label_name_in_features_handles_multiple_hits():
    df = _make_clean_df()
    df["valence"] = 0.0
    df["arousal"] = 0.0
    df["emotion"] = 0.0
    feats = ["valence", "arousal", "emotion", "feat_alpha"]
    summary = validate_dataset(df, feats)
    w = next(w for w in summary.warnings if w["code"] == "label_name_in_features")
    assert w["message"]


def test_feature_label_high_correlation_warns_on_near_perfect_feature():
    df = _make_clean_df()
    # Create a feature that is effectively the target with tiny noise.
    rng = np.random.default_rng(5)
    df["leaky"] = df["target_label"].astype(float) + rng.normal(0, 0.001, len(df))
    feats = ["feat_alpha", "leaky"]
    codes = warning_codes(validate_dataset(df, feats))
    assert "feature_label_high_correlation" in codes


def test_feature_label_high_correlation_warns_on_exact_copy():
    df = _make_clean_df()
    df["exact_copy"] = df["target_label"].astype(float)
    feats = ["feat_alpha", "exact_copy"]
    codes = warning_codes(validate_dataset(df, feats))
    assert "feature_label_high_correlation" in codes


def test_feature_label_high_correlation_not_triggered_for_moderate_correlation():
    rng = np.random.default_rng(6)
    df = _make_clean_df()
    # Moderate correlation (r ≈ 0.5): should not trigger the leakage warning.
    df["moderate"] = df["target_label"].astype(float) + rng.standard_normal(len(df))
    feats = ["feat_alpha", "moderate"]
    codes = warning_codes(validate_dataset(df, feats))
    assert "feature_label_high_correlation" not in codes


def test_feature_label_high_correlation_message_is_informative():
    rng = np.random.default_rng(7)
    df = _make_clean_df()
    df["leaky"] = df["target_label"].astype(float) + rng.normal(0, 0.001, len(df))
    feats = ["feat_alpha", "leaky"]
    summary = validate_dataset(df, feats)
    w = next(
        w for w in summary.warnings if w["code"] == "feature_label_high_correlation"
    )
    assert "leaky" in w["message"]
    assert "leakage" in w["message"].lower()
    assert "prototype heuristic" in w["message"].lower()


def test_constant_feature_does_not_crash_correlation_check():
    df = _make_clean_df()
    df["constant"] = 3.14
    feats = ["feat_alpha", "constant"]
    # std=0 — must not raise, and must not trigger the correlation warning.
    codes = warning_codes(validate_dataset(df, feats))
    assert "feature_label_high_correlation" not in codes


def test_leakage_warnings_have_messages():
    df = _make_clean_df()
    df["valence"] = df["target_label"].astype(float)
    feats = ["valence"]
    for warning in validate_dataset(df, feats).warnings:
        assert warning["message"], f"warning {warning['code']!r} has no message"
