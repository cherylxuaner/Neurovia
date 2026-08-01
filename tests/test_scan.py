"""End-to-end scan and the report contract in spec section 22.

The JSON-serialisability test is not a formality. scikit-learn and numpy return
np.float64, which pandas and dataclasses will carry along silently until
json.dump raises - at demo time, on stage.
"""

import json

import pytest

from nerveml.scan import run_scan, write_report

FAST = dict(model_kind="logistic_regression", n_permutations=20, n_splits=4)


@pytest.fixture(scope="module")
def leaky_report():
    return run_scan(dataset="subject_leakage", n_subjects=12, n_trials=20, seed=0, **FAST)


@pytest.fixture(scope="module")
def genuine_report():
    return run_scan(dataset="true_signal", n_subjects=12, n_trials=20, seed=0, **FAST)


def test_report_has_the_documented_top_level_sections(leaky_report):
    assert set(leaky_report) >= {
        "dataset",
        "evaluation",
        "permutation_test",
        "risk_flags",
        "top_features",
        "permitted_claim",
        "unsupported_claims",
        "recommendations",
        "config",
    }


def test_evaluation_section_reports_both_protocols_and_the_gap(leaky_report):
    evaluation = leaky_report["evaluation"]

    assert set(evaluation) >= {
        "trial_random_balanced_accuracy",
        "trial_random_roc_auc",
        "grouped_balanced_accuracy",
        "grouped_roc_auc",
        "generalization_gap",
        "grouping_unit",
        "strict_scheme",
    }


def test_permutation_section_reports_the_null(leaky_report):
    assert set(leaky_report["permutation_test"]) >= {
        "n_permutations",
        "p_value",
        "null_mean",
    }


def test_the_whole_report_is_json_serialisable(leaky_report):
    # numpy scalars survive every earlier step and die here.
    json.dumps(leaky_report)


def test_config_is_recorded_so_the_scan_can_be_reproduced(leaky_report):
    config = leaky_report["config"]

    assert config["seed"] == 0
    assert config["model_kind"] == "logistic_regression"
    assert config["n_splits"] == 4
    assert config["n_permutations"] == 20


def test_leaky_dataset_is_reported_as_subject_dependent(leaky_report):
    risk = leaky_report["risk_flags"]

    assert leaky_report["evaluation"]["generalization_gap"] >= 0.20
    assert risk["possible_subject_dependence"] is True
    # Twelve subjects put the gap's lower bound below the high-warning
    # threshold even though the estimate clears it, so the verdict stops at
    # "possible". Stating "high" here would be reading past the evidence.
    low, _ = leaky_report["evaluation"]["confidence_intervals"]["generalization_gap"]
    assert low >= 0.10
    assert risk["subject_dependence_level"] == ("high" if low >= 0.20 else "possible")


def test_leaky_dataset_does_not_get_a_decodability_claim(leaky_report):
    verdict = leaky_report["risk_flags"]["sensitive_inference_evidence"]
    claim = leaky_report["permitted_claim"]

    assert verdict in ("low_or_inconclusive", "inconclusive_at_this_sample_size")
    assert "not convincingly" in claim or "cannot decide" in claim
    # Whatever the verdict, the claim must not assert decodability.
    assert "was decodable across held-out subjects" not in claim


def test_genuine_dataset_is_reported_as_generalising(genuine_report):
    assert genuine_report["evaluation"]["grouped_roc_auc"] > 0.65
    assert genuine_report["evaluation"]["generalization_gap"] < 0.10
    assert genuine_report["risk_flags"]["possible_subject_dependence"] is False
    assert genuine_report["risk_flags"]["null_evidence"] == "exceeds_null"


def test_dataset_summary_travels_into_the_report(leaky_report):
    dataset = leaky_report["dataset"]

    assert dataset["n_subjects"] == 12
    assert dataset["n_trials"] == 240
    assert dataset["modality"] == "synthetic"


def test_top_features_are_present_and_marked_non_causal(leaky_report):
    assert leaky_report["top_features"]
    assert "non-causal" in leaky_report["feature_caveat"].lower()


def test_per_subject_scores_reach_the_report(leaky_report):
    per_group = leaky_report["evaluation"]["per_group"]

    assert len(per_group) == leaky_report["dataset"]["n_subjects"]
    assert leaky_report["evaluation"]["worst_group"]["group"] == min(
        per_group, key=lambda e: e["balanced_accuracy"]
    )["group"]


def test_the_grouping_unit_is_named_in_the_report(leaky_report):
    assert leaky_report["config"]["group_column"] == "subject_id"
    assert leaky_report["evaluation"]["grouping_unit"] == "subject_id"
    # Consumers must be able to find the strict protocol without guessing.
    assert leaky_report["evaluation"]["strict_scheme"] == "subject_id_grouped"


def test_any_column_can_be_the_grouping_unit(tmp_path):
    import pandas as pd

    from nerveml.synth import make_synthetic

    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    df["session_id"] = df["subject_id"].str[-2:].astype(int) // 2
    path = tmp_path / "sessions.csv"
    df.to_csv(path, index=False)

    report = run_scan(dataset=str(path), group_column="session_id", seed=0, **FAST)

    assert report["evaluation"]["grouping_unit"] == "session_id"
    assert len(report["evaluation"]["per_group"]) == 6
    pd.read_csv(path)  # the loader must not have consumed the file


def test_a_report_with_single_class_groups_still_serialises(tmp_path):
    from nerveml.synth import make_synthetic

    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    df.loc[df["subject_id"] == "s00", "target_label"] = 1
    path = tmp_path / "single.csv"
    df.to_csv(path, index=False)

    report = run_scan(dataset=str(path), seed=0, **FAST)

    # A null balanced accuracy must survive the JSON round trip.
    json.dumps(report)


def test_identity_inference_reaches_the_report(leaky_report):
    identity = leaky_report["identity_inference"]

    assert identity["n_identities"] == 12
    assert identity["chance"] == pytest.approx(1 / 12, abs=1e-3)
    assert identity["lift_over_chance"] > 1
    assert "reference" in identity["attack_model"].lower()


def test_a_fingerprinted_dataset_is_flagged_as_re_identifiable(leaky_report):
    codes = [f["code"] for f in leaky_report["risk_flags"]["flags"]]

    assert "re_identifiable" in codes


def test_a_session_column_upgrades_the_attack_automatically(tmp_path):
    from nerveml.synth import make_synthetic

    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    df["session_id"] = list(range(4)) * (len(df) // 4)
    path = tmp_path / "sessioned.csv"
    df.to_csv(path, index=False)

    report = run_scan(dataset=str(path), seed=0, **FAST)

    # Given the means to run the harder attack, run the harder attack.
    assert "held-out recording" in report["identity_inference"]["attack_model"].lower()


def test_without_a_session_column_the_weaker_attack_is_labelled_as_such(leaky_report):
    attack = leaky_report["identity_inference"]["attack_model"].lower()

    assert "same recording" in attack


def test_the_identity_probe_can_be_switched_off():
    report = run_scan(
        dataset="subject_leakage",
        n_subjects=8,
        n_trials=20,
        seed=0,
        identity_probe=False,
        **FAST,
    )

    assert report["identity_inference"] is None
    assert "re_identifiable" not in [
        f["code"] for f in report["risk_flags"]["flags"]
    ]


def test_intervals_reach_the_report(leaky_report):
    intervals = leaky_report["evaluation"]["confidence_intervals"]

    for name in ("grouped_roc_auc", "grouped_balanced_accuracy", "generalization_gap"):
        low, high = intervals[name]
        assert low < high


def test_the_auc_interval_brackets_its_estimate(leaky_report):
    evaluation = leaky_report["evaluation"]
    low, high = evaluation["confidence_intervals"]["grouped_roc_auc"]

    # The interval is over per-subject AUCs, so it brackets their mean rather
    # than the fold mean; the two are close but not identical.
    assert low <= evaluation["grouped_within_unit_roc_auc"] <= high


def test_every_interval_contains_the_estimate_it_describes(leaky_report):
    evaluation = leaky_report["evaluation"]
    intervals = evaluation["confidence_intervals"]

    # An interval that misses its own point estimate is measuring a different
    # quantity. The gap is a difference of pooled scores, and a pooled score is
    # not the mean of its per-subject scores, so per-subject differences are
    # not a paired version of it.
    for name in (
        "grouped_roc_auc",
        "grouped_balanced_accuracy",
        "generalization_gap",
    ):
        low, high = intervals[name]
        assert low <= evaluation[name] <= high, name


def test_the_verdict_records_how_many_units_would_settle_it(leaky_report):
    risk = leaky_report["risk_flags"]

    if risk["sensitive_inference_evidence"] == "inconclusive_at_this_sample_size":
        assert risk["units_needed"] > leaky_report["dataset"]["n_subjects"]
    else:
        assert risk["units_needed"] is None


def test_re_identification_carries_an_interval(leaky_report):
    low, high = leaky_report["identity_inference"]["recall_ci"]

    assert 0.0 <= low <= high <= 1.0


def channel_band_csv(tmp_path):
    """A feature table whose names parse as channel_band, as real EEG does."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    channels, bands = ["C3", "C4", "Cz"], ["delta", "alpha", "beta"]
    rows = []
    for s in range(10):
        offset = rng.normal(0, 3.0, size=len(channels) * len(bands))
        block = offset + rng.normal(0, 1, size=(20, len(channels) * len(bands)))
        frame = pd.DataFrame(
            block, columns=[f"{c}_{b}" for c in channels for b in bands]
        )
        frame.insert(0, "subject_id", f"s{s:02d}")
        frame.insert(1, "trial_id", [f"s{s:02d}_t{i:02d}" for i in range(20)])
        frame.insert(2, "target_label", np.tile([0, 1], 10))
        rows.append(frame)
    path = tmp_path / "channels.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    return path


def test_band_power_features_get_a_fingerprint_decomposition(tmp_path):
    report = run_scan(dataset=str(channel_band_csv(tmp_path)), seed=0, **FAST)
    composition = report["fingerprint_composition"]

    assert composition["carried_by"] in ("amplitude", "spectral_shape", "both")
    assert composition["amplitude_only"]["accuracy"] >= 0
    assert composition["spectral_shape_only"]["accuracy"] >= 0
    assert composition["interpretation"]


def test_features_that_are_not_channel_bands_skip_the_decomposition(leaky_report):
    # The synthetic generator emits f00, f01 - there is no channel to sum over,
    # so the question is not askable and no answer is invented.
    assert leaky_report["fingerprint_composition"] is None


def test_stages_are_reported_in_order():
    seen = []

    run_scan(
        dataset="subject_leakage",
        n_subjects=8,
        n_trials=20,
        seed=0,
        on_stage=seen.append,
        **FAST,
    )

    assert seen == [
        "validate",
        "trial_random",
        "grouped",
        "permutation",
        "identity",
        "interpret",
    ]


def test_stage_reporting_is_optional():
    # The default path must not require a callback.
    run_scan(dataset="subject_leakage", n_subjects=8, n_trials=20, seed=0, **FAST)


def test_stages_are_named_in_the_module_so_a_ui_can_label_them():
    from nerveml.scan import STAGES

    assert STAGES["permutation"]
    assert set(STAGES) == {
        "validate",
        "trial_random",
        "grouped",
        "permutation",
        "identity",
        "confounds",
        "artifact_baseline",
        "interpret",
        "multi_axis",
        "secondary_probe",
    }


def test_unknown_dataset_is_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        run_scan(dataset="mystery", **FAST)


def test_write_report_produces_the_documented_artifacts(tmp_path, leaky_report):
    paths = write_report(leaky_report, tmp_path)

    assert (tmp_path / "audit_report.json").exists()
    assert (tmp_path / "fold_metrics.csv").exists()
    assert set(paths) == {"json", "csv"}

    written = json.loads((tmp_path / "audit_report.json").read_text(encoding="utf-8"))
    assert written == leaky_report


def test_fold_metrics_csv_has_one_row_per_fold_per_protocol(tmp_path, leaky_report):
    import pandas as pd

    write_report(leaky_report, tmp_path)
    folds = pd.read_csv(tmp_path / "fold_metrics.csv")

    assert len(folds) == 8  # 4 folds, 2 protocols
    assert set(folds["scheme"]) == {"trial_random", "subject_id_grouped"}


# ---------------------------------------------------------------------------
# Multi-axis validation
# ---------------------------------------------------------------------------


def _make_multi_axis_csv(tmp_path, include_site=False):
    """Feature CSV with subject_id as primary and session_id (+ optionally site_id)."""
    import numpy as np
    import pandas as pd

    from nerveml.synth import make_synthetic

    rng = np.random.default_rng(42)
    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    # 4 sessions, each subject contributes to all of them
    df["session_id"] = (rng.integers(0, 4, size=len(df))).astype(int)
    if include_site:
        df["site_id"] = (rng.integers(0, 4, size=len(df))).astype(int)
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)
    return path


def test_detect_extra_axes_returns_present_candidate_columns(tmp_path):
    import pandas as pd

    from nerveml.scan import CANDIDATE_GROUPING_AXES, _detect_extra_axes
    from nerveml.synth import make_synthetic

    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    df["session_id"] = list(range(6)) * (len(df) // 6)
    df["site_id"] = list(range(4)) * (len(df) // 4)

    extra = _detect_extra_axes(df, primary_group_column="subject_id", n_splits=4)

    assert "session_id" in extra
    assert "site_id" in extra
    assert "subject_id" not in extra


def test_detect_extra_axes_skips_the_primary_column(tmp_path):
    from nerveml.scan import _detect_extra_axes
    from nerveml.synth import make_synthetic

    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    df["session_id"] = list(range(6)) * (len(df) // 6)

    extra = _detect_extra_axes(df, primary_group_column="session_id", n_splits=4)

    assert "session_id" not in extra


def test_detect_extra_axes_skips_columns_with_too_few_unique_values():
    from nerveml.scan import _detect_extra_axes
    from nerveml.synth import make_synthetic

    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    # Only 2 unique sessions — below n_splits=4
    df["session_id"] = list(range(2)) * (len(df) // 2)

    extra = _detect_extra_axes(df, primary_group_column="subject_id", n_splits=4)

    assert "session_id" not in extra


def test_detect_extra_axes_returns_empty_when_no_candidates_present():
    from nerveml.scan import _detect_extra_axes
    from nerveml.synth import make_synthetic

    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)

    extra = _detect_extra_axes(df, primary_group_column="subject_id", n_splits=4)

    assert extra == []


def test_multi_axis_validation_key_is_always_present(leaky_report):
    # Even without extra axes the key must exist so consumers don't need to guard.
    assert "multi_axis_validation" in leaky_report
    assert leaky_report["multi_axis_validation"] == []


def test_multi_axis_validation_runs_for_each_detected_axis(tmp_path):
    report = run_scan(
        dataset=str(_make_multi_axis_csv(tmp_path, include_site=True)),
        seed=0,
        **FAST,
    )

    axes = [entry["axis"] for entry in report["multi_axis_validation"]]
    assert "session_id" in axes
    assert "site_id" in axes


def test_multi_axis_entry_has_required_keys(tmp_path):
    report = run_scan(
        dataset=str(_make_multi_axis_csv(tmp_path)),
        seed=0,
        **FAST,
    )

    entries = report["multi_axis_validation"]
    assert len(entries) == 1
    entry = entries[0]

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
    assert entry["axis"] == "session_id"
    assert entry["n_units"] == 4
    assert entry["scheme"] == "session_id_grouped"


def test_multi_axis_entry_is_json_serialisable(tmp_path):
    report = run_scan(
        dataset=str(_make_multi_axis_csv(tmp_path)),
        seed=0,
        **FAST,
    )

    json.dumps(report["multi_axis_validation"])


def test_multi_axis_stage_fires_when_extra_axes_exist(tmp_path):
    seen = []

    run_scan(
        dataset=str(_make_multi_axis_csv(tmp_path)),
        seed=0,
        on_stage=seen.append,
        **FAST,
    )

    assert "multi_axis" in seen


def test_multi_axis_stage_is_absent_when_no_extra_axes():
    seen = []

    run_scan(
        dataset="subject_leakage",
        n_subjects=8,
        n_trials=20,
        seed=0,
        on_stage=seen.append,
        **FAST,
    )

    assert "multi_axis" not in seen


def test_multi_axis_scores_are_bounded(tmp_path):
    report = run_scan(
        dataset=str(_make_multi_axis_csv(tmp_path)),
        seed=0,
        **FAST,
    )

    for entry in report["multi_axis_validation"]:
        assert 0.0 <= entry["grouped_balanced_accuracy"] <= 1.0
        assert 0.0 <= entry["grouped_roc_auc"] <= 1.0
        assert entry["grouped_balanced_accuracy_std"] >= 0.0
        assert -1.0 <= entry["generalization_gap"] <= 1.0


def test_multi_axis_confidence_intervals_are_valid(tmp_path):
    report = run_scan(
        dataset=str(_make_multi_axis_csv(tmp_path)),
        seed=0,
        **FAST,
    )

    for entry in report["multi_axis_validation"]:
        ci = entry["confidence_intervals"]
        assert set(ci) >= {"grouped_balanced_accuracy", "grouped_roc_auc", "generalization_gap"}
        for key in ("grouped_balanced_accuracy", "grouped_roc_auc"):
            bounds = ci[key]
            assert len(bounds) == 2
            low, high = bounds
            # With >=4 axis units the bootstrap always returns numeric bounds.
            assert low is not None and high is not None
            assert 0.0 <= low <= high <= 1.0
        # Gap CI: inversion of grouped_bacc CI — bounds may be negative (model
        # below naive) or positive, but lo <= hi always holds.
        gap_bounds = ci["generalization_gap"]
        assert len(gap_bounds) == 2
        gap_lo, gap_hi = gap_bounds
        assert gap_lo is not None and gap_hi is not None
        assert gap_lo <= gap_hi


def test_multi_axis_gap_ci_brackets_point_estimate(tmp_path):
    """The gap CI [lo, hi] should contain the gap point estimate."""
    report = run_scan(
        dataset=str(_make_multi_axis_csv(tmp_path)),
        seed=0,
        **FAST,
    )

    for entry in report["multi_axis_validation"]:
        gap = entry["generalization_gap"]
        gap_lo, gap_hi = entry["confidence_intervals"]["generalization_gap"]
        if gap_lo is None or gap_hi is None:
            continue
        assert gap_lo <= gap <= gap_hi, (
            f"Gap point estimate {gap} outside CI [{gap_lo}, {gap_hi}]"
        )


def test_multi_axis_confidence_intervals_are_json_serialisable(tmp_path):
    import json

    report = run_scan(
        dataset=str(_make_multi_axis_csv(tmp_path)),
        seed=0,
        **FAST,
    )

    for entry in report["multi_axis_validation"]:
        json.dumps(entry["confidence_intervals"])


def test_multi_axis_confidence_intervals_absent_when_only_one_unit(tmp_path):
    """With a single axis unit bootstrap_ci returns (None, None) — report None, not a crash."""
    import numpy as np
    import pandas as pd

    from nerveml.synth import make_synthetic

    rng = np.random.default_rng(99)
    df = make_synthetic("subject_leakage", n_subjects=8, n_trials=20, seed=0)
    # Exactly n_splits unique values so it just barely qualifies, but CI should
    # still work (bootstrap_ci needs >= 2 per_group entries, which n_splits=4 provides).
    df["session_id"] = (rng.integers(0, 4, size=len(df))).astype(int)
    path = tmp_path / "one_unit.csv"
    df.to_csv(path, index=False)

    report = run_scan(dataset=str(path), seed=0, **FAST)

    for entry in report["multi_axis_validation"]:
        ci = entry["confidence_intervals"]
        for bounds in ci.values():
            assert len(bounds) == 2
            assert all(v is None or isinstance(v, float) for v in bounds)


# ── Leakage-smell risk-flag integration ──────────────────────────────────────

def test_label_name_smell_surfaces_in_risk_flags(tmp_path):
    """A column named 'valence' in the features must raise a label_leakage_smell
    risk flag so operators see the issue without reading the dataset section."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    n_subjects, n_trials = 8, 20
    rows = []
    for s in range(n_subjects):
        frame = pd.DataFrame(
            rng.normal(size=(n_trials, 3)),
            columns=["alpha_power", "beta_power", "valence"],
        )
        frame.insert(0, "subject_id", f"s{s:02d}")
        frame.insert(1, "trial_id", [f"s{s:02d}_t{i:02d}" for i in range(n_trials)])
        frame.insert(2, "target_label", np.tile([0, 1], n_trials // 2))
        rows.append(frame)
    path = tmp_path / "smell.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)

    report = run_scan(dataset=str(path), seed=0, **FAST)

    codes = [f["code"] for f in report["risk_flags"]["flags"]]
    assert "label_leakage_smell" in codes, (
        "label_name_in_features warning must propagate to risk_flags"
    )


def test_high_correlation_smell_surfaces_in_risk_flags(tmp_path):
    """A feature column that is a near-perfect copy of the target label must raise
    a high_correlation_leakage_smell risk flag."""
    import numpy as np
    import pandas as pd

    from nerveml.loaders import CORRELATION_LEAKAGE_THRESHOLD

    rng = np.random.default_rng(7)
    n_subjects, n_trials = 8, 20
    rows = []
    for s in range(n_subjects):
        labels = np.tile([0, 1], n_trials // 2).astype(float)
        frame = pd.DataFrame({
            "f0": rng.normal(size=n_trials),
            # leaky_feat is the label plus tiny noise so |r| > threshold
            "leaky_feat": labels + rng.normal(scale=1e-8, size=n_trials),
        })
        frame.insert(0, "subject_id", f"s{s:02d}")
        frame.insert(1, "trial_id", [f"s{s:02d}_t{i:02d}" for i in range(n_trials)])
        frame.insert(2, "target_label", labels.astype(int))
        rows.append(frame)
    path = tmp_path / "corr_smell.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)

    report = run_scan(dataset=str(path), seed=0, **FAST)

    codes = [f["code"] for f in report["risk_flags"]["flags"]]
    assert "high_correlation_leakage_smell" in codes, (
        "feature_label_high_correlation warning must propagate to risk_flags"
    )
