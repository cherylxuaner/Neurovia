"""Regression scan: compare two audit reports and detect metric degradation.

The tests work with minimal dict stubs that satisfy the same dotted-path
contract as a real audit_report.json. No scan needs to run.
"""

import json

import pytest

from nerveml.regression import (
    ARTIFACT_METRIC_SPECS,
    MULTI_AXIS_CI_HI_METRICS,
    MULTI_AXIS_CI_METRICS,
    MULTI_AXIS_METRIC_SPECS,
    SECONDARY_PROBE_METRIC_SPECS,
    SIGNIFICANCE_THRESHOLD,
    TRACKED_METRICS,
    MetricDiff,
    RegressionResult,
    _artifact_diffs,
    _classify,
    _get,
    _multi_axis_diffs,
    _secondary_probe_diffs,
    compare_reports,
)


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

def _stub(
    grouped_bacc=0.70,
    grouped_auc=0.75,
    trial_random_bacc=0.80,
    gap=0.10,
    p_value=0.02,
    identity_accuracy=0.30,
    permitted_claim="weakly decodable under this protocol",
    worst_group_bacc=None,
    worst_group_group=1,
):
    """Minimal report dict for comparison tests."""
    evaluation = {
        "grouped_balanced_accuracy": grouped_bacc,
        "grouped_roc_auc": grouped_auc,
        "trial_random_balanced_accuracy": trial_random_bacc,
        "generalization_gap": gap,
    }
    if worst_group_bacc is not None:
        evaluation["worst_group"] = {
            "group": worst_group_group,
            "n_trials": 20,
            "balanced_accuracy": worst_group_bacc,
            "roc_auc": worst_group_bacc,
            "accuracy": worst_group_bacc,
            "single_class": False,
        }
    return {
        "evaluation": evaluation,
        "permutation_test": {"p_value": p_value},
        "identity_inference": {"accuracy": identity_accuracy},
        "permitted_claim": permitted_claim,
    }


# ---------------------------------------------------------------------------
# _get
# ---------------------------------------------------------------------------

def test_get_walks_nested_dict():
    report = {"a": {"b": {"c": 42}}}
    assert _get(report, "a.b.c") == 42


def test_get_returns_none_for_missing_key():
    assert _get({"a": {}}, "a.b") is None


def test_get_returns_none_when_intermediate_is_none():
    assert _get({"a": None}, "a.b") is None


def test_get_returns_none_for_empty_path_steps():
    assert _get({}, "missing.key") is None


# ---------------------------------------------------------------------------
# _classify
# ---------------------------------------------------------------------------

def test_classify_regression_higher_is_better():
    status, delta, note = _classify(0.70, 0.65, higher_is_better=True, tolerance=0.02)
    assert status == "regression"
    assert abs(delta - (-0.05)) < 1e-9


def test_classify_improvement_higher_is_better():
    status, delta, _ = _classify(0.65, 0.72, higher_is_better=True, tolerance=0.02)
    assert status == "improvement"
    assert delta > 0


def test_classify_stable_within_tolerance():
    status, _, _ = _classify(0.70, 0.71, higher_is_better=True, tolerance=0.02)
    assert status == "stable"


def test_classify_regression_lower_is_better():
    status, delta, note = _classify(0.10, 0.15, higher_is_better=False, tolerance=0.03)
    assert status == "regression"
    assert delta > 0
    assert "rose" in note


def test_classify_improvement_lower_is_better():
    status, delta, _ = _classify(0.10, 0.04, higher_is_better=False, tolerance=0.03)
    assert status == "improvement"


def test_classify_unavailable_when_baseline_none():
    status, delta, _ = _classify(None, 0.70, higher_is_better=True, tolerance=0.02)
    assert status == "unavailable"
    assert delta is None


def test_classify_unavailable_when_candidate_none():
    status, delta, _ = _classify(0.70, None, higher_is_better=True, tolerance=0.02)
    assert status == "unavailable"
    assert delta is None


def test_classify_significance_flip_regression():
    # Was significant, now not.
    status, _, note = _classify(0.03, 0.08, higher_is_better=False, tolerance=None)
    assert status == "regression"
    assert "significance lost" in note


def test_classify_significance_flip_improvement():
    # Was not significant, now is.
    status, _, note = _classify(0.10, 0.03, higher_is_better=False, tolerance=None)
    assert status == "improvement"
    assert "significance gained" in note


def test_classify_significance_stable_both_significant():
    status, _, _ = _classify(0.01, 0.04, higher_is_better=False, tolerance=None)
    assert status == "stable"


def test_classify_significance_stable_both_not_significant():
    status, _, _ = _classify(0.10, 0.20, higher_is_better=False, tolerance=None)
    assert status == "stable"


# ---------------------------------------------------------------------------
# compare_reports
# ---------------------------------------------------------------------------

def test_identical_reports_produce_no_regressions():
    report = _stub()
    result = compare_reports(report, report)

    assert result.passed
    assert len(result.regressions) == 0


def test_worse_candidate_flags_regression_in_grouped_accuracy():
    baseline = _stub(grouped_bacc=0.70)
    # Drop by 0.05, above the 0.02 tolerance.
    candidate = _stub(grouped_bacc=0.65)

    result = compare_reports(baseline, candidate)

    regression_metrics = {m.metric for m in result.regressions}
    assert "evaluation.grouped_balanced_accuracy" in regression_metrics
    assert not result.passed


def test_improved_candidate_does_not_flag_regression():
    baseline = _stub(grouped_bacc=0.65)
    candidate = _stub(grouped_bacc=0.72)

    result = compare_reports(baseline, candidate)

    assert result.passed
    assert any(
        m.metric == "evaluation.grouped_balanced_accuracy" and m.status == "improvement"
        for m in result.metrics
    )


def test_growing_generalization_gap_is_a_regression():
    baseline = _stub(gap=0.10)
    candidate = _stub(gap=0.16)  # rose by 0.06, above 0.03 tolerance

    result = compare_reports(baseline, candidate)

    regression_metrics = {m.metric for m in result.regressions}
    assert "evaluation.generalization_gap" in regression_metrics
    assert not result.passed


def test_significance_lost_is_a_regression():
    baseline = _stub(p_value=0.03)
    candidate = _stub(p_value=0.08)

    result = compare_reports(baseline, candidate)

    regression_metrics = {m.metric for m in result.regressions}
    assert "permutation_test.p_value" in regression_metrics
    assert not result.passed


def test_significance_retained_is_not_a_regression():
    baseline = _stub(p_value=0.04)
    candidate = _stub(p_value=0.01)

    result = compare_reports(baseline, candidate)

    assert result.passed


def test_identity_accuracy_rise_is_a_regression():
    baseline = _stub(identity_accuracy=0.30)
    candidate = _stub(identity_accuracy=0.40)  # rose by 0.10, above 0.05 tolerance

    result = compare_reports(baseline, candidate)

    regression_metrics = {m.metric for m in result.regressions}
    assert "identity_inference.accuracy" in regression_metrics


def test_missing_identity_section_does_not_cause_regression():
    baseline = _stub()
    candidate = _stub()
    candidate["identity_inference"] = None

    result = compare_reports(baseline, candidate)

    # identity metric should be unavailable, not a regression
    unavailable = {m.metric for m in result.unavailable}
    assert "identity_inference.accuracy" in unavailable


def test_custom_thresholds_are_respected():
    baseline = _stub(grouped_bacc=0.70)
    candidate = _stub(grouped_bacc=0.69)  # drop of 0.01, within default 0.02

    # With a tighter threshold the tiny drop should be a regression.
    result = compare_reports(
        baseline,
        candidate,
        thresholds={"evaluation.grouped_balanced_accuracy": 0.005},
    )

    regression_metrics = {m.metric for m in result.regressions}
    assert "evaluation.grouped_balanced_accuracy" in regression_metrics


def test_permitted_claim_change_is_noted():
    baseline = _stub(permitted_claim="weakly decodable under this protocol")
    candidate = _stub(permitted_claim="no significant evidence of decodability")

    result = compare_reports(baseline, candidate)

    assert result.permitted_claim_changed
    assert result.baseline_claim != result.candidate_claim


def test_permitted_claim_unchanged_is_not_flagged():
    report = _stub()
    result = compare_reports(report, report)

    assert not result.permitted_claim_changed


def test_result_to_dict_is_json_serialisable():
    baseline = _stub(grouped_bacc=0.70, p_value=0.03)
    candidate = _stub(grouped_bacc=0.63, p_value=0.07)

    result = compare_reports(baseline, candidate)

    # Must not raise.
    serialised = json.dumps(result.to_dict())
    parsed = json.loads(serialised)
    assert parsed["passed"] is False
    assert parsed["n_regressions"] >= 2


def test_result_has_all_tracked_metrics():
    result = compare_reports(_stub(), _stub())

    tracked_paths = {path for path, _, _ in TRACKED_METRICS}
    reported_paths = {m.metric for m in result.metrics}
    assert tracked_paths == reported_paths


def test_metric_diff_to_dict_rounds_values():
    diff = MetricDiff(
        metric="evaluation.grouped_balanced_accuracy",
        baseline=0.700001,
        candidate=0.699998,
        delta=-0.000003,
        threshold=0.02,
        status="stable",
    )
    d = diff.to_dict()
    assert isinstance(d["baseline"], float)
    # 4 decimal places
    assert d["baseline"] == round(0.700001, 4)


def test_regression_result_properties_partition_metrics():
    baseline = _stub(grouped_bacc=0.70, p_value=0.03)
    candidate = _stub(grouped_bacc=0.63, p_value=0.08)
    candidate["identity_inference"] = None

    result = compare_reports(baseline, candidate)

    total = (
        len(result.regressions)
        + len(result.improvements)
        + len(result.stable)
        + len(result.unavailable)
    )
    assert total == len(result.metrics)


# ---------------------------------------------------------------------------
# compare_cli integration
# ---------------------------------------------------------------------------

def test_compare_cli_exits_0_on_no_regressions(tmp_path):
    from nerveml.compare_cli import main

    report = _stub()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    exit_code = main([str(path), str(path)])
    assert exit_code == 0


def test_compare_cli_exits_1_on_regressions(tmp_path):
    from nerveml.compare_cli import main

    baseline = _stub(grouped_bacc=0.70)
    candidate = _stub(grouped_bacc=0.63)

    b_path = tmp_path / "baseline.json"
    c_path = tmp_path / "candidate.json"
    b_path.write_text(json.dumps(baseline), encoding="utf-8")
    c_path.write_text(json.dumps(candidate), encoding="utf-8")

    exit_code = main([str(b_path), str(c_path)])
    assert exit_code == 1


def test_compare_cli_writes_output_file(tmp_path):
    from nerveml.compare_cli import main

    report = _stub()
    r_path = tmp_path / "report.json"
    r_path.write_text(json.dumps(report), encoding="utf-8")
    out_path = tmp_path / "comparison.json"

    main([str(r_path), str(r_path), "--out", str(out_path)])

    assert out_path.exists()
    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert "passed" in result
    assert "regressions" in result


def test_compare_cli_exits_2_on_missing_file(tmp_path):
    from nerveml.compare_cli import main

    exit_code = main(["nonexistent.json", "also_nonexistent.json"])
    assert exit_code == 2


def test_compare_cli_strict_flag_triggers_on_claim_change(tmp_path):
    from nerveml.compare_cli import main

    baseline = _stub(permitted_claim="claim A")
    candidate = _stub(permitted_claim="claim B")
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    b_path.write_text(json.dumps(baseline), encoding="utf-8")
    c_path.write_text(json.dumps(candidate), encoding="utf-8")

    exit_code = main([str(b_path), str(c_path), "--strict"])
    assert exit_code == 1


def test_compare_cli_without_strict_passes_on_claim_change_only(tmp_path):
    from nerveml.compare_cli import main

    baseline = _stub(permitted_claim="claim A")
    candidate = _stub(permitted_claim="claim B")
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    b_path.write_text(json.dumps(baseline), encoding="utf-8")
    c_path.write_text(json.dumps(candidate), encoding="utf-8")

    exit_code = main([str(b_path), str(c_path)])
    # No metric regressions, strict not set — should pass
    assert exit_code == 0


# ---------------------------------------------------------------------------
# multi-axis regression tracking
# ---------------------------------------------------------------------------

def _multi_axis_entry(axis, bacc=0.70, auc=0.75, gap=0.10):
    return {
        "axis": axis,
        "n_units": 4,
        "scheme": "GroupKFold",
        "grouped_balanced_accuracy": bacc,
        "grouped_roc_auc": auc,
        "generalization_gap": gap,
        "worst_group": None,
        "per_group": {},
    }


def _stub_with_axes(*entries):
    """Minimal report with multi_axis_validation populated."""
    s = _stub()
    s["multi_axis_validation"] = list(entries)
    return s


def test_multi_axis_diffs_empty_when_both_lists_empty():
    baseline = _stub()
    baseline["multi_axis_validation"] = []
    candidate = _stub()
    candidate["multi_axis_validation"] = []
    diffs = _multi_axis_diffs(baseline, candidate, {})
    assert diffs == []


def test_multi_axis_diffs_empty_when_key_absent():
    # Reports that predate multi_axis_validation (no key at all)
    diffs = _multi_axis_diffs(_stub(), _stub(), {})
    assert diffs == []


def test_multi_axis_diffs_produces_three_metrics_per_axis():
    b = _stub_with_axes(_multi_axis_entry("session_id"))
    c = _stub_with_axes(_multi_axis_entry("session_id"))
    diffs = _multi_axis_diffs(b, c, {})
    assert len(diffs) == 3
    paths = {d.metric for d in diffs}
    assert "multi_axis_validation[session_id].grouped_balanced_accuracy" in paths
    assert "multi_axis_validation[session_id].grouped_roc_auc" in paths
    assert "multi_axis_validation[session_id].generalization_gap" in paths


def test_multi_axis_diffs_stable_when_values_identical():
    b = _stub_with_axes(_multi_axis_entry("session_id", bacc=0.72, auc=0.78, gap=0.08))
    c = _stub_with_axes(_multi_axis_entry("session_id", bacc=0.72, auc=0.78, gap=0.08))
    diffs = _multi_axis_diffs(b, c, {})
    assert all(d.status == "stable" for d in diffs)


def test_multi_axis_bacc_drop_is_regression():
    b = _stub_with_axes(_multi_axis_entry("session_id", bacc=0.72))
    c = _stub_with_axes(_multi_axis_entry("session_id", bacc=0.68))  # drop 0.04 > tol 0.02
    diffs = _multi_axis_diffs(b, c, {})
    bacc_diff = next(d for d in diffs if "grouped_balanced_accuracy" in d.metric)
    assert bacc_diff.status == "regression"


def test_multi_axis_bacc_small_drop_is_stable():
    b = _stub_with_axes(_multi_axis_entry("session_id", bacc=0.72))
    c = _stub_with_axes(_multi_axis_entry("session_id", bacc=0.71))  # drop 0.01 within tol 0.02
    diffs = _multi_axis_diffs(b, c, {})
    bacc_diff = next(d for d in diffs if "grouped_balanced_accuracy" in d.metric)
    assert bacc_diff.status == "stable"


def test_multi_axis_gap_rise_is_regression():
    b = _stub_with_axes(_multi_axis_entry("session_id", gap=0.10))
    c = _stub_with_axes(_multi_axis_entry("session_id", gap=0.15))  # rise 0.05 > tol 0.03
    diffs = _multi_axis_diffs(b, c, {})
    gap_diff = next(d for d in diffs if "generalization_gap" in d.metric)
    assert gap_diff.status == "regression"


def test_multi_axis_gap_fall_is_improvement():
    b = _stub_with_axes(_multi_axis_entry("session_id", gap=0.15))
    c = _stub_with_axes(_multi_axis_entry("session_id", gap=0.08))  # fell 0.07 > tol 0.03
    diffs = _multi_axis_diffs(b, c, {})
    gap_diff = next(d for d in diffs if "generalization_gap" in d.metric)
    assert gap_diff.status == "improvement"


def test_multi_axis_axis_only_in_baseline_is_unavailable():
    b = _stub_with_axes(_multi_axis_entry("session_id"), _multi_axis_entry("site_id"))
    c = _stub_with_axes(_multi_axis_entry("session_id"))  # site_id missing in candidate
    diffs = _multi_axis_diffs(b, c, {})
    site_diffs = [d for d in diffs if "site_id" in d.metric]
    assert len(site_diffs) == 3
    assert all(d.status == "unavailable" for d in site_diffs)


def test_multi_axis_axis_only_in_candidate_is_unavailable():
    b = _stub_with_axes(_multi_axis_entry("session_id"))
    c = _stub_with_axes(_multi_axis_entry("session_id"), _multi_axis_entry("site_id"))
    diffs = _multi_axis_diffs(b, c, {})
    site_diffs = [d for d in diffs if "site_id" in d.metric]
    assert all(d.status == "unavailable" for d in site_diffs)


def test_multi_axis_tolerance_override_respected():
    # With default tol 0.02 a drop of 0.015 is stable; with override 0.01 it's a regression.
    b = _stub_with_axes(_multi_axis_entry("session_id", bacc=0.72))
    c = _stub_with_axes(_multi_axis_entry("session_id", bacc=0.705))  # drop 0.015
    path = "multi_axis_validation[session_id].grouped_balanced_accuracy"
    diffs_default = _multi_axis_diffs(b, c, {})
    diffs_strict = _multi_axis_diffs(b, c, {path: 0.01})
    bacc_default = next(d for d in diffs_default if "grouped_balanced_accuracy" in d.metric)
    bacc_strict = next(d for d in diffs_strict if "grouped_balanced_accuracy" in d.metric)
    assert bacc_default.status == "stable"
    assert bacc_strict.status == "regression"


def test_compare_reports_multi_axis_regression_fails():
    """compare_reports passes=False when a multi-axis metric regresses."""
    b = _stub_with_axes(_multi_axis_entry("session_id", bacc=0.72))
    c = _stub_with_axes(_multi_axis_entry("session_id", bacc=0.65))  # big drop
    result = compare_reports(b, c)
    assert result.passed is False
    multi_regressions = [m for m in result.regressions
                         if m.metric.startswith("multi_axis_validation")]
    assert len(multi_regressions) >= 1


def test_compare_reports_multi_axis_stable_does_not_fail():
    b = _stub_with_axes(_multi_axis_entry("session_id"))
    c = _stub_with_axes(_multi_axis_entry("session_id"))
    result = compare_reports(b, c)
    assert result.passed is True


def test_compare_reports_no_multi_axis_key_unchanged_behaviour():
    """Reports without multi_axis_validation behave identically to before the feature."""
    b = _stub()
    c = _stub()
    result = compare_reports(b, c)
    assert result.passed is True
    # No multi-axis diffs produced
    assert not any(m.metric.startswith("multi_axis_validation") for m in result.metrics)


def test_multi_axis_diffs_multiple_axes_sorted():
    b = _stub_with_axes(
        _multi_axis_entry("site_id", bacc=0.70),
        _multi_axis_entry("session_id", bacc=0.70),
    )
    c = _stub_with_axes(
        _multi_axis_entry("site_id", bacc=0.70),
        _multi_axis_entry("session_id", bacc=0.70),
    )
    diffs = _multi_axis_diffs(b, c, {})
    # Should be 6 diffs (3 per axis), axes in alphabetical order
    assert len(diffs) == 6
    assert diffs[0].metric.startswith("multi_axis_validation[session_id]")
    assert diffs[3].metric.startswith("multi_axis_validation[site_id]")


def test_compare_reports_to_dict_includes_multi_axis_metrics():
    b = _stub_with_axes(_multi_axis_entry("session_id"))
    c = _stub_with_axes(_multi_axis_entry("session_id"))
    result = compare_reports(b, c)
    d = result.to_dict()
    all_metrics = d["stable"] + d["regressions"] + d["improvements"] + d["unavailable"]
    paths = {m["metric"] for m in all_metrics}
    assert "multi_axis_validation[session_id].grouped_balanced_accuracy" in paths


def test_multi_axis_metric_specs_covers_expected_keys():
    keys = {spec[0] for spec in MULTI_AXIS_METRIC_SPECS}
    assert "grouped_balanced_accuracy" in keys
    assert "grouped_roc_auc" in keys
    assert "generalization_gap" in keys


# ---------------------------------------------------------------------------
# multi-axis CI lower-bound regression tracking
# ---------------------------------------------------------------------------

def _multi_axis_entry_with_ci(axis, bacc=0.70, auc=0.75, gap=0.10,
                               ci_bacc=None, ci_auc=None, ci_gap=None):
    """Like _multi_axis_entry but includes confidence_intervals.

    ci_bacc / ci_auc / ci_gap are [lo, hi] lists; pass None to omit the key.
    """
    entry = _multi_axis_entry(axis, bacc=bacc, auc=auc, gap=gap)
    ci = {}
    if ci_bacc is not None:
        ci["grouped_balanced_accuracy"] = ci_bacc
    if ci_auc is not None:
        ci["grouped_roc_auc"] = ci_auc
    if ci_gap is not None:
        ci["generalization_gap"] = ci_gap
    if ci:
        entry["confidence_intervals"] = ci
    return entry


def test_multi_axis_ci_metrics_constant_covers_expected_keys():
    keys = {spec[0] for spec in MULTI_AXIS_CI_METRICS}
    assert "grouped_balanced_accuracy" in keys
    assert "grouped_roc_auc" in keys


def test_multi_axis_ci_lo_not_emitted_when_neither_has_ci():
    """Old-format entries without confidence_intervals produce no ci_lo rows."""
    b = _stub_with_axes(_multi_axis_entry("session_id"))
    c = _stub_with_axes(_multi_axis_entry("session_id"))
    diffs = _multi_axis_diffs(b, c, {})
    ci_paths = [d.metric for d in diffs if "ci_lo" in d.metric]
    assert ci_paths == []


def test_multi_axis_ci_lo_emitted_when_both_have_ci():
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.60, 0.80], ci_auc=[0.65, 0.85])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.60, 0.80], ci_auc=[0.65, 0.85])
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_paths = {d.metric for d in diffs if "ci_lo" in d.metric}
    assert "multi_axis_validation[session_id].ci_lo.grouped_balanced_accuracy" in ci_paths
    assert "multi_axis_validation[session_id].ci_lo.grouped_roc_auc" in ci_paths


def test_multi_axis_ci_lo_drop_is_regression():
    """A CI lower-bound drop exceeding tolerance is a regression."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.62, 0.80])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.55, 0.78])  # lo dropped 0.07 > tol 0.05
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_lo.grouped_balanced_accuracy"))
    assert ci_diff.status == "regression"
    assert ci_diff.baseline == pytest.approx(0.62)
    assert ci_diff.candidate == pytest.approx(0.55)


def test_multi_axis_ci_lo_small_drop_is_stable():
    """A CI lower-bound drop within tolerance is stable."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.62, 0.80])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.60, 0.80])  # drop 0.02 < tol 0.05
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_lo.grouped_balanced_accuracy"))
    assert ci_diff.status == "stable"


def test_multi_axis_ci_lo_rise_is_improvement():
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.58, 0.78])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.66, 0.83])  # lo rose 0.08 > tol 0.05
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_lo.grouped_balanced_accuracy"))
    assert ci_diff.status == "improvement"


def test_multi_axis_ci_lo_unavailable_when_only_baseline_has_ci():
    """If baseline has CI data but candidate doesn't, metric is unavailable."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.60, 0.80])
    )
    c = _stub_with_axes(
        _multi_axis_entry("session_id")  # no confidence_intervals
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_lo.grouped_balanced_accuracy"))
    assert ci_diff.status == "unavailable"


def test_multi_axis_ci_lo_unavailable_when_only_candidate_has_ci():
    b = _stub_with_axes(
        _multi_axis_entry("session_id")  # no confidence_intervals
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.60, 0.80])
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_lo.grouped_balanced_accuracy"))
    assert ci_diff.status == "unavailable"


def test_multi_axis_ci_lo_tolerance_override():
    """CI lower-bound tolerance can be overridden via the thresholds dict."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.62, 0.80])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.59, 0.80])  # drop 0.03
    )
    path = "multi_axis_validation[session_id].ci_lo.grouped_balanced_accuracy"
    # default tol 0.05: drop 0.03 → stable
    diffs_default = _multi_axis_diffs(b, c, {})
    ci_default = next(d for d in diffs_default if d.metric == path)
    assert ci_default.status == "stable"
    # override tol 0.02: drop 0.03 > 0.02 → regression
    diffs_strict = _multi_axis_diffs(b, c, {path: 0.02})
    ci_strict = next(d for d in diffs_strict if d.metric == path)
    assert ci_strict.status == "regression"


def test_multi_axis_ci_lo_regression_propagates_to_compare_reports():
    """compare_reports fails when a CI lower bound regresses."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.64, 0.82])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[0.56, 0.82])  # lo drop 0.08 > tol 0.05
    )
    result = compare_reports(b, c)
    assert result.passed is False
    ci_regressions = [m for m in result.regressions if "ci_lo" in m.metric]
    assert len(ci_regressions) >= 1


def test_multi_axis_none_ci_values_not_emitted():
    """[None, None] CI bounds (too few axis units) produce no ci_lo rows."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[None, None])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_bacc=[None, None])
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_paths = [d.metric for d in diffs if "ci_lo" in d.metric]
    assert ci_paths == []


# ---------------------------------------------------------------------------
# ci_hi: generalization_gap upper-bound regression tracking
# ---------------------------------------------------------------------------

def test_multi_axis_ci_hi_metrics_constant_covers_generalization_gap():
    keys = {spec[0] for spec in MULTI_AXIS_CI_HI_METRICS}
    assert "generalization_gap" in keys


def test_multi_axis_ci_hi_not_emitted_when_neither_has_gap_ci():
    """No ci_hi rows when neither entry carries a generalization_gap CI."""
    b = _stub_with_axes(_multi_axis_entry("session_id"))
    c = _stub_with_axes(_multi_axis_entry("session_id"))
    diffs = _multi_axis_diffs(b, c, {})
    ci_hi_paths = [d.metric for d in diffs if "ci_hi" in d.metric]
    assert ci_hi_paths == []


def test_multi_axis_ci_hi_rise_is_regression():
    """Gap CI upper bound rising beyond tolerance → regression."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.05, 0.15])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.05, 0.22])  # hi rose 0.07 > tol 0.05
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_hi.generalization_gap"))
    assert ci_diff.status == "regression"
    assert ci_diff.baseline == pytest.approx(0.15)
    assert ci_diff.candidate == pytest.approx(0.22)


def test_multi_axis_ci_hi_small_rise_is_stable():
    """Gap CI upper bound within tolerance → stable."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.05, 0.15])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.05, 0.18])  # rise 0.03 < tol 0.05
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_hi.generalization_gap"))
    assert ci_diff.status == "stable"


def test_multi_axis_ci_hi_fall_is_improvement():
    """Gap CI upper bound falling → improvement (gap tightening is good)."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.08, 0.22])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.04, 0.12])  # hi fell 0.10 > tol 0.05
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_hi.generalization_gap"))
    assert ci_diff.status == "improvement"


def test_multi_axis_ci_hi_unavailable_when_only_baseline_has_gap_ci():
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.05, 0.15])
    )
    c = _stub_with_axes(
        _multi_axis_entry("session_id")  # no confidence_intervals
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_hi.generalization_gap"))
    assert ci_diff.status == "unavailable"


def test_multi_axis_ci_hi_unavailable_when_only_candidate_has_gap_ci():
    b = _stub_with_axes(
        _multi_axis_entry("session_id")  # no confidence_intervals
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.05, 0.15])
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_hi.generalization_gap"))
    assert ci_diff.status == "unavailable"


def test_multi_axis_ci_hi_regression_propagates_to_compare_reports():
    """compare_reports fails when gap CI upper bound regresses."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.04, 0.14])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.04, 0.22])  # hi rose 0.08 > tol 0.05
    )
    result = compare_reports(b, c)
    assert result.passed is False
    ci_hi_regressions = [m for m in result.regressions if "ci_hi" in m.metric]
    assert len(ci_hi_regressions) >= 1


def test_multi_axis_ci_hi_none_gap_bounds_not_emitted():
    """[None, None] gap CI bounds produce no ci_hi rows."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[None, None])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[None, None])
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_hi_paths = [d.metric for d in diffs if "ci_hi" in d.metric]
    assert ci_hi_paths == []


# ---------------------------------------------------------------------------
# ci_lo: generalization_gap lower-bound regression tracking
# A rising lower bound means even the best-case gap has widened — regression.
# ---------------------------------------------------------------------------

def test_multi_axis_ci_lo_metrics_constant_covers_generalization_gap():
    keys = {spec[0] for spec in MULTI_AXIS_CI_METRICS}
    assert "generalization_gap" in keys


def test_multi_axis_ci_lo_gap_rise_is_regression():
    """Gap CI lower bound rising beyond tolerance → regression (best-case gap widened)."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.03, 0.15])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.10, 0.20])  # lo rose 0.07 > tol 0.05
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_lo.generalization_gap"))
    assert ci_diff.status == "regression"
    assert ci_diff.baseline == pytest.approx(0.03)
    assert ci_diff.candidate == pytest.approx(0.10)


def test_multi_axis_ci_lo_gap_small_rise_is_stable():
    """Gap CI lower bound within tolerance → stable."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.05, 0.15])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.08, 0.18])  # lo rose 0.03 < tol 0.05
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_lo.generalization_gap"))
    assert ci_diff.status == "stable"


def test_multi_axis_ci_lo_gap_fall_is_improvement():
    """Gap CI lower bound falling → improvement (best-case gap narrowed)."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.10, 0.22])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.02, 0.12])  # lo fell 0.08 > tol 0.05
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_lo.generalization_gap"))
    assert ci_diff.status == "improvement"


def test_multi_axis_ci_lo_gap_unavailable_when_only_baseline_has_gap_ci():
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.04, 0.14])
    )
    c = _stub_with_axes(
        _multi_axis_entry("session_id")  # no confidence_intervals
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_lo.generalization_gap"))
    assert ci_diff.status == "unavailable"


def test_multi_axis_ci_lo_gap_unavailable_when_only_candidate_has_gap_ci():
    b = _stub_with_axes(
        _multi_axis_entry("session_id")  # no confidence_intervals
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.04, 0.14])
    )
    diffs = _multi_axis_diffs(b, c, {})
    ci_diff = next(d for d in diffs if d.metric.endswith("ci_lo.generalization_gap"))
    assert ci_diff.status == "unavailable"


def test_multi_axis_ci_lo_gap_regression_propagates_to_compare_reports():
    """compare_reports fails when gap CI lower bound regresses."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.02, 0.12])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[0.10, 0.18])  # lo rose 0.08 > tol 0.05
    )
    result = compare_reports(b, c)
    assert result.passed is False
    ci_lo_regressions = [m for m in result.regressions if "ci_lo.generalization_gap" in m.metric]
    assert len(ci_lo_regressions) >= 1


def test_multi_axis_ci_lo_gap_none_bounds_not_emitted():
    """[None, None] gap CI bounds produce no ci_lo.generalization_gap rows."""
    b = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[None, None])
    )
    c = _stub_with_axes(
        _multi_axis_entry_with_ci("session_id", ci_gap=[None, None])
    )
    diffs = _multi_axis_diffs(b, c, {})
    gap_lo_paths = [d.metric for d in diffs if "ci_lo.generalization_gap" in d.metric]
    assert gap_lo_paths == []


# ---------------------------------------------------------------------------
# _print_result table format
# ---------------------------------------------------------------------------

def _make_result(regressions=None, improvements=None, stable=None, unavailable=None,
                 passed=True, claim_changed=False):
    """Build a minimal RegressionResult for output tests."""
    from nerveml.regression import MetricDiff, RegressionResult

    def _diff(metric, status, baseline=0.7, candidate=0.7, delta=0.0, note=""):
        return MetricDiff(
            metric=metric,
            baseline=baseline,
            candidate=candidate,
            delta=delta,
            threshold=0.02,
            status=status,
            note=note,
        )

    metrics = []
    for m in (regressions or []):
        metrics.append(_diff(m["metric"], "regression",
                             m.get("baseline", 0.70), m.get("candidate", 0.63),
                             m.get("delta", -0.07), m.get("note", "dropped")))
    for m in (improvements or []):
        metrics.append(_diff(m if isinstance(m, str) else m["metric"], "improvement",
                             delta=0.05))
    for m in (stable or []):
        metrics.append(_diff(m if isinstance(m, str) else m["metric"], "stable",
                             delta=0.001))
    for m in (unavailable or []):
        metrics.append(MetricDiff(metric=m, baseline=None, candidate=None,
                                  delta=None, threshold=0.02, status="unavailable"))
    return RegressionResult(
        metrics=metrics,
        passed=passed,
        permitted_claim_changed=claim_changed,
        baseline_claim="claim A" if claim_changed else "",
        candidate_claim="claim B" if claim_changed else "",
    )


def test_print_result_header_shows_passed(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(stable=["evaluation.grouped_balanced_accuracy"])
    _print_result(result)
    out = capsys.readouterr().out
    assert "PASSED" in out


def test_print_result_header_shows_failed(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(
        regressions=[{"metric": "evaluation.grouped_balanced_accuracy"}],
        passed=False,
    )
    _print_result(result)
    out = capsys.readouterr().out
    assert "FAILED" in out


def test_print_result_table_has_column_headers(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(stable=["evaluation.grouped_balanced_accuracy"])
    _print_result(result)
    out = capsys.readouterr().out
    assert "Metric" in out
    assert "Baseline" in out
    assert "Candidate" in out


def test_print_result_table_has_separator_line(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(stable=["evaluation.grouped_balanced_accuracy"])
    _print_result(result)
    out = capsys.readouterr().out
    assert "─" in out


def test_print_result_regression_row_uppercase(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(
        regressions=[{"metric": "evaluation.grouped_balanced_accuracy"}],
        passed=False,
    )
    _print_result(result)
    out = capsys.readouterr().out
    assert "REGRESSION" in out


def test_print_result_stable_row_lowercase(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(stable=["evaluation.grouped_balanced_accuracy"])
    _print_result(result)
    out = capsys.readouterr().out
    assert "stable" in out


def test_print_result_improvement_row_label(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(improvements=["evaluation.grouped_balanced_accuracy"])
    _print_result(result)
    out = capsys.readouterr().out
    assert "improved" in out


def test_print_result_unavailable_shows_na(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(unavailable=["evaluation.grouped_balanced_accuracy"])
    _print_result(result)
    out = capsys.readouterr().out
    # The metric name must appear
    assert "evaluation.grouped_balanced_accuracy" in out
    # Unavailable rows have n/a in numeric columns
    assert "n/a" in out


def test_print_result_metric_values_formatted(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(
        stable=[{"metric": "evaluation.grouped_balanced_accuracy",
                 "baseline": None, "candidate": None, "delta": None}],
    )
    # Build a specific result with known values
    from nerveml.regression import MetricDiff, RegressionResult
    m = MetricDiff(
        metric="evaluation.grouped_balanced_accuracy",
        baseline=0.7123,
        candidate=0.6789,
        delta=-0.0334,
        threshold=0.02,
        status="regression",
        note="dropped by 0.0334",
    )
    result = RegressionResult(metrics=[m], passed=False)
    _print_result(result)
    out = capsys.readouterr().out
    assert "0.7123" in out
    assert "0.6789" in out
    assert "-0.0334" in out


def test_print_result_note_annotated_below_regression_row(capsys):
    from nerveml.compare_cli import _print_result
    from nerveml.regression import MetricDiff, RegressionResult
    m = MetricDiff(
        metric="evaluation.grouped_balanced_accuracy",
        baseline=0.70,
        candidate=0.63,
        delta=-0.07,
        threshold=0.02,
        status="regression",
        note="dropped by 0.0700 (tolerance 0.02)",
    )
    result = RegressionResult(metrics=[m], passed=False)
    _print_result(result)
    out = capsys.readouterr().out
    assert "dropped by 0.0700" in out


def test_print_result_summary_counts_line(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(
        regressions=[{"metric": "evaluation.grouped_balanced_accuracy"}],
        stable=["evaluation.grouped_roc_auc"],
        passed=False,
    )
    _print_result(result)
    out = capsys.readouterr().out
    assert "Regressions:" in out
    assert "Stable:" in out


def test_print_result_permitted_claim_section(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(stable=["evaluation.grouped_balanced_accuracy"],
                          claim_changed=True)
    _print_result(result)
    out = capsys.readouterr().out
    assert "PERMITTED CLAIM CHANGED" in out
    assert "claim A" in out
    assert "claim B" in out


def test_print_result_regressions_appear_before_stable(capsys):
    from nerveml.compare_cli import _print_result
    result = _make_result(
        regressions=[{"metric": "evaluation.grouped_balanced_accuracy"}],
        stable=["evaluation.grouped_roc_auc"],
        passed=False,
    )
    _print_result(result)
    out = capsys.readouterr().out
    reg_pos = out.index("REGRESSION")
    stable_pos = out.index("stable")
    assert reg_pos < stable_pos


def test_print_result_empty_metrics_graceful(capsys):
    from nerveml.compare_cli import _print_result
    from nerveml.regression import RegressionResult
    result = RegressionResult(metrics=[], passed=True)
    _print_result(result)
    out = capsys.readouterr().out
    assert "PASSED" in out


# ---------------------------------------------------------------------------
# artifact-baseline regression tracking
# ---------------------------------------------------------------------------

def _artifact_baseline(share_eog=0.20, share_emg=0.15,
                       eog_bacc=0.60, emg_bacc=0.57):
    """Minimal artifact_baseline dict matching ArtifactBaseline.to_dict()."""
    return {
        "full_balanced_accuracy": 0.75,
        "eog_proxy": {
            "name": "eog",
            "n_features": 20,
            "balanced_accuracy": eog_bacc,
            "roc_auc": 0.62,
            "note": "frontal channels",
        },
        "emg_proxy": {
            "name": "emg",
            "n_features": 7,
            "balanced_accuracy": emg_bacc,
            "roc_auc": 0.58,
            "note": "gamma-band features",
        },
        "artifact_share_eog": share_eog,
        "artifact_share_emg": share_emg,
        "interpretation": "Upper-bound estimate; consistent with artifact confounds.",
    }


def _stub_with_artifact(share_eog=0.20, share_emg=0.15):
    s = _stub()
    s["artifact_baseline"] = _artifact_baseline(share_eog=share_eog,
                                                 share_emg=share_emg)
    return s


def test_artifact_diffs_empty_when_both_reports_have_no_section():
    b = _stub()
    c = _stub()
    diffs = _artifact_diffs(b, c, {})
    assert diffs == []


def test_artifact_diffs_empty_when_both_sections_are_none():
    b = _stub()
    b["artifact_baseline"] = None
    c = _stub()
    c["artifact_baseline"] = None
    diffs = _artifact_diffs(b, c, {})
    assert diffs == []


def test_artifact_diffs_produces_two_metrics_per_report_pair():
    b = _stub_with_artifact()
    c = _stub_with_artifact()
    diffs = _artifact_diffs(b, c, {})
    assert len(diffs) == 2
    paths = {d.metric for d in diffs}
    assert "artifact_baseline.artifact_share_eog" in paths
    assert "artifact_baseline.artifact_share_emg" in paths


def test_artifact_diffs_stable_when_values_identical():
    b = _stub_with_artifact(share_eog=0.20, share_emg=0.15)
    c = _stub_with_artifact(share_eog=0.20, share_emg=0.15)
    diffs = _artifact_diffs(b, c, {})
    assert all(d.status == "stable" for d in diffs)


def test_artifact_share_rise_is_regression():
    # eog share rises 0.20 → 0.35 (delta 0.15, tol 0.10) — regression
    b = _stub_with_artifact(share_eog=0.20)
    c = _stub_with_artifact(share_eog=0.35)
    diffs = _artifact_diffs(b, c, {})
    eog = next(d for d in diffs if "eog" in d.metric)
    assert eog.status == "regression"


def test_artifact_share_small_rise_is_stable():
    # eog share rises 0.20 → 0.25 (delta 0.05, tol 0.10) — within tolerance
    b = _stub_with_artifact(share_eog=0.20)
    c = _stub_with_artifact(share_eog=0.25)
    diffs = _artifact_diffs(b, c, {})
    eog = next(d for d in diffs if "eog" in d.metric)
    assert eog.status == "stable"


def test_artifact_share_fall_is_improvement():
    # emg share falls 0.30 → 0.10 (delta -0.20, tol 0.10) — improvement
    b = _stub_with_artifact(share_emg=0.30)
    c = _stub_with_artifact(share_emg=0.10)
    diffs = _artifact_diffs(b, c, {})
    emg = next(d for d in diffs if "emg" in d.metric)
    assert emg.status == "improvement"


def test_artifact_section_only_in_candidate_gives_unavailable():
    b = _stub()  # no artifact_baseline
    c = _stub_with_artifact()
    diffs = _artifact_diffs(b, c, {})
    # b has no section → b values are None → unavailable
    assert all(d.status == "unavailable" for d in diffs)
    assert len(diffs) == 2


def test_artifact_section_only_in_baseline_gives_unavailable():
    b = _stub_with_artifact()
    c = _stub()  # no artifact_baseline
    diffs = _artifact_diffs(b, c, {})
    assert all(d.status == "unavailable" for d in diffs)
    assert len(diffs) == 2


def test_artifact_tolerance_override_respected():
    # Default tol 0.10; a rise of 0.08 is stable; with override 0.05 it is a regression.
    b = _stub_with_artifact(share_eog=0.20)
    c = _stub_with_artifact(share_eog=0.28)
    path = "artifact_baseline.artifact_share_eog"
    diffs_default = _artifact_diffs(b, c, {})
    diffs_strict = _artifact_diffs(b, c, {path: 0.05})
    eog_default = next(d for d in diffs_default if "eog" in d.metric)
    eog_strict = next(d for d in diffs_strict if "eog" in d.metric)
    assert eog_default.status == "stable"
    assert eog_strict.status == "regression"


def test_compare_reports_artifact_regression_fails():
    b = _stub_with_artifact(share_eog=0.20)
    c = _stub_with_artifact(share_eog=0.45)  # large rise — regression
    result = compare_reports(b, c)
    assert result.passed is False
    art_regressions = [m for m in result.regressions
                       if m.metric.startswith("artifact_baseline")]
    assert len(art_regressions) >= 1


def test_compare_reports_artifact_stable_does_not_fail():
    b = _stub_with_artifact(share_eog=0.20, share_emg=0.15)
    c = _stub_with_artifact(share_eog=0.20, share_emg=0.15)
    result = compare_reports(b, c)
    # No artifact regressions
    art_regressions = [m for m in result.regressions
                       if m.metric.startswith("artifact_baseline")]
    assert len(art_regressions) == 0


def test_compare_reports_no_artifact_section_unchanged_behaviour():
    """Reports without artifact_baseline produce no artifact diffs."""
    b = _stub()
    c = _stub()
    result = compare_reports(b, c)
    art_metrics = [m for m in result.metrics
                   if m.metric.startswith("artifact_baseline")]
    assert art_metrics == []


def test_compare_reports_to_dict_includes_artifact_metrics():
    b = _stub_with_artifact()
    c = _stub_with_artifact()
    result = compare_reports(b, c)
    d = result.to_dict()
    all_metrics = d["stable"] + d["regressions"] + d["improvements"] + d["unavailable"]
    paths = {m["metric"] for m in all_metrics}
    assert "artifact_baseline.artifact_share_eog" in paths
    assert "artifact_baseline.artifact_share_emg" in paths


def test_artifact_metric_specs_covers_expected_keys():
    keys = {spec[0] for spec in ARTIFACT_METRIC_SPECS}
    assert "artifact_share_eog" in keys
    assert "artifact_share_emg" in keys


def test_artifact_metric_specs_all_lower_is_better():
    # Rising artifact share is always bad (higher_is_better=False)
    for key, higher_is_better, _ in ARTIFACT_METRIC_SPECS:
        assert higher_is_better is False, f"{key} should be lower-is-better"


# ---------------------------------------------------------------------------
# worst-unit tracking (evaluation.worst_group.balanced_accuracy)
# ---------------------------------------------------------------------------

WORST_UNIT_PATH = "evaluation.worst_group.balanced_accuracy"


def test_worst_unit_unavailable_when_both_reports_have_no_worst_group():
    b = _stub()   # no worst_group_bacc — field absent
    c = _stub()
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "unavailable"


def test_worst_unit_stable_when_values_identical():
    b = _stub(worst_group_bacc=0.60)
    c = _stub(worst_group_bacc=0.60)
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "stable"


def test_worst_unit_regression_when_bacc_drops_beyond_tolerance():
    # Drop of 0.07 > tolerance 0.05 → regression
    b = _stub(worst_group_bacc=0.62)
    c = _stub(worst_group_bacc=0.55)
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "regression"


def test_worst_unit_stable_when_drop_within_tolerance():
    # Drop of 0.03 ≤ tolerance 0.05 → stable
    b = _stub(worst_group_bacc=0.62)
    c = _stub(worst_group_bacc=0.59)
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "stable"


def test_worst_unit_improvement_when_bacc_rises_beyond_tolerance():
    # Rise of 0.06 > tolerance 0.05 → improvement
    b = _stub(worst_group_bacc=0.55)
    c = _stub(worst_group_bacc=0.61)
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "improvement"


def test_worst_unit_regression_sets_passed_false():
    b = _stub(worst_group_bacc=0.65)
    c = _stub(worst_group_bacc=0.55)   # drop 0.10 > tol 0.05
    result = compare_reports(b, c)
    assert result.passed is False


def test_worst_unit_regression_when_aggregate_is_stable():
    # Aggregate within tolerance, but worst unit regresses — should still fail.
    b = _stub(grouped_bacc=0.70, worst_group_bacc=0.62)
    c = _stub(grouped_bacc=0.70, worst_group_bacc=0.55)   # drop 0.07 > tol 0.05
    result = compare_reports(b, c)
    assert result.passed is False
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "regression"


def test_worst_unit_unavailable_when_only_baseline_has_worst_group():
    # Candidate report predates worst_group tracking — field absent.
    b = _stub(worst_group_bacc=0.60)
    c = _stub()   # no worst_group_bacc
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "unavailable"
    assert result.passed is True   # unavailable ≠ regression


def test_worst_unit_tracked_in_tracked_metrics():
    paths = {path for path, _, _ in TRACKED_METRICS}
    assert WORST_UNIT_PATH in paths


def test_worst_unit_is_higher_is_better():
    spec = next(
        (spec for spec in TRACKED_METRICS if spec[0] == WORST_UNIT_PATH), None
    )
    assert spec is not None
    _, higher_is_better, _ = spec
    assert higher_is_better is True


# ---------------------------------------------------------------------------
# worst-unit identity annotation in regression note
# ---------------------------------------------------------------------------

def test_worst_unit_regression_note_names_unchanged_unit():
    # Same group in both reports — note says "unchanged: <group>".
    b = _stub(worst_group_bacc=0.65, worst_group_group="subject_3")
    c = _stub(worst_group_bacc=0.55, worst_group_group="subject_3")
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "regression"
    assert "worst unit unchanged" in worst.note
    assert "subject_3" in worst.note


def test_worst_unit_regression_note_names_changed_unit():
    # Different groups — note shows baseline and candidate group.
    b = _stub(worst_group_bacc=0.65, worst_group_group="subject_3")
    c = _stub(worst_group_bacc=0.55, worst_group_group="subject_7")
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "regression"
    assert "worst unit:" in worst.note
    assert "baseline=subject_3" in worst.note
    assert "candidate=subject_7" in worst.note


def test_worst_unit_no_identity_note_on_stable():
    # No regression → no worst-unit identity annotation added.
    b = _stub(worst_group_bacc=0.62, worst_group_group="subject_3")
    c = _stub(worst_group_bacc=0.60, worst_group_group="subject_7")   # drop 0.02 ≤ tol
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "stable"
    assert "worst unit" not in worst.note


def test_worst_unit_no_identity_note_on_improvement():
    b = _stub(worst_group_bacc=0.55, worst_group_group="subject_3")
    c = _stub(worst_group_bacc=0.65, worst_group_group="subject_7")
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "improvement"
    assert "worst unit" not in worst.note


def test_worst_unit_regression_note_with_integer_group():
    # Group ID stored as int (the common case) should appear in note cleanly.
    b = _stub(worst_group_bacc=0.65, worst_group_group=3)
    c = _stub(worst_group_bacc=0.55, worst_group_group=7)
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "regression"
    assert "baseline=3" in worst.note
    assert "candidate=7" in worst.note


def test_worst_unit_regression_note_when_group_key_absent():
    # worst_group dict exists (has balanced_accuracy) but is missing the group
    # key — regression note should still fire without the identity annotation
    # rather than crash.
    b = {"evaluation": {"grouped_balanced_accuracy": 0.70, "grouped_roc_auc": 0.75,
                        "trial_random_balanced_accuracy": 0.80, "generalization_gap": 0.10,
                        "worst_group": {"balanced_accuracy": 0.65}},
         "permutation_test": {"p_value": 0.02},
         "identity_inference": {"accuracy": 0.30},
         "permitted_claim": "weakly decodable under this protocol"}
    c = {"evaluation": {"grouped_balanced_accuracy": 0.70, "grouped_roc_auc": 0.75,
                        "trial_random_balanced_accuracy": 0.80, "generalization_gap": 0.10,
                        "worst_group": {"balanced_accuracy": 0.55}},
         "permutation_test": {"p_value": 0.02},
         "identity_inference": {"accuracy": 0.30},
         "permitted_claim": "weakly decodable under this protocol"}
    result = compare_reports(b, c)
    worst = next(m for m in result.metrics if m.metric == WORST_UNIT_PATH)
    assert worst.status == "regression"
    # group key absent in both → no identity annotation appended
    assert "worst unit" not in worst.note


# ---------------------------------------------------------------------------
# secondary-probe regression tracking
# ---------------------------------------------------------------------------

def _sp_entry(attribute, bacc=0.55, auc=0.60):
    """Minimal secondary-probe dict mirroring scan.py output."""
    return {
        "attribute": attribute,
        "n_units": 10,
        "grouped_balanced_accuracy": bacc,
        "grouped_roc_auc": auc,
        "interpretation": "weakly decodable under this protocol",
    }


def _stub_with_probes(*attrs_kwargs):
    """Build a full stub report with secondary_probes list."""
    probes = [_sp_entry(**kw) for kw in attrs_kwargs]
    s = _stub()
    s["secondary_probes"] = probes
    return s


def test_secondary_probe_no_probes_returns_no_diffs():
    # Neither report has secondary_probes — no diffs produced.
    b, c = _stub(), _stub()
    diffs = _secondary_probe_diffs(b, c, {})
    assert diffs == []


def test_secondary_probe_empty_lists_returns_no_diffs():
    b, c = _stub(), _stub()
    b["secondary_probes"] = []
    c["secondary_probes"] = []
    diffs = _secondary_probe_diffs(b, c, {})
    assert diffs == []


def test_secondary_probe_stable_when_within_tolerance():
    b = _stub_with_probes({"attribute": "arousal", "bacc": 0.55})
    c = _stub_with_probes({"attribute": "arousal", "bacc": 0.57})
    diffs = _secondary_probe_diffs(b, c, {})
    bacc_diff = next(d for d in diffs if "grouped_balanced_accuracy" in d.metric)
    assert bacc_diff.status == "stable"


def test_secondary_probe_regression_when_bacc_rises_beyond_tolerance():
    # bacc rises by 0.08 > tol 0.05 — privacy regression.
    b = _stub_with_probes({"attribute": "fatigue", "bacc": 0.55})
    c = _stub_with_probes({"attribute": "fatigue", "bacc": 0.63})
    diffs = _secondary_probe_diffs(b, c, {})
    bacc_diff = next(
        d for d in diffs
        if "fatigue" in d.metric and "grouped_balanced_accuracy" in d.metric
    )
    assert bacc_diff.status == "regression"


def test_secondary_probe_improvement_when_bacc_falls_beyond_tolerance():
    # bacc drops by 0.10 > tol 0.05 — attribute became less decodable (improvement).
    b = _stub_with_probes({"attribute": "sleep_stage", "bacc": 0.70})
    c = _stub_with_probes({"attribute": "sleep_stage", "bacc": 0.58})
    diffs = _secondary_probe_diffs(b, c, {})
    bacc_diff = next(
        d for d in diffs
        if "sleep_stage" in d.metric and "grouped_balanced_accuracy" in d.metric
    )
    assert bacc_diff.status == "improvement"


def test_secondary_probe_auc_tracked():
    b = _stub_with_probes({"attribute": "medication", "bacc": 0.55, "auc": 0.58})
    c = _stub_with_probes({"attribute": "medication", "bacc": 0.55, "auc": 0.68})
    diffs = _secondary_probe_diffs(b, c, {})
    auc_diff = next(
        d for d in diffs
        if "medication" in d.metric and "grouped_roc_auc" in d.metric
    )
    assert auc_diff.status == "regression"


def test_secondary_probe_missing_in_candidate_is_unavailable():
    # Attribute in baseline but absent from candidate.
    b = _stub_with_probes({"attribute": "stress", "bacc": 0.60})
    c = _stub()
    c["secondary_probes"] = []
    diffs = _secondary_probe_diffs(b, c, {})
    unavail = [d for d in diffs if "stress" in d.metric]
    assert all(d.status == "unavailable" for d in unavail)


def test_secondary_probe_missing_in_baseline_is_unavailable():
    # New attribute detected in candidate that was not in baseline.
    b = _stub()
    b["secondary_probes"] = []
    c = _stub_with_probes({"attribute": "motor_state", "bacc": 0.65})
    diffs = _secondary_probe_diffs(b, c, {})
    unavail = [d for d in diffs if "motor_state" in d.metric]
    assert all(d.status == "unavailable" for d in unavail)


def test_secondary_probe_multiple_attributes_independent():
    b = _stub_with_probes(
        {"attribute": "arousal", "bacc": 0.55},
        {"attribute": "fatigue", "bacc": 0.60},
    )
    c = _stub_with_probes(
        {"attribute": "arousal", "bacc": 0.64},   # regression (+0.09)
        {"attribute": "fatigue", "bacc": 0.60},   # stable
    )
    diffs = _secondary_probe_diffs(b, c, {})
    arousal_bacc = next(d for d in diffs if "arousal" in d.metric and "balanced_accuracy" in d.metric)
    fatigue_bacc = next(d for d in diffs if "fatigue" in d.metric and "balanced_accuracy" in d.metric)
    assert arousal_bacc.status == "regression"
    assert fatigue_bacc.status == "stable"


def test_secondary_probe_metric_path_format():
    b = _stub_with_probes({"attribute": "medication", "bacc": 0.55})
    c = _stub_with_probes({"attribute": "medication", "bacc": 0.55})
    diffs = _secondary_probe_diffs(b, c, {})
    paths = {d.metric for d in diffs}
    assert "secondary_probes[medication].grouped_balanced_accuracy" in paths
    assert "secondary_probes[medication].grouped_roc_auc" in paths


def test_secondary_probe_threshold_override():
    # Default tol is 0.05; override to 0.20 so a +0.08 rise stays "stable".
    b = _stub_with_probes({"attribute": "arousal", "bacc": 0.55})
    c = _stub_with_probes({"attribute": "arousal", "bacc": 0.63})
    overrides = {"secondary_probes[arousal].grouped_balanced_accuracy": 0.20}
    diffs = _secondary_probe_diffs(b, c, overrides)
    bacc_diff = next(d for d in diffs if "grouped_balanced_accuracy" in d.metric)
    assert bacc_diff.status == "stable"


def test_secondary_probe_regression_propagates_to_compare_reports():
    b = _stub_with_probes({"attribute": "fatigue", "bacc": 0.55})
    c = _stub_with_probes({"attribute": "fatigue", "bacc": 0.63})
    result = compare_reports(b, c)
    assert result.passed is False
    reg_paths = {m.metric for m in result.regressions}
    assert "secondary_probes[fatigue].grouped_balanced_accuracy" in reg_paths


def test_secondary_probe_stable_does_not_affect_passed():
    b = _stub_with_probes({"attribute": "arousal", "bacc": 0.55})
    c = _stub_with_probes({"attribute": "arousal", "bacc": 0.57})
    result = compare_reports(b, c)
    # Only secondary probe metrics involved, none regressed.
    sec_regs = [m for m in result.regressions if "secondary_probes" in m.metric]
    assert sec_regs == []


def test_secondary_probe_no_probes_does_not_add_metrics():
    # Reports without secondary_probes should produce no secondary_probes entries.
    b, c = _stub(), _stub()
    result = compare_reports(b, c)
    sec_metrics = [m for m in result.metrics if "secondary_probes" in m.metric]
    assert sec_metrics == []


def test_secondary_probe_sorted_alphabetically():
    # Attributes must be sorted so diff output is deterministic.
    b = _stub_with_probes(
        {"attribute": "stress"},
        {"attribute": "arousal"},
        {"attribute": "fatigue"},
    )
    c = _stub_with_probes(
        {"attribute": "stress"},
        {"attribute": "arousal"},
        {"attribute": "fatigue"},
    )
    diffs = _secondary_probe_diffs(b, c, {})
    attrs_seen = []
    for d in diffs:
        for attr in ["arousal", "fatigue", "stress"]:
            if attr in d.metric and attr not in attrs_seen:
                attrs_seen.append(attr)
    assert attrs_seen == sorted(attrs_seen)


def test_secondary_probe_specs_have_correct_direction():
    for _key, higher_is_better, _tol in SECONDARY_PROBE_METRIC_SPECS:
        assert higher_is_better is False, (
            f"{_key}: expected lower-is-better for secondary probes"
        )


def test_secondary_probe_to_dict_serialisable():
    b = _stub_with_probes({"attribute": "fatigue", "bacc": 0.55})
    c = _stub_with_probes({"attribute": "fatigue", "bacc": 0.63})
    result = compare_reports(b, c)
    # Should not raise
    import json
    json.dumps(result.to_dict())


# ---------------------------------------------------------------------------
# --list-metrics flag
# ---------------------------------------------------------------------------

def test_list_metrics_exits_zero(capsys):
    from nerveml.compare_cli import main

    code = main(["--list-metrics"])
    assert code == 0


def test_list_metrics_output_contains_static_metric(capsys):
    from nerveml.compare_cli import main

    main(["--list-metrics"])
    out = capsys.readouterr().out
    assert "evaluation.grouped_balanced_accuracy" in out


def test_list_metrics_output_contains_worst_group_metric(capsys):
    from nerveml.compare_cli import main

    main(["--list-metrics"])
    out = capsys.readouterr().out
    assert "evaluation.worst_group.balanced_accuracy" in out


def test_list_metrics_output_contains_axis_pattern(capsys):
    from nerveml.compare_cli import main

    main(["--list-metrics"])
    out = capsys.readouterr().out
    assert "{axis}" in out


def test_list_metrics_output_contains_attribute_pattern(capsys):
    from nerveml.compare_cli import main

    main(["--list-metrics"])
    out = capsys.readouterr().out
    assert "{attribute}" in out


def test_list_metrics_output_contains_artifact_metric(capsys):
    from nerveml.compare_cli import main

    main(["--list-metrics"])
    out = capsys.readouterr().out
    assert "artifact_baseline.artifact_share_eog" in out


def test_list_metrics_output_shows_sig_flip_for_p_value(capsys):
    from nerveml.compare_cli import main

    main(["--list-metrics"])
    out = capsys.readouterr().out
    assert "sig-flip" in out


def test_list_metrics_output_shows_direction_labels(capsys):
    from nerveml.compare_cli import main

    main(["--list-metrics"])
    out = capsys.readouterr().out
    assert "higher-better" in out
    assert "lower-better" in out


def test_list_metrics_no_file_args_required(capsys):
    from nerveml.compare_cli import main

    # Must succeed without positional file arguments.
    code = main(["--list-metrics"])
    assert code == 0


# ---------------------------------------------------------------------------
# --threshold CLI flag
# ---------------------------------------------------------------------------

def test_threshold_override_widens_tolerance_preventing_regression(tmp_path, capsys):
    from nerveml.compare_cli import main

    # Candidate drops by 0.04, normally flagged (default tol=0.02).
    baseline = _stub(grouped_bacc=0.70)
    candidate = _stub(grouped_bacc=0.66)
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    b_path.write_text(json.dumps(baseline), encoding="utf-8")
    c_path.write_text(json.dumps(candidate), encoding="utf-8")

    # Without override: regression → exit 1.
    code_default = main([str(b_path), str(c_path)])
    assert code_default == 1

    # With a wider tolerance (0.05): 0.04 drop is within tolerance → exit 0.
    code_override = main([
        str(b_path), str(c_path),
        "--threshold", "evaluation.grouped_balanced_accuracy=0.05",
    ])
    assert code_override == 0


def test_threshold_invalid_format_exits_two(tmp_path):
    from nerveml.compare_cli import main

    report = _stub()
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    code = main([str(path), str(path), "--threshold", "notakeyvalue"])
    assert code == 2


def test_threshold_non_numeric_value_exits_two(tmp_path):
    from nerveml.compare_cli import main

    report = _stub()
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    code = main([str(path), str(path), "--threshold", "evaluation.grouped_balanced_accuracy=abc"])
    assert code == 2


def test_threshold_multiple_overrides_accepted(tmp_path):
    from nerveml.compare_cli import main

    report = _stub()
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    code = main([
        str(path), str(path),
        "--threshold", "evaluation.grouped_balanced_accuracy=0.05",
        "--threshold", "evaluation.grouped_roc_auc=0.05",
    ])
    assert code == 0


def test_threshold_tightens_tolerance_causing_regression(tmp_path):
    from nerveml.compare_cli import main

    # Candidate drops by 0.01, normally stable (default tol=0.02).
    baseline = _stub(grouped_bacc=0.70)
    candidate = _stub(grouped_bacc=0.69)
    b_path = tmp_path / "b.json"
    c_path = tmp_path / "c.json"
    b_path.write_text(json.dumps(baseline), encoding="utf-8")
    c_path.write_text(json.dumps(candidate), encoding="utf-8")

    # Default tolerance: stable → exit 0.
    code_default = main([str(b_path), str(c_path)])
    assert code_default == 0

    # Tighter tolerance (0.005): 0.01 drop now exceeds it → regression → exit 1.
    code_tight = main([
        str(b_path), str(c_path),
        "--threshold", "evaluation.grouped_balanced_accuracy=0.005",
    ])
    assert code_tight == 1


def test_no_file_args_without_list_metrics_exits_two(capsys):
    from nerveml.compare_cli import main

    code = main([])
    assert code == 2


# ---------------------------------------------------------------------------
# --verbose flag
# ---------------------------------------------------------------------------

def test_verbose_flag_is_false_by_default():
    from nerveml.compare_cli import build_parser

    args = build_parser().parse_args(["baseline.json", "candidate.json"])
    assert args.verbose is False


def test_verbose_flag_parsed():
    from nerveml.compare_cli import build_parser

    args = build_parser().parse_args(["baseline.json", "candidate.json", "--verbose"])
    assert args.verbose is True


def test_verbose_prints_loading_messages_to_stderr(tmp_path, capsys):
    from nerveml.compare_cli import main

    report = _stub()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    main([str(path), str(path), "--verbose"])

    captured = capsys.readouterr()
    assert "[nerveml-compare]" in captured.err
    assert "Loading baseline" in captured.err
    assert "Loading candidate" in captured.err


def test_verbose_prints_stage_labels_to_stderr(tmp_path, capsys):
    from nerveml.compare_cli import main

    report = _stub()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    main([str(path), str(path), "--verbose"])

    err = capsys.readouterr().err
    assert "Aggregate evaluation metrics" in err
    assert "Per-axis validation metrics" in err
    assert "Artifact-baseline metrics" in err
    assert "Secondary sensitive-attribute probes" in err
    assert "Permitted-claim text" in err


def test_without_verbose_no_stage_output_to_stderr(tmp_path, capsys):
    from nerveml.compare_cli import main

    report = _stub()
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    main([str(path), str(path)])

    captured = capsys.readouterr()
    assert "[nerveml-compare]" not in captured.err


def test_on_stage_callback_receives_all_keys():
    from nerveml.regression import compare_reports

    stages_seen = []
    compare_reports(_stub(), _stub(), on_stage=stages_seen.append)

    assert "aggregate" in stages_seen
    assert "multi_axis" in stages_seen
    assert "artifact" in stages_seen
    assert "secondary_probes" in stages_seen
    assert "claim" in stages_seen
