"""Demonstration datasets shaped like a customer's, not like a test fixture.

A scan of a public dataset shows what the tool does. It does not show what it
is for. These scenarios put the tool in the situation it is sold into: a team
has built something, has a number they believe, and the scan has to say whether
the number survives.

They are constructed, and labelled as constructed. What makes that honest is
that the failure each one contains is a real and common one, and the scan is
never told which.
"""

import pytest

from nerveml.scan import run_scan
from nerveml.scenarios import SCENARIOS, describe, load_scenario


def flags_of(report):
    return [f["code"] for f in report["risk_flags"]["flags"]]


def test_every_scenario_is_described():
    for name in SCENARIOS:
        described = describe(name)
        assert described["customer"]
        assert described["pitch"]
        assert described["reported_metric"]
        assert described["planted_failure"]


def test_the_customer_is_obviously_fictional():
    # A demonstration must not be mistakable for an audit of a real company.
    for name in SCENARIOS:
        assert "(fictional)" in describe(name)["customer"].lower()


def test_the_scenario_context_travels_into_the_report():
    from nerveml.scan import run_scan

    report = run_scan(
        dataset="focus_tracker", model_kind="logistic_regression",
        n_permutations=10, n_splits=4, seed=0, identity_probe=False,
    )
    context = report["dataset"]["context"]

    assert context["customer"] == describe("focus_tracker")["customer"]
    assert context["reported_metric"]
    assert report["dataset"]["modality"] == "demonstration"


def test_a_real_dataset_carries_no_scenario_context():
    from nerveml.scan import run_scan

    report = run_scan(
        dataset="subject_leakage", n_subjects=8, n_trials=20,
        model_kind="logistic_regression", n_permutations=10, n_splits=4,
        seed=0, identity_probe=False,
    )

    assert report["dataset"].get("context") is None


def test_an_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        load_scenario("not_a_scenario")


def test_the_focus_tracker_looks_excellent_to_its_own_team():
    df, feats = load_scenario("focus_tracker")

    from nerveml.models import build_model
    from nerveml.validation import evaluate_trial_random

    naive = evaluate_trial_random(
        df, feats, model=build_model("logistic_regression", seed=0), seed=0
    )

    # The number the team would put in a deck.
    assert naive.balanced_accuracy > 0.85


@pytest.fixture(scope="module")
def focus_report():
    return run_scan(
        dataset="focus_tracker",
        model_kind="logistic_regression",
        n_permutations=40,
        seed=0,
        identity_probe=False,
    )


def test_the_scan_reaches_a_verdict_rather_than_hedging(focus_report):
    risk = focus_report["risk_flags"]

    # A demonstration whose every verdict is "cannot decide" demonstrates
    # nothing. The planted effect is large enough that the intervals settle it.
    assert risk["subject_dependence_level"] == "high"
    assert risk["sensitive_inference_evidence"] != "inconclusive_at_this_sample_size"
    assert "high_generalization_warning" in flags_of(focus_report)


def test_the_scan_finds_the_planted_failure(focus_report):
    evaluation = focus_report["evaluation"]

    assert evaluation["trial_random_balanced_accuracy"] > 0.85
    assert evaluation["grouped_balanced_accuracy"] < 0.60
    assert evaluation["generalization_gap"] > 0.30


def test_the_within_unit_reading_exposes_it_from_one_protocol(focus_report):
    evaluation = focus_report["evaluation"]

    # The pooled number is high and every participant individually is at
    # chance: the model learned who, not what.
    assert evaluation["trial_random_within_unit_balanced_accuracy"] == pytest.approx(
        0.5, abs=0.05
    )
    assert "between_unit_discrimination_only" in flags_of(focus_report)


def test_the_gap_interval_clears_the_warning_threshold(focus_report):
    low, _ = focus_report["evaluation"]["confidence_intervals"]["generalization_gap"]

    assert low > 0.20


def test_the_working_detector_passes(focus_report):
    working = run_scan(
        dataset="calibrated_detector",
        model_kind="logistic_regression",
        n_permutations=40,
        seed=0,
        identity_probe=False,
    )

    # The same pipeline must clear a model that does generalise, or a warning
    # from it means nothing.
    assert working["evaluation"]["generalization_gap"] < 0.10
    assert working["risk_flags"]["subject_dependence_level"] == "none"
    assert "high_generalization_warning" not in flags_of(working)


def test_scenario_features_are_named_like_real_recordings():
    _, feats = load_scenario("focus_tracker")

    assert any(name.endswith("_alpha") for name in feats)
    assert all("_" in name for name in feats)


def test_scenarios_are_reproducible():
    a, _ = load_scenario("focus_tracker", seed=1)
    b, _ = load_scenario("focus_tracker", seed=1)

    assert a.equals(b)
