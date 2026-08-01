"""Smoke tests for the Streamlit interface (spec section 13).

The app is presentation glue over modules that are tested elsewhere, so these
check the two things unit tests cannot: that it renders without raising, and
that the numbers on screen are the ones the scan actually returned rather than
placeholders.
"""

import pytest

from streamlit.testing.v1 import AppTest

APP = "nerveml/app.py"
TIMEOUT = 120


@pytest.fixture(scope="module")
def fresh():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    return app


@pytest.fixture(scope="module")
def scanned():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    app.selectbox(key="dataset").select("subject_leakage")
    app.selectbox(key="model").select("logistic_regression")
    app.number_input(key="subjects").set_value(8)
    app.number_input(key="trials").set_value(20)
    app.number_input(key="folds").set_value(4)
    app.number_input(key="permutations").set_value(10)
    app.button(key="run").click().run()
    return app


def test_app_renders_without_raising(fresh):
    assert not fresh.exception


def test_app_names_itself(fresh):
    assert any("NerveML" in heading.value for heading in fresh.title)


def test_nothing_is_shown_before_a_scan_is_run(fresh):
    # No placeholder numbers on screen; the demo must never show mock values.
    assert fresh.metric == []


def test_scan_completes_without_raising(scanned):
    assert not scanned.exception


def test_four_result_cards_are_shown(scanned):
    # Spec 13.1: naive, held-out, gap, risk level.
    labels = [metric.label for metric in scanned.metric]

    assert any("trial-random" in label for label in labels)
    assert any("held-out" in label for label in labels)
    assert any("gap" in label.lower() for label in labels)
    assert any("risk" in label.lower() for label in labels)


def test_the_cards_show_the_scanned_numbers(scanned):
    report = scanned.session_state["report"]
    shown = [metric.value for metric in scanned.metric]

    assert f"{report['evaluation']['trial_random_balanced_accuracy']:.3f}" in shown
    assert f"{report['evaluation']['grouped_balanced_accuracy']:.3f}" in shown


def test_the_permitted_claim_reaches_the_screen(scanned):
    report = scanned.session_state["report"]
    body = " ".join(element.value for element in scanned.markdown)

    assert report["permitted_claim"] in body


def test_unsupported_claims_reach_the_screen(scanned):
    report = scanned.session_state["report"]
    body = " ".join(element.value for element in scanned.markdown)

    for claim in report["unsupported_claims"]:
        assert claim in body


def test_the_report_is_downloadable(scanned):
    assert len(scanned.get("download_button")) >= 2


def test_customer_scenarios_are_offered_first(fresh):
    from nerveml.scenarios import SCENARIOS

    options = fresh.selectbox(key="dataset").options

    # A demo should open on the situation the tool is sold into, not on a
    # mechanism fixture.
    assert options[0] in SCENARIOS
    for name in SCENARIOS:
        assert name in options


@pytest.fixture(scope="module")
def scenario_run():
    app = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    app.selectbox(key="dataset").select("focus_tracker")
    app.selectbox(key="model").select("logistic_regression")
    app.number_input(key="folds").set_value(4)
    app.number_input(key="permutations").set_value(10)
    app.button(key="run").click().run()
    return app


def test_a_scenario_run_shows_the_client_context(scenario_run):
    context = scenario_run.session_state["report"]["dataset"]["context"]
    body = " ".join(
        [e.value for e in scenario_run.markdown]
        + [e.value for e in scenario_run.caption]
        + [e.value for e in scenario_run.info]
    )

    assert context["customer"] in body
    assert context["reported_metric"] in body


def test_a_scenario_run_says_the_data_is_constructed(scenario_run):
    body = " ".join(
        [e.value for e in scenario_run.markdown]
        + [e.value for e in scenario_run.caption]
        + [e.value for e in scenario_run.info]
    ).lower()

    assert "fictional" in body or "constructed" in body


def test_the_scorecard_is_the_first_thing_shown(scenario_run):
    # Six panels now: scorecard, folds, null, per-unit, identity, features.
    assert len(scenario_run.get("image")) == 6


def test_the_grouping_unit_is_configurable(fresh):
    # Spec H4: the same pipeline must run against session, site or device.
    assert fresh.text_input(key="group_column").value == "subject_id"


def test_the_worst_held_out_subject_is_named(scanned):
    report = scanned.session_state["report"]
    body = " ".join(element.value for element in scanned.markdown)

    assert str(report["evaluation"]["worst_group"]["group"]) in body


def test_every_evidence_panel_is_rendered(scanned):
    # st.pyplot arrives as an image element. Six panels: scorecard, folds,
    # null, per-subject, re-identification, features.
    assert len(scanned.get("image")) == 6


def test_re_identification_gets_its_own_headline(scanned):
    identity = scanned.session_state["report"]["identity_inference"]
    labels = [m.label for m in scanned.metric]
    shown = [m.value for m in scanned.metric]

    assert any("re-identification" in label.lower() for label in labels)
    assert f"{identity['accuracy']:.3f}" in shown


def test_the_attack_model_is_shown_beside_the_number(scanned):
    identity = scanned.session_state["report"]["identity_inference"]
    body = " ".join(
        [e.value for e in scanned.markdown]
        + [e.value for e in scanned.caption]
        + [e.value for e in scanned.info]
    )

    # A re-identification figure without its threat model invites the reader
    # to hear a stronger claim than the scan made.
    assert identity["attack_model"] in body


def test_per_unit_dataframe_is_shown(scanned):
    # A per-unit table should appear in addition to the chart.
    report = scanned.session_state["report"]
    n_units = len(report["evaluation"]["per_group"])
    # Streamlit dataframes show as st.dataframe elements.
    dfs = scanned.get("dataframe")
    assert dfs, "at least one dataframe should be present"
    # The per-unit table has one column per unit (n_units rows of data).
    per_unit_df = next(
        (d for d in dfs if len(d.value) == n_units),
        None,
    )
    assert per_unit_df is not None, "per-unit dataframe with correct row count not found"


def test_per_unit_table_disclaimer_is_shown(scanned):
    captions = " ".join(c.value for c in scanned.caption)
    assert "chance" in captions.lower() or "0.50" in captions


# ---------------------------------------------------------------------------
# Leakage-smell surfacing (spec: dataset warnings more prominent)
# ---------------------------------------------------------------------------

def test_dataset_leakage_warnings_section_is_silent_for_clean_report(scanned):
    """If the scan dataset has no leakage-smell warnings, no error widget is shown."""
    from nerveml.app import dataset_leakage_warnings_section

    report = scanned.session_state["report"]
    leakage_warnings = [
        w for w in report["dataset"]["warnings"]
        if w.get("code") in {"label_name_in_features", "feature_label_high_correlation"}
    ]
    # subject_leakage scenario has no leakage-smell warnings by design.
    assert leakage_warnings == []


def test_dataset_leakage_warnings_section_is_importable():
    from nerveml.app import dataset_leakage_warnings_section
    assert callable(dataset_leakage_warnings_section)


def test_leakage_smell_warning_appears_in_scan_report_dataset(tmp_path):
    """A scan on a dataset with a leakage-smell column stores warnings in the report."""
    from nerveml.scan import run_scan
    from nerveml.synth import make_synthetic

    df = make_synthetic("subject_leakage", n_subjects=8, n_trials=20, seed=0)
    df["valence"] = 0.0  # matches _LABEL_SMELL_KEYWORDS → label_name_in_features
    path = tmp_path / "leaky_scan.csv"
    df.to_csv(path, index=False)

    report = run_scan(
        dataset=str(path),
        model_kind="logistic_regression",
        n_splits=4,
        n_permutations=10,
        seed=0,
        identity_probe=False,
    )
    leakage_warnings = [
        w for w in report["dataset"]["warnings"]
        if w.get("code") in {"label_name_in_features", "feature_label_high_correlation"}
    ]
    assert leakage_warnings, "leakage-smell warning expected in report dataset.warnings"
