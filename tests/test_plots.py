"""Spec P1 - the null distribution and feature associations have to be visible.

These tests check that each figure is built from the report's real values and
saved where the report says it is. They deliberately do not assert on pixels;
what they protect is the wiring, so a chart can never quietly show stale or
placeholder numbers.
"""

import matplotlib
import pytest

matplotlib.use("Agg")

from nerveml.plots import (  # noqa: E402
    plot_fold_scores,
    plot_identity,
    plot_null_distribution,
    plot_per_group,
    plot_scorecard,
    plot_top_features,
    write_figures,
)
from nerveml.scan import run_scan  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    """Tests build figures directly; only write_figures closes its own."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")


@pytest.fixture(scope="module")
def report():
    return run_scan(
        dataset="subject_leakage",
        n_subjects=8,
        n_trials=20,
        model_kind="logistic_regression",
        n_splits=4,
        n_permutations=15,
        seed=0,
    )


def all_text(figure):
    """Everything readable on the figure, wherever it was anchored."""
    parts = [t.get_text() for t in figure.texts]
    for axes in figure.axes:
        parts.append(caption(axes))
        parts += [t.get_text() for t in axes.texts]
    return " ".join(parts)


def test_null_distribution_marks_the_observed_score(report):
    figure = plot_null_distribution(report)
    axes = figure.axes[0]

    observed = report["permutation_test"]["observed_balanced_accuracy"]
    marked = [line.get_xdata()[0] for line in axes.lines]

    assert any(abs(x - observed) < 1e-6 for x in marked)


def test_null_distribution_plots_every_permutation(report):
    figure = plot_null_distribution(report)
    axes = figure.axes[0]

    bar_heights = [patch.get_height() for patch in axes.patches]
    assert sum(bar_heights) == report["permutation_test"]["n_permutations"]


def caption(axes):
    """Every title slot plus the axis label - the chart's readable text."""
    titles = [axes.get_title(loc=loc) for loc in ("left", "center", "right")]
    return " ".join(titles + [axes.get_xlabel()])


def test_null_distribution_reports_the_p_value_in_the_title(report):
    figure = plot_null_distribution(report)

    assert f"{report['permutation_test']['p_value']:.3f}" in caption(figure.axes[0])


def test_fold_chart_shows_both_protocols(report):
    figure = plot_fold_scores(report)
    axes = figure.axes[0]

    labels = {text.get_text() for text in axes.get_legend().get_texts()}
    assert any("trial-random" in label for label in labels)
    assert any("subject-held-out" in label for label in labels)


def test_fold_chart_plots_one_point_per_fold_per_protocol(report):
    figure = plot_fold_scores(report)
    axes = figure.axes[0]

    plotted = sum(len(collection.get_offsets()) for collection in axes.collections)
    assert plotted == 2 * report["config"]["n_splits"]


def test_fold_chart_marks_the_chance_line(report):
    figure = plot_fold_scores(report)
    axes = figure.axes[0]

    assert any(line.get_xdata()[0] == 0.5 for line in axes.lines)


def test_feature_chart_shows_the_ranked_features(report):
    figure = plot_top_features(report)
    axes = figure.axes[0]

    expected = [item["feature"] for item in report["top_features"]]
    labels = [text.get_text() for text in axes.get_yticklabels()]
    assert set(labels) == set(expected)


def test_feature_chart_states_that_associations_are_not_causal(report):
    figure = plot_top_features(report)

    text = caption(figure.axes[0]).lower()
    assert "non-causal" in text or "not causal" in text


def test_per_group_chart_caps_its_length_and_says_what_it_dropped():
    from nerveml.plots import PER_GROUP_SHOWN

    many = {
        "evaluation": {
            "grouping_unit": "subject_id",
            "per_group": [
                {"group": f"P{i:02d}", "n_trials": 20,
                 "balanced_accuracy": round(0.30 + i * 0.01, 3),
                 "roc_auc": 0.5, "accuracy": 0.5, "single_class": False}
                for i in range(60)
            ],
            "worst_group": {"group": "P00", "n_trials": 20,
                            "balanced_accuracy": 0.30},
        }
    }
    figure = plot_per_group(many)
    axes = figure.axes[0]

    # Sixty rows is a page, and a page of rows is not read. The worst are what
    # matter, but a chart that quietly drops the rest is lying by omission.
    labels = [t.get_text() for t in axes.get_yticklabels()]
    assert len(labels) <= PER_GROUP_SHOWN
    assert "P00" in labels
    assert "omitted" in caption(axes)
    assert str(60 - len(labels)) in caption(axes)


def test_per_group_chart_shows_every_unit_when_there_are_few(report):
    figure = plot_per_group(report)
    axes = figure.axes[0]

    labels = [t.get_text() for t in axes.get_yticklabels()]
    assert "omitted" not in caption(axes)
    assert len(labels) == len(report["evaluation"]["per_group"])


def test_per_group_chart_shows_every_held_out_unit(report):
    figure = plot_per_group(report)
    axes = figure.axes[0]

    labels = [text.get_text() for text in axes.get_yticklabels()]
    assert set(labels) == {e["group"] for e in report["evaluation"]["per_group"]}


def test_per_group_chart_is_anchored_at_chance(report):
    figure = plot_per_group(report)
    axes = figure.axes[0]

    # Marks diverge from 0.5, not from 0: below chance and above chance are
    # opposite outcomes, and a zero baseline would not show that.
    assert axes.lines
    assert all(line.get_xdata()[0] == pytest.approx(0.5) for line in axes.lines)


def test_a_unit_scoring_exactly_at_chance_is_still_visible(report):
    figure = plot_per_group(report)
    axes = figure.axes[0]

    # A zero-length bar would render as nothing and read as missing data, so
    # every unit carries a marker regardless of its distance from chance.
    scored = [e for e in report["evaluation"]["per_group"] if e["balanced_accuracy"]]
    plotted = sum(len(collection.get_offsets()) for collection in axes.collections)
    assert plotted == len(scored)


def test_the_worst_unit_is_listed_first(report):
    figure = plot_per_group(report)
    axes = figure.axes[0]

    worst = report["evaluation"]["worst_group"]["group"]
    top_to_bottom = [t.get_text() for t in axes.get_yticklabels()]
    if axes.yaxis_inverted():
        assert top_to_bottom[0] == worst
    else:
        assert top_to_bottom[-1] == worst


def test_per_group_chart_names_the_worst_unit(report):
    figure = plot_per_group(report)

    worst = report["evaluation"]["worst_group"]["group"]
    assert worst in caption(figure.axes[0])


def test_identity_chart_shows_one_mark_per_identity(report):
    figure = plot_identity(report)
    axes = figure.axes[0]

    plotted = sum(len(c.get_offsets()) for c in axes.collections)
    assert plotted == len(report["identity_inference"]["per_identity"])


def test_identity_chart_stays_compact_as_identities_grow(report):
    small = plot_identity(report)
    many = dict(report)
    identity = dict(many["identity_inference"])
    identity["per_identity"] = [
        {"group": f"P{i:03d}", "n_records": 20, "recall": 1.0} for i in range(120)
    ]
    identity["n_identities"] = 120
    many["identity_inference"] = identity

    # A row per identity turns a hundred people into a page of identical bars.
    # A grid gains columns instead, so it stays inside half a page at any size.
    assert plot_identity(many).get_size_inches()[1] <= 5.5
    assert small.get_size_inches()[1] <= 5.5


def test_identity_chart_counts_how_many_were_recovered(report):
    figure = plot_identity(report)

    entries = report["identity_inference"]["per_identity"]
    recovered = sum(1 for e in entries if e["recall"] >= 0.5)
    assert f"{recovered} of {len(entries)}" in all_text(figure)


def test_identity_chart_states_what_chance_would_recover(report):
    figure = plot_identity(report)

    # A count means nothing without the count a guess would have produced.
    identity = report["identity_inference"]
    expected_by_chance = identity["chance"] * identity["n_identities"]
    assert f"{expected_by_chance:.0f}" in all_text(figure)


def test_identity_chart_states_the_headline_and_the_lift(report):
    figure = plot_identity(report)

    identity = report["identity_inference"]
    text = all_text(figure)
    assert f"{identity['accuracy']:.3f}" in text
    assert f"{identity['lift_over_chance']:.1f}" in text


def test_identity_chart_names_the_hardest_identity_to_recover(report):
    figure = plot_identity(report)
    axes = figure.axes[0]

    entries = report["identity_inference"]["per_identity"]
    floor = min(e["recall"] for e in entries)
    hardest = {str(e["group"]) for e in entries if e["recall"] == floor}
    # With every recall tied, any of them is the hardest; the point is that one
    # is named rather than that a particular one is.
    assert any(group in all_text(figure) for group in hardest)


def test_no_identity_chart_when_the_probe_did_not_run(report):
    without = dict(report, identity_inference=None)

    assert plot_identity(without) is None


def test_a_sparkline_is_small_enough_to_sit_in_a_table_row():
    from nerveml.plots import sparkline

    figure = sparkline(0.53, reference=0.5)

    # Tufte's test for a sparkline: small enough to be embedded in the text.
    width, height = figure.get_size_inches()
    assert height <= 0.22
    assert width <= 1.4


def test_a_sparkline_marks_its_value_and_its_reference():
    from nerveml.plots import sparkline

    axes = sparkline(0.8, reference=0.5).axes[0]

    # A mark without its reference is not a reading, at any size.
    assert axes.collections
    assert len(axes.collections) >= 2


def test_a_sparkline_carries_no_axes_or_labels():
    from nerveml.plots import sparkline

    axes = sparkline(0.8, reference=0.5).axes[0]

    # Data-ink only: every stroke that is not the value or its reference is
    # ink spent saying nothing.
    assert axes.get_xticks().size == 0
    assert not axes.get_xlabel()
    assert not any(side.get_visible() for side in axes.spines.values())


def test_the_scorecard_carries_one_tile_per_headline_number(report):
    figure = plot_scorecard(report)

    # Four tiles: the weaker protocol, the strict one, the distance between
    # them, and re-identification.
    assert len(figure.axes) == 4


def test_every_tile_prints_its_own_number(report):
    figure = plot_scorecard(report)
    evaluation = report["evaluation"]

    printed = " ".join(
        text.get_text() for axes in figure.axes for text in axes.texts
    )
    assert f"{evaluation['trial_random_balanced_accuracy']:.3f}" in printed
    assert f"{evaluation['grouped_balanced_accuracy']:.3f}" in printed
    assert f"{evaluation['generalization_gap']:+.3f}" in printed
    assert f"{report['identity_inference']['accuracy']:.3f}" in printed


def test_a_tile_shows_where_its_reference_sits(report):
    figure = plot_scorecard(report)

    # A number without its reference is not a reading. Each tile draws the
    # value it would take if nothing were there, as a line collection.
    for axes in figure.axes:
        assert axes.collections


def test_the_scorecard_drops_the_identity_tile_when_it_did_not_run(report):
    figure = plot_scorecard(dict(report, identity_inference=None))

    assert len(figure.axes) == 3


def test_write_figures_saves_every_panel(tmp_path, report):
    paths = write_figures(report, tmp_path)

    assert set(paths) == {
        "scorecard",
        "null_distribution",
        "fold_scores",
        "top_features",
        "per_group",
        "identity",
    }
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
        assert path.suffix == ".png"


def session_report(tmp_path):
    """A scan grouped by something other than subject."""
    from nerveml.synth import make_synthetic

    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    df["session_id"] = df["subject_id"].str[-2:].astype(int) // 2
    path = tmp_path / "sessions.csv"
    df.to_csv(path, index=False)
    return run_scan(
        dataset=str(path),
        group_column="session_id",
        model_kind="logistic_regression",
        n_splits=4,
        n_permutations=10,
        seed=0,
    )


def test_figures_render_for_any_grouping_unit(tmp_path):
    # The strict protocol is named after its grouping column, so a figure that
    # looks for "subject_grouped" cannot render a session-grouped scan.
    paths = write_figures(session_report(tmp_path), tmp_path / "out")

    assert len(paths) == 6
    for path in paths.values():
        assert path.stat().st_size > 0


def test_fold_chart_labels_the_actual_grouping_unit(tmp_path):
    figure = plot_fold_scores(session_report(tmp_path))

    labels = " ".join(t.get_text() for t in figure.axes[0].get_legend().get_texts())
    assert "session" in labels


def test_figures_are_closed_after_writing(tmp_path, report):
    import matplotlib.pyplot as plt

    plt.close("all")
    write_figures(report, tmp_path)

    # A long scan session must not leak figure handles.
    assert plt.get_fignums() == []
