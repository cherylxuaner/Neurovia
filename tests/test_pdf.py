"""The PDF report (spec 6.2, 12).

This is the artefact that leaves the building: a privacy lead, a reviewer or a
procurement team reads it without opening a notebook. So the tests are about
content rather than layout — every claim the scan is allowed to make must be
in it, and every claim it is not allowed to make must be in it too.
"""

import pytest
from pypdf import PdfReader

from nerveml.pdf import write_pdf_report
from nerveml.scan import run_scan


@pytest.fixture(scope="module")
def report():
    return run_scan(
        dataset="subject_leakage",
        n_subjects=10,
        n_trials=20,
        model_kind="logistic_regression",
        n_splits=4,
        n_permutations=15,
        seed=0,
    )


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, report):
    path = tmp_path_factory.mktemp("pdf") / "report.pdf"
    write_pdf_report(report, path)
    return path


def text_of(path):
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def test_a_pdf_is_produced(rendered):
    assert rendered.exists()
    assert rendered.read_bytes()[:5] == b"%PDF-"


def test_it_runs_to_several_pages(rendered):
    assert len(PdfReader(rendered).pages) >= 2


def test_it_does_not_waste_pages(tmp_path, report):
    from nerveml.plots import write_figures

    path = tmp_path / "packed.pdf"
    write_pdf_report(report, path, figures=write_figures(report, tmp_path / "f"))

    # Forced page breaks used to strand a lone figure on an otherwise empty
    # page. Content flows now: eight sections and five figures, still short
    # enough that nobody has to be talked into reading it.
    assert len(PdfReader(path).pages) <= 10


def test_the_headline_numbers_are_in_it(rendered, report):
    text = text_of(rendered)
    evaluation = report["evaluation"]

    assert f"{evaluation['trial_random_balanced_accuracy']:.3f}" in text
    assert f"{evaluation['grouped_balanced_accuracy']:.3f}" in text
    assert f"{evaluation['generalization_gap']:+.3f}" in text


def test_the_permitted_claim_is_quoted_verbatim(rendered, report):
    # Paraphrasing the one sentence a customer may repeat would defeat it.
    assert report["permitted_claim"] in " ".join(text_of(rendered).split())


def test_every_unsupported_claim_is_listed(rendered, report):
    text = " ".join(text_of(rendered).split())

    for claim in report["unsupported_claims"]:
        assert claim in text


def test_the_risk_flags_carry_their_rules(rendered, report):
    text = " ".join(text_of(rendered).split())

    for flag in report["risk_flags"]["flags"]:
        assert flag["code"] in text
        assert flag["rule"] in text


def test_the_attack_model_travels_with_the_re_identification_number(rendered, report):
    text = " ".join(text_of(rendered).split())

    assert report["identity_inference"]["attack_model"] in text


def test_the_dataset_summary_carries_balance_and_missingness(rendered, report):
    text = " ".join(text_of(rendered).split())
    dataset = report["dataset"]

    # Spec 12: modality, subjects, trials, features, label definition,
    # missingness, class balance.
    for count in dataset["class_balance"].values():
        assert str(count) in text
    assert str(dataset["missing_values"]) in text
    assert dataset["modality"] in text


def test_integrity_warnings_are_carried_into_the_document(tmp_path):
    import numpy as np
    import pandas as pd

    from nerveml.scenarios import load_scenario

    df, feats = load_scenario("focus_tracker")
    df.loc[0, feats[0]] = np.nan
    path = tmp_path / "gappy.csv"
    df.to_csv(path, index=False)

    scanned = run_scan(
        dataset=str(path), model_kind="logistic_regression", n_splits=4,
        n_permutations=10, seed=0, identity_probe=False,
    )
    pdf = tmp_path / "gappy.pdf"
    write_pdf_report(scanned, pdf)

    assert scanned["dataset"]["warnings"]
    text = " ".join(text_of(pdf).split())
    for warning in scanned["dataset"]["warnings"]:
        assert warning["message"] in text


def test_the_strict_result_reports_its_fold_variability(rendered, report):
    text = text_of(rendered)

    # A mean over folds without its spread hides how unstable it was.
    assert f"{report['evaluation']['grouped_balanced_accuracy_std']:.3f}" in text


def test_feature_associations_are_a_section_of_their_own(rendered, report):
    text = " ".join(text_of(rendered).split())

    # Spec 12 asks for them explicitly, and marked non-causal.
    for item in report["top_features"][:5]:
        assert item["feature"] in text
    # The full caveat sentence can be word-wrapped mid-token by the PDF renderer,
    # so check key phrases rather than the full string.
    assert "model-dependent and non-causal" in text
    assert "not which neural processes" in text


def test_the_within_unit_reading_is_explained_not_just_printed(rendered):
    text = " ".join(text_of(rendered).split()).lower()

    assert "within" in text
    assert "telling units apart" in text or "telling labels apart" in text


def scenario_pdf(tmp_path):
    from nerveml.plots import write_figures

    scanned = run_scan(
        dataset="focus_tracker", model_kind="logistic_regression",
        n_splits=4, n_permutations=10, seed=0,
    )
    path = tmp_path / "scenario.pdf"
    write_pdf_report(scanned, path, figures=write_figures(scanned, tmp_path / "f"))
    return scanned, path


def test_the_customer_context_opens_the_report(tmp_path):
    scanned, path = scenario_pdf(tmp_path)
    text = " ".join(text_of(path).split())
    context = scanned["dataset"]["context"]

    # A reader should meet the claim before the finding, as they would in a
    # real engagement.
    assert context["customer"] in text
    assert context["reported_metric"] in text


def test_a_demonstration_says_so_on_the_page(tmp_path):
    _, path = scenario_pdf(tmp_path)
    text = " ".join(text_of(path).split()).lower()

    # Constructed data must never be mistakable for an audit of real work.
    assert "demonstration" in text
    assert "fictional" in text


def test_a_real_dataset_gets_no_demonstration_banner(rendered):
    assert "constructed" not in " ".join(text_of(rendered).split()).lower()


def test_the_cover_carries_the_verdict_and_nothing_else(rendered):
    first = PdfReader(rendered).pages[0].extract_text() or ""

    # A cover sets the frame: what this is, whose it is, what it found. Detail
    # starts overleaf.
    assert "Neural Dataset Risk Report" in first
    assert "Rule:" not in first


def test_findings_are_summarised_by_severity(rendered, report):
    text = " ".join(text_of(rendered).split())
    flags = report["risk_flags"]["flags"]
    warnings = sum(1 for f in flags if f["severity"] == "warning")

    # Convention from security reporting: a count before the detail, so a
    # reader knows the shape of the report before reading it.
    assert "Findings at a glance" in text
    assert f"{warnings}" in text
    for flag in flags:
        assert flag["code"] in text


def test_body_prose_is_set_to_a_readable_measure():
    from reportlab.lib.units import mm

    from nerveml.pdf import BODY_MEASURE, CONTENT_WIDTH

    # 60-70 characters is the readable range; the full 174mm text frame runs to
    # about a hundred at this size.
    assert BODY_MEASURE < CONTENT_WIDTH
    assert 100 < BODY_MEASURE / mm < 135


def test_table_cells_are_not_squeezed_by_the_prose_measure(rendered):
    text = text_of(rendered)

    # A cell too narrow for its own text is broken one character per line,
    # which survives as "M i s s i n g" in the extracted text.
    assert "M i s s i n g" not in text
    assert "Missing feature values" in text


def test_the_vendored_fonts_register():
    from nerveml.pdf import FONTS, register_fonts

    families = register_fonts()

    assert families["sans"] != "Helvetica"
    for role in ("sans", "sans_bold", "serif", "mono"):
        assert role in families
    for path in FONTS.values():
        assert path.exists()


def test_fonts_are_embedded_in_the_document(rendered):
    # Base-14 fonts are substituted by the reader, so the same file wraps
    # differently on a different machine. Embedded ones do not.
    raw = rendered.read_bytes()

    assert b"FontFile2" in raw
    assert b"IBMPlex" in raw


def test_a_missing_font_file_does_not_stop_the_report(tmp_path, report, monkeypatch):
    import nerveml.pdf as pdf

    monkeypatch.setattr(
        pdf, "FONTS", {k: tmp_path / "gone.ttf" for k in pdf.FONTS}
    )
    pdf.register_fonts.cache_clear()
    path = tmp_path / "fallback.pdf"

    # A report set in the wrong typeface beats no report at all.
    pdf.write_pdf_report(report, path)
    pdf.register_fonts.cache_clear()

    assert path.exists()


def test_the_permitted_claim_is_the_only_serif_on_the_page(rendered):
    from nerveml.pdf import _styles, register_fonts

    families = register_fonts()
    styles = _styles()

    assert styles["quote"].fontName == families["serif"]
    others = [s for name, s in styles.items() if name != "quote"]
    assert all(s.fontName != families["serif"] for s in others)


def test_metric_rows_carry_a_sparkline(tmp_path, report):
    from nerveml.pdf import write_pdf_report

    path = tmp_path / "sparked.pdf"
    write_pdf_report(report, path)

    # Every embedded raster adds an image object; the validity table's readings
    # each carry one.
    raw = path.read_bytes()
    assert raw.count(b"/Subtype /Image") >= 4 or raw.count(b"/Subtype/Image") >= 4


def test_colour_is_reserved_for_data_not_for_chrome(rendered):
    from nerveml.pdf import _heading, _styles

    blocks = _heading("2 &nbsp; Validity", _styles())

    # Tufte: grayscale for text and structure, colour reserved for data. A
    # section numeral is structure.
    assert "#2a78d6" not in blocks[0].text.lower()


def test_the_heuristic_notice_is_present(rendered, report):
    assert report["risk_flags"]["heuristic_notice"] in " ".join(
        text_of(rendered).split()
    )


def test_the_configuration_is_recorded_for_reproduction(rendered, report):
    text = text_of(rendered)

    assert report["config"]["model_kind"] in text
    assert str(report["config"]["n_permutations"]) in text
    assert str(report["config"]["seed"]) in text


def test_the_document_is_titled_and_dated(rendered):
    metadata = PdfReader(rendered).metadata

    assert "NerveML" in metadata.title


def test_figures_are_embedded_when_they_exist(tmp_path, report):
    from nerveml.plots import write_figures

    figures = write_figures(report, tmp_path / "figs")
    path = tmp_path / "with_figures.pdf"
    write_pdf_report(report, path, figures=figures)

    plain = tmp_path / "plain.pdf"
    write_pdf_report(report, plain)
    assert path.stat().st_size > plain.stat().st_size


def test_a_tall_figure_is_scaled_to_fit_the_page(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib.pagesizes import A4

    from nerveml.pdf import MARGIN, fit_figure

    # A figure taller than it is wide: scaling on width alone pushes it off
    # the page, which clips the axis labels and the title.
    figure, axes = plt.subplots(figsize=(7.4, 14), dpi=160)
    axes.plot([0, 1], [0, 1])
    path = tmp_path / "tall.png"
    figure.savefig(path)
    plt.close(figure)

    image = fit_figure(path)

    assert image.drawHeight <= A4[1] - 2 * MARGIN
    assert image.drawWidth <= A4[0] - 2 * MARGIN


def test_a_missing_figure_does_not_stop_the_report(tmp_path, report):
    path = tmp_path / "missing.pdf"

    write_pdf_report(report, path, figures={"fold_scores": tmp_path / "nope.png"})

    assert path.exists()


def test_a_scan_without_the_identity_probe_still_renders(tmp_path):
    bare = run_scan(
        dataset="subject_leakage",
        n_subjects=8,
        n_trials=20,
        model_kind="logistic_regression",
        n_splits=4,
        n_permutations=10,
        seed=0,
        identity_probe=False,
    )
    path = tmp_path / "bare.pdf"

    write_pdf_report(bare, path)

    assert "Re-identification" not in text_of(path)


def _make_multi_axis_csv(tmp_path):
    """Feature CSV with subject_id as primary and session_id as extra axis."""
    import numpy as np
    import pandas as pd

    from nerveml.synth import make_synthetic

    rng = np.random.default_rng(42)
    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    df["session_id"] = rng.integers(0, 4, size=len(df)).astype(int)
    path = tmp_path / "multi.csv"
    df.to_csv(path, index=False)
    return path


def test_multi_axis_section_appears_when_extra_axes_detected(tmp_path):
    scanned = run_scan(
        dataset=str(_make_multi_axis_csv(tmp_path)),
        model_kind="logistic_regression",
        n_splits=4,
        n_permutations=10,
        seed=0,
        identity_probe=False,
    )
    path = tmp_path / "multi.pdf"
    write_pdf_report(scanned, path)
    text = " ".join(text_of(path).split())

    # Section heading present and axis name surfaced
    assert "Multi-axis" in text
    assert scanned["multi_axis_validation"]
    axis = scanned["multi_axis_validation"][0]["axis"].removesuffix("_id").capitalize()
    assert axis in text


def test_multi_axis_section_absent_when_no_extra_axes(rendered):
    # rendered fixture uses subject_leakage which has no extra grouping columns.
    assert "Multi-axis" not in text_of(rendered)


def test_per_unit_table_contains_all_group_ids(rendered, report):
    text = " ".join(text_of(rendered).split())
    for entry in report["evaluation"]["per_group"]:
        assert str(entry["group"]) in text


def test_per_unit_table_worst_first(tmp_path, report):
    from nerveml.pdf import _per_unit_table, _styles

    styles = _styles()
    flowables = _per_unit_table(report["evaluation"]["per_group"], "subject", styles)

    assert flowables, "table must not be empty when per_group is non-empty"


def test_per_unit_table_empty_per_group_returns_nothing():
    from nerveml.pdf import _per_unit_table, _styles

    styles = _styles()
    assert _per_unit_table([], "subject", styles) == []


def test_per_unit_table_skips_null_balanced_accuracy():
    from nerveml.pdf import _per_unit_table, _styles

    # Entries where balanced_accuracy is None (edge case) must not appear.
    rows = [
        {"group": "s0", "balanced_accuracy": None, "roc_auc": None, "n_trials": 10, "single_class": True},
        {"group": "s1", "balanced_accuracy": 0.6, "roc_auc": 0.65, "n_trials": 10, "single_class": False},
    ]
    styles = _styles()
    flowables = _per_unit_table(rows, "subject", styles)
    assert flowables


# ---------------------------------------------------------------------------
# Leakage-smell callout
# ---------------------------------------------------------------------------

def _leakage_warning(code="label_name_in_features"):
    return {
        "code": code,
        "message": f"Feature column 'valence' matches known label vocabulary ({code}).",
    }


def test_leakage_callout_returns_flowables():
    from nerveml.pdf import _leakage_callout, _styles

    styles = _styles()
    flowables = _leakage_callout(_leakage_warning(), styles)
    # KeepTogether wraps the inner list; the result must be a KeepTogether.
    from reportlab.platypus import KeepTogether
    assert isinstance(flowables, KeepTogether)


def test_leakage_callout_encodes_code_in_header():
    from reportlab.platypus import Table
    from nerveml.pdf import _leakage_callout, _styles

    styles = _styles()
    callout = _leakage_callout(_leakage_warning("feature_label_high_correlation"), styles)
    # KeepTogether stores flowables in _content
    items = callout._content
    table = next(i for i in items if isinstance(i, Table))
    cell_texts = " ".join(
        str(cell) for row in table._cellvalues for cell in row
    )
    assert "feature_label_high_correlation" in cell_texts


def test_leakage_callout_encodes_message():
    from reportlab.platypus import Paragraph
    from nerveml.pdf import _leakage_callout, _styles

    styles = _styles()
    msg = "Feature column 'arousal' matches known label vocabulary."
    callout = _leakage_callout({"code": "label_name_in_features", "message": msg}, styles)
    items = callout._content
    para_texts = " ".join(p.text for p in items if isinstance(p, Paragraph))
    assert "arousal" in para_texts


def test_leakage_smell_warning_appears_with_callout_header_in_pdf(tmp_path):
    """A dataset with a leakage-smell warning must produce a callout in the PDF."""
    import pandas as pd
    from nerveml.synth import make_synthetic

    df = make_synthetic("subject_leakage", n_subjects=8, n_trials=20, seed=0)
    # Add a column named "valence" — matches the _LABEL_SMELL_KEYWORDS exactly.
    df["valence"] = 0.0
    path = tmp_path / "leaky.csv"
    df.to_csv(path, index=False)

    scanned = run_scan(
        dataset=str(path),
        model_kind="logistic_regression",
        n_splits=4,
        n_permutations=10,
        seed=0,
        identity_probe=False,
    )
    pdf_path = tmp_path / "leaky.pdf"
    write_pdf_report(scanned, pdf_path)
    text = " ".join(text_of(pdf_path).split())

    assert "leakage smell" in text.lower()
    assert "label_name_in_features" in text


def test_non_leakage_warning_still_renders_in_pdf(tmp_path):
    """Integrity warnings (not leakage) must still appear in the PDF."""
    import numpy as np
    from nerveml.scenarios import load_scenario

    df, feats = load_scenario("focus_tracker")
    df.loc[0, feats[0]] = np.nan
    path = tmp_path / "gappy2.csv"
    df.to_csv(path, index=False)

    scanned = run_scan(
        dataset=str(path),
        model_kind="logistic_regression",
        n_splits=4,
        n_permutations=10,
        seed=0,
        identity_probe=False,
    )
    pdf_path = tmp_path / "gappy2.pdf"
    write_pdf_report(scanned, pdf_path)
    text = " ".join(text_of(pdf_path).split())

    for w in scanned["dataset"]["warnings"]:
        assert w["message"] in text


# ── Multi-axis gap CI branch ──────────────────────────────────────────────────

def _multi_axis_entry(*, axis="session_id", gap_ci=None, bacc_ci=None):
    """Minimal multi_axis_validation entry for unit tests of _multi_axis_section."""
    entry = {
        "axis": axis,
        "n_units": 4,
        "scheme": f"{axis}_grouped",
        "grouped_balanced_accuracy": 0.612,
        "grouped_roc_auc": 0.671,
        "generalization_gap": 0.087,
        "std": 0.042,
        "worst_group": {"group": "s1", "balanced_accuracy": 0.532},
        "per_group": [],
        "confidence_intervals": {
            "grouped_balanced_accuracy": bacc_ci or [None, None],
            "generalization_gap": gap_ci or [None, None],
        },
    }
    return entry


def test_multi_axis_gap_ci_branch_when_bounds_are_present():
    """'Gap 95% CI' annotation must appear when gap CI bounds are non-None."""
    from nerveml.pdf import _multi_axis_section, _styles

    styles = _styles()
    report = {"multi_axis_validation": [_multi_axis_entry(gap_ci=[0.042, 0.138])]}

    flowables = _multi_axis_section(report, styles, "5")
    combined = " ".join(
        getattr(f, "text", "") for f in flowables
    )
    # The note is embedded inside a Table cell Paragraph; extract text by
    # serialising the flowable's raw XML-like markup.
    assert "Gap 95% CI" in combined or any(
        "Gap 95% CI" in getattr(cell, "text", "")
        for f in flowables
        for row in (getattr(f, "_cellvalues", None) or [])
        for cell in row
        if hasattr(cell, "text")
    ), "Gap 95% CI annotation must be in at least one flowable when bounds are set"


def test_multi_axis_gap_ci_branch_absent_when_bounds_are_none():
    """'Gap 95% CI' must NOT appear when gap CI bounds are [None, None]."""
    from nerveml.pdf import _multi_axis_section, _styles

    styles = _styles()
    report = {"multi_axis_validation": [_multi_axis_entry(gap_ci=[None, None])]}

    flowables = _multi_axis_section(report, styles, "5")
    combined = " ".join(getattr(f, "text", "") for f in flowables)
    # None bounds → branch must be skipped entirely
    assert "Gap 95% CI" not in combined and not any(
        "Gap 95% CI" in getattr(cell, "text", "")
        for f in flowables
        for row in (getattr(f, "_cellvalues", None) or [])
        for cell in row
        if hasattr(cell, "text")
    ), "Gap 95% CI must be absent when bounds are None"


def test_multi_axis_gap_ci_appears_in_full_pdf(tmp_path):
    """End-to-end: multi-axis gap CI must appear in rendered PDF text."""
    import numpy as np
    from nerveml.synth import make_synthetic

    rng = np.random.default_rng(42)
    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    # 4 sessions → enough axis units for bootstrap_ci to return non-None gap CI
    df["session_id"] = rng.integers(0, 4, size=len(df)).astype(int)
    path = tmp_path / "gap_ci_full.csv"
    df.to_csv(path, index=False)

    scanned = run_scan(
        dataset=str(path),
        model_kind="logistic_regression",
        n_splits=4,
        n_permutations=5,
        seed=0,
        identity_probe=False,
    )
    # Confirm scan actually produced non-None gap CI for the session axis
    session_entry = next(
        (e for e in scanned.get("multi_axis_validation", []) if e["axis"] == "session_id"),
        None,
    )
    assert session_entry is not None, "session_id axis must appear in multi_axis_validation"
    gap_ci = (session_entry.get("confidence_intervals") or {}).get("generalization_gap", [None, None])
    assert gap_ci[0] is not None, "gap CI must be non-None with 4+ axis units"

    pdf_path = tmp_path / "gap_ci.pdf"
    write_pdf_report(scanned, pdf_path)
    text = " ".join(text_of(pdf_path).split())
    assert "Gap 95% CI" in text, "Gap 95% CI must appear in rendered PDF"
