"""The report as a document someone outside the team can read (spec 12).

This is the artefact that leaves the building. It is laid out for a privacy
lead, a reviewer or a procurement team, so it opens with the lines they would
keep if they read nothing else, prints the rule behind every warning, and gives
the claims the scan does *not* support the same weight as the ones it does. A
report that buried those would be doing the overclaiming this product exists to
catch.

Colours come from the plotting module, so a page and a chart can never disagree
about what blue means.
"""

from functools import lru_cache
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

import matplotlib.pyplot as plt

from nerveml.plots import (
    CRITICAL,
    GROUPED,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    NAIVE,
    sparkline,
)
from nerveml.risk_rules import LEAKAGE_TYPE_HIGH_CORR, LEAKAGE_TYPE_LABEL_NAME

_LEAKAGE_SMELL_CODES = frozenset({LEAKAGE_TYPE_LABEL_NAME, LEAKAGE_TYPE_HIGH_CORR})

SURFACE = colors.HexColor("#fcfcfb")
PANEL = colors.HexColor("#f1f4f9")
INK_C = colors.HexColor(INK)
INK_2 = colors.HexColor(INK_SECONDARY)
MUTED = colors.HexColor(INK_MUTED)
RULE = colors.HexColor("#e1e0d9")
BLUE = colors.HexColor(NAIVE)
ORANGE = colors.HexColor(GROUPED)
RED = colors.HexColor(CRITICAL)

FONT_DIR = Path(__file__).parent / "fonts"
FONTS = {
    "sans": FONT_DIR / "IBMPlexSans-Regular.ttf",
    "sans_bold": FONT_DIR / "IBMPlexSans-SemiBold.ttf",
    "serif": FONT_DIR / "IBMPlexSerif-Regular.ttf",
    "mono": FONT_DIR / "IBMPlexMono-Regular.ttf",
}
# What to fall back to if the files are missing. A report set in the wrong
# typeface beats no report at all.
BASE_14 = {
    "sans": "Helvetica",
    "sans_bold": "Helvetica-Bold",
    "serif": "Times-Roman",
    "mono": "Courier",
}


@lru_cache(maxsize=1)
def register_fonts():
    """Register the vendored faces and return the name to use for each role.

    Embedding matters beyond looks: a base-14 font is not carried in the file,
    so the reader substitutes one and the same document wraps differently on a
    different machine. A document sent to a client should look the same when it
    gets there.
    """
    names = {}
    for role, path in FONTS.items():
        face = f"Plex-{role}"
        try:
            pdfmetrics.registerFont(TTFont(face, str(path)))
            names[role] = face
        except Exception:
            names[role] = BASE_14[role]
    return names


MARGIN = 18 * mm
CONTENT_WIDTH = A4[0] - 2 * MARGIN
# Running prose is set narrower than the frame. Sixty to seventy characters is
# the readable range; the full 174mm text block runs to about a hundred at this
# size, and tables keep the full width because a table is scanned, not read.
BODY_MEASURE = 122 * mm
FIGURE_WIDTH = CONTENT_WIDTH * 0.84
# Room left under a full-height figure for its caption and the page furniture.
FIGURE_MAX_HEIGHT = A4[1] - 2 * MARGIN - 28

VERDICT_WORDING = {
    "elevated": "Elevated inference signal",
    "moderate": "Moderate inference signal",
    "low_or_inconclusive": "Low or inconclusive",
    "inconclusive_at_this_sample_size": "Undecided at this sample size",
}

VERDICT_TONE = {
    "elevated": RED,
    "moderate": ORANGE,
    "low_or_inconclusive": BLUE,
    "inconclusive_at_this_sample_size": MUTED,
}

DEMONSTRATION_NOTICE = (
    "This scan was run on a constructed demonstration dataset. The customer is "
    "fictional and the recordings are synthetic. What is real is the failure "
    "the data contains and the fact that the scan was not told about it."
)

CAPTIONS = {
    "scorecard": "The headline numbers, each beside the value it would take if "
    "there were nothing to find.",
    "fold_scores": "Every fold of both protocols, against an explicit chance "
    "reference.",
    "null_distribution": "The strict score against labels shuffled within each "
    "unit, which is where chance actually sits for this pipeline.",
    "per_group": "Each held-out unit scored on its own, so a mean cannot hide a "
    "participant the model fails completely.",
    "identity": "Recall for each identity, with chance drawn in the same frame.",
    "top_features": "Ranked feature associations. Model-dependent and non-causal.",
}


SEVERITY_WORD = {"warning": "Warning", "info": "Informational"}


def _styles():
    base = getSampleStyleSheet()
    faces = register_fonts()
    sans, bold = faces["sans"], faces["sans_bold"]
    serif, mono = faces["serif"], faces["mono"]
    measure_indent = CONTENT_WIDTH - BODY_MEASURE
    return {
        "cover_eyebrow": ParagraphStyle(
            "cover_eyebrow", parent=base["Normal"], fontName=sans, fontSize=8,
            leading=11.5, textColor=MUTED, alignment=TA_LEFT, spaceAfter=8,
        ),
        "cover_title": ParagraphStyle(
            "cover_title", parent=base["Title"], fontName=bold,
            fontSize=29, leading=33, textColor=INK_C, alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "cover_lede": ParagraphStyle(
            "cover_lede", parent=base["Normal"], fontName=sans, fontSize=12,
            leading=18.5, textColor=INK_2, alignment=TA_LEFT, spaceAfter=10,
            rightIndent=CONTENT_WIDTH - BODY_MEASURE,
        ),
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName=bold, fontSize=23,
            leading=27, textColor=INK_C, alignment=TA_LEFT, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName=sans, fontSize=10,
            leading=15, textColor=INK_2, spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName=bold, fontSize=12,
            leading=17, textColor=INK_C, spaceBefore=20, spaceAfter=2,
            borderPadding=0,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName=sans, fontSize=9.5,
            leading=15, textColor=INK_2, spaceAfter=6,
            rightIndent=measure_indent,
        ),
        "body_wide": ParagraphStyle(
            "body_wide", parent=base["Normal"], fontName=sans, fontSize=9.5,
            leading=15, textColor=INK_2, spaceAfter=5,
        ),
        "quote": ParagraphStyle(
            "quote", parent=base["Normal"], fontName=serif,
            fontSize=12, leading=18.5, textColor=INK_C, leftIndent=12,
            rightIndent=measure_indent, spaceBefore=4, spaceAfter=6,
        ),
        "panel_key": ParagraphStyle(
            "panel_key", parent=base["Normal"], fontName=bold,
            fontSize=9, leading=13.5, textColor=INK_C,
        ),
        "panel_val": ParagraphStyle(
            "panel_val", parent=base["Normal"], fontName=sans, fontSize=9,
            leading=13.5, textColor=INK_2,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontName=sans, fontSize=8,
            leading=11.5, textColor=MUTED,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["Normal"], fontName=sans, fontSize=8,
            leading=11, textColor=MUTED, alignment=1,
        ),
        "rule": ParagraphStyle(
            "rule", parent=base["Normal"], fontName=mono, fontSize=7.5,
            leading=11, textColor=MUTED, leftIndent=10,
        ),
    }


def _cover(report, styles, figures):
    """A cover states what this is, whose it is, and what it found.

    Detail starts overleaf. A reader who only ever sees this page should still
    leave with the finding rather than with a title.
    """
    dataset = report["dataset"]
    context = dataset.get("context") or {}
    risk = report["risk_flags"]
    evaluation = report["evaluation"]
    unit = evaluation["grouping_unit"].removesuffix("_id")

    blocks = [
        Spacer(1, 26),
        Paragraph("NEURAL DATA VALIDATION SCAN", styles["cover_eyebrow"]),
        Paragraph("Neural Dataset Risk Report", styles["cover_title"]),
        Paragraph(
            context.get("customer") or dataset["name"],
            styles["cover_lede"],
        ),
        HRFlowable(width="100%", thickness=0.7, color=RULE,
                   spaceBefore=8, spaceAfter=18),
        Paragraph(
            f"A model reported {evaluation['trial_random_balanced_accuracy']:.3f} "
            f"on a shuffled split of its own trials. Held out a whole "
            f"{unit}, it scores "
            f"{evaluation['grouped_balanced_accuracy']:.3f}.",
            styles["cover_lede"],
        ),
        Spacer(1, 8),
        _verdict_panel(report, styles),
        Spacer(1, 20),
    ]

    scorecard = _figure(figures, "scorecard", styles)
    if scorecard is not None:
        blocks.append(scorecard)

    blocks += [
        Spacer(1, 14),
        _findings_at_a_glance(report, styles),
        Spacer(1, 18),
        Paragraph(
            DEMONSTRATION_NOTICE if dataset.get("context")
            else risk["heuristic_notice"],
            styles["small"],
        ),
        PageBreak(),
    ]
    return blocks


def _findings_at_a_glance(report, styles):
    """Counts before detail, so a reader knows the shape before reading it."""
    flags = report["risk_flags"]["flags"]
    counts = {}
    for flag in flags:
        counts.setdefault(flag["severity"], []).append(flag["code"])

    rows = [[
        Paragraph("<b>Findings at a glance</b>", styles["panel_key"]),
        Paragraph("", styles["panel_val"]),
        Paragraph("", styles["panel_val"]),
    ]]
    for severity in ("warning", "info"):
        codes = counts.get(severity, [])
        rows.append([
            Paragraph(_chip(SEVERITY_WORD[severity], severity), styles["panel_val"]),
            Paragraph(f"<b>{len(codes)}</b>", styles["panel_val"]),
            Paragraph(", ".join(codes) or "none", styles["small"]),
        ])

    table = Table(
        rows,
        colWidths=[CONTENT_WIDTH * 0.22, CONTENT_WIDTH * 0.06,
                   CONTENT_WIDTH * 0.72],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
    ]))
    return table


def _chip(text, severity):
    colour = "#d03b3b" if severity == "warning" else "#2a78d6"
    return f'<font color="{colour}"><b>{text}</b></font>'


def _verdict_panel(report, styles):
    """The lines a reader keeps if they read nothing else."""
    risk = report["risk_flags"]
    evaluation = report["evaluation"]
    identity = report.get("identity_inference")
    level = risk["sensitive_inference_evidence"]
    unit = evaluation["grouping_unit"].removesuffix("_id")

    lines = [
        ("Cross-person validity",
         f"{evaluation['trial_random_balanced_accuracy']:.3f} on a shuffled "
         f"split becomes {evaluation['grouped_balanced_accuracy']:.3f} on a "
         f"held-out {unit}."),
        ("Sensitive inference", VERDICT_WORDING.get(level, level)),
    ]
    if identity is not None:
        lines.append((
            "Re-identification",
            f"{identity['accuracy']:.3f} of held-out records linked to the "
            f"individual who produced them, "
            f"{identity['lift_over_chance']:.1f}x chance.",
        ))

    table = Table(
        [[Paragraph(f"<b>{label}</b>", styles["panel_key"]),
          Paragraph(value, styles["panel_val"])] for label, value in lines],
        colWidths=[CONTENT_WIDTH * 0.26, CONTENT_WIDTH * 0.74 - 14],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (0, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LINEBEFORE", (0, 0), (0, -1), 3, VERDICT_TONE.get(level, BLUE)),
    ]))
    return table


def _spark(reading, cache_dir):
    """Render one sparkline to a file and return it as a flowable."""
    if reading is None or cache_dir is None:
        return ""
    value, reference, span, color = reading
    figure = sparkline(value, reference, span=span, color=color)
    path = Path(cache_dir) / f"spark_{abs(hash((value, reference, span))):x}.png"
    figure.savefig(path, transparent=True, dpi=220)
    plt.close(figure)
    # Size through the constructor: assigning drawWidth afterwards does not
    # survive, and the flowable then renders at its pixel size and spills
    # across the column beside it.
    probe = Image(str(path))
    width = 30 * mm
    image = Image(
        str(path), width=width, height=probe.imageHeight * (width / probe.imageWidth)
    )
    image.hAlign = "LEFT"
    return image


def _metric_table(rows, styles, cache_dir=None):
    """Label, value, an optional reading, and a note.

    Cells use the unindented style: the measure indent that keeps running prose
    readable would leave a 56mm column with almost no usable width, and
    ReportLab answers that by breaking one character per line.
    """
    body = []
    for row in rows:
        label, value, note = row[:3]
        reading = row[3] if len(row) > 3 else None
        body.append([
            Paragraph(f"<b>{label}</b>", styles["body_wide"]),
            Paragraph(value, styles["body_wide"]),
            _spark(reading, cache_dir),
            Paragraph(note, styles["small"]),
        ])

    table = Table(
        body,
        colWidths=[CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.15,
                   CONTENT_WIDTH * 0.19, CONTENT_WIDTH * 0.41],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]))
    return table


def _flag_block(flag, styles):
    accent = RED if flag["severity"] == "warning" else BLUE
    header = Table(
        [[Paragraph(
            f"{_chip(SEVERITY_WORD[flag['severity']], flag['severity'])}"
            f" &nbsp;&nbsp;{flag['code']}",
            styles["body_wide"],
        )]],
        colWidths=[CONTENT_WIDTH], hAlign="LEFT",
    )
    header.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return KeepTogether([
        Spacer(1, 6),
        header,
        Paragraph(flag["message"], styles["body"]),
        Paragraph(f"Rule: {flag['rule']}", styles["rule"]),
    ])


def _leakage_callout(warning, styles):
    """Distinct callout box for label-leakage-smell warnings.

    Uses an orange left-border accent (same as grouped-accuracy colour) to
    visually separate these from routine integrity notes. The header names the
    code so operators can grep CI logs for it; the body carries the full message.
    """
    label = "Label leakage smell"
    header = Table(
        [[Paragraph(
            f'<font color="{GROUPED}">'
            f"<b>⚠ {label} — {warning['code']}</b></font>",
            styles["body_wide"],
        )]],
        colWidths=[CONTENT_WIDTH], hAlign="LEFT",
    )
    header.setStyle(TableStyle([
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ORANGE),
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return KeepTogether([
        Spacer(1, 8),
        header,
        Paragraph(warning["message"], styles["body"]),
        Spacer(1, 4),
    ])


def _interval(report, name):
    bounds = report["evaluation"]["confidence_intervals"].get(name, (None, None))
    if None in bounds:
        return ""
    return f"95% CI {bounds[0]:.3f} to {bounds[1]:.3f}."


def _per_unit_table(per_group, unit_label, styles):
    """All held-out units ranked worst-first by balanced accuracy.

    A chart communicates the distribution shape; a table lets the reader read
    the exact score for the one unit they are responsible for. Both belong in a
    report that leaves the building.
    """
    scored = [e for e in per_group if e.get("balanced_accuracy") is not None]
    if not scored:
        return []

    rows = sorted(scored, key=lambda e: e["balanced_accuracy"])
    header = [
        Paragraph(f"<b>{unit_label.capitalize()}</b>", styles["body_wide"]),
        Paragraph("<b>Balanced accuracy</b>", styles["body_wide"]),
        Paragraph("<b>ROC-AUC</b>", styles["body_wide"]),
        Paragraph("<b>Trials</b>", styles["body_wide"]),
    ]
    body = [header]
    for e in rows:
        roc = f"{e['roc_auc']:.3f}" if e.get("roc_auc") is not None else "—"
        single = e.get("single_class", False)
        suffix = " ¹" if single else ""
        body.append([
            Paragraph(str(e["group"]), styles["body_wide"]),
            Paragraph(f"{e['balanced_accuracy']:.3f}{suffix}", styles["body_wide"]),
            Paragraph(roc, styles["body_wide"]),
            Paragraph(str(e["n_trials"]), styles["body_wide"]),
        ])

    col_w = CONTENT_WIDTH / 4
    table = Table(body, colWidths=[col_w] * 4, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, INK_C),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, RULE),
        ("BACKGROUND", (0, 0), (-1, 0), PANEL),
        ("TEXTCOLOR", (1, 1), (1, 1), RED),
    ]))
    footnote = []
    if any(e.get("single_class") for e in rows):
        footnote = [Paragraph(
            "¹ All trials in this unit share one label; balanced accuracy "
            "defaults to 0.50.",
            styles["small"],
        )]
    return [Spacer(1, 6), table] + footnote + [Spacer(1, 4)]


def fit_figure(path):
    """Scale a figure to fit the text frame in both directions.

    A per-unit chart with twenty rows is nearly square. Scaling on width alone
    then pushes it taller than the frame, and it comes out clipped - axis
    labels gone, title cut in half.
    """
    image = Image(str(path))
    scale = min(
        FIGURE_WIDTH / image.imageWidth, FIGURE_MAX_HEIGHT / image.imageHeight
    )
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return image


def _figure(figures, name, styles):
    """A figure with its caption, kept on one page or moved together."""
    if not figures or name not in figures:
        return None
    path = Path(figures[name])
    if not path.exists():
        return None

    label = Paragraph(CAPTIONS[name], styles["caption"])
    label.hAlign = "CENTER"
    return KeepTogether(
        [Spacer(1, 10), fit_figure(path), Spacer(1, 3), label, Spacer(1, 6)]
    )


class _Tracker(SimpleDocTemplate):
    """Remembers the last heading drawn, so the header can name the section."""

    current_section = ""

    def afterFlowable(self, flowable):
        title = getattr(flowable, "_section_title", None)
        if title:
            self.current_section = title


def _page_furniture(canvas, doc):
    faces = register_fonts()
    canvas.saveState()
    canvas.setFillColor(SURFACE)
    canvas.rect(0, 0, A4[0], A4[1], stroke=0, fill=1)

    section = getattr(doc, "current_section", "")
    if doc.page > 1 and section:
        canvas.setFont(faces["sans"], 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, A4[1] - MARGIN + 10, section)
        canvas.drawRightString(
            A4[0] - MARGIN, A4[1] - MARGIN + 10, "Neural Dataset Risk Report"
        )
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, A4[1] - MARGIN + 5, A4[0] - MARGIN, A4[1] - MARGIN + 5)

    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, MARGIN - 6, A4[0] - MARGIN, MARGIN - 6)
    canvas.setFont(faces["sans"], 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(
        MARGIN, MARGIN - 16,
        "NerveML prototype scan - not a clinical, legal or regulatory certification",
    )
    canvas.drawRightString(A4[0] - MARGIN, MARGIN - 16, f"page {doc.page}")
    canvas.restoreState()


def _context_block(report, styles):
    """The customer's own account of the data, before any finding."""
    context = report["dataset"].get("context")
    if not context:
        return []

    table = Table(
        [[Paragraph("<b>Client</b>", styles["panel_key"]),
          Paragraph(context["customer"], styles["panel_val"])],
         [Paragraph("<b>What it is</b>", styles["panel_key"]),
          Paragraph(context["pitch"], styles["panel_val"])],
         [Paragraph("<b>Reported</b>", styles["panel_key"]),
          Paragraph(context["reported_metric"], styles["panel_val"])]],
        colWidths=[CONTENT_WIDTH * 0.20, CONTENT_WIDTH * 0.80 - 14],
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (0, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]))
    return [
        *_heading("Submitted for review", styles),
        table,
        Spacer(1, 6),
        Paragraph(DEMONSTRATION_NOTICE, styles["small"]),
    ]


def _heading(text, styles):
    """A heading, its hairline, and a bookmark the running header can read.

    The number is set in the accent colour and the words in ink: the numeral
    guides a reader scanning for a section, the words carry what it is.
    """
    number, _, rest = text.partition("&nbsp;")
    if rest.strip() and number.strip().rstrip(".").isdigit():
        # Grey, not accent. Tufte's rule: grayscale carries text and structure,
        # colour is reserved for data. A section numeral is structure, and
        # spending the accent on it makes it mean less where it is data.
        marked = f'<font color="#898781">{number}</font>&nbsp;{rest}'
    else:
        marked = text
    heading = Paragraph(marked, styles["h2"])
    heading._section_title = _plain(text)
    return [
        heading,
        HRFlowable(width="100%", thickness=0.5, color=RULE, spaceAfter=8),
    ]


def _plain(markup):
    """Strip the entities a heading carries, for the running header."""
    return " ".join(markup.replace("&nbsp;", " ").split())


def _dataset_section(report, styles):
    dataset = report["dataset"]
    blocks = [
        *_heading("1 &nbsp; Dataset", styles),
        _metric_table([
            ("Modality", dataset["modality"],
             f"Target: {dataset['target']}."),
            ("Subjects / trials",
             f"{dataset['n_subjects']} / {dataset['n_trials']}",
             f"Between {dataset['min_trials_per_subject']} and "
             f"{dataset['max_trials_per_subject']} trials per subject."),
            ("Features", str(dataset["n_features"]), "One row per trial."),
            ("Class balance",
             " / ".join(f"{k}: {v}" for k, v in dataset["class_balance"].items()),
             "Balanced accuracy and ROC-AUC are used throughout, so an "
             "imbalance cannot flatter the result."),
            ("Missing feature values", str(dataset["missing_values"]),
             "Any imputation is fitted inside training folds only."),
            ("Duplicate trial ids", str(dataset["duplicate_trial_ids"]),
             "Repeated rows can place one measurement on both sides of a split."),
        ], styles),
    ]
    for warning in dataset["warnings"]:
        if warning.get("code") in _LEAKAGE_SMELL_CODES:
            blocks.append(_leakage_callout(warning, styles))
        else:
            blocks += [
                Spacer(1, 6),
                Paragraph(f"&nbsp;&nbsp;!&nbsp;&nbsp;{warning['message']}", styles["body"]),
            ]
    return blocks


def _multi_axis_section(report, styles, section):
    """Per-axis rows when extra grouping dimensions were evaluated.

    Each axis ran the same strict grouped protocol as the primary split, so
    the numbers are directly comparable. The caveat paragraph is mandatory: a
    table of accuracy values without it reads as four independent guarantees
    rather than four facets of one scan.
    """
    entries = report.get("multi_axis_validation", [])
    if not entries:
        return []

    rows = []
    for entry in entries:
        axis_label = entry["axis"].removesuffix("_id").replace("_", " ").capitalize()
        worst = entry.get("worst_group") or {}
        note = (
            f"ROC-AUC {entry['grouped_roc_auc']:.3f}. "
            f"Gap vs trial-random: {entry['generalization_gap']:+.3f}. "
            f"{entry['n_units']} units."
        )
        ci = entry.get("confidence_intervals", {})
        bacc_ci = ci.get("grouped_balanced_accuracy", [None, None])
        if bacc_ci and bacc_ci[0] is not None and bacc_ci[1] is not None:
            note += f" Bal-acc 95% CI [{bacc_ci[0]:.3f}, {bacc_ci[1]:.3f}]."
        gap_ci = ci.get("generalization_gap", [None, None])
        if gap_ci and gap_ci[0] is not None and gap_ci[1] is not None:
            note += f" Gap 95% CI [{gap_ci[0]:+.3f}, {gap_ci[1]:+.3f}]."
        if worst:
            note += (
                f" Worst {axis_label.lower()}: {worst['group']} "
                f"at {worst['balanced_accuracy']:.3f} balanced accuracy."
            )
        rows.append((
            f"{axis_label}-held-out balanced accuracy",
            f"{entry['grouped_balanced_accuracy']:.3f}",
            note,
        ))

    return [
        *_heading(f"{section} &nbsp; Multi-axis held-out validation", styles),
        Paragraph(
            "Each row below was evaluated under the same strict grouped protocol "
            "as the primary split, holding the named axis out whole. These are "
            "descriptive associations under the stated attack model, not "
            "independent validity claims.",
            styles["body"],
        ),
        Spacer(1, 4),
        _metric_table(rows, styles),
    ]


def _secondary_probes_section(report, styles, section):
    """Secondary sensitive-attribute probes — extra attributes found in the data."""
    probes = report.get("secondary_probes", [])
    if not probes:
        return []

    rows = []
    for p in probes:
        rows.append((
            p["attribute"],
            f"{p['grouped_balanced_accuracy']:.3f}",
            f"ROC-AUC {p['grouped_roc_auc']:.3f}. {p['n_units']} held-out units. "
            f"{p['interpretation']}",
        ))

    return [
        *_heading(f"{section} &nbsp; Secondary sensitive-attribute probes", styles),
        Paragraph(
            "These columns were found alongside the primary label and probed "
            "using the same features and held-out protocol. A high balanced "
            "accuracy indicates the primary features also carry information about "
            "that attribute — an association under this attack model, not a causal "
            "or diagnostic claim.",
            styles["body"],
        ),
        Spacer(1, 4),
        _metric_table(rows, styles),
    ]


def _artifact_section(report, styles, section):
    """EOG- and EMG-proxy baseline — how much performance do artifact channels explain."""
    ab = report.get("artifact_baseline")
    if not ab:
        return []

    rows = []
    full_bacc = ab["full_balanced_accuracy"]
    rows.append((
        "Full model (reference)",
        f"{full_bacc:.3f}",
        "Subject-held-out balanced accuracy on all features.",
    ))
    for key, label in (("eog_proxy", "EOG proxy (frontal channels Fp*, AF*)"),
                       ("emg_proxy", "EMG proxy (gamma band 30-45 Hz)")):
        p = ab.get(key)
        if p:
            share = ab.get(f"artifact_share_{p['name']}")
            share_str = f" | artifact share ≤{share:.0%}" if share is not None else ""
            rows.append((
                label,
                f"{p['balanced_accuracy']:.3f}",
                f"{p['n_features']} features.{share_str}",
            ))

    return [
        *_heading(f"{section} &nbsp; Artifact-confound baseline", styles),
        _metric_table(rows, styles),
        Spacer(1, 4),
        Paragraph(ab["interpretation"], styles["body"]),
    ]


def _validity_section(report, styles, unit, cache_dir=None):
    evaluation = report["evaluation"]
    risk = report["risk_flags"]
    null = report["permutation_test"]
    full = (0.0, 1.0)
    return [
        *_heading("2 &nbsp; Validity", styles),
        Paragraph(
            "Both protocols run over identical features and an identical model "
            "family. Only the splitting rule differs, so the difference between "
            "them is attributable to the protocol and to nothing else.",
            styles["body"],
        ),
        Spacer(1, 4),
        _metric_table([
            ("Trial-random balanced accuracy",
             f"{evaluation['trial_random_balanced_accuracy']:.3f}",
             "Weaker protocol: trials from one subject fall on both sides of "
             "the split, so a model that recognises people scores well without "
             "learning anything transferable.",
             (evaluation["trial_random_balanced_accuracy"], 0.5, full, NAIVE)),
            (f"{unit.capitalize()}-held-out balanced accuracy",
             f"{evaluation['grouped_balanced_accuracy']:.3f}",
             f"Strict protocol. Fold SD "
             f"{evaluation['grouped_balanced_accuracy_std']:.3f}. "
             + _interval(report, "grouped_balanced_accuracy"),
             (evaluation["grouped_balanced_accuracy"], 0.5, full, GROUPED)),
            (f"{unit.capitalize()}-held-out ROC-AUC",
             f"{evaluation['grouped_roc_auc']:.3f}",
             _interval(report, "grouped_roc_auc"),
             (evaluation["grouped_roc_auc"], 0.55, full, GROUPED)),
            ("Generalization gap",
             f"{evaluation['generalization_gap']:+.3f}",
             _interval(report, "generalization_gap")
             + " A description of the evaluation, not an accusation.",
             (evaluation["generalization_gap"], 0.20, (0.0, 0.6), CRITICAL)),
            ("Within-unit balanced accuracy",
             f"{evaluation['trial_random_within_unit_balanced_accuracy']:.3f}",
             "The same trial-random predictions scored inside each unit "
             "separately. Where this falls to chance while the pooled score "
             "does not, the pooled score was telling units apart rather than "
             "telling labels apart.",
             (evaluation["trial_random_within_unit_balanced_accuracy"], 0.5,
              full, NAIVE)),
            ("Permutation p", f"{null['p_value']:.3f}",
             f"Observed {null['observed_balanced_accuracy']:.3f} against "
             f"{null['n_permutations']} within-unit label shuffles, null mean "
             f"{null['null_mean']:.3f}. A finite permutation count cannot "
             "evidence an arbitrarily small p."),
            ("Verdict",
             VERDICT_WORDING.get(
                 risk["sensitive_inference_evidence"],
                 risk["sensitive_inference_evidence"],
             ),
             f"About {risk['units_needed']} units would settle it."
             if risk.get("units_needed")
             else "Stated only because the whole interval sits on one side of "
                  "the threshold."),
        ], styles, cache_dir),
    ]


def write_pdf_report(report, path, figures=None):
    """Render the scan as a PDF. Returns the path written."""
    styles = _styles()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sparklines are written beside the document and are part of it; a reader
    # never sees them as files.
    cache_dir = path.parent / ".sparklines"
    cache_dir.mkdir(exist_ok=True)

    dataset = report["dataset"]
    evaluation = report["evaluation"]
    risk = report["risk_flags"]
    identity = report.get("identity_inference")
    composition = report.get("fingerprint_composition")
    unit = evaluation["grouping_unit"].removesuffix("_id")

    story = _cover(report, styles, figures)
    story += [
        *_heading("What this scan concludes", styles),
        Paragraph(report["permitted_claim"], styles["quote"]),
    ]
    story += _context_block(report, styles)
    story += _dataset_section(report, styles)
    story += _validity_section(report, styles, unit, cache_dir)

    for name in ("fold_scores", "null_distribution"):
        block = _figure(figures, name, styles)
        if block is not None:
            story.append(block)

    section = 3
    story += _multi_axis_section(report, styles, section)
    if report.get("multi_axis_validation"):
        section += 1

    story += _heading(f"{section} &nbsp; Per-{unit} performance", styles)
    worst = evaluation.get("worst_group")
    if worst is not None:
        story.append(Paragraph(
            f"A mean over {unit}s is the statistic most able to hide one the "
            f"model fails. The weakest here is <b>{worst['group']}</b>, at "
            f"{worst['balanced_accuracy']:.3f} balanced accuracy over "
            f"{worst['n_trials']} trials.",
            styles["body"],
        ))
    story += _per_unit_table(evaluation.get("per_group", []), unit, styles)
    block = _figure(figures, "per_group", styles)
    if block is not None:
        story.append(block)
    section += 1
    if identity is not None:
        story += [
            *_heading(f"{section} &nbsp; Re-identification", styles),
            _metric_table([
                ("Held-out records linked", f"{identity['accuracy']:.3f}",
                 f"{identity['lift_over_chance']:.1f}x a chance of "
                 f"{identity['chance']:.3f} over {identity['n_identities']} "
                 f"identities and {identity['n_records']} records."),
                ("Most identifiable", str(identity["most_identifiable"]["group"]),
                 f"{identity['most_identifiable']['recall']:.3f} recall."),
            ] + ([
                ("Amplitude only",
                 f"{composition['amplitude_only']['accuracy']:.3f}",
                 "Per-channel level alone, which is what skull thickness, "
                 "electrode impedance and cap fit move."),
                ("Spectral shape only",
                 f"{composition['spectral_shape_only']['accuracy']:.3f}",
                 "Each channel's overall level divided out."),
            ] if composition else []), styles),
            Spacer(1, 8),
            Paragraph(f"<b>Attack model.</b> {identity['attack_model']}",
                      styles["body"]),
        ]
        if composition:
            story.append(Paragraph(composition["interpretation"], styles["body"]))
        block = _figure(figures, "identity", styles)
        if block is not None:
            story.append(block)
        section += 1

    if report.get("artifact_baseline"):
        story += _artifact_section(report, styles, section)
        section += 1

    if report.get("secondary_probes"):
        story += _secondary_probes_section(report, styles, section)
        section += 1

    if report.get("top_features"):
        story += [
            *_heading(f"{section} &nbsp; Feature associations", styles),
            Paragraph(report["feature_caveat"], styles["body"]),
            Spacer(1, 4),
            _metric_table([
                (item["feature"], f"{item['importance']:.4f}",
                 f"Direction of association: {item['direction']}."
                 if item.get("direction") else "Unsigned importance.")
                for item in report["top_features"][:8]
            ], styles),
        ]
        block = _figure(figures, "top_features", styles)
        if block is not None:
            story.append(block)
        section += 1

    story += _heading(f"{section} &nbsp; Risk flags", styles)
    story += [_flag_block(flag, styles) for flag in risk["flags"]]
    section += 1

    story += [
        Paragraph(f"{section} &nbsp; Claims this scan does not support",
                  styles["h2"]),
        Paragraph(
            "Listed whatever the result. A scan is evidence under a stated "
            "attack model; it is not a certificate.",
            styles["body"],
        ),
    ]
    story += [
        Paragraph(f"&nbsp;&nbsp;—&nbsp;&nbsp;{claim}", styles["body"])
        for claim in report["unsupported_claims"]
    ]
    section += 1

    story += _heading(f"{section} &nbsp; Recommended next steps", styles)
    story += [
        Paragraph(f"&nbsp;&nbsp;{index}.&nbsp;&nbsp;{item}", styles["body"])
        for index, item in enumerate(report["recommendations"], start=1)
    ]

    if report["dataset"].get("context"):
        story += [
            *_heading("About this demonstration", styles),
            Paragraph(DEMONSTRATION_NOTICE, styles["body"]),
            Paragraph(
                "<b>The failure that was planted:</b> "
                + report["dataset"]["context"].get("planted_failure", ""),
                styles["body"],
            ),
        ]

    story += [
        *_heading("How this scan was run", styles),
        Paragraph(
            " &nbsp;·&nbsp; ".join(
                f"{key}: {value}" for key, value in report["config"].items()
            ),
            styles["small"],
        ),
        Spacer(1, 8),
        Paragraph(risk["heuristic_notice"], styles["small"]),
    ]

    document = _Tracker(
        str(path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN + 6,
        bottomMargin=MARGIN + 6,
        title=f"NerveML Neural Dataset Risk Report - {dataset['name']}",
        author="NerveML",
        subject="Neural data privacy and validation scan",
    )
    document.build(story, onFirstPage=_page_furniture, onLaterPages=_page_furniture)
    return path
