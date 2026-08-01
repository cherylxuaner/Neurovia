"""Evidence panels for the report (spec 13.5).

Six figures, each built from values already present in the report dict so a
chart can never disagree with the JSON beside it. Balanced accuracy is plotted
against an explicit chance reference rather than a zero baseline, because zero
is not a meaningful floor for this metric and anchoring there would compress
the only range anyone cares about.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

NAIVE = "#2a78d6"
GROUPED = "#eb6834"
CRITICAL = "#d03b3b"
# A lighter step of the same blue: the null is context, the observed score is
# the subject of the chart and has to win the eye.
NULL_FILL = "#5598e7"
# Diverging partner for the blue, per the palette's blue-red pair: a unit below
# chance is the opposite outcome from one above it, not merely a smaller one.
BELOW_CHANCE = "#e34948"

CHANCE = 0.5

# A record linked to the right person more often than not counts as recovered;
# the higher step separates "usually" from "essentially always".
RECOVERED = 0.5
STRONGLY_RECOVERED = 0.9
# Beyond this the grid gains columns rather than rows, so it stays on a page.
MAX_GRID_ROWS = 8
# Rows shown on the per-unit chart before it starts omitting the middle.
PER_GROUP_SHOWN = 18

FIGURE_DPI = 160

TRIAL_RANDOM_LABEL = "trial-random (weaker)"


def unit_name(group_column):
    """subject_id -> subject, session_id -> session."""
    return group_column[:-3] if group_column.endswith("_id") else group_column


def _new_figure(width, height):
    figure, axes = plt.subplots(figsize=(width, height), dpi=FIGURE_DPI)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    axes.title.set_color(INK)
    return figure, axes


def _protocol(report, scheme):
    for protocol in report["evaluation"]["protocols"]:
        if protocol["scheme"] == scheme:
            return protocol
    raise KeyError(f"protocol {scheme!r} missing from report")


def plot_null_distribution(report):
    """Where chance actually sits for this pipeline, and where the result landed."""
    null = report["permutation_test"]
    scores = report["null_distribution"]
    observed = null["observed_balanced_accuracy"]

    figure, axes = _new_figure(7.0, 3.6)
    axes.hist(
        scores,
        bins=min(20, max(6, len(scores) // 3)),
        color=NULL_FILL,
        edgecolor=SURFACE,
        linewidth=1.2,
        label=f"permuted labels (n={null['n_permutations']})",
    )
    beats_null = null["p_value"] < 0.05
    # Surface halo first, so the marker stays legible where it crosses a bar.
    axes.axvline(observed, color=SURFACE, linewidth=5.5, zorder=2)
    axes.axvline(
        observed,
        color=GROUPED if beats_null else CRITICAL,
        linewidth=2.5,
        zorder=3,
        label=f"observed  {observed:.3f}",
    )

    unit = unit_name(report["evaluation"]["grouping_unit"])
    axes.set_title(
        f"{unit.capitalize()}-held-out score against its own null    "
        f"permutation p = {null['p_value']:.3f}",
        fontsize=11,
        loc="left",
        pad=12,
    )
    axes.set_xlabel("balanced accuracy", fontsize=9, color=INK_SECONDARY)
    axes.set_ylabel("permutations", fontsize=9, color=INK_SECONDARY)
    axes.grid(axis="y", color=GRID, linewidth=1.0)
    axes.set_axisbelow(True)
    legend = axes.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    figure.tight_layout()
    return figure


def plot_fold_scores(report):
    """Every fold from both protocols, so a mean cannot hide a bad fold."""
    figure, axes = _new_figure(7.0, 3.6)

    axes.axvline(CHANCE, color=AXIS, linewidth=1.5, zorder=1)
    axes.annotate(
        "chance",
        xy=(CHANCE, 0.0),
        xycoords=("data", "axes fraction"),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
    )

    unit = unit_name(report["evaluation"]["grouping_unit"])
    panels = (
        ("trial_random", TRIAL_RANDOM_LABEL, NAIVE),
        (report["evaluation"]["strict_scheme"], f"{unit}-held-out (strict)", GROUPED),
    )
    for scheme, label, color in panels:
        protocol = _protocol(report, scheme)
        scores = protocol["fold_balanced_accuracy"]
        axes.scatter(
            scores,
            range(len(scores)),
            s=90,
            color=color,
            edgecolor=SURFACE,
            linewidth=2.0,
            zorder=3,
            label=f"{label}   mean {protocol['balanced_accuracy']:.3f}",
        )

    n_folds = report["config"]["n_splits"]
    axes.set_yticks(range(n_folds))
    axes.set_yticklabels([f"fold {i + 1}" for i in range(n_folds)])
    axes.set_xlim(0.0, 1.0)
    axes.invert_yaxis()
    axes.set_title(
        f"Per-fold balanced accuracy    gap "
        f"{report['evaluation']['generalization_gap']:+.3f}",
        fontsize=11,
        loc="left",
        pad=36,
    )
    axes.set_xlabel("balanced accuracy", fontsize=9, color=INK_SECONDARY)
    axes.grid(axis="x", color=GRID, linewidth=1.0)
    axes.set_axisbelow(True)
    # Above the plot area: at lower right the legend sat on top of the folds
    # whose scores are highest, which are the ones the reader looks at first.
    legend = axes.legend(
        frameon=False,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        handletextpad=0.4,
        columnspacing=2.0,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    figure.tight_layout()
    return figure


def plot_top_features(report):
    """Which inputs the model leaned on. Association only."""
    features = report["top_features"]
    names = [item["feature"] for item in features]
    values = [item["importance"] for item in features]

    figure, axes = _new_figure(7.0, max(2.6, 0.34 * len(features) + 1.6))
    positions = range(len(features))
    axes.barh(positions, values, height=0.55, color=NAIVE)

    axes.set_yticks(list(positions))
    axes.set_yticklabels(names)
    axes.invert_yaxis()
    for position, value in zip(positions, values):
        axes.annotate(
            f"{value:.3f}",
            xy=(value, position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )

    axes.set_title("Top feature associations", fontsize=11, loc="left", pad=12)
    axes.set_xlabel(
        "model-dependent and non-causal; describes this classifier on this data",
        fontsize=9,
        color=INK_SECONDARY,
    )
    axes.set_xlim(0, max(values) * 1.18 if values else 1)
    axes.grid(axis="x", color=GRID, linewidth=1.0)
    axes.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_per_group(report):
    """Every held-out unit, so no single failing participant can hide.

    Bars diverge from chance rather than from zero. Below chance and above
    chance are opposite outcomes for a participant, and a zero baseline would
    render them as merely shorter and longer.
    """
    evaluation = report["evaluation"]
    scored = [e for e in evaluation["per_group"] if e["balanced_accuracy"] is not None]
    scored = sorted(scored, key=lambda entry: entry["balanced_accuracy"])
    single_class = len(evaluation["per_group"]) - len(scored)
    worst = evaluation["worst_group"]

    # Sixty rows is a page, and a page of rows is not read. The weakest units
    # are the ones the mean is hiding, so those are kept; dropping the rest
    # silently would be lying by omission, so the count is printed.
    omitted = max(0, len(scored) - PER_GROUP_SHOWN)
    if omitted:
        keep = PER_GROUP_SHOWN - 3
        scored = scored[:keep] + scored[-3:]

    figure, axes = _new_figure(7.0, max(2.6, 0.28 * len(scored) + 1.8))
    positions = list(range(len(scored)))
    scores = [entry["balanced_accuracy"] for entry in scored]
    colors = [NAIVE if score >= CHANCE else BELOW_CHANCE for score in scores]

    axes.axvline(CHANCE, color=AXIS, linewidth=1.5, zorder=1)
    for position, score, color in zip(positions, scores, colors):
        axes.plot([CHANCE, score], [position, position], color=color, linewidth=2.0,
                  zorder=2)
    # A unit landing exactly on chance has a zero-length stem; the marker is
    # what stops it reading as missing data.
    axes.scatter(scores, positions, s=70, color=colors, edgecolor=SURFACE,
                 linewidth=1.5, zorder=3)

    axes.set_yticks(positions)
    axes.set_yticklabels([str(entry["group"]) for entry in scored])
    axes.set_xlim(0.0, 1.0)
    # Worst first: the unit the mean is hiding belongs where the eye lands.
    axes.invert_yaxis()

    headline = f"Per-{unit_name(evaluation['grouping_unit'])} balanced accuracy"
    if worst is not None:
        headline += (
            f"    worst: {worst['group']} at {worst['balanced_accuracy']:.3f}"
        )
    axes.set_title(headline, fontsize=11, loc="left", pad=12)

    label = "balanced accuracy, measured from chance"
    if omitted:
        label += (
            f"  ·  weakest {PER_GROUP_SHOWN - 3} and strongest 3 shown, "
            f"{omitted} omitted"
        )
    if single_class:
        label += f"  ·  {single_class} unit(s) scored no balanced accuracy"
    axes.set_xlabel(label, fontsize=9, color=INK_SECONDARY)
    axes.grid(axis="x", color=GRID, linewidth=1.0)
    axes.set_axisbelow(True)
    figure.tight_layout()
    return figure


def plot_identity(report):
    """One dot per person, so the finding is counted rather than measured.

    A row per identity was the wrong form here. Where everyone is recovered the
    rows are identical and the chart says one fact forty times; where there are
    a hundred people it says it a hundred times, over a page. "How many of them
    could be picked out" is a count, and a count is read fastest as a grid of
    countable things.

    Fill carries how reliably each one was recovered, so the spread that a bar
    chart showed is still there - it is just no longer costing a row apiece.
    """
    identity = report.get("identity_inference")
    if identity is None:
        return None

    entries = sorted(identity["per_identity"], key=lambda e: -e["recall"])
    recalls = [e["recall"] for e in entries]
    total = len(entries)
    recovered = sum(1 for r in recalls if r >= RECOVERED)
    weakest = entries[-1]
    by_chance = identity["chance"] * total

    # Rows are capped, columns grow. A grid that added a row per ten people
    # would climb off the page again at a hundred, which is the fault this
    # form was chosen to fix.
    # Roughly square while the count is small, gaining columns rather than rows
    # once it is not: a grid that added a row per ten people would climb off the
    # page at a hundred, which is the fault this form was chosen to fix.
    columns = max(int(round((total * 1.4) ** 0.5)), -(-total // MAX_GRID_ROWS))
    rows = -(-total // columns)

    figure, axes = _new_figure(7.0, 1.4 + 0.46 * rows)
    axes.set_xlim(-0.7, columns - 0.3)
    axes.set_ylim(-0.9, rows - 0.3)
    axes.set_xticks([])
    axes.set_yticks([])
    for side in axes.spines.values():
        side.set_visible(False)
    # Equal aspect keeps the marks evenly spaced in both directions, so the
    # grid reads as one block to be counted rather than as a stretched row.
    axes.set_aspect("equal")
    # Equal aspect shrinks the axes box; anchor it centrally so the block of
    # marks sits in the middle of the figure rather than against one edge.
    axes.set_anchor("C")
    axes.invert_yaxis()

    for index, entry in enumerate(entries):
        x, y = index % columns, index // columns
        recall = entry["recall"]
        if recall >= STRONGLY_RECOVERED:
            face, edge = NAIVE, NAIVE
        elif recall >= RECOVERED:
            face, edge = NULL_FILL, NULL_FILL
        else:
            face, edge = SURFACE, AXIS
        axes.scatter([x], [y], s=190, facecolor=face, edgecolor=edge,
                     linewidth=1.6, zorder=2)

    # The one that resisted most is the only one worth naming.
    weakest_index = total - 1
    axes.annotate(
        f"{weakest['group']}  {weakest['recall']:.2f}",
        xy=(weakest_index % columns, weakest_index // columns),
        xytext=(0, -15),
        textcoords="offset points",
        ha="center",
        fontsize=7.5,
        color=INK_MUTED,
    )

    # Title at figure level too: the axes is narrower than the page under equal
    # aspect, so a title anchored to it runs off the edge.
    figure.suptitle(
        f"Re-identification    {recovered} of {total} identities recovered    "
        f"chance would recover {by_chance:.0f}",
        fontsize=11,
        color=INK,
        y=0.97,
    )
    figure.tight_layout(rect=(0, 0.15, 1, 0.92))
    # Figure coordinates, not axes: equal aspect leaves the axes narrower than
    # the page, and a caption centred on it was clipped at both ends.
    figure.text(
        0.5, 0.085,
        f"{identity['accuracy']:.3f} of held-out records linked, "
        f"{identity['lift_over_chance']:.1f}x chance",
        ha="center", fontsize=9, color=INK_SECONDARY,
    )
    figure.text(
        0.5, 0.025,
        f"one mark per identity  ·  filled {STRONGLY_RECOVERED:.0%}+ linked  ·  "
        f"pale {RECOVERED:.0%}-{STRONGLY_RECOVERED:.0%}  ·  hollow below "
        f"{RECOVERED:.0%}",
        ha="center", fontsize=8, color=INK_MUTED,
    )
    return figure


def sparkline(value, reference, span=(0.0, 1.0), color=NAIVE):
    """A reading small enough to sit inside a line of text.

    Tufte's test for a sparkline is that it is small enough to be embedded in
    the text, and that every stroke is data. So there are no axes, no labels
    and no frame - a track for the range, a bar to the value, a tick at the
    reference. Beside a number in a table it does what a sentence of
    explanation would, in a glance and in less space.
    """
    figure, axes = plt.subplots(figsize=(1.15, 0.17), dpi=FIGURE_DPI)
    figure.patch.set_alpha(0)
    axes.set_facecolor("none")
    axes.set_xlim(*span)
    axes.set_ylim(0, 1)
    axes.set_xticks([])
    axes.set_yticks([])
    for side in axes.spines.values():
        side.set_visible(False)

    low, high = span
    clamped = min(max(value, low), high)
    axes.hlines(0.5, low, high, color=GRID, linewidth=3.4, zorder=1)
    axes.hlines(0.5, low, clamped, color=color, linewidth=3.4, zorder=2)
    axes.vlines(reference, 0.05, 0.95, color=INK_MUTED, linewidth=1.1, zorder=3)

    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    return figure


def plot_scorecard(report):
    """The headline numbers as tiles, each beside the reference it is read against.

    A row of bare figures invites the reader to supply their own baseline, and
    the baseline is the whole argument here: 0.53 means nothing until chance is
    drawn next to it. Each tile is a number, a track running the full range it
    could have taken, a mark where it landed, and a tick at the value it would
    take if there were nothing to find.
    """
    evaluation = report["evaluation"]
    identity = report.get("identity_inference")
    unit = unit_name(evaluation["grouping_unit"])

    tiles = [
        ("trial-random", evaluation["trial_random_balanced_accuracy"],
         CHANCE, "chance", NAIVE, (0.0, 1.0)),
        (f"{unit}-held-out", evaluation["grouped_balanced_accuracy"],
         CHANCE, "chance", GROUPED, (0.0, 1.0)),
        ("generalization gap", evaluation["generalization_gap"],
         0.20, "warning", CRITICAL, (0.0, 0.6)),
    ]
    if identity is not None:
        tiles.append((
            "re-identification", identity["accuracy"],
            identity["chance"], "chance", NAIVE, (0.0, 1.0),
        ))

    figure, panels = plt.subplots(
        1, len(tiles), figsize=(1.85 * len(tiles), 1.9), dpi=FIGURE_DPI
    )
    figure.patch.set_facecolor(SURFACE)

    for axes, (label, value, reference, reference_label, color, span) in zip(
        panels, tiles
    ):
        axes.set_facecolor(SURFACE)
        for side in axes.spines.values():
            side.set_visible(False)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_xlim(*span)
        axes.set_ylim(0, 1)

        formatted = f"{value:+.3f}" if label == "generalization gap" else f"{value:.3f}"
        axes.text(0.02, 0.86, label, transform=axes.transAxes, fontsize=8,
                  color=INK_MUTED, va="top")
        axes.text(0.02, 0.62, formatted, transform=axes.transAxes, fontsize=19,
                  color=INK, va="top", fontweight="medium")

        axes.hlines(0.16, span[0], span[1], color=GRID, linewidth=4.0,
                    transform=axes.get_yaxis_transform(), zorder=1)
        axes.hlines(0.16, span[0], min(max(value, span[0]), span[1]), color=color,
                    linewidth=4.0, transform=axes.get_yaxis_transform(), zorder=2)
        axes.vlines(reference, 0.08, 0.24, color=INK_MUTED, linewidth=1.4,
                    transform=axes.get_yaxis_transform(), zorder=3)
        axes.text(reference, 0.0, f" {reference_label} {reference:.2f}",
                  transform=axes.get_yaxis_transform(), fontsize=6.5,
                  color=INK_MUTED, va="bottom")

    figure.tight_layout(pad=0.9)
    return figure


def write_figures(report, output_dir):
    """Render every evidence panel to PNG. Returns the paths."""
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "scorecard": plot_scorecard,
        "null_distribution": plot_null_distribution,
        "fold_scores": plot_fold_scores,
        "per_group": plot_per_group,
        "identity": plot_identity,
        "top_features": plot_top_features,
    }

    paths = {}
    for name, builder in builders.items():
        figure = builder(report)
        if figure is None:
            continue
        path = output_dir / f"{name}.png"
        figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
        plt.close(figure)
        paths[name] = path
    return paths
