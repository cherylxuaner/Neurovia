"""nerveml-compare: compare two audit reports and exit non-zero on regressions.

Usage
-----
    nerveml-compare baseline.json candidate.json [--out comparison.json] [--strict]
    nerveml-compare --list-metrics
    nerveml-compare ... --threshold evaluation.grouped_balanced_accuracy=0.05

Exit codes
----------
0   No regressions found (improvements and stable metrics are allowed).
1   At least one regression found.
2   Bad arguments or unreadable files.

Designed for CI: put it after the scan step in a pipeline and it will break
the build when a code or preprocessing change degrades auditable metrics.
"""

import argparse
import json
import sys
from pathlib import Path

from nerveml.regression import (
    ARTIFACT_METRIC_SPECS,
    MULTI_AXIS_CI_HI_METRICS,
    MULTI_AXIS_CI_METRICS,
    MULTI_AXIS_METRIC_SPECS,
    SECONDARY_PROBE_METRIC_SPECS,
    TRACKED_METRICS,
    compare_reports,
)


def _list_metrics():
    """Print all tracked metric paths with direction and default tolerance."""
    rows = []

    for path, higher_is_better, tol in TRACKED_METRICS:
        direction = "higher-better" if higher_is_better else "lower-better"
        tol_str = "sig-flip" if tol is None else f"{tol:.4f}"
        rows.append((path, direction, tol_str))

    for key, higher_is_better, tol in ARTIFACT_METRIC_SPECS:
        path = f"artifact_baseline.{key}"
        direction = "higher-better" if higher_is_better else "lower-better"
        rows.append((path, direction, f"{tol:.4f}"))

    for key, higher_is_better, tol in MULTI_AXIS_METRIC_SPECS:
        path = f"multi_axis_validation[{{axis}}].{key}"
        direction = "higher-better" if higher_is_better else "lower-better"
        rows.append((path, direction, f"{tol:.4f}"))

    for key, higher_is_better, tol in MULTI_AXIS_CI_METRICS:
        path = f"multi_axis_validation[{{axis}}].ci_lo.{key}"
        direction = "higher-better" if higher_is_better else "lower-better"
        rows.append((path, direction, f"{tol:.4f}"))

    for key, higher_is_better, tol in MULTI_AXIS_CI_HI_METRICS:
        path = f"multi_axis_validation[{{axis}}].ci_hi.{key}"
        direction = "higher-better" if higher_is_better else "lower-better"
        rows.append((path, direction, f"{tol:.4f}"))

    for key, higher_is_better, tol in SECONDARY_PROBE_METRIC_SPECS:
        path = f"secondary_probes[{{attribute}}].{key}"
        direction = "higher-better" if higher_is_better else "lower-better"
        rows.append((path, direction, f"{tol:.4f}"))

    h0, h1, h2 = "Metric Path", "Direction", "Default Tolerance"
    w0 = max(len(h0), max(len(r[0]) for r in rows))
    w1 = max(len(h1), max(len(r[1]) for r in rows))

    print()
    print(f"  {h0:<{w0}}  {h1:<{w1}}  {h2}")
    print(f"  {'─' * w0}  {'─' * w1}  {'─' * len(h2)}")
    for path, direction, tol_str in rows:
        print(f"  {path:<{w0}}  {direction:<{w1}}  {tol_str}")
    print()
    print(
        "  Override with --threshold PATH=VALUE "
        "(e.g. --threshold evaluation.grouped_balanced_accuracy=0.05)."
    )
    print(
        "  Dynamic placeholders: replace {axis} with session_id/site_id/device_id/"
        "stimulus_id; replace {attribute} with the attribute name (arousal, etc.)."
    )
    print()


def build_parser():
    parser = argparse.ArgumentParser(
        prog="nerveml-compare",
        description=(
            "Compare two NerveML audit reports and flag metric regressions. "
            "Exits 0 when no regressions are found, 1 otherwise."
        ),
    )
    parser.add_argument(
        "baseline",
        nargs="?",
        help="path to baseline audit_report.json",
    )
    parser.add_argument(
        "candidate",
        nargs="?",
        help="path to candidate audit_report.json",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the comparison result to this JSON file (optional)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also exit 1 when the permitted_claim text changes",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help=(
            "print all tracked metric paths with their direction and default "
            "tolerance, then exit 0 (no file arguments needed)"
        ),
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help=(
            "override the default tolerance for a metric path, e.g. "
            "--threshold evaluation.grouped_balanced_accuracy=0.05. "
            "Can be repeated. Use --list-metrics to see available paths."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print progress to stderr as each metric category is evaluated",
    )
    return parser


def _load(path_str):
    path = Path(path_str)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"error: could not parse {path}: {exc}", file=sys.stderr)
        return None


def _fmt_val(v):
    return f"{v:.4f}" if v is not None else "n/a"


def _fmt_delta(d):
    return f"{d:+.4f}" if d is not None else "n/a"


_STATUS_LABELS = {
    "regression": "REGRESSION",
    "improvement": "improved",
    "stable": "stable",
    "unavailable": "n/a",
}


def _status_label(status):
    return _STATUS_LABELS.get(status, status)


def _print_result(result):
    """Print an aligned Metric | Baseline | Candidate | Δ | Status table."""
    status_word = "PASSED" if result.passed else "FAILED"
    print()
    print(f"NerveML Regression Check — {status_word}")
    print()

    # Regressions first, then improvements, stable, unavailable.
    ordered = (
        result.regressions + result.improvements + result.stable + result.unavailable
    )

    if not ordered:
        print("  (no metrics to compare)")
        print()
        return

    headers = ("Metric", "Baseline", "Candidate", "Δ", "Status")

    rows = []
    for m in ordered:
        rows.append((
            m.metric,
            _fmt_val(m.baseline),
            _fmt_val(m.candidate),
            _fmt_delta(m.delta),
            _status_label(m.status),
            m.note,
        ))

    # Column widths driven by the widest value in each column (header or data).
    col_w = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[:5]):
            col_w[i] = max(col_w[i], len(cell))

    # Right-align numeric columns (1-3), left-align label columns (0, 4).
    def _fmt_row(cells):
        m, b, c, d, s = cells[:5]
        return (
            f"  {m:<{col_w[0]}}  "
            f"{b:>{col_w[1]}}  "
            f"{c:>{col_w[2]}}  "
            f"{d:>{col_w[3]}}  "
            f"{s}"
        )

    header_line = _fmt_row(headers)
    sep = "  " + "─" * (len(header_line) - 2)
    print(header_line)
    print(sep)

    note_indent = " " * (col_w[0] + col_w[1] + col_w[2] + col_w[3] + 12)
    for row in rows:
        print(_fmt_row(row))
        # Append the regression note on the next indented line.
        if row[4] == "REGRESSION" and row[5]:
            print(f"{note_indent}^ {row[5]}")

    print()

    counts = (
        f"Regressions: {len(result.regressions)}  "
        f"Improvements: {len(result.improvements)}  "
        f"Stable: {len(result.stable)}  "
        f"Unavailable: {len(result.unavailable)}"
    )
    print(f"  {counts}")
    print()

    if result.permitted_claim_changed:
        print("  PERMITTED CLAIM CHANGED")
        print(f"  baseline:  {result.baseline_claim}")
        print(f"  candidate: {result.candidate_claim}")
        print()


def _parse_thresholds(pairs):
    """Parse a list of 'PATH=VALUE' strings into a {path: float} dict.

    Returns (thresholds_dict, error_message).  error_message is None on success.
    """
    thresholds = {}
    for pair in pairs:
        if "=" not in pair:
            return None, f"--threshold must be PATH=VALUE, got: {pair!r}"
        path, _, value_str = pair.partition("=")
        path = path.strip()
        value_str = value_str.strip()
        try:
            thresholds[path] = float(value_str)
        except ValueError:
            return None, f"--threshold VALUE must be a number, got: {value_str!r}"
    return thresholds, None


_STAGE_LABELS = {
    "aggregate": "Aggregate evaluation metrics",
    "multi_axis": "Per-axis validation metrics",
    "artifact": "Artifact-baseline metrics",
    "secondary_probes": "Secondary sensitive-attribute probes",
    "claim": "Permitted-claim text",
}


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.list_metrics:
        _list_metrics()
        return 0

    if not args.baseline or not args.candidate:
        print("error: baseline and candidate paths are required", file=sys.stderr)
        build_parser().print_usage(sys.stderr)
        return 2

    thresholds, err = _parse_thresholds(args.threshold)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    verbose = getattr(args, "verbose", False)

    if verbose:
        print(f"  [nerveml-compare] Loading baseline from {args.baseline} …", file=sys.stderr, flush=True)
    baseline = _load(args.baseline)
    if baseline is None:
        return 2

    if verbose:
        print(f"  [nerveml-compare] Loading candidate from {args.candidate} …", file=sys.stderr, flush=True)
    candidate = _load(args.candidate)
    if candidate is None:
        return 2

    def _on_stage(key):
        label = _STAGE_LABELS.get(key, key)
        print(f"  [nerveml-compare] {label} …", file=sys.stderr, flush=True)

    result = compare_reports(
        baseline, candidate,
        thresholds=thresholds,
        on_stage=_on_stage if verbose else None,
    )
    _print_result(result)

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  wrote {out_path}")
        print()

    failed = not result.passed
    if args.strict and result.permitted_claim_changed:
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
