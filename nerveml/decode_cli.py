"""Command-line entry point: nerveml-decode [options]

Benchmarks the CSP motor-imagery decoder on PhysioNet EEG and, optionally,
prints it beside the band-power baseline. Separate from `nerveml` (the audit)
because CSP needs the raw epochs, not the reduced feature table, so it cannot
ride the same feature-CSV path.

The point it makes for a reader: an *optimised* task decoder still lands far
below the re-identification score the audit reports. Task decoding being weak is
not an artefact of a weak decoder — it survives a good one.
"""

import argparse
import json
import sys
from pathlib import Path

_METADATA_COLUMNS = ("subject_id", "session_id", "trial_id", "target_label")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="nerveml-decode",
        description="Benchmark the CSP motor-imagery decoder (raw epochs) and "
        "compare it against the band-power baseline.",
    )
    parser.add_argument(
        "--subjects", type=int, default=20,
        help="how many PhysioNet subjects to download/epoch (default: 20)",
    )
    parser.add_argument(
        "--components", type=int, default=6,
        help="number of CSP spatial filters (default: 6)",
    )
    parser.add_argument(
        "--method", default="csp", choices=["csp", "fbcsp"],
        help="single-band CSP (default) or filterbank CSP across mu+beta sub-bands",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--baseline-csv", default=None, metavar="PATH",
        help="feature CSV to score the band-power+logistic baseline for comparison "
        "(e.g. sample_data/eegbci_20.csv)",
    )
    parser.add_argument(
        "--out", default=None, metavar="PATH",
        help="write the comparison as JSON to this path",
    )
    return parser


def baseline_grouped(csv_path, n_splits=5, seed=0):
    """Subject-held-out band-power+logistic scores from a feature CSV.

    The honest reference the CSP decoder is measured against: the same
    subject-held-out protocol the audit runs, on the reduced feature table.
    """
    import pandas as pd

    from nerveml.models import build_model
    from nerveml.validation import evaluate_grouped

    df = pd.read_csv(csv_path)
    feats = [c for c in df.columns if c not in _METADATA_COLUMNS]
    grouped = evaluate_grouped(
        df, feats,
        model=build_model("logistic_regression", seed=seed),
        n_splits=min(n_splits, df["subject_id"].nunique()),
        seed=seed,
    )
    return {
        "model": "band_power+logistic_regression",
        "grouped_roc_auc": round(grouped.roc_auc, 4),
        "grouped_balanced_accuracy": round(grouped.balanced_accuracy, 4),
    }


def _print_report(report):
    csp = report["decoder"]
    print()
    print("  CSP MOTOR-IMAGERY DECODER  (raw epochs, mu+beta band-pass, in-fold)")
    print(f"  epochs / channels / subjects   {csp['n_trials']} / "
          f"{csp['n_channels']} / {csp['n_subjects']}")
    print(f"  trial-random       AUC {csp['trial_random_roc_auc']:.3f}"
          f"   bacc {csp['trial_random_balanced_accuracy']:.3f}")
    print(f"  subject-held-out   AUC {csp['grouped_roc_auc']:.3f}"
          f"   bacc {csp['grouped_balanced_accuracy']:.3f}"
          f"   (SD {csp['grouped_roc_auc_std']:.3f})")

    baseline = report.get("baseline")
    if baseline is not None:
        print()
        print("  vs BAND-POWER BASELINE  (subject-held-out AUC)")
        print(f"  baseline  {baseline['model']:<32s} {baseline['grouped_roc_auc']:.3f}")
        print(f"  optimised csp+lda{'':<24s} {csp['grouped_roc_auc']:.3f}")
        print(f"  lift{'':<37s} {report['auc_lift']:+.3f}")
    print()
    print("  Bounded reading: a stronger decoder lifts task decoding, but the "
          "held-out number stays a task-decoding estimate under this protocol — "
          "not a claim about anyone's mental state, and far below the "
          "re-identification score the audit reports on the same data.")
    print()


def main(argv=None):
    args = build_parser().parse_args(argv)

    from nerveml.csp import decode_benchmark
    from nerveml.eegbci import load_eegbci_epochs

    try:
        epochs = load_eegbci_epochs(args.subjects)
    except Exception as error:  # network / MNE failure — report, don't crash
        print(f"error: could not load EEG epochs: {error}", file=sys.stderr)
        return 1

    result = decode_benchmark(
        epochs.X, epochs.y, epochs.subjects,
        n_splits=args.folds, seed=args.seed,
        n_components=args.components, sampling_rate=epochs.sampling_rate,
        method=args.method,
    )
    report = {"decoder": result.to_dict()}

    if args.baseline_csv:
        try:
            report["baseline"] = baseline_grouped(
                args.baseline_csv, n_splits=args.folds, seed=args.seed
            )
            report["auc_lift"] = round(
                result.grouped_roc_auc - report["baseline"]["grouped_roc_auc"], 4
            )
        except (FileNotFoundError, ValueError, KeyError) as error:
            print(f"warning: baseline comparison skipped: {error}", file=sys.stderr)

    _print_report(report)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
