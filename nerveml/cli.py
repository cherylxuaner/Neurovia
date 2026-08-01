"""Command-line entry point: nerveml [options]

Prints the four headline numbers and writes the machine-readable report.
"""

import argparse
import sys
from pathlib import Path

from nerveml.models import MODEL_KINDS
from nerveml.scan import STAGES, run_scan, write_report
from nerveml.scenarios import SCENARIOS
from nerveml.synth import MODES


def build_parser():
    parser = argparse.ArgumentParser(
        prog="nerveml",
        description="Scan a multi-subject dataset for subject leakage and "
        "unintended sensitive inference.",
    )
    parser.add_argument(
        "--dataset",
        default="subject_leakage",
        help=f"one of {tuple(SCENARIOS) + MODES}, or a path to a feature CSV",
    )
    parser.add_argument("--model", default="random_forest", choices=MODEL_KINDS)
    parser.add_argument(
        "--group-column",
        default="subject_id",
        help="independent unit to hold out: subject_id, session_id, site_id, ...",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--subjects", type=int, default=20, help="synthetic data only")
    parser.add_argument("--trials", type=int, default=40, help="synthetic data only")
    parser.add_argument("--features", type=int, default=16, help="synthetic data only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs")
    parser.add_argument(
        "--no-pdf",
        dest="pdf",
        action="store_false",
        help="skip the PDF risk report",
    )
    parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="skip the evidence figures",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help=(
            "print each pipeline stage to stderr as it starts "
            "(useful when scanning large datasets)"
        ),
    )

    # ── CSV format selection ──────────────────────────────────────────────────
    parser.add_argument(
        "--format",
        dest="csv_format",
        default="auto",
        choices=["auto", "nerveml", "deap", "seed"],
        help=(
            "Format of the input CSV. 'auto' (default) sniffs the column names. "
            "'nerveml' uses the generic loader (requires subject_id and target_label). "
            "'deap' normalises DEAP-convention CSVs. "
            "'seed' normalises SEED-convention CSVs."
        ),
    )
    # DEAP-specific options
    parser.add_argument(
        "--deap-target",
        default="valence",
        choices=["valence", "arousal", "dominance", "liking"],
        help="DEAP rating column to binarise as the classification target (default: valence).",
    )
    parser.add_argument(
        "--deap-threshold",
        type=float,
        default=5.0,
        metavar="THRESH",
        help=(
            "Binarisation cut-off for the DEAP target rating (1-9 scale). "
            "Ratings strictly above this value become class 1 (default: 5.0)."
        ),
    )
    # SEED-specific options
    parser.add_argument(
        "--seed-label-column",
        default=None,
        metavar="COL",
        help=(
            "Column containing SEED emotion labels. "
            "Inferred from common SEED names (emotion, emotion_label, label) if omitted."
        ),
    )
    parser.add_argument(
        "--seed-positive-class",
        default=None,
        metavar="VAL",
        help=(
            "Value to treat as the positive class in a one-vs-rest scheme. "
            "When given, all other label values become class 0 and no rows are dropped. "
            "Accepts an integer or a string matching a label value."
        ),
    )
    parser.add_argument(
        "--seed-no-drop-neutral",
        dest="seed_drop_neutral",
        action="store_false",
        help=(
            "Keep neutral trials instead of dropping them. "
            "Requires --seed-positive-class when the label has 3 classes."
        ),
    )
    return parser


def _print_summary(report):
    evaluation = report["evaluation"]
    risk = report["risk_flags"]

    print()
    print(f"  dataset                     {report['dataset']['name']}")
    print(
        f"  subjects / trials           {report['dataset']['n_subjects']} / "
        f"{report['dataset']['n_trials']}"
    )
    print()
    print("  RESULT CARDS")
    print(
        f"  trial-random  balanced acc  {evaluation['trial_random_balanced_accuracy']:.3f}"
        f"   (AUC {evaluation['trial_random_roc_auc']:.3f})"
    )
    unit = evaluation["grouping_unit"].removesuffix("_id")
    intervals = evaluation["confidence_intervals"]

    def span(name):
        low, high = intervals.get(name, (None, None))
        return "" if None in (low, high) else f"   95% CI [{low:.3f}, {high:.3f}]"

    print(
        f"  {unit + '-held-out':<26s}  {evaluation['grouped_balanced_accuracy']:.3f}"
        f"   (AUC {evaluation['grouped_roc_auc']:.3f}"
        f", SD {evaluation['grouped_balanced_accuracy_std']:.3f})"
    )
    print(f"  {'held-out AUC':<26s}  {evaluation['grouped_roc_auc']:.3f}"
          f"{span('grouped_roc_auc')}")
    print(
        f"  {'within-' + unit + ' (same preds)':<26s}  "
        f"{evaluation['trial_random_within_unit_balanced_accuracy']:.3f}"
    )
    print(
        f"  generalization gap          {evaluation['generalization_gap']:.3f}"
        f"{span('generalization_gap')}"
    )
    worst = evaluation.get("worst_group")
    if worst is not None:
        print(
            f"  worst held-out unit         {worst['group']} at "
            f"{worst['balanced_accuracy']:.3f}"
        )
    print(f"  sensitive inference risk    {risk['sensitive_inference_evidence']}")
    if risk.get("units_needed"):
        print(
            f"  {'units for a verdict':<26s}  about {risk['units_needed']} "
            f"(have {report['dataset']['n_subjects']})"
        )
    print(
        f"  permutation p               {report['permutation_test']['p_value']:.3f}"
        f"   (null mean {report['permutation_test']['null_mean']:.3f})"
    )
    identity = report.get("identity_inference")
    if identity is not None:
        print()
        print("  RE-IDENTIFICATION")
        print(
            f"  held-out records linked     {identity['accuracy']:.3f}"
            f"   ({identity['lift_over_chance']:.1f}x chance of "
            f"{identity['chance']:.3f}, {identity['n_identities']} identities)"
        )
        recall_low, recall_high = identity["recall_ci"]
        if None not in (recall_low, recall_high):
            print(
                f"  mean per-identity recall    "
                f"95% CI [{recall_low:.3f}, {recall_high:.3f}]"
            )
        composition = report.get("fingerprint_composition")
        if composition is not None:
            print(
                f"  amplitude only              "
                f"{composition['amplitude_only']['accuracy']:.3f}"
                f"   (skull, impedance, cap fit)"
            )
            print(
                f"  spectral shape only         "
                f"{composition['spectral_shape_only']['accuracy']:.3f}"
                f"   (amplitude divided out)"
            )
            print(f"  fingerprint carried by      {composition['carried_by']}")

        top = identity["most_identifiable"]
        print(
            f"  most identifiable           {top['group']} at "
            f"{top['recall']:.3f} recall"
        )

    if report["dataset"]["warnings"]:
        print()
        print("  DATA WARNINGS")
        for warning in report["dataset"]["warnings"]:
            print(f"  - {warning['message']}")

    print()
    print("  RISK FLAGS")
    for flag in risk["flags"]:
        print(f"  [{flag['severity']}] {flag['code']}")
        print(f"      {flag['message']}")
        print(f"      rule: {flag['rule']}")

    print()
    print("  PERMITTED CLAIM")
    print(f"  {report['permitted_claim']}")
    print()
    print("  NOT SUPPORTED BY THIS SCAN")
    for claim in report["unsupported_claims"]:
        print(f"  - {claim}")
    print()
    print("  RECOMMENDED NEXT STEPS")
    for item in report["recommendations"]:
        print(f"  - {item}")
    print()


def main(argv=None):
    args = build_parser().parse_args(argv)

    # --seed-positive-class arrives as a string; coerce to int when possible so
    # it matches the integer label values that SEED feature tables typically use.
    seed_positive_class = args.seed_positive_class
    if seed_positive_class is not None:
        try:
            seed_positive_class = int(seed_positive_class)
        except ValueError:
            pass  # keep as string — label column may contain string labels

    def _on_stage(name):
        label = STAGES.get(name, name)
        print(f"  [nerveml] {label} …", file=sys.stderr, flush=True)

    def _on_fold(fold_idx, n_folds, bacc):
        print(
            f"  [nerveml]   fold {fold_idx + 1}/{n_folds}  bacc {bacc:.3f}",
            file=sys.stderr,
            flush=True,
        )

    try:
        report = run_scan(
            dataset=args.dataset,
            n_subjects=args.subjects,
            n_trials=args.trials,
            n_features=args.features,
            model_kind=args.model,
            group_column=args.group_column,
            n_splits=args.folds,
            n_permutations=args.permutations,
            seed=args.seed,
            on_stage=_on_stage if args.verbose else None,
            on_fold=_on_fold if args.verbose else None,
            csv_format=args.csv_format,
            deap_target=args.deap_target,
            deap_threshold=args.deap_threshold,
            seed_label_column=args.seed_label_column,
            seed_positive_class=seed_positive_class,
            seed_drop_neutral=args.seed_drop_neutral,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _print_summary(report)
    paths = write_report(report, args.out)
    print(f"  wrote {paths['json']}")
    print(f"  wrote {paths['csv']}")

    figures = {}
    if args.plots:
        from nerveml.plots import write_figures

        figures = write_figures(report, args.out)
        for path in figures.values():
            print(f"  wrote {path}")

    if args.pdf:
        from nerveml.pdf import write_pdf_report

        print(f"  wrote {write_pdf_report(report, Path(args.out) / 'risk_report.pdf', figures=figures)}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
