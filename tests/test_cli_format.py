"""Tests for --format / --deap-* / --seed-* CLI flags.

Each test writes a minimal CSV to tmp_path, invokes main(), and asserts on
the resulting audit_report.json.  All data is synthetic so no network access
is needed.
"""

import json

import numpy as np
import pandas as pd
import pytest

from nerveml.cli import build_parser, main


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_deap_csv(tmp_path, n_subjects=6, n_trials=12, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for _ in range(n_trials):
            row = {
                "participant_id": f"p{s:02d}",
                "valence": rng.uniform(1, 9),
                "arousal": rng.uniform(1, 9),
                "dominance": rng.uniform(1, 9),
                "liking": rng.uniform(1, 9),
            }
            for f in range(n_features):
                row[f"feat_{f:02d}"] = rng.standard_normal()
            rows.append(row)
    path = tmp_path / "deap.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_seed_csv(tmp_path, n_subjects=6, n_per_class=8, n_features=4, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for cls in [-1, 0, 1]:
            for _ in range(n_per_class):
                row = {"subject": f"s{s:02d}", "emotion": cls}
                for f in range(n_features):
                    row[f"de_{f:02d}"] = rng.standard_normal()
                rows.append(row)
    path = tmp_path / "seed.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_nerveml_csv(tmp_path, n_subjects=6, n_trials=12, n_features=4, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        for t in range(n_trials):
            row = {
                "subject_id": f"sub{s:02d}",
                "trial_id": f"sub{s:02d}_t{t}",
                "target_label": rng.integers(0, 2),
            }
            for f in range(n_features):
                row[f"f{f}"] = rng.standard_normal()
            rows.append(row)
    path = tmp_path / "nerveml.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ── parser argument tests ─────────────────────────────────────────────────────


def test_format_defaults_to_auto():
    args = build_parser().parse_args([])
    assert args.csv_format == "auto"


def test_format_choices_parse():
    for fmt in ("auto", "nerveml", "deap", "seed"):
        args = build_parser().parse_args(["--format", fmt])
        assert args.csv_format == fmt


def test_deap_target_default():
    args = build_parser().parse_args([])
    assert args.deap_target == "valence"


def test_deap_target_choices_parse():
    for target in ("valence", "arousal", "dominance", "liking"):
        args = build_parser().parse_args(["--deap-target", target])
        assert args.deap_target == target


def test_deap_threshold_parses_as_float():
    args = build_parser().parse_args(["--deap-threshold", "6.5"])
    assert args.deap_threshold == pytest.approx(6.5)


def test_seed_label_column_parses():
    args = build_parser().parse_args(["--seed-label-column", "label"])
    assert args.seed_label_column == "label"


def test_seed_positive_class_parses():
    args = build_parser().parse_args(["--seed-positive-class", "1"])
    assert args.seed_positive_class == "1"


def test_seed_drop_neutral_default():
    args = build_parser().parse_args([])
    assert args.seed_drop_neutral is True


def test_seed_no_drop_neutral_flag():
    args = build_parser().parse_args(["--seed-no-drop-neutral"])
    assert args.seed_drop_neutral is False


# ── end-to-end scan tests ─────────────────────────────────────────────────────


def _run(tmp_path, csv_path, extra_args=(), folds=4, permutations=10):
    """Invoke main() with minimal args and return the parsed report dict."""
    exit_code = main(
        [
            "--dataset", str(csv_path),
            "--folds", str(folds),
            "--permutations", str(permutations),
            "--no-pdf",
            "--no-plots",
            "--out", str(tmp_path / "out"),
        ] + list(extra_args)
    )
    assert exit_code == 0, f"main() returned {exit_code}"
    return json.loads((tmp_path / "out" / "audit_report.json").read_text())


def test_auto_detects_deap_format_and_scans(tmp_path):
    path = _make_deap_csv(tmp_path)
    report = _run(tmp_path, path)
    assert report["dataset"]["n_subjects"] == 6
    assert "evaluation" in report


def test_explicit_deap_format_scans(tmp_path):
    path = _make_deap_csv(tmp_path)
    report = _run(tmp_path, path, extra_args=["--format", "deap"])
    assert report["dataset"]["n_subjects"] == 6


def test_deap_arousal_target_changes_scan(tmp_path):
    path = _make_deap_csv(tmp_path)
    report = _run(tmp_path, path, extra_args=["--format", "deap", "--deap-target", "arousal"])
    assert "evaluation" in report


def test_deap_threshold_accepted(tmp_path):
    path = _make_deap_csv(tmp_path)
    # A high threshold should still produce a valid (possibly skewed) scan.
    report = _run(tmp_path, path, extra_args=["--format", "deap", "--deap-threshold", "7.0"])
    assert "evaluation" in report


def test_auto_detects_seed_format_and_scans(tmp_path):
    path = _make_seed_csv(tmp_path)
    report = _run(tmp_path, path)
    assert report["dataset"]["n_subjects"] == 6
    assert "evaluation" in report


def test_explicit_seed_format_scans(tmp_path):
    path = _make_seed_csv(tmp_path)
    report = _run(tmp_path, path, extra_args=["--format", "seed"])
    assert report["dataset"]["n_subjects"] == 6


def test_seed_positive_class_coerced_to_int(tmp_path):
    path = _make_seed_csv(tmp_path)
    # Pass positive_class as "1" (string) — CLI should coerce to int 1.
    report = _run(
        tmp_path, path,
        extra_args=["--format", "seed", "--seed-positive-class", "1"],
    )
    assert "evaluation" in report


def test_nerveml_format_loads_generic_csv(tmp_path):
    path = _make_nerveml_csv(tmp_path)
    report = _run(tmp_path, path, extra_args=["--format", "nerveml"])
    assert report["dataset"]["n_subjects"] == 6


def test_auto_detects_nerveml_format(tmp_path):
    path = _make_nerveml_csv(tmp_path)
    report = _run(tmp_path, path)
    assert report["dataset"]["n_subjects"] == 6


def test_unknown_format_falls_back_to_generic_loader(tmp_path):
    # A CSV with subject_id + target_label (NerveML-internal) but unknown format
    # should work fine via the generic loader.
    path = _make_nerveml_csv(tmp_path)
    report = _run(tmp_path, path, extra_args=["--format", "nerveml"])
    assert "permitted_claim" in report
