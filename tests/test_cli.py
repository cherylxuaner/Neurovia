"""The CLI is a thin wrapper, so these tests only check the wiring."""

import json

import pytest

from nerveml.cli import build_parser, main
from nerveml.scan import STAGES


def test_defaults_match_the_documented_demo_scan():
    args = build_parser().parse_args([])

    assert args.dataset == "subject_leakage"
    assert args.model == "random_forest"
    assert args.folds == 5
    assert args.permutations == 100
    assert args.seed == 0


def test_flags_are_parsed():
    args = build_parser().parse_args(
        ["--dataset", "true_signal", "--model", "logistic_regression",
         "--folds", "4", "--permutations", "20", "--seed", "7"]
    )

    assert args.dataset == "true_signal"
    assert args.model == "logistic_regression"
    assert args.folds == 4
    assert args.permutations == 20
    assert args.seed == 7


def test_main_writes_a_report_and_exits_cleanly(tmp_path, capsys):
    exit_code = main(
        ["--dataset", "subject_leakage", "--model", "logistic_regression",
         "--folds", "4", "--permutations", "10", "--subjects", "8",
         "--trials", "20", "--out", str(tmp_path)]
    )

    assert exit_code == 0
    report = json.loads((tmp_path / "audit_report.json").read_text(encoding="utf-8"))
    assert report["evaluation"]["generalization_gap"] > 0

    # The headline numbers must reach the terminal, not just the file.
    printed = capsys.readouterr().out
    assert "generalization gap" in printed.lower()


def test_main_writes_the_evidence_figures(tmp_path):
    main(
        ["--model", "logistic_regression", "--folds", "4", "--permutations", "10",
         "--subjects", "8", "--trials", "20", "--out", str(tmp_path)]
    )

    for name in ("null_distribution", "fold_scores", "top_features"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0


def test_no_plots_skips_the_figures(tmp_path):
    main(
        ["--model", "logistic_regression", "--folds", "4", "--permutations", "10",
         "--subjects", "8", "--trials", "20", "--out", str(tmp_path), "--no-plots"]
    )

    assert (tmp_path / "audit_report.json").exists()
    assert not (tmp_path / "fold_scores.png").exists()


def test_main_reports_a_bad_dataset_without_a_traceback(tmp_path, capsys):
    exit_code = main(["--dataset", "nope", "--out", str(tmp_path)])

    assert exit_code == 1
    assert "unknown dataset" in capsys.readouterr().err.lower()


# ── --verbose flag ───────────────────────────────────────────────────────────

def test_verbose_defaults_to_false():
    args = build_parser().parse_args([])
    assert args.verbose is False


def test_verbose_flag_parsed():
    args = build_parser().parse_args(["--verbose"])
    assert args.verbose is True


_SMALL_SCAN = [
    "--dataset", "subject_leakage",
    "--model", "logistic_regression",
    "--folds", "3",
    "--permutations", "5",
    "--subjects", "6",
    "--trials", "12",
    "--no-pdf",
    "--no-plots",
]


def test_verbose_prints_stage_names_to_stderr(tmp_path, capsys):
    exit_code = main(_SMALL_SCAN + ["--verbose", "--out", str(tmp_path)])

    assert exit_code == 0
    err = capsys.readouterr().err
    # Every stage key that ran should appear as its human-readable label.
    for key in ("validate", "trial_random", "grouped", "permutation"):
        assert STAGES[key].lower() in err.lower(), (
            f"expected stage '{key}' label in stderr but got:\n{err}"
        )


def test_verbose_prefixes_each_line_with_nerveml_tag(tmp_path, capsys):
    main(_SMALL_SCAN + ["--verbose", "--out", str(tmp_path)])

    err = capsys.readouterr().err
    stage_lines = [ln for ln in err.splitlines() if "[nerveml]" in ln]
    assert len(stage_lines) >= 4, (
        f"expected at least 4 [nerveml] progress lines, got {len(stage_lines)}"
    )
    for line in stage_lines:
        assert line.strip().startswith("[nerveml]"), (
            f"progress line has unexpected format: {line!r}"
        )


def test_without_verbose_no_stage_output_on_stderr(tmp_path, capsys):
    main(_SMALL_SCAN + ["--out", str(tmp_path)])

    err = capsys.readouterr().err
    assert "[nerveml]" not in err


def test_verbose_prints_fold_progress_lines_to_stderr(tmp_path, capsys):
    # _SMALL_SCAN uses --folds 3; two protocols × 3 folds = 6 fold lines
    main(_SMALL_SCAN + ["--verbose", "--out", str(tmp_path)])

    err = capsys.readouterr().err
    fold_lines = [ln for ln in err.splitlines() if "fold" in ln and "[nerveml]" in ln]
    assert len(fold_lines) == 6, (
        f"expected 6 fold progress lines (2 protocols × 3 folds), got {len(fold_lines)}"
    )


def test_verbose_fold_lines_show_running_fraction(tmp_path, capsys):
    main(_SMALL_SCAN + ["--verbose", "--out", str(tmp_path)])

    err = capsys.readouterr().err
    fold_lines = [ln for ln in err.splitlines() if "fold" in ln and "[nerveml]" in ln]
    # Each line should say e.g. "fold 1/3", "fold 2/3", "fold 3/3"
    for line in fold_lines:
        assert "/3" in line, f"expected 'fold X/3' in: {line!r}"


def test_verbose_fold_lines_include_bacc(tmp_path, capsys):
    main(_SMALL_SCAN + ["--verbose", "--out", str(tmp_path)])

    err = capsys.readouterr().err
    fold_lines = [ln for ln in err.splitlines() if "fold" in ln and "[nerveml]" in ln]
    for line in fold_lines:
        assert "bacc" in line, f"expected 'bacc' in fold progress line: {line!r}"


def test_without_verbose_no_fold_lines_on_stderr(tmp_path, capsys):
    main(_SMALL_SCAN + ["--out", str(tmp_path)])

    err = capsys.readouterr().err
    fold_lines = [ln for ln in err.splitlines() if "fold" in ln]
    assert fold_lines == [], f"expected no fold lines without --verbose, got: {fold_lines}"
