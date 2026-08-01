"""nerveml-decode CLI: parser and report formatting, without touching the network.

The real decode path downloads EEG and is covered by the slow integration test.
Here we test argument parsing and that the printed / JSON report is well formed
from a synthetic result, plus the band-power baseline helper on a tiny frame.
"""

import json

import numpy as np
import pandas as pd

from nerveml.decode_cli import (
    _print_report,
    baseline_grouped,
    build_parser,
    main,
)


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.subjects == 20
    assert args.components == 6
    assert args.folds == 5
    assert args.baseline_csv is None


def test_parser_accepts_options():
    args = build_parser().parse_args(
        ["--subjects", "5", "--components", "8", "--baseline-csv", "x.csv", "--out", "r.json"]
    )
    assert args.subjects == 5
    assert args.components == 8
    assert args.baseline_csv == "x.csv"
    assert args.out == "r.json"


def test_print_report_with_baseline(capsys):
    report = {
        "decoder": {
            "decoder": "bandpass+csp+lda",
            "n_trials": 900, "n_channels": 64, "n_subjects": 20,
            "trial_random_roc_auc": 0.55, "trial_random_balanced_accuracy": 0.53,
            "grouped_roc_auc": 0.62, "grouped_balanced_accuracy": 0.58,
            "grouped_roc_auc_std": 0.03, "generalization_gap": -0.05,
        },
        "baseline": {"model": "band_power+logistic_regression",
                     "grouped_roc_auc": 0.555, "grouped_balanced_accuracy": 0.524},
        "auc_lift": 0.065,
    }
    _print_report(report)
    out = capsys.readouterr().out
    assert "CSP MOTOR-IMAGERY DECODER" in out
    assert "0.620" in out          # optimised held-out AUC
    assert "+0.065" in out         # signed lift
    assert "re-identification" in out  # the bounded reading is present


def test_print_report_without_baseline(capsys):
    report = {
        "decoder": {
            "decoder": "bandpass+csp+lda",
            "n_trials": 100, "n_channels": 8, "n_subjects": 5,
            "trial_random_roc_auc": 0.9, "trial_random_balanced_accuracy": 0.88,
            "grouped_roc_auc": 0.85, "grouped_balanced_accuracy": 0.83,
            "grouped_roc_auc_std": 0.04, "generalization_gap": 0.05,
        }
    }
    _print_report(report)
    out = capsys.readouterr().out
    assert "subject-held-out" in out
    assert "BASELINE" not in out  # no baseline section without one


def test_baseline_grouped_on_small_frame(tmp_path):
    """Band-power baseline helper returns bounded scores on a tiny feature CSV."""
    rng = np.random.default_rng(0)
    rows = []
    for subj in range(6):
        for trial in range(20):
            label = trial % 2
            rows.append({
                "subject_id": f"S{subj:02d}",
                "trial_id": f"S{subj:02d}_{trial:03d}",
                "target_label": label,
                "C3_alpha": rng.standard_normal() + label,
                "C4_beta": rng.standard_normal(),
            })
    csv = tmp_path / "feats.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    base = baseline_grouped(csv, n_splits=3, seed=0)
    assert base["model"] == "band_power+logistic_regression"
    assert 0.0 <= base["grouped_roc_auc"] <= 1.0
    assert 0.0 <= base["grouped_balanced_accuracy"] <= 1.0


def test_main_reports_load_failure(monkeypatch, capsys):
    """A network/MNE failure exits 1 with a message, never a traceback."""
    import nerveml.eegbci as eegbci

    def _boom(*a, **k):
        raise RuntimeError("no network")

    monkeypatch.setattr(eegbci, "load_eegbci_epochs", _boom)
    code = main(["--subjects", "3"])
    assert code == 1
    assert "could not load EEG epochs" in capsys.readouterr().err
