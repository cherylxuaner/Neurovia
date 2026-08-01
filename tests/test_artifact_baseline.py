"""Artifact-confound baseline for the classification probe.

Checks that EOG/EMG proxy feature selection is correct, that the artifact
share computation is bounded, and that run_artifact_baseline integrates with
a realistic band-power feature table.
"""

import numpy as np
import pandas as pd
import pytest

from nerveml.artifact_baseline import (
    ArtifactBaseline,
    ProxyResult,
    eog_proxy_features,
    emg_proxy_features,
    run_artifact_baseline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_band_table(n_subjects=10, n_trials=20, seed=0):
    """Band-power feature table with frontal, lateral-frontal and motor channels."""
    channels = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F8", "C3", "C4", "Cz"]
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        X = rng.normal(0, 1, size=(n_trials, len(channels) * len(bands)))
        # plant a weak label signal on alpha across all channels
        label = rng.integers(0, 2, size=n_trials)
        X[:, 2] += label * 0.5  # push Fp1_alpha
        frame = pd.DataFrame(
            X,
            columns=[f"{c}_{b}" for c in channels for b in bands],
        )
        frame.insert(0, "subject_id", f"s{s:02d}")
        frame.insert(1, "trial_id", [f"s{s:02d}_t{t:03d}" for t in range(n_trials)])
        frame.insert(2, "target_label", label)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _feature_cols(df):
    return [c for c in df.columns
            if c not in ("subject_id", "trial_id", "target_label")]


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

class TestEogProxyFeatures:
    def test_selects_fp_and_af_channels(self):
        cols = [
            "Fp1_alpha", "Fp2_beta", "AF3_gamma",
            "C3_alpha", "C4_delta", "Cz_theta",
        ]
        result = eog_proxy_features(cols)
        assert set(result) == {"Fp1_alpha", "Fp2_beta", "AF3_gamma"}

    def test_case_insensitive(self):
        cols = ["fp1_alpha", "FP2_beta", "aF3_gamma", "C3_alpha"]
        result = eog_proxy_features(cols)
        assert set(result) == {"fp1_alpha", "FP2_beta", "aF3_gamma"}

    def test_no_frontal_channels_returns_empty(self):
        cols = ["C3_alpha", "C4_beta", "Cz_gamma"]
        assert eog_proxy_features(cols) == []

    def test_plain_features_without_underscore_return_empty(self):
        # Synth-style features f00, f01 have no underscore → not band-power
        cols = ["f00", "f01", "f02"]
        assert eog_proxy_features(cols) == []

    def test_fc_prefix_not_included(self):
        # Fc (frontocentral) is not an EOG channel
        cols = ["Fc5_alpha", "Fc3_beta", "Fp1_gamma"]
        result = eog_proxy_features(cols)
        assert set(result) == {"Fp1_gamma"}

    def test_selects_f7_f8_lateral_eog_channels(self):
        # F7/F8 are standard lateral-frontal EOG channels for horizontal saccades
        cols = ["F7_alpha", "F8_beta", "F3_alpha", "F4_beta", "Fz_gamma"]
        result = eog_proxy_features(cols)
        assert set(result) == {"F7_alpha", "F8_beta"}

    def test_selects_ft7_ft8_lateral_channels(self):
        # FT7/FT8 are extended-10-20 fronto-temporal lateral EOG channels
        cols = ["FT7_theta", "FT8_alpha", "FC5_alpha", "T7_beta"]
        result = eog_proxy_features(cols)
        assert set(result) == {"FT7_theta", "FT8_alpha"}

    def test_f3_f4_fz_excluded_not_lateral_eog(self):
        # F3, F4, Fz are central frontal — NOT lateral EOG channels
        cols = ["F3_alpha", "F4_beta", "Fz_gamma", "F5_theta", "F6_delta"]
        assert eog_proxy_features(cols) == []

    def test_case_insensitive_lateral_channels(self):
        cols = ["f7_alpha", "F8_BETA", "ft7_theta", "FT8_delta"]
        result = eog_proxy_features(cols)
        assert set(result) == {"f7_alpha", "F8_BETA", "ft7_theta", "FT8_delta"}

    def test_combined_vertical_and_lateral_channels(self):
        cols = ["Fp1_alpha", "AF3_beta", "F7_theta", "F8_delta", "C3_alpha"]
        result = eog_proxy_features(cols)
        assert set(result) == {"Fp1_alpha", "AF3_beta", "F7_theta", "F8_delta"}


class TestEmgProxyFeatures:
    def test_selects_gamma_band_only(self):
        cols = [
            "C3_delta", "C3_theta", "C3_alpha", "C3_beta", "C3_gamma",
            "Fp1_gamma", "Cz_gamma",
        ]
        result = emg_proxy_features(cols)
        assert set(result) == {"C3_gamma", "Fp1_gamma", "Cz_gamma"}

    def test_case_insensitive(self):
        cols = ["C3_GAMMA", "C4_gamma", "Cz_Alpha"]
        result = emg_proxy_features(cols)
        assert set(result) == {"C3_GAMMA", "C4_gamma"}

    def test_no_gamma_returns_empty(self):
        cols = ["C3_alpha", "C4_beta", "Cz_delta"]
        assert emg_proxy_features(cols) == []

    def test_plain_features_without_underscore_return_empty(self):
        cols = ["f00", "f01", "f02"]
        assert emg_proxy_features(cols) == []


# ---------------------------------------------------------------------------
# ArtifactBaseline
# ---------------------------------------------------------------------------

class TestArtifactBaselineArtifactShare:
    def _proxy(self, bacc):
        return ProxyResult(
            name="eog", n_features=5, balanced_accuracy=bacc, roc_auc=0.6,
            note="test",
        )

    def test_full_above_chance_share_between_zero_and_one(self):
        ab = ArtifactBaseline(full_balanced_accuracy=0.8, eog=self._proxy(0.65), emg=None)
        share = ab.eog_artifact_share
        assert share is not None
        assert 0.0 <= share <= 1.0

    def test_proxy_at_chance_gives_zero_share(self):
        ab = ArtifactBaseline(full_balanced_accuracy=0.7, eog=self._proxy(0.5), emg=None)
        assert ab.eog_artifact_share == 0.0

    def test_proxy_equals_full_gives_one_share(self):
        ab = ArtifactBaseline(full_balanced_accuracy=0.7, eog=self._proxy(0.7), emg=None)
        assert ab.eog_artifact_share == pytest.approx(1.0, abs=0.01)

    def test_proxy_above_full_clamped_to_one(self):
        ab = ArtifactBaseline(full_balanced_accuracy=0.6, eog=self._proxy(0.9), emg=None)
        assert ab.eog_artifact_share == pytest.approx(1.0, abs=0.01)

    def test_full_at_chance_returns_none(self):
        ab = ArtifactBaseline(full_balanced_accuracy=0.5, eog=self._proxy(0.6), emg=None)
        assert ab.eog_artifact_share is None

    def test_none_proxy_gives_none_share(self):
        ab = ArtifactBaseline(full_balanced_accuracy=0.7, eog=None, emg=None)
        assert ab.eog_artifact_share is None
        assert ab.emg_artifact_share is None


class TestArtifactBaselineToDict:
    def test_keys_present(self):
        proxy = ProxyResult("eog", 5, 0.6, 0.62, "note")
        ab = ArtifactBaseline(full_balanced_accuracy=0.75, eog=proxy, emg=None)
        d = ab.to_dict()
        assert "full_balanced_accuracy" in d
        assert "eog_proxy" in d
        assert "emg_proxy" in d
        assert "artifact_share_eog" in d
        assert "artifact_share_emg" in d
        assert "interpretation" in d

    def test_null_proxy_serialised_as_none(self):
        ab = ArtifactBaseline(full_balanced_accuracy=0.7, eog=None, emg=None)
        d = ab.to_dict()
        assert d["eog_proxy"] is None
        assert d["emg_proxy"] is None

    def test_interpretation_contains_bounded_claim_language(self):
        proxy = ProxyResult("eog", 5, 0.65, 0.63, "note")
        ab = ArtifactBaseline(full_balanced_accuracy=0.8, eog=proxy, emg=None)
        interp = ab.to_dict()["interpretation"]
        # must not overclaim — must say "consistent with", "upper bound"
        assert "consistent with" in interp.lower()
        assert "upper bound" in interp.lower()

    def test_to_dict_is_json_serialisable(self):
        import json
        proxy = ProxyResult("emg", 7, 0.58, 0.61, "note")
        ab = ArtifactBaseline(full_balanced_accuracy=0.72, eog=None, emg=proxy)
        json.dumps(ab.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# run_artifact_baseline
# ---------------------------------------------------------------------------

class TestRunArtifactBaseline:
    def test_returns_baseline_when_frontal_and_gamma_present(self):
        df = _make_band_table()
        feats = _feature_cols(df)
        result = run_artifact_baseline(
            df, feats, group_column="subject_id",
            full_bacc=0.65, n_splits=5, seed=0,
        )
        assert result is not None
        assert result.eog is not None  # Fp1, Fp2, AF3, AF4 are frontal
        assert result.emg is not None  # gamma band present

    def test_eog_proxy_feature_count_matches_frontal(self):
        df = _make_band_table()
        feats = _feature_cols(df)
        result = run_artifact_baseline(
            df, feats, "subject_id", full_bacc=0.65, n_splits=5,
        )
        # 6 EOG channels (Fp1, Fp2, AF3, AF4 vertical + F7, F8 lateral) × 5 bands = 30
        assert result.eog.n_features == 30

    def test_emg_proxy_feature_count_matches_gamma(self):
        df = _make_band_table()
        feats = _feature_cols(df)
        result = run_artifact_baseline(
            df, feats, "subject_id", full_bacc=0.65, n_splits=5,
        )
        # 9 channels × 1 gamma band = 9
        assert result.emg.n_features == 9

    def test_returns_none_when_no_frontal_no_gamma(self):
        df = _make_band_table()
        # Only keep theta/alpha bands of C3/C4 — no frontal, no gamma
        cols = [c for c in _feature_cols(df)
                if c.startswith("C") and c.endswith(("_theta", "_alpha"))]
        assert cols, "test helper produced no columns"
        result = run_artifact_baseline(
            df, cols, "subject_id", full_bacc=0.65, n_splits=5,
        )
        assert result is None

    def test_balanced_accuracy_is_between_zero_and_one(self):
        df = _make_band_table()
        feats = _feature_cols(df)
        result = run_artifact_baseline(
            df, feats, "subject_id", full_bacc=0.65, n_splits=5,
        )
        for proxy in (result.eog, result.emg):
            if proxy is not None:
                assert 0.0 <= proxy.balanced_accuracy <= 1.0
                assert 0.0 <= proxy.roc_auc <= 1.0

    def test_artifact_share_is_bounded(self):
        df = _make_band_table()
        feats = _feature_cols(df)
        result = run_artifact_baseline(
            df, feats, "subject_id", full_bacc=0.65, n_splits=5,
        )
        for share in (result.eog_artifact_share, result.emg_artifact_share):
            if share is not None:
                assert 0.0 <= share <= 1.0

    def test_to_dict_round_trips_through_json(self):
        import json
        df = _make_band_table()
        feats = _feature_cols(df)
        result = run_artifact_baseline(
            df, feats, "subject_id", full_bacc=0.65, n_splits=5,
        )
        assert result is not None
        json.dumps(result.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# Integration: artifact_baseline appears in the scan report for EEGBCI-style data
# ---------------------------------------------------------------------------

class TestScanReportIntegration:
    def test_report_has_artifact_baseline_key_for_band_power_data(self):
        """Scan on band-power data must include 'artifact_baseline' in the report."""
        from nerveml.scan import run_scan
        import tempfile, csv
        from pathlib import Path

        df = _make_band_table(n_subjects=10, n_trials=20, seed=42)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w",
                                         delete=False, newline="") as f:
            df.to_csv(f, index=False)
            path = f.name

        report = run_scan(path, n_splits=5, seed=0)
        assert "artifact_baseline" in report
        ab = report["artifact_baseline"]
        # With frontal channels present, at least one proxy should be non-null
        assert ab is not None
        assert ab.get("eog_proxy") is not None or ab.get("emg_proxy") is not None

    def test_report_artifact_baseline_none_for_non_band_power(self):
        """Scan on synth data (f00, f01, ...) must have artifact_baseline=None."""
        from nerveml.scan import run_scan
        report = run_scan("true_signal", n_subjects=10, n_trials=20, n_splits=5, seed=0)
        # Synth features are not band-power, so artifact_baseline is None
        assert report.get("artifact_baseline") is None
