"""CSP decoder tests, on synthetic epochs with a known planted signal.

Real EEG cannot tell you whether a decoder works, because you do not know how
much signal the data held. A planted spatial signal does: class 0 puts its
variance in one channel pair, class 1 in another, so CSP has a correct answer to
recover and the tests can assert it reaches it — the same reason synth.py exists
for the feature-table pipeline.
"""

import numpy as np
import pytest

from nerveml.csp import (
    CSP,
    DecodeResult,
    FilterbankCSP,
    build_csp_decoder,
    build_fbcsp_decoder,
    decode_benchmark,
)


def _planted_epochs(n_subjects=6, per_class=12, n_channels=8, n_samples=200, seed=0):
    """Epochs where class variance lives in different channels per class.

    Class 0 carries high variance in channels {0, 1}; class 1 in channels
    {6, 7}. The pattern is shared across subjects (so it generalises to a
    held-out subject) but each trial is an independent noise realisation, so a
    model cannot win by memorising trials.
    """
    rng = np.random.default_rng(seed)
    X, y, subjects = [], [], []
    # Class 0's variance lives in the first two channels, class 1's in the last
    # two, so the hot channels stay in range for any channel count.
    hot0 = [0, 1]
    hot1 = [n_channels - 2, n_channels - 1]
    for s in range(n_subjects):
        for label in (0, 1):
            for _ in range(per_class):
                trial = rng.standard_normal((n_channels, n_samples)) * 0.5
                trial[hot0 if label == 0 else hot1] *= 4.0
                X.append(trial)
                y.append(label)
                subjects.append(f"S{s:02d}")
    return np.asarray(X), np.asarray(y), np.asarray(subjects)


def _noise_epochs(n_subjects=6, per_subject=24, n_channels=8, n_samples=200, seed=1):
    """Pure noise with random labels — no signal, so honest scores sit at chance."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_subjects * per_subject, n_channels, n_samples))
    y = rng.integers(0, 2, size=n_subjects * per_subject)
    subjects = np.repeat([f"S{s:02d}" for s in range(n_subjects)], per_subject)
    return X, y, subjects


# --- the transform itself -------------------------------------------------

def test_csp_transform_shape():
    X, y, _ = _planted_epochs()
    csp = CSP(n_components=4).fit(X, y)
    out = csp.transform(X)
    assert out.shape == (X.shape[0], 4)
    assert np.isfinite(out).all()


def test_csp_caps_components_at_channel_count():
    X, y, _ = _planted_epochs(n_channels=6)
    csp = CSP(n_components=20).fit(X, y)
    assert csp.filters_.shape[1] == 6


def test_csp_rejects_non_binary():
    X = np.random.default_rng(0).standard_normal((10, 8, 50))
    y = np.array([0, 1, 2] + [0] * 7)
    with pytest.raises(ValueError, match="binary"):
        CSP().fit(X, y)


def test_csp_rejects_wrong_ndim():
    X = np.random.default_rng(0).standard_normal((10, 8))  # 2D, not epochs
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError, match="n_trials"):
        CSP().fit(X, y)


def test_csp_separates_planted_classes():
    """On planted data the log-variance features must differ between classes."""
    X, y, _ = _planted_epochs()
    feats = CSP(n_components=4).fit(X, y).transform(X)
    sep = abs(feats[y == 0].mean(axis=0) - feats[y == 1].mean(axis=0)).max()
    assert sep > 0.5


# --- the benchmark --------------------------------------------------------

def test_decode_benchmark_recovers_planted_signal():
    """CSP+LDA should decode the planted signal well, held-out subjects and all."""
    X, y, subjects = _planted_epochs()
    result = decode_benchmark(X, y, subjects, n_splits=3, n_components=4)
    assert isinstance(result, DecodeResult)
    # Signal is shared and strong, so the honest held-out number is high.
    assert result.grouped_roc_auc > 0.9
    assert result.trial_random_roc_auc > 0.9


def test_decode_benchmark_at_chance_on_noise():
    """No signal → both protocols near chance. In-fold fitting cannot invent one."""
    X, y, subjects = _noise_epochs()
    result = decode_benchmark(X, y, subjects, n_splits=3, n_components=4)
    assert result.grouped_roc_auc < 0.70
    assert result.grouped_balanced_accuracy < 0.70


def test_decode_benchmark_reports_shape_metadata():
    X, y, subjects = _planted_epochs(n_subjects=5, n_channels=8)
    result = decode_benchmark(X, y, subjects, n_splits=3, n_components=4)
    assert result.n_channels == 8
    assert result.n_subjects == 5
    assert result.n_trials == X.shape[0]


def test_decode_benchmark_needs_two_subjects():
    X, y, _ = _planted_epochs(n_subjects=1)
    one = np.array(["S00"] * len(y))
    with pytest.raises(ValueError, match="at least 2 subjects"):
        decode_benchmark(X, y, one, n_splits=3)


def test_decode_result_to_dict_is_serializable():
    import json

    X, y, subjects = _planted_epochs()
    result = decode_benchmark(X, y, subjects, n_splits=3, n_components=4)
    d = result.to_dict()
    json.dumps(d)  # must not raise
    assert "csp" in d["decoder"] and "lda" in d["decoder"]
    assert "generalization_gap" in d


def test_decode_benchmark_is_deterministic():
    X, y, subjects = _planted_epochs()
    a = decode_benchmark(X, y, subjects, n_splits=3, seed=7, n_components=4)
    b = decode_benchmark(X, y, subjects, n_splits=3, seed=7, n_components=4)
    assert a.grouped_roc_auc == b.grouped_roc_auc
    assert a.trial_random_roc_auc == b.trial_random_roc_auc


# --- filterbank CSP -------------------------------------------------------

def test_fbcsp_transform_concatenates_bands():
    X, y, _ = _planted_epochs(n_samples=256)
    bands = ((8.0, 12.0), (12.0, 16.0), (16.0, 24.0))
    fb = FilterbankCSP(sampling_rate=128, bands=bands, n_components=2).fit(X, y)
    out = fb.transform(X)
    # One block of n_components features per band.
    assert out.shape == (X.shape[0], len(bands) * 2)
    assert np.isfinite(out).all()


def test_fbcsp_recovers_planted_signal():
    X, y, subjects = _planted_epochs(n_samples=256)
    result = decode_benchmark(
        X, y, subjects, n_splits=3, n_components=2,
        sampling_rate=128, method="fbcsp",
    )
    assert result.grouped_roc_auc > 0.9


def test_decode_benchmark_rejects_unknown_method():
    X, y, subjects = _planted_epochs()
    with pytest.raises(ValueError, match="unknown method"):
        decode_benchmark(X, y, subjects, n_splits=3, method="deep_learning")


def test_build_fbcsp_decoder_is_one_pipeline():
    from sklearn.pipeline import Pipeline

    dec = build_fbcsp_decoder(sampling_rate=128, n_components=2)
    assert isinstance(dec, Pipeline)
    assert list(dec.named_steps) == ["fbcsp", "lda"]


def test_build_csp_decoder_is_in_fold_safe():
    """The decoder is a single pipeline, so cloning per fold refits CSP in-fold."""
    from sklearn.base import clone

    dec = build_csp_decoder(n_components=4)
    # Two independent clones must not share fitted state.
    X, y, _ = _planted_epochs()
    a = clone(dec).fit(X, y)
    b = clone(dec)
    assert hasattr(a.named_steps["csp"], "filters_")
    assert not hasattr(b.named_steps["csp"], "filters_")
