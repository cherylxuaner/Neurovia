"""The null test, and the two ways it is commonly got wrong.

First: labels must be shuffled *within* each group. Shuffling globally also
destroys the per-subject label imbalance, which produces a null distribution
that is easier to beat than the real data - and therefore a p-value that looks
significant when nothing was learned.

Second: every permutation must rerun the whole grouped pipeline, refitting
preprocessing and model. Reusing anything fitted on the real labels leaks them
into the null.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nerveml.permutation import permutation_test, permute_labels_within_group
from nerveml.synth import feature_columns, make_synthetic

N_PERMUTATIONS = 40

_FIT_ROW_COUNTS = []


class CountingSpy(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        _FIT_ROW_COUNTS.append(len(X))
        return self

    def transform(self, X):
        return X


def fast_model():
    """Logistic regression keeps a 40-permutation run to a couple of seconds."""
    return Pipeline(
        [("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000))]
    )


@pytest.fixture
def leaky():
    df = make_synthetic("subject_leakage", n_subjects=12, n_trials=20, seed=0)
    return df, feature_columns(df)


@pytest.fixture
def genuine():
    df = make_synthetic("true_signal", n_subjects=12, n_trials=20, seed=0)
    return df, feature_columns(df)


def test_shuffling_preserves_each_groups_label_counts(leaky):
    df, _ = leaky
    y, groups = df["target_label"].to_numpy(), df["subject_id"].to_numpy()

    shuffled = permute_labels_within_group(y, groups, np.random.default_rng(0))

    for subject in set(groups):
        mask = groups == subject
        assert shuffled[mask].sum() == y[mask].sum()


def test_shuffling_actually_reorders_labels(leaky):
    df, _ = leaky
    y, groups = df["target_label"].to_numpy(), df["subject_id"].to_numpy()

    shuffled = permute_labels_within_group(y, groups, np.random.default_rng(0))

    assert not np.array_equal(shuffled, y)


def test_shuffling_does_not_mutate_the_original_labels(leaky):
    df, _ = leaky
    y, groups = df["target_label"].to_numpy(), df["subject_id"].to_numpy()
    before = y.copy()

    permute_labels_within_group(y, groups, np.random.default_rng(0))

    assert np.array_equal(y, before)


def test_leaky_data_does_not_beat_its_null(leaky):
    df, feats = leaky

    result = permutation_test(
        df, feats, model=fast_model(), n_permutations=N_PERMUTATIONS, n_splits=4, seed=0
    )

    # Subject-held-out performance here is chance, so it must not look special.
    assert result.p_value >= 0.05
    assert result.null_mean == pytest.approx(0.5, abs=0.08)


def test_genuine_signal_beats_its_null(genuine):
    df, feats = genuine

    result = permutation_test(
        df, feats, model=fast_model(), n_permutations=N_PERMUTATIONS, n_splits=4, seed=0
    )

    assert result.p_value < 0.05
    assert result.observed > result.null_mean


def test_p_value_is_never_reported_as_zero(genuine):
    df, feats = genuine

    result = permutation_test(
        df, feats, model=fast_model(), n_permutations=N_PERMUTATIONS, n_splits=4, seed=0
    )

    # A finite permutation count cannot evidence p = 0.
    assert result.p_value >= 1 / (N_PERMUTATIONS + 1)
    assert len(result.null_scores) == N_PERMUTATIONS


def test_every_permutation_refits_the_entire_pipeline(leaky):
    df, feats = leaky
    _FIT_ROW_COUNTS.clear()
    spied = Pipeline(
        [
            ("spy", CountingSpy()),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )

    permutation_test(df, feats, model=spied, n_permutations=5, n_splits=4, seed=0)

    # 1 observed run + 5 permutations, four folds each.
    assert len(_FIT_ROW_COUNTS) == 6 * 4


def test_thread_count_does_not_change_the_null(genuine):
    df, feats = genuine

    serial = permutation_test(
        df, feats, model=fast_model(), n_permutations=20, n_splits=4, seed=0, n_jobs=1
    )
    threaded = permutation_test(
        df, feats, model=fast_model(), n_permutations=20, n_splits=4, seed=0, n_jobs=4
    )

    # Permuted label vectors are drawn up front from one seeded generator, so
    # scheduling cannot reorder them into a different null.
    assert serial.null_scores == threaded.null_scores
    assert serial.p_value == threaded.p_value


def test_same_seed_reproduces_the_null_distribution(genuine):
    df, feats = genuine

    a = permutation_test(df, feats, model=fast_model(), n_permutations=10, n_splits=4, seed=1)
    b = permutation_test(df, feats, model=fast_model(), n_permutations=10, n_splits=4, seed=1)

    assert a.null_scores == b.null_scores
    assert a.p_value == b.p_value


# ---------------------------------------------------------------------------
# Property tests: mathematical invariants of the permutation protocol
# ---------------------------------------------------------------------------

@pytest.fixture
def strong_signal():
    """Dataset with genuine, strong within-subject signal.

    Observed grouped balanced-accuracy (~0.92) beats every null permutation
    drawn with seed=42, so p_value == 1 / (n_permutations + 1) exactly.
    """
    df = make_synthetic("true_signal", n_subjects=12, n_trials=60, seed=0)
    return df, feature_columns(df)


def test_null_distribution_is_centered_near_chance(leaky):
    """The within-group shuffle protocol must produce a null at chance (0.5).

    For data where grouped CV yields chance performance, the permutation null
    must also sit near chance — if it drifted significantly, the p-value would
    be miscalibrated.  100 permutations gives a stable-enough estimate.
    """
    df, feats = leaky

    result = permutation_test(
        df, feats, model=fast_model(), n_permutations=100, n_splits=4, seed=0
    )

    assert abs(result.null_mean - 0.5) < 0.07, (
        f"null_mean={result.null_mean:.4f} drifted too far from 0.5"
    )


def test_null_prefix_consistency(genuine):
    """Permuted label vectors are drawn from a single seeded generator up front.

    A run with n=30 permutations and seed=7 must produce exactly the same first
    10 null scores as a run with n=10 permutations and seed=7.  Violating this
    would mean the null distribution changes depending on how many permutations
    are requested — a reproducibility failure.
    """
    df, feats = genuine
    N_SMALL = 10

    small = permutation_test(
        df, feats, model=fast_model(), n_permutations=N_SMALL, n_splits=4, seed=7
    )
    large = permutation_test(
        df, feats, model=fast_model(), n_permutations=30, n_splits=4, seed=7
    )

    assert large.null_scores[:N_SMALL] == small.null_scores, (
        "First N_SMALL null scores differ between short and long run with same seed"
    )


def test_p_value_equals_one_over_n_plus_one_when_observed_beats_all(strong_signal):
    """When the observed score exceeds every null permutation, p = 1/(n+1).

    This is the minimum achievable p-value for a given n and is exact (not
    approximate) because n_at_least_as_good == 0 in the numerator formula
    (1 + 0) / (1 + n).  This test validates both the formula and that the
    strong-signal fixture actually achieves k=0.
    """
    df, feats = strong_signal

    for n in (10, 20, 30):
        result = permutation_test(
            df, feats, model=fast_model(), n_permutations=n, n_splits=4, seed=42
        )
        expected_p = 1.0 / (1 + n)
        assert result.observed > max(result.null_scores), (
            f"n={n}: observed {result.observed:.3f} did not beat all null scores"
        )
        assert result.p_value == pytest.approx(expected_p, abs=1e-9), (
            f"n={n}: p_value={result.p_value} != 1/(n+1)={expected_p}"
        )


def test_p_value_strictly_decreases_as_permutations_increase_when_k_zero(strong_signal):
    """p-value is strictly monotone-decreasing in n when observed beats all nulls.

    When k=0 (no null permutation reaches the observed score), p = 1/(n+1)
    which is strictly decreasing.  Adding more permutations provides sharper
    evidence — the claim becomes stronger, not weaker — which is the key
    guarantee users rely on when choosing n.
    """
    df, feats = strong_signal
    ns = [10, 20, 30, 40]

    p_values = [
        permutation_test(
            df, feats, model=fast_model(), n_permutations=n, n_splits=4, seed=42
        ).p_value
        for n in ns
    ]

    for prev, curr in zip(p_values, p_values[1:]):
        assert curr < prev, (
            f"p-value did not decrease: {prev:.4f} -> {curr:.4f}"
        )
