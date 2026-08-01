"""The product's central claim, as executable assertions.

On data whose labels are a property of the person, trial-random validation
should look good and subject-held-out validation should collapse. On data whose
labels come from a shared signal, both should hold up. If the engine cannot
reproduce that difference on data where we know the answer, it cannot be
trusted on data where we do not.
"""

import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nerveml.synth import feature_columns, make_synthetic
from nerveml.validation import (
    evaluate_grouped,
    evaluate_subject_grouped,
    evaluate_trial_random,
    generalization_gap,
)

CHANCE = 0.5

_FIT_ROW_COUNTS = []


class RowCountSpy(BaseEstimator, TransformerMixin):
    """Records how many rows each fit() saw. Passes data through untouched."""

    def fit(self, X, y=None):
        _FIT_ROW_COUNTS.append(len(X))
        return self

    def transform(self, X):
        return X


@pytest.fixture
def leaky():
    df = make_synthetic("subject_leakage", n_subjects=20, n_trials=40, seed=0)
    return df, feature_columns(df)


@pytest.fixture
def genuine():
    df = make_synthetic("true_signal", n_subjects=20, n_trials=40, seed=0)
    return df, feature_columns(df)


def test_trial_random_validation_looks_strong_on_leaky_data(leaky):
    df, feats = leaky

    result = evaluate_trial_random(df, feats, seed=0)

    # Nothing here generalises, yet the naive protocol reports success.
    assert result.balanced_accuracy > 0.75


def test_subject_held_out_validation_collapses_on_leaky_data(leaky):
    df, feats = leaky

    result = evaluate_subject_grouped(df, feats, seed=0)

    assert result.balanced_accuracy < 0.60


def test_leaky_data_produces_a_large_generalization_gap(leaky):
    df, feats = leaky

    gap = generalization_gap(
        evaluate_trial_random(df, feats, seed=0),
        evaluate_subject_grouped(df, feats, seed=0),
    )

    # The spec treats >= 0.20 as a high generalisation warning.
    assert gap >= 0.20


def test_subject_held_out_validation_survives_on_genuine_signal(genuine):
    df, feats = genuine

    result = evaluate_subject_grouped(df, feats, seed=0)

    assert result.balanced_accuracy > 0.70
    assert result.roc_auc > 0.70


def test_genuine_signal_produces_a_small_generalization_gap(genuine):
    df, feats = genuine

    gap = generalization_gap(
        evaluate_trial_random(df, feats, seed=0),
        evaluate_subject_grouped(df, feats, seed=0),
    )

    assert abs(gap) < 0.10


def test_no_subject_appears_in_both_sides_of_a_grouped_split(leaky):
    df, feats = leaky

    result = evaluate_subject_grouped(df, feats, n_splits=5, seed=0)

    seen = [set(subjects) for subjects in result.held_out_subjects]
    assert sum(len(s) for s in seen) == df["subject_id"].nunique()
    assert set.union(*seen) == set(df["subject_id"].unique())
    for i, a in enumerate(seen):
        for b in seen[i + 1 :]:
            assert not (a & b)


def test_preprocessing_is_never_fitted_on_held_out_rows(leaky):
    df, feats = leaky
    _FIT_ROW_COUNTS.clear()
    spied = Pipeline(
        [
            ("spy", RowCountSpy()),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=0)),
        ]
    )

    evaluate_subject_grouped(df, feats, model=spied, n_splits=5, seed=0)

    assert len(_FIT_ROW_COUNTS) == 5
    # 20 subjects, 5 folds: each fit sees 16 subjects' worth of trials, never 20.
    assert all(count == 640 for count in _FIT_ROW_COUNTS)


def test_per_fold_scores_are_reported(leaky):
    df, feats = leaky

    result = evaluate_subject_grouped(df, feats, n_splits=5, seed=0)

    # A global mean can hide a subject the model fails badly on.
    assert len(result.fold_balanced_accuracy) == 5
    assert result.balanced_accuracy == pytest.approx(
        sum(result.fold_balanced_accuracy) / 5
    )
    assert result.balanced_accuracy_std >= 0


def test_every_subject_gets_its_own_score(leaky):
    df, feats = leaky

    result = evaluate_subject_grouped(df, feats, n_splits=5, seed=0)

    # Spec 11.1: a global mean must not be able to hide a failing participant.
    scored = {entry["group"] for entry in result.per_group}
    assert scored == set(df["subject_id"].unique())
    for entry in result.per_group:
        assert entry["n_trials"] == 40


def test_per_subject_scores_expose_subjects_the_mean_hides(leaky):
    df, feats = leaky

    result = evaluate_subject_grouped(df, feats, n_splits=5, seed=0)

    worst = min(entry["balanced_accuracy"] for entry in result.per_group)
    assert worst < 0.5


def test_no_subject_is_left_behind_when_the_signal_is_real(genuine):
    df, feats = genuine

    result = evaluate_subject_grouped(df, feats, n_splits=5, seed=0)

    assert all(entry["balanced_accuracy"] > 0.5 for entry in result.per_group)


def test_a_subject_seen_in_only_one_class_is_marked(leaky):
    df, feats = leaky
    df = df.copy()
    df.loc[df["subject_id"] == "s00", "target_label"] = 1

    result = evaluate_subject_grouped(df, feats, n_splits=5, seed=0)

    marked = {e["group"] for e in result.per_group if e["single_class"]}
    assert marked == {"s00"}


def test_every_subject_gets_its_own_auc(genuine):
    df, feats = genuine

    result = evaluate_subject_grouped(df, feats, n_splits=5, seed=0)

    # The risk thresholds are applied to AUC, so an interval on the verdict
    # needs AUC computed at the unit of independence.
    aucs = [e["roc_auc"] for e in result.per_group]
    assert all(0.0 <= a <= 1.0 for a in aucs)
    assert result.within_unit_roc_auc == pytest.approx(sum(aucs) / len(aucs))


def test_a_single_class_subject_reports_no_auc(leaky):
    df, feats = leaky
    df = df.copy()
    df.loc[df["subject_id"] == "s00", "target_label"] = 1

    result = evaluate_subject_grouped(df, feats, n_splits=5, seed=0)
    entry = next(e for e in result.per_group if e["group"] == "s00")

    assert entry["roc_auc"] is None


def test_a_single_class_subject_reports_no_balanced_accuracy(leaky):
    df, feats = leaky
    df = df.copy()
    df.loc[df["subject_id"] == "s00", "target_label"] = 1

    result = evaluate_subject_grouped(df, feats, n_splits=5, seed=0)
    entry = next(e for e in result.per_group if e["group"] == "s00")

    # Against one class it degenerates to that class's recall, which is not
    # comparable with the other subjects' scores. Plain accuracy still is.
    assert entry["balanced_accuracy"] is None
    assert 0.0 <= entry["accuracy"] <= 1.0


def test_a_memorising_model_scores_at_chance_inside_every_subject(leaky):
    df, feats = leaky

    result = evaluate_trial_random(df, feats, seed=0)

    # Pooled, it looks strong. Scored inside each person separately, it has no
    # discriminative power at all - the pooled number was telling people apart,
    # not labels apart.
    assert result.balanced_accuracy > 0.75
    assert result.within_unit_balanced_accuracy == pytest.approx(0.5, abs=0.05)


def test_a_real_signal_scores_the_same_pooled_and_within_subject(genuine):
    df, feats = genuine

    result = evaluate_trial_random(df, feats, seed=0)

    assert result.within_unit_balanced_accuracy == pytest.approx(
        result.balanced_accuracy, abs=0.05
    )


def test_trial_random_reports_a_score_for_every_subject(leaky):
    df, feats = leaky

    result = evaluate_trial_random(df, feats, seed=0)

    assert {e["group"] for e in result.per_group} == set(df["subject_id"].unique())


def test_any_column_can_be_the_independent_grouping_unit(leaky):
    df, feats = leaky
    # Two subjects per recording session: holding out sessions is stricter.
    df = df.copy()
    df["session_id"] = df["subject_id"].str[-2:].astype(int) // 2

    result = evaluate_grouped(df, feats, group_column="session_id", n_splits=5, seed=0)

    assert result.scheme == "session_id_grouped"
    assert len(result.per_group) == 10


def test_grouped_evaluation_names_the_missing_column(leaky):
    df, feats = leaky

    with pytest.raises(ValueError, match="site_id"):
        evaluate_grouped(df, feats, group_column="site_id", n_splits=5, seed=0)


def test_grouped_evaluation_requires_more_subjects_than_folds(leaky):
    df, feats = leaky
    two_subjects = df[df["subject_id"].isin(["s00", "s01"])]

    with pytest.raises(ValueError, match="at least 5 subjects"):
        evaluate_subject_grouped(two_subjects, feats, n_splits=5, seed=0)


# ── on_fold callback ─────────────────────────────────────────────────────────


def test_on_fold_callback_called_once_per_fold_trial_random(leaky):
    df, feats = leaky
    calls = []
    evaluate_trial_random(df, feats, n_splits=3, seed=0, on_fold=lambda *a: calls.append(a))
    assert len(calls) == 3


def test_on_fold_callback_receives_correct_n_folds_trial_random(leaky):
    df, feats = leaky
    calls = []
    evaluate_trial_random(df, feats, n_splits=4, seed=0, on_fold=lambda *a: calls.append(a))
    assert all(n_folds == 4 for _, n_folds, _ in calls)


def test_on_fold_callback_fold_idx_is_zero_based_sequential(leaky):
    df, feats = leaky
    calls = []
    evaluate_trial_random(df, feats, n_splits=3, seed=0, on_fold=lambda *a: calls.append(a))
    assert [fold_idx for fold_idx, _, _ in calls] == [0, 1, 2]


def test_on_fold_callback_bacc_is_float_in_range(leaky):
    df, feats = leaky
    calls = []
    evaluate_trial_random(df, feats, n_splits=3, seed=0, on_fold=lambda *a: calls.append(a))
    assert all(isinstance(bacc, float) and 0.0 <= bacc <= 1.0 for _, _, bacc in calls)


def test_on_fold_callback_called_once_per_fold_grouped(leaky):
    df, feats = leaky
    calls = []
    evaluate_grouped(df, feats, n_splits=4, seed=0, on_fold=lambda *a: calls.append(a))
    assert len(calls) == 4


def test_on_fold_callback_none_is_default_no_error(leaky):
    df, feats = leaky
    result = evaluate_trial_random(df, feats, n_splits=3, seed=0)
    assert result.n_splits == 3
