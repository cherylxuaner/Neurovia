"""Feature ranking, with a dataset where the right answer is known.

If exactly one feature carries the label, that feature has to come first. Any
ranking that cannot recover a planted signal cannot be trusted to describe a
real one.
"""

import pytest

from nerveml.interpret import rank_features
from nerveml.models import MODEL_KINDS
from nerveml.synth import feature_columns, make_synthetic


@pytest.fixture
def planted():
    """true_signal data relabelled so that only f00 predicts the target."""
    df = make_synthetic("true_signal", n_subjects=10, n_trials=40, n_features=8, seed=0)
    df = df.copy()
    df["target_label"] = (df["f00"] > df["f00"].median()).astype(int)
    return df, feature_columns(df)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_the_planted_feature_ranks_first(planted, kind):
    df, feats = planted

    ranked = rank_features(df, feats, kind=kind, seed=0)

    assert ranked[0]["feature"] == "f00"


def test_ranking_is_sorted_by_descending_importance(planted):
    df, feats = planted

    ranked = rank_features(df, feats, seed=0)

    scores = [item["importance"] for item in ranked]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_the_output(planted):
    df, feats = planted

    assert len(rank_features(df, feats, top_k=3, seed=0)) == 3


def test_top_k_beyond_the_feature_count_returns_everything(planted):
    df, feats = planted

    assert len(rank_features(df, feats, top_k=999, seed=0)) == len(feats)


def test_logistic_ranking_reports_the_direction_of_association(planted):
    df, feats = planted

    ranked = rank_features(df, feats, kind="logistic_regression", seed=0)

    # Higher f00 was defined to mean label 1.
    assert ranked[0]["direction"] == "positive"


def test_random_forest_ranking_reports_no_direction(planted):
    df, feats = planted

    ranked = rank_features(df, feats, kind="random_forest", seed=0)

    # Impurity importance is unsigned; claiming a direction would be invented.
    assert ranked[0]["direction"] is None


def test_importances_are_plain_floats(planted):
    df, feats = planted

    for item in rank_features(df, feats, seed=0):
        assert type(item["importance"]) is float
