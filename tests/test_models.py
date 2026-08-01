"""Baselines must arrive unfitted and carry their preprocessing inside them.

A bare classifier would force callers to scale features themselves, and the
obvious place to do that is before cross-validation - which is exactly the
leakage this product exists to detect.
"""

import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from nerveml.models import MODEL_KINDS, build_model


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_model_is_a_pipeline_that_owns_its_preprocessing(kind):
    model = build_model(kind)

    assert isinstance(model, Pipeline)
    assert "scaler" in model.named_steps
    assert "classifier" in model.named_steps


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_model_is_returned_unfitted(kind):
    model = build_model(kind)

    with pytest.raises(NotFittedError):
        check_is_fitted(model.named_steps["classifier"])


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_model_can_estimate_probabilities(kind):
    # ROC-AUC needs scores, not just hard labels.
    assert hasattr(build_model(kind), "predict_proba")


def test_two_models_of_the_same_kind_are_identical():
    # Fold-to-fold and run-to-run comparisons are meaningless otherwise.
    a = build_model("random_forest", seed=3).get_params()
    b = build_model("random_forest", seed=3).get_params()

    assert a["classifier__random_state"] == b["classifier__random_state"] == 3


def test_forest_thread_count_is_configurable():
    # A scan that fits 500 forests parallelises over permutations instead, so
    # it needs to be able to ask for single-threaded trees.
    assert build_model("random_forest", n_jobs=1).get_params()["classifier__n_jobs"] == 1
    assert build_model("random_forest").get_params()["classifier__n_jobs"] == -1


def test_thread_count_does_not_change_what_the_forest_learns():
    from nerveml.synth import feature_columns, make_synthetic

    df = make_synthetic("true_signal", n_subjects=6, n_trials=20, n_features=8, seed=0)
    feats = feature_columns(df)
    X, y = df[feats].to_numpy(), df["target_label"].to_numpy()

    wide = build_model("random_forest", seed=0, n_jobs=-1).fit(X, y).predict(X)
    narrow = build_model("random_forest", seed=0, n_jobs=1).fit(X, y).predict(X)

    assert (wide == narrow).all()


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_the_pipeline_survives_missing_values(kind):
    import numpy as np

    from nerveml.synth import feature_columns, make_synthetic

    df = make_synthetic("true_signal", n_subjects=6, n_trials=20, n_features=8, seed=0)
    feats = feature_columns(df)
    X = df[feats].to_numpy().copy()
    X[0, 0] = np.nan
    X[5, 3] = np.nan

    # The integrity scan promises imputation fitted inside training folds. That
    # promise needs a step in the pipeline to be true, and real feature tables
    # arrive with gaps.
    predictions = build_model(kind, seed=0).fit(X, df["target_label"]).predict(X)

    assert len(predictions) == len(X)


@pytest.mark.parametrize("kind", MODEL_KINDS)
def test_imputation_lives_inside_the_pipeline(kind):
    model = build_model(kind)

    assert "imputer" in model.named_steps
    assert list(model.named_steps) == ["imputer", "scaler", "classifier"]


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model("neural_net_please")


# --- the registry seam ----------------------------------------------------

def test_model_kinds_is_derived_from_the_registry():
    from nerveml.models import MODEL_REGISTRY

    assert tuple(MODEL_REGISTRY) == MODEL_KINDS
    # The two originals plus the LDA baseline the registry demonstrates.
    assert set(MODEL_KINDS) >= {"random_forest", "logistic_regression", "lda"}


def test_register_model_adds_a_usable_kind():
    from sklearn.tree import DecisionTreeClassifier

    from nerveml.models import MODEL_REGISTRY, register_model

    name = "_test_only_tree"
    try:
        @register_model(name)
        def _factory(seed, n_jobs):
            return DecisionTreeClassifier(random_state=seed)

        model = build_model(name, seed=0)
        assert model.named_steps["classifier"].__class__ is DecisionTreeClassifier
    finally:
        MODEL_REGISTRY.pop(name, None)


def test_register_model_rejects_duplicate_names():
    from nerveml.models import register_model

    with pytest.raises(ValueError, match="already registered"):
        @register_model("random_forest")
        def _dupe(seed, n_jobs):
            return None


def test_lda_kind_builds_and_predicts():
    import numpy as np

    X = np.random.default_rng(0).standard_normal((40, 6))
    y = (X[:, 0] > 0).astype(int)
    model = build_model("lda", seed=0).fit(X, y)
    assert hasattr(model, "predict_proba")
    assert len(model.predict(X)) == len(X)
