"""Ranking which features a model leaned on.

Fitted on the full table, because this describes the model's behaviour on the
data it was given rather than estimating out-of-sample performance - the
subject-held-out numbers are what carry the performance claim. The ranking is
model-dependent and non-causal, and the report says so next to the numbers.
"""

import numpy as np

from nerveml.models import build_model
from nerveml.validation import TARGET_COLUMN

FEATURE_CAVEAT = (
    "Feature associations are model-dependent and non-causal. They describe "
    "which inputs this classifier used on this dataset, not which neural "
    "processes generate the target attribute."
)


def rank_features(df, feature_columns, kind="random_forest", top_k=10, seed=0):
    """Rank features by their contribution, strongest first."""
    model = build_model(kind, seed=seed)
    model.fit(df[feature_columns].to_numpy(), df[TARGET_COLUMN].to_numpy())
    classifier = model.named_steps["classifier"]

    if hasattr(classifier, "coef_"):
        # Inputs were standardised inside the pipeline, so coefficients are
        # directly comparable across features.
        weights = classifier.coef_.ravel()
        importances = np.abs(weights)
        directions = ["positive" if w > 0 else "negative" for w in weights]
    else:
        importances = classifier.feature_importances_
        directions = [None] * len(feature_columns)

    ranked = sorted(
        (
            {
                "feature": name,
                "importance": float(importance),
                "direction": direction,
            }
            for name, importance, direction in zip(
                feature_columns, importances, directions
            )
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return ranked[:top_k]
