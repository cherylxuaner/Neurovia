"""Baseline classifiers, always wrapped with their own preprocessing.

Scaling lives inside the pipeline so that cross-validation refits it on every
training fold. Scaling the full table once, before splitting, would leak test
statistics into training - the failure mode this product reports on.

Feature-table decoders are declared in a registry rather than an if/elif chain,
so adding one is a single ``@register_model`` decorator next to the factory,
and every caller that lists or validates ``MODEL_KINDS`` picks it up for free.
The registry holds only decoders that consume the 2D feature table; CSP works on
raw epochs and lives in ``csp.py``, off this seam by design.
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# name -> callable(seed, n_jobs) -> unfitted sklearn classifier. Populated by the
# @register_model decorator below; MODEL_KINDS is derived from it so the two can
# never drift apart.
MODEL_REGISTRY = {}


def register_model(name):
    """Register a classifier factory under name. Returns the factory unchanged."""
    def decorator(factory):
        if name in MODEL_REGISTRY:
            raise ValueError(f"model kind {name!r} is already registered")
        MODEL_REGISTRY[name] = factory
        return factory
    return decorator


@register_model("random_forest")
def _random_forest(seed, n_jobs):
    # n_jobs affects only how the forest is fitted, never what it fits: the same
    # seed yields the same trees at any thread count. A caller that already
    # parallelises at a coarser level - the permutation null fits hundreds of
    # forests - should pass n_jobs=1 so the two levels do not contend for cores.
    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=n_jobs)


@register_model("logistic_regression")
def _logistic_regression(seed, n_jobs):
    return LogisticRegression(max_iter=2000, random_state=seed)


@register_model("lda")
def _lda(seed, n_jobs):
    # Linear discriminant analysis: no randomness, so seed and n_jobs are inert.
    # A fast, closed-form linear baseline and the natural classifier to pair with
    # CSP features elsewhere.
    return LinearDiscriminantAnalysis()


# Tuple of registered kinds, in registration order. Kept as the public constant
# every other module imports, so its contents follow the registry automatically.
MODEL_KINDS = tuple(MODEL_REGISTRY)


def build_model(kind="random_forest", seed=0, n_jobs=-1):
    """Return an unfitted scaler + classifier pipeline for a registered kind."""
    if kind not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model kind {kind!r}; expected one of {tuple(MODEL_REGISTRY)}"
        )

    classifier = MODEL_REGISTRY[kind](seed=seed, n_jobs=n_jobs)

    # Imputation before scaling, both inside the pipeline: a real feature
    # table arrives with gaps, and the column means used to fill them are
    # learned statistics like any other. Computed over the whole table they
    # would carry test information into training.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])
