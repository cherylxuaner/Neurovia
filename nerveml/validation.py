"""The two evaluation protocols whose disagreement is the product.

evaluate_trial_random splits trials without regard to who produced them, so the
same person can sit on both sides of the split. evaluate_grouped keeps every
unit of a chosen grouping column wholly in train or wholly in test. Running
both over identical features and an identical model family makes the difference
attributable to the protocol rather than to anything else.

The grouping column is a parameter, not a constant: subject is the first and
most important unit, but session, site, device and stimulus leak the same way
and are checked by pointing this at a different column.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from nerveml.models import build_model

SUBJECT_COLUMN = "subject_id"
TARGET_COLUMN = "target_label"


def _unit_name(group_column):
    """subject_id -> subjects, session_id -> sessions."""
    stem = group_column[:-3] if group_column.endswith("_id") else group_column
    return f"{stem}s"


@dataclass
class CVResult:
    """Fold-level outcome of one evaluation protocol."""

    scheme: str
    n_splits: int
    balanced_accuracy: float
    roc_auc: float
    balanced_accuracy_std: float
    roc_auc_std: float
    fold_balanced_accuracy: list = field(default_factory=list)
    fold_roc_auc: list = field(default_factory=list)
    held_out_subjects: list = field(default_factory=list)
    per_group: list = field(default_factory=list)
    # {unit: (truth, predicted)} over its held-out trials. Kept out of
    # to_dict(): needed to resample units, far too bulky for the report.
    unit_predictions: dict = field(default_factory=dict)

    @property
    def within_unit_roc_auc(self):
        """Mean of each unit's own ROC-AUC. See within_unit_balanced_accuracy."""
        scored = [
            e["roc_auc"] for e in self.per_group if e.get("roc_auc") is not None
        ]
        return float(np.mean(scored)) if scored else float("nan")

    @property
    def within_unit_balanced_accuracy(self):
        """Mean of each unit's own score, rather than one score over all trials.

        Pooling lets a model earn credit for telling people apart. Scoring each
        person separately removes that route: what is left is whether the model
        can tell the labels apart inside a single person. When the pooled score
        is high and this one sits at chance, the pooled score was never about
        the label.
        """
        scored = [
            e["balanced_accuracy"] for e in self.per_group
            if e["balanced_accuracy"] is not None
        ]
        return float(np.mean(scored)) if scored else float("nan")

    def to_dict(self):
        return {
            "scheme": self.scheme,
            "n_splits": self.n_splits,
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "roc_auc": round(self.roc_auc, 4),
            "balanced_accuracy_std": round(self.balanced_accuracy_std, 4),
            "roc_auc_std": round(self.roc_auc_std, 4),
            "fold_balanced_accuracy": [round(v, 4) for v in self.fold_balanced_accuracy],
            "fold_roc_auc": [round(v, 4) for v in self.fold_roc_auc],
            "within_unit_balanced_accuracy": round(
                self.within_unit_balanced_accuracy, 4
            ),
            "within_unit_roc_auc": round(self.within_unit_roc_auc, 4),
            "per_group": self.per_group,
        }


def _fit_fold(model, X, y, train_idx, test_idx):
    """Fit one fold. Returns (balanced accuracy, roc auc, predictions, scores)."""
    estimator = clone(model)
    estimator.fit(X[train_idx], y[train_idx])
    predicted = estimator.predict(X[test_idx])
    scores = estimator.predict_proba(X[test_idx])[:, 1]

    bacc = balanced_accuracy_score(y[test_idx], predicted)
    if len(np.unique(y[test_idx])) < 2:
        # ROC-AUC is undefined against a single-class fold.
        auc = float("nan")
    else:
        auc = roc_auc_score(y[test_idx], scores)
    return bacc, auc, predicted, scores


class GroupedScorer:
    """Reusable subject-held-out scorer over pre-extracted arrays.

    The permutation null reruns this pipeline once per permutation. Preparing
    the feature matrix, the grouping vector and the splitter a single time
    removes a full dataframe copy and a numpy conversion from every one of
    those iterations, which is most of their cost at prototype data sizes.
    """

    def __init__(self, X, groups, splitter, model):
        self.X = X
        self.groups = groups
        self.splitter = splitter
        self.model = model
        self._splits = None

    def splits(self, y):
        if self._splits is None:
            self._splits = list(self.splitter.split(self.X, y, self.groups))
        return self._splits

    def score(self, y):
        """Mean balanced accuracy across folds for this label vector."""
        return float(
            np.mean(
                [
                    _fit_fold(self.model, self.X, y, train_idx, test_idx)[0]
                    for train_idx, test_idx in self.splits(y)
                ]
            )
        )


def _per_group_scores(y, groups, fold_predictions):
    """Balanced accuracy and AUC for each unit, in the fold that held it out."""
    entries = []
    for test_idx, predicted, scored in fold_predictions:
        held = groups[test_idx]
        for unit in dict.fromkeys(held):
            mask = held == unit
            truth = y[test_idx][mask]
            guess = predicted[mask]
            single_class = len(np.unique(truth)) < 2
            entries.append(
                {
                    "group": unit.item() if hasattr(unit, "item") else unit,
                    "n_trials": int(mask.sum()),
                    # Undefined against one class: it collapses to that class's
                    # recall, which does not compare with the other units.
                    "balanced_accuracy": (
                        None
                        if single_class
                        else round(float(balanced_accuracy_score(truth, guess)), 4)
                    ),
                    "roc_auc": (
                        None
                        if single_class
                        else round(float(roc_auc_score(truth, scored[mask])), 4)
                    ),
                    "accuracy": round(float(np.mean(truth == guess)), 4),
                    "single_class": bool(single_class),
                }
            )
    return sorted(entries, key=lambda entry: str(entry["group"]))


def _unit_predictions(y, groups, fold_predictions):
    """Held-out truth and predictions for each unit, keyed by unit."""
    kept = {}
    for test_idx, predicted, _scored in fold_predictions:
        held = groups[test_idx]
        for unit in dict.fromkeys(held):
            mask = held == unit
            key = unit.item() if hasattr(unit, "item") else unit
            kept.setdefault(key, ([], []))
            kept[key][0].append(y[test_idx][mask])
            kept[key][1].append(predicted[mask])
    return {
        unit: (np.concatenate(truth), np.concatenate(guess))
        for unit, (truth, guess) in kept.items()
    }


def _run_folds(X, y, groups, splitter, model, scheme, per_group_column, split_groups, on_fold=None):
    """Run one protocol.

    groups labels every row for per-unit scoring. split_groups is what the
    splitter is allowed to see - None for trial-random, where the whole point
    is that the splitter does not respect unit boundaries.

    on_fold, when provided, is called after each fold as
    on_fold(fold_idx, n_folds, bacc) with 0-based fold_idx.
    """
    fold_bacc, fold_auc, held_out, fold_predictions = [], [], [], []
    n_folds = splitter.get_n_splits()
    fold_cb = on_fold if on_fold is not None else lambda *_: None

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X, y, split_groups)):
        bacc, auc, predicted, scored = _fit_fold(model, X, y, train_idx, test_idx)
        fold_bacc.append(bacc)
        fold_auc.append(auc)
        fold_predictions.append((test_idx, predicted, scored))
        if per_group_column is not None:
            held_out.append(sorted(set(groups[test_idx])))
        fold_cb(fold_idx, n_folds, bacc)

    per_group = (
        _per_group_scores(y, groups, fold_predictions)
        if per_group_column is not None
        else []
    )
    predictions = (
        _unit_predictions(y, groups, fold_predictions)
        if per_group_column is not None
        else {}
    )

    return CVResult(
        scheme=scheme,
        n_splits=splitter.get_n_splits(),
        balanced_accuracy=float(np.mean(fold_bacc)),
        roc_auc=float(np.nanmean(fold_auc)),
        balanced_accuracy_std=float(np.std(fold_bacc)),
        roc_auc_std=float(np.nanstd(fold_auc)),
        fold_balanced_accuracy=[float(v) for v in fold_bacc],
        fold_roc_auc=[float(v) for v in fold_auc],
        held_out_subjects=held_out,
        per_group=per_group,
        unit_predictions=predictions,
    )


def evaluate_trial_random(
    df, feature_columns, group_column=SUBJECT_COLUMN, model=None, n_splits=5, seed=0,
    on_fold=None,
):
    """Shuffled trial-level cross-validation. The weaker protocol.

    Trials from one subject can land in both train and test, so a model that
    recognises people can score well without learning anything transferable.

    Per-unit scores are still reported. They are not a generalisation estimate
    here - no unit is ever fully held out - they decompose the pooled score,
    separating what the model knows about labels from what it knows about
    people.
    """
    model = build_model(seed=seed) if model is None else model
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return _run_folds(
        df[feature_columns].to_numpy(),
        df[TARGET_COLUMN].to_numpy(),
        df[group_column].to_numpy(),
        splitter,
        model,
        "trial_random",
        group_column,
        split_groups=None,
        on_fold=on_fold,
    )


def evaluate_grouped(
    df, feature_columns, group_column=SUBJECT_COLUMN, model=None, n_splits=5, seed=0,
    on_fold=None,
):
    """Held-out-unit cross-validation. The protocol that supports claims."""
    if group_column not in df.columns:
        raise ValueError(
            f"grouping column {group_column!r} is not present in the data; "
            "an independent grouping variable is required to tell "
            "generalisation apart from memorisation"
        )

    n_groups = df[group_column].nunique()
    if n_groups < n_splits:
        raise ValueError(
            f"grouped evaluation with {n_splits} folds needs at least "
            f"{n_splits} {_unit_name(group_column)}, found {n_groups}"
        )

    model = build_model(seed=seed) if model is None else model
    scheme = f"{group_column}_grouped"
    return _run_folds(
        df[feature_columns].to_numpy(),
        df[TARGET_COLUMN].to_numpy(),
        df[group_column].to_numpy(),
        GroupKFold(n_splits=n_splits),
        model,
        scheme,
        group_column,
        split_groups=df[group_column].to_numpy(),
        on_fold=on_fold,
    )


def evaluate_subject_grouped(df, feature_columns, model=None, n_splits=5, seed=0, on_fold=None):
    """Subject-held-out cross-validation (spec 10.5)."""
    return evaluate_grouped(
        df,
        feature_columns,
        group_column=SUBJECT_COLUMN,
        model=model,
        n_splits=n_splits,
        seed=seed,
        on_fold=on_fold,
    )


def build_grouped_scorer(
    df, feature_columns, group_column=SUBJECT_COLUMN, model=None, n_splits=5, seed=0
):
    """A GroupedScorer over this dataset, for callers that rescore many times."""
    return GroupedScorer(
        X=df[feature_columns].to_numpy(),
        groups=df[group_column].to_numpy(),
        splitter=GroupKFold(n_splits=n_splits),
        model=build_model(seed=seed) if model is None else model,
    )


def generalization_gap(naive, grouped):
    """Naive score minus held-out score, in balanced accuracy.

    A large positive gap means within-unit structure inflated the naive result.
    It is a description of the evaluation, not an accusation.
    """
    return naive.balanced_accuracy - grouped.balanced_accuracy
