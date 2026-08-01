"""Re-identification probe (spec 6.2): can the signal name who produced it?

The threat model is linkage, not clairvoyance. An attacker who holds labelled
reference recordings — a prior session, a public release, a research cohort —
wants to attach new anonymous records to those same people. So every identity
stays in training and unseen records of them are held out. Holding a person out
entirely would measure nothing: no method can name a stranger.

This is deliberately the opposite arrangement from the validity protocols. There,
a subject appearing on both sides is the flaw being detected. Here it is the
attack being modelled.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from nerveml.models import build_model
from nerveml.uncertainty import bootstrap_ci
from nerveml.validation import SUBJECT_COLUMN

_LINKAGE = (
    "Linkage against a labelled reference set: the adversary already holds "
    "identified recordings of these individuals and attaches held-out records "
    "to them. Measures whether the signal itself carries identity after direct "
    "identifiers are removed. It does not measure whether a stranger can be "
    "named, and it is evidence under this attack only, not proof of anonymity."
)

SAME_RECORDING = _LINKAGE + (
    " Reference and held-out records come from the same recording, so a "
    "fingerprint of the session - electrode placement, impedance, drift - is "
    "not separated from a fingerprint of the person."
)

CROSS_RECORDING = _LINKAGE + (
    " Reference and held-out recording are different, so whatever is specific "
    "to one recording cannot carry the result. What survives belongs to the "
    "person rather than to the session."
)

ATTACK_MODEL = SAME_RECORDING


@dataclass
class IdentityResult:
    accuracy: float
    chance: float
    lift: float
    n_identities: int
    n_records: int
    per_identity: list = field(default_factory=list)
    attack_model: str = ATTACK_MODEL

    @property
    def recall_ci(self):
        """Interval over identities: each person is one independent unit."""
        return bootstrap_ci([e["recall"] for e in self.per_identity])

    @property
    def most_identifiable(self):
        return max(self.per_identity, key=lambda entry: entry["recall"])

    def to_dict(self):
        return {
            "accuracy": round(self.accuracy, 4),
            "chance": round(self.chance, 4),
            "lift_over_chance": round(self.lift, 2),
            "n_identities": self.n_identities,
            "n_records": self.n_records,
            "recall_ci": [
                None if v is None else round(v, 4) for v in self.recall_ci
            ],
            "most_identifiable": self.most_identifiable,
            "per_identity": self.per_identity,
            "attack_model": self.attack_model,
        }


def identity_attack(
    df,
    feature_columns,
    identity_column=SUBJECT_COLUMN,
    session_column=None,
    model_kind="random_forest",
    n_splits=5,
    seed=0,
):
    """Try to name the producer of each held-out record.

    Without session_column, held-out records are drawn from the same recordings
    as the reference set. That measures linkage, but it cannot separate a
    person's fingerprint from their electrode placement on the day.

    With session_column, whole recordings are held out while every identity
    stays in the reference set. Anything specific to a single recording is then
    useless to the attacker, so a result that survives belongs to the person.
    """
    identities = df[identity_column].to_numpy()
    unique = np.unique(identities)
    if len(unique) < 2:
        raise ValueError(
            f"re-identification needs at least 2 identities, found {len(unique)}"
        )

    smallest = df[identity_column].value_counts().min()
    if smallest < n_splits:
        raise ValueError(
            f"re-identification with {n_splits} folds needs at least {n_splits} "
            f"records per identity; the smallest has {smallest}"
        )

    X = df[feature_columns].to_numpy()
    model = build_model(model_kind, seed=seed)

    if session_column is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        sessions = None
        attack_model = SAME_RECORDING
    else:
        n_sessions = df[session_column].nunique()
        if n_sessions < 2:
            raise ValueError(
                f"a cross-recording attack needs at least 2 recordings per "
                f"identity, found {n_sessions} in {session_column!r}"
            )
        splitter = LeaveOneGroupOut()
        sessions = df[session_column].to_numpy()
        attack_model = CROSS_RECORDING

    truth, guessed = [], []
    for train_idx, test_idx in splitter.split(X, identities, sessions):
        estimator = clone(model)
        estimator.fit(X[train_idx], identities[train_idx])
        truth.append(identities[test_idx])
        guessed.append(estimator.predict(X[test_idx]))
    truth, guessed = np.concatenate(truth), np.concatenate(guessed)

    accuracy = float(np.mean(truth == guessed))
    chance = 1.0 / len(unique)

    per_identity = [
        {
            "group": identity.item() if hasattr(identity, "item") else identity,
            "n_records": int((truth == identity).sum()),
            "recall": round(float(np.mean(guessed[truth == identity] == identity)), 4),
        }
        for identity in unique
    ]

    return IdentityResult(
        accuracy=accuracy,
        chance=chance,
        lift=accuracy / chance,
        n_identities=len(unique),
        n_records=len(df),
        per_identity=sorted(per_identity, key=lambda e: -e["recall"]),
        attack_model=attack_model,
    )
