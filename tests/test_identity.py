"""Spec 6.2 — can the signal be used to tell who produced it?

The attack model matters and is not the same as the validity protocol. Holding
a person out entirely is meaningless here: nobody can name a stranger. The
threat is linkage — an attacker holds labelled reference recordings and wants
to attach new anonymous records to those same people. So the split keeps every
identity in training and holds out unseen *records* of them.

California's de-identification exemption turns on whether data can be linked to
a person, and the statute permits attempting re-identification precisely to
check that. This is the measurement that answers it.
"""

import pytest

from nerveml.identity import identity_attack
from nerveml.synth import feature_columns, make_synthetic


@pytest.fixture(scope="module")
def fingerprinted():
    """Strong per-person offsets: identity should be trivially recoverable."""
    df = make_synthetic("subject_leakage", n_subjects=20, n_trials=40, seed=0)
    return df, feature_columns(df)


@pytest.fixture(scope="module")
def faint():
    """Per-person offsets one fifth the size of trial noise."""
    df = make_synthetic("true_signal", n_subjects=20, n_trials=40, seed=0)
    return df, feature_columns(df)


def test_a_fingerprinted_dataset_is_re_identifiable(fingerprinted):
    df, feats = fingerprinted

    result = identity_attack(df, feats, model_kind="logistic_regression", seed=0)

    assert result.accuracy > 0.9
    assert result.chance == pytest.approx(1 / 20)


def test_re_identification_is_reported_relative_to_chance(fingerprinted):
    df, feats = fingerprinted

    result = identity_attack(df, feats, model_kind="logistic_regression", seed=0)

    # Raw accuracy is meaningless without the number of identities in play.
    assert result.lift == pytest.approx(result.accuracy / result.chance)
    assert result.n_identities == 20


def test_a_faint_fingerprint_scores_far_lower(fingerprinted, faint):
    strong = identity_attack(*fingerprinted, model_kind="logistic_regression", seed=0)
    weak = identity_attack(*faint, model_kind="logistic_regression", seed=0)

    assert weak.accuracy < strong.accuracy


def test_every_identity_gets_its_own_recall(fingerprinted):
    df, feats = fingerprinted

    result = identity_attack(df, feats, model_kind="logistic_regression", seed=0)

    # One person being highly identifiable is a finding even if the mean is low.
    assert {e["group"] for e in result.per_identity} == set(df["subject_id"].unique())
    assert all(0.0 <= e["recall"] <= 1.0 for e in result.per_identity)
    assert result.most_identifiable["group"] in set(df["subject_id"].unique())


def test_the_attack_model_is_stated_in_the_result(fingerprinted):
    df, feats = fingerprinted

    result = identity_attack(df, feats, model_kind="logistic_regression", seed=0)

    # A number without its threat model is not evidence of anything.
    assert "reference" in result.attack_model.lower()


def test_result_serialises_to_plain_types(fingerprinted):
    import json

    df, feats = fingerprinted

    json.dumps(identity_attack(df, feats, model_kind="logistic_regression", seed=0).to_dict())


def sessioned(person_bound, n_subjects=10, n_sessions=4, n_records=15, n_features=12):
    """Records tagged with the recording they came from.

    person_bound decides where the fingerprint lives. True: one offset per
    person, identical in every recording, so identity survives a change of
    recording. False: a fresh offset per person *per recording*, which looks
    identical inside one recording and carries nothing across them - the
    electrode-placement artefact that a same-recording result cannot rule out.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    person = rng.normal(0, 3.0, size=(n_subjects, n_features))
    rows = []
    for s in range(n_subjects):
        for session in range(n_sessions):
            offset = person[s] if person_bound else rng.normal(0, 3.0, n_features)
            noise = rng.normal(0, 1, size=(n_records, n_features))
            frame = pd.DataFrame(
                offset + noise, columns=[f"f{i:02d}" for i in range(n_features)]
            )
            frame.insert(0, "subject_id", f"s{s:02d}")
            frame.insert(1, "session_id", session)
            rows.append(frame)
    df = pd.concat(rows, ignore_index=True)
    return df, [c for c in df.columns if c.startswith("f")]


def test_a_person_bound_fingerprint_survives_a_change_of_recording():
    df, feats = sessioned(person_bound=True)

    result = identity_attack(
        df, feats, session_column="session_id", model_kind="logistic_regression", seed=0
    )

    assert result.accuracy > 0.9
    assert "held-out recording" in result.attack_model.lower()


def test_a_recording_bound_fingerprint_does_not_survive():
    df, feats = sessioned(person_bound=False)

    result = identity_attack(
        df, feats, session_column="session_id", model_kind="logistic_regression", seed=0
    )

    # Same data would score near-perfectly with a same-recording split. This is
    # the control that tells a person apart from their electrode placement.
    assert result.accuracy < 0.3


def test_a_same_recording_split_cannot_rule_out_the_artefact():
    stable, feats = sessioned(person_bound=True)
    unstable, _ = sessioned(person_bound=False)

    same_stable = identity_attack(stable, feats, model_kind="logistic_regression", seed=0)
    same_unstable = identity_attack(
        unstable, feats, model_kind="logistic_regression", seed=0
    )
    cross_unstable = identity_attack(
        unstable, feats, session_column="session_id",
        model_kind="logistic_regression", seed=0,
    )

    # Both look like a strong result against a chance of 0.1, so a
    # same-recording number on its own is consistent with either explanation.
    assert same_stable.accuracy > 5 * same_stable.chance
    assert same_unstable.accuracy > 5 * same_unstable.chance

    # Holding out the recording is what separates them, and it costs the
    # artefact-driven dataset most of its apparent identifiability.
    assert cross_unstable.accuracy < same_unstable.accuracy / 2


def test_session_aware_attack_needs_more_than_one_recording():
    df, feats = sessioned(person_bound=True, n_sessions=1)

    with pytest.raises(ValueError, match="at least 2 recordings"):
        identity_attack(df, feats, session_column="session_id", seed=0)


def test_too_few_records_per_identity_is_rejected(fingerprinted):
    df, feats = fingerprinted
    thin = df.groupby("subject_id", group_keys=False).head(2)

    with pytest.raises(ValueError, match="records per identity"):
        identity_attack(thin, feats, n_splits=5, seed=0)


def test_a_single_identity_is_rejected(fingerprinted):
    df, feats = fingerprinted

    with pytest.raises(ValueError, match="at least 2 identities"):
        identity_attack(df[df["subject_id"] == "s00"], feats, seed=0)
