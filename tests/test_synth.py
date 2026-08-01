"""The synthetic generator is the ground truth for every other test.

Two datasets, two known answers:

  subject_leakage - labels depend only on WHO the subject is, never on the
                    signal itself. A naive trial-random split can score well by
                    memorising subjects; a subject-held-out split cannot.
  true_signal     - labels depend on a feature direction shared by everyone.
                    Both splits should score well.

If these tests do not hold, the pipeline tests downstream prove nothing.
"""

import numpy as np
import pytest

from nerveml.synth import feature_columns, make_synthetic


def subject_label_rates(df):
    return df.groupby("subject_id")["target_label"].mean()


def identity_encoding_ratio(df):
    """Between-subject variance divided by within-subject variance.

    High means the features carry a strong per-person fingerprint.
    """
    feats = df[feature_columns(df)]
    subject_means = feats.groupby(df["subject_id"]).mean()
    between = subject_means.var(axis=0).mean()
    within = feats.groupby(df["subject_id"]).var().mean(axis=0).mean()
    return between / within


def test_subject_leakage_dataset_has_expected_shape():
    df = make_synthetic("subject_leakage", n_subjects=10, n_trials=20, n_features=8, seed=0)

    assert len(df) == 200
    assert df["subject_id"].nunique() == 10
    assert df["trial_id"].nunique() == 200
    assert len(feature_columns(df)) == 8
    assert set(df["target_label"].unique()) == {0, 1}


def test_subject_leakage_labels_are_driven_by_the_subject():
    df = make_synthetic("subject_leakage", n_subjects=20, n_trials=40, n_features=16, seed=0)

    rates = subject_label_rates(df)

    # Every subject leans hard one way, so knowing the subject nearly
    # determines the label.
    assert (np.abs(rates - 0.5) > 0.25).all()
    # But across the whole dataset the classes stay balanced, so a majority
    # classifier gains nothing.
    assert df["target_label"].mean() == pytest.approx(0.5, abs=0.02)


def test_subject_leakage_features_carry_a_subject_fingerprint():
    df = make_synthetic("subject_leakage", n_subjects=20, n_trials=40, n_features=16, seed=0)

    # Person-to-person differences dominate trial-to-trial noise, which is what
    # makes subject identity learnable in the first place.
    assert identity_encoding_ratio(df) > 3.0


def test_true_signal_labels_are_balanced_inside_every_subject():
    df = make_synthetic("true_signal", n_subjects=20, n_trials=40, n_features=16, seed=0)

    rates = subject_label_rates(df)

    # No subject leans one way, so subject identity is useless as a predictor.
    assert (np.abs(rates - 0.5) < 0.2).all()


def test_true_signal_features_do_not_carry_a_strong_subject_fingerprint():
    df = make_synthetic("true_signal", n_subjects=20, n_trials=40, n_features=16, seed=0)

    assert identity_encoding_ratio(df) < 1.0


def test_same_seed_produces_an_identical_dataset():
    a = make_synthetic("subject_leakage", n_subjects=5, n_trials=10, n_features=4, seed=7)
    b = make_synthetic("subject_leakage", n_subjects=5, n_trials=10, n_features=4, seed=7)

    assert a.equals(b)


def test_different_seeds_produce_different_datasets():
    a = make_synthetic("subject_leakage", n_subjects=5, n_trials=10, n_features=4, seed=7)
    b = make_synthetic("subject_leakage", n_subjects=5, n_trials=10, n_features=4, seed=8)

    assert not a[feature_columns(a)].equals(b[feature_columns(b)])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown mode"):
        make_synthetic("banana", n_subjects=5, n_trials=10, n_features=4, seed=0)
