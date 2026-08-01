"""What is the fingerprint actually made of? (spec 6.2, 18)

A re-identification result invites one obvious deflation: that the model is
reading skull thickness, electrode impedance, cap fit or hair rather than
anything neural. Those show up as per-channel amplitude. Spectral shape - how a
channel's power is distributed across bands - survives dividing amplitude out.

Splitting the features into those two parts and attacking each separately is
what makes the deflation testable instead of arguable.
"""

import numpy as np
import pandas as pd
import pytest

from nerveml.confounds import amplitude_features, shape_features, split_fingerprint


def band_table(person_amplitude, person_shape, n_subjects=8, n_records=20, seed=0):
    """Log band power where the fingerprint sits in amplitude, shape, or neither.

    Four channels x five bands, matching the real feature layout. Amplitude is
    a per-channel offset applied to every band alike; shape tilts power between
    bands while leaving each channel's total alone.
    """
    channels = ["C3", "C4", "Cz", "Pz"]
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    rng = np.random.default_rng(seed)

    amp = rng.normal(0, 2.0, size=(n_subjects, len(channels)))
    tilt = rng.normal(0, 2.0, size=(n_subjects, len(bands)))
    # A tilt whose bands did not sum to zero would raise every channel's total
    # by the same per-person amount, which is an amplitude fingerprint wearing
    # a shape costume. Centring it is what makes "shape only" mean that.
    tilt -= tilt.mean(axis=1, keepdims=True)

    rows = []
    for s in range(n_subjects):
        block = rng.normal(0, 0.3, size=(n_records, len(channels), len(bands)))
        if person_amplitude:
            block += amp[s][None, :, None]
        if person_shape:
            block += tilt[s][None, None, :]
        frame = pd.DataFrame(
            block.reshape(n_records, -1),
            columns=[f"{c}_{b}" for c in channels for b in bands],
        )
        frame.insert(0, "subject_id", f"s{s:02d}")
        rows.append(frame)

    df = pd.concat(rows, ignore_index=True)
    return df, [c for c in df.columns if c != "subject_id"]


def test_amplitude_features_are_one_per_channel():
    df, feats = band_table(True, True)

    amplitude = amplitude_features(df, feats)

    assert list(amplitude.columns) == ["C3", "C4", "Cz", "Pz"]
    assert len(amplitude) == len(df)


def test_shape_features_keep_every_channel_band():
    df, feats = band_table(True, True)

    shape = shape_features(df, feats)

    assert list(shape.columns) == feats


def test_shape_removes_a_channels_overall_level():
    df, feats = band_table(True, True)

    shape = shape_features(df, feats)

    # Each channel's bands sum to the same thing for every row, so nothing
    # about its overall level can survive into the shape features.
    totals = shape[[c for c in feats if c.startswith("C3_")]].sum(axis=1)
    assert totals.std() < 1e-9


def test_shifting_a_channels_level_does_not_move_its_shape():
    df, feats = band_table(True, True)
    louder = df.copy()
    for column in [c for c in feats if c.startswith("Cz_")]:
        louder[column] = louder[column] + 3.0

    before = shape_features(df, feats)
    after = shape_features(louder, feats)

    assert np.allclose(before.to_numpy(), after.to_numpy())


def test_an_amplitude_fingerprint_is_found_by_amplitude_and_not_by_shape():
    df, feats = band_table(person_amplitude=True, person_shape=False)

    result = split_fingerprint(df, feats, model_kind="logistic_regression", seed=0)

    assert result.amplitude.accuracy > 0.9
    assert result.shape.accuracy < 0.4


def test_a_shape_fingerprint_is_found_by_shape_and_not_by_amplitude():
    df, feats = band_table(person_amplitude=False, person_shape=True)

    result = split_fingerprint(df, feats, model_kind="logistic_regression", seed=0)

    assert result.shape.accuracy > 0.9
    assert result.amplitude.accuracy < 0.4


def test_the_full_feature_set_finds_either():
    amp_only, feats = band_table(person_amplitude=True, person_shape=False)
    shape_only, _ = band_table(person_amplitude=False, person_shape=True)

    a = split_fingerprint(amp_only, feats, model_kind="logistic_regression", seed=0)
    b = split_fingerprint(shape_only, feats, model_kind="logistic_regression", seed=0)

    assert a.full.accuracy > 0.9
    assert b.full.accuracy > 0.9


def test_the_verdict_names_which_component_carries_it():
    amp_only, feats = band_table(person_amplitude=True, person_shape=False)
    shape_only, _ = band_table(person_amplitude=False, person_shape=True)

    assert split_fingerprint(
        amp_only, feats, model_kind="logistic_regression", seed=0
    ).carried_by == "amplitude"
    assert split_fingerprint(
        shape_only, feats, model_kind="logistic_regression", seed=0
    ).carried_by == "spectral_shape"


def test_a_fingerprint_in_both_is_reported_as_both():
    df, feats = band_table(person_amplitude=True, person_shape=True)

    assert split_fingerprint(
        df, feats, model_kind="logistic_regression", seed=0
    ).carried_by == "both"


def test_result_serialises_to_plain_types():
    import json

    df, feats = band_table(True, True)

    json.dumps(split_fingerprint(df, feats, model_kind="logistic_regression", seed=0).to_dict())


def test_features_that_are_not_channel_band_pairs_are_rejected():
    df = pd.DataFrame({"subject_id": ["a"] * 4, "f00": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="channel_band"):
        amplitude_features(df, ["f00"])
