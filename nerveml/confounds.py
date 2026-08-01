"""Splitting a fingerprint into the part that is physical and the part that is not.

A re-identification result has an obvious deflation waiting for it: that the
model reads skull thickness, electrode impedance, cap fit or hair rather than
anything neural. Every one of those changes how much power a channel records,
not how that power is distributed across frequencies.

So the features split in two. Amplitude is each channel's overall level.
Spectral shape is what is left once that level is divided out - the same
channel, the same total, only the balance between bands. A fingerprint that
survives in shape alone cannot be explained by how well the electrode was
seated.

This does not make the surviving part neural. It removes one specific and very
likely alternative, which is what a confound baseline is for.
"""

from dataclasses import dataclass

import pandas as pd

from nerveml.identity import identity_attack

# How far the weaker component has to trail the stronger before the fingerprint
# is attributed to one of them rather than to both.
DOMINANCE_MARGIN = 0.15


def _channels_and_bands(feature_columns):
    """Parse channel_band feature names into an ordered channel -> columns map."""
    grouped = {}
    for column in feature_columns:
        channel, _, band = column.rpartition("_")
        if not channel or not band:
            raise ValueError(
                f"expected channel_band feature names, got {column!r}; this "
                "decomposition only applies to band-power features"
            )
        grouped.setdefault(channel, []).append(column)
    return grouped


def amplitude_features(df, feature_columns):
    """One column per channel: its overall level, with band structure summed away.

    Log band power sums to a monotone function of total power, which is what
    impedance and anatomy move.
    """
    grouped = _channels_and_bands(feature_columns)
    return pd.DataFrame(
        {channel: df[columns].sum(axis=1) for channel, columns in grouped.items()},
        index=df.index,
    )


def shape_features(df, feature_columns):
    """The same features with each channel's overall level divided out.

    Centring a channel's bands on their own mean is division in the log domain.
    Every channel then sums to zero for every row, so nothing about how loudly
    it recorded can survive.
    """
    grouped = _channels_and_bands(feature_columns)
    parts = {}
    for columns in grouped.values():
        block = df[columns]
        centred = block.sub(block.mean(axis=1), axis=0)
        for column in columns:
            parts[column] = centred[column]
    return pd.DataFrame(parts, index=df.index)[list(feature_columns)]


@dataclass
class FingerprintSplit:
    full: object
    amplitude: object
    shape: object

    @property
    def carried_by(self):
        """Which component the identifiability rides on."""
        amplitude, shape = self.amplitude.accuracy, self.shape.accuracy
        if amplitude - shape >= DOMINANCE_MARGIN:
            return "amplitude"
        if shape - amplitude >= DOMINANCE_MARGIN:
            return "spectral_shape"
        return "both"

    @property
    def interpretation(self):
        return {
            "amplitude": (
                "Identity rides on per-channel amplitude, which is what skull "
                "thickness, electrode impedance and cap fit move. The result is "
                "consistent with a physical fingerprint rather than a neural one."
            ),
            "spectral_shape": (
                "Identity survives with each channel's overall level divided "
                "out, so it is not explained by how well the electrodes were "
                "seated. That does not make it neural; it removes the most "
                "likely alternative."
            ),
            "both": (
                "Amplitude and spectral shape each identify individuals on "
                "their own, so a physical explanation covers part of the result "
                "but not all of it."
            ),
        }[self.carried_by]

    def to_dict(self):
        return {
            "carried_by": self.carried_by,
            "interpretation": self.interpretation,
            "full": self.full.to_dict(),
            "amplitude_only": self.amplitude.to_dict(),
            "spectral_shape_only": self.shape.to_dict(),
        }


def split_fingerprint(
    df,
    feature_columns,
    identity_column="subject_id",
    session_column=None,
    model_kind="random_forest",
    n_splits=5,
    seed=0,
):
    """Attack identity three times: on everything, on amplitude, on shape."""
    keep = [identity_column] + (
        [session_column] if session_column and session_column in df.columns else []
    )

    def attack(features):
        table = pd.concat([df[keep].reset_index(drop=True),
                           features.reset_index(drop=True)], axis=1)
        return identity_attack(
            table,
            list(features.columns),
            identity_column=identity_column,
            session_column=session_column,
            model_kind=model_kind,
            n_splits=n_splits,
            seed=seed,
        )

    return FingerprintSplit(
        full=attack(df[list(feature_columns)]),
        amplitude=attack(amplitude_features(df, feature_columns)),
        shape=attack(shape_features(df, feature_columns)),
    )
