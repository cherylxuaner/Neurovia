"""Log band power per channel, from raw epochs (spec 10.2).

Welch's method on each epoch, integrated over the standard EEG bands, then
logged. The log matters: raw power spans orders of magnitude across channels
and subjects, so without it a single high-amplitude channel dominates every
distance and every split, and amplitude differences between people masquerade
as signal.

scipy alone is enough here. MNE is only needed to read raw recordings, not to
turn epochs into features.
"""

import numpy as np
from scipy.signal import welch

# Standard EEG bands, contiguous and non-overlapping, in Hz.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# Guards the logarithm against an empty band at the Nyquist edge.
_FLOOR = 1e-12


def feature_names(channel_names):
    """One name per channel-band pair, in the order band_power_features emits."""
    return [f"{channel}_{band}" for channel in channel_names for band in BANDS]


def band_power_features(epochs, sampling_rate, channel_names=None):
    """Log band power for each epoch.

    epochs is (n_epochs, n_channels, n_samples). Returns
    (n_epochs, n_channels * len(BANDS)), column order matching feature_names.
    """
    epochs = np.asarray(epochs, dtype=float)
    if epochs.ndim != 3:
        raise ValueError(
            f"expected epochs shaped (n_epochs, n_channels, n_samples), got "
            f"{epochs.shape}"
        )

    n_epochs, n_channels, n_samples = epochs.shape
    if channel_names is not None and len(channel_names) != n_channels:
        raise ValueError(
            f"got {len(channel_names)} channel names for {n_channels} channels"
        )

    # A window that never exceeds the epoch keeps short epochs usable.
    nperseg = min(n_samples, int(sampling_rate * 2))
    freqs, power = welch(epochs, fs=sampling_rate, nperseg=nperseg, axis=-1)

    columns = []
    for channel in range(n_channels):
        for low, high in BANDS.values():
            inside = (freqs >= low) & (freqs < high)
            columns.append(power[:, channel, inside].sum(axis=-1))

    return np.log(np.column_stack(columns) + _FLOOR)
