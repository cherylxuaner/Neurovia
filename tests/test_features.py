"""Band-power features from raw epochs (spec 9, feature pipeline).

Tested against synthetic signals whose spectral content is known by
construction: a pure 10 Hz sine must put its power in alpha and nowhere else.
Real EEG cannot check this, because nobody knows what is in it.
"""

import numpy as np
import pytest

from nerveml.features import BANDS, band_power_features, feature_names


@pytest.fixture
def sampling_rate():
    return 128.0


def sine_epochs(freq, n_epochs, n_channels, n_times, rate, seed=0):
    """Epochs of a pure sine at freq, shaped (epochs, channels, samples)."""
    t = np.arange(n_times) / rate
    wave = np.sin(2 * np.pi * freq * t)
    rng = np.random.default_rng(seed)
    return wave + rng.normal(0, 0.01, size=(n_epochs, n_channels, n_times))


def test_feature_names_pair_every_channel_with_every_band():
    names = feature_names(["Fp1", "Cz"])

    assert len(names) == 2 * len(BANDS)
    assert "Fp1_alpha" in names
    assert "Cz_gamma" in names


def test_output_is_one_row_per_epoch_and_one_column_per_channel_band(sampling_rate):
    epochs = sine_epochs(10.0, n_epochs=7, n_channels=3, n_times=256, rate=sampling_rate)

    features = band_power_features(epochs, sampling_rate)

    assert features.shape == (7, 3 * len(BANDS))


def test_a_ten_hertz_sine_puts_its_power_in_alpha(sampling_rate):
    epochs = sine_epochs(10.0, n_epochs=4, n_channels=1, n_times=512, rate=sampling_rate)

    features = band_power_features(epochs, sampling_rate)
    names = feature_names(["ch0"])
    by_band = dict(zip(names, features.mean(axis=0)))

    assert max(by_band, key=by_band.get) == "ch0_alpha"


def test_a_twenty_five_hertz_sine_puts_its_power_in_beta(sampling_rate):
    epochs = sine_epochs(25.0, n_epochs=4, n_channels=1, n_times=512, rate=sampling_rate)

    features = band_power_features(epochs, sampling_rate)
    names = feature_names(["ch0"])
    by_band = dict(zip(names, features.mean(axis=0)))

    assert max(by_band, key=by_band.get) == "ch0_beta"


def test_features_are_log_scaled_so_they_are_comparable_across_channels(sampling_rate):
    loud = sine_epochs(10.0, 4, 1, 512, sampling_rate) * 100
    quiet = sine_epochs(10.0, 4, 1, 512, sampling_rate)

    louder = band_power_features(loud, sampling_rate)
    quieter = band_power_features(quiet, sampling_rate)

    # A hundredfold amplitude is a constant offset in log power, not a
    # hundredfold feature value.
    assert (louder - quieter).std() < 0.5
    assert louder.mean() > quieter.mean()


def test_bands_cover_the_standard_ranges_without_overlapping():
    edges = [BANDS[name] for name in BANDS]

    for low, high in edges:
        assert low < high
    for (_, high), (low, _) in zip(edges, edges[1:]):
        assert high <= low


def test_channel_count_must_match_the_epochs(sampling_rate):
    epochs = sine_epochs(10.0, 4, 3, 256, sampling_rate)

    with pytest.raises(ValueError, match="channel names"):
        band_power_features(epochs, sampling_rate, channel_names=["only_one"])
