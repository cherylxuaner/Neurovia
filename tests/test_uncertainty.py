"""Confidence intervals over the unit of independence.

Subjects are independent; the trials inside one are not. So the interval is
built by resampling subjects, not trials or folds, and it needs no refitting
because each subject's score has already been computed.
"""

import numpy as np
import pytest

from nerveml.uncertainty import bootstrap_ci, units_needed


def test_the_interval_brackets_the_mean():
    values = [0.4, 0.5, 0.6, 0.55, 0.45, 0.52, 0.48, 0.61]

    low, high = bootstrap_ci(values, seed=0)

    assert low < np.mean(values) < high


def test_more_spread_gives_a_wider_interval():
    tight = [0.50, 0.51, 0.49, 0.50, 0.51, 0.49, 0.50, 0.51]
    loose = [0.10, 0.90, 0.30, 0.70, 0.20, 0.80, 0.40, 0.60]

    tight_low, tight_high = bootstrap_ci(tight, seed=0)
    loose_low, loose_high = bootstrap_ci(loose, seed=0)

    assert (tight_high - tight_low) < (loose_high - loose_low)


def test_more_units_gives_a_narrower_interval():
    rng = np.random.default_rng(0)
    few = rng.normal(0.6, 0.1, size=8)
    many = rng.normal(0.6, 0.1, size=200)

    few_low, few_high = bootstrap_ci(few, seed=0)
    many_low, many_high = bootstrap_ci(many, seed=0)

    assert (many_high - many_low) < (few_high - few_low)


def test_a_higher_confidence_level_gives_a_wider_interval():
    values = [0.4, 0.5, 0.6, 0.55, 0.45, 0.52, 0.48, 0.61]

    ninety_five = bootstrap_ci(values, confidence=0.95, seed=0)
    ninety_nine = bootstrap_ci(values, confidence=0.99, seed=0)

    assert ninety_nine[0] <= ninety_five[0]
    assert ninety_nine[1] >= ninety_five[1]


def test_the_same_seed_reproduces_the_interval():
    values = [0.4, 0.5, 0.6, 0.55, 0.45]

    assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)


def test_missing_values_are_dropped_not_propagated():
    values = [0.4, float("nan"), 0.6, 0.5, None, 0.55]

    low, high = bootstrap_ci(values, seed=0)

    assert not np.isnan(low) and not np.isnan(high)


def test_too_few_units_yields_no_interval():
    # One number carries no information about how it would vary.
    assert bootstrap_ci([0.5], seed=0) == (None, None)
    assert bootstrap_ci([], seed=0) == (None, None)


def test_a_paired_difference_is_just_the_interval_of_the_differences():
    naive = np.array([0.80, 0.85, 0.78, 0.82])
    grouped = np.array([0.50, 0.52, 0.48, 0.51])

    low, high = bootstrap_ci(naive - grouped, seed=0)

    assert low > 0  # every subject lost ground, so the gap is clearly positive


def test_units_needed_scales_with_the_square_of_the_shortfall():
    # Halving an interval's half-width takes four times the units.
    assert units_needed(n_units=20, half_width=0.20, target_half_width=0.10) == 80


def test_units_needed_is_never_below_what_you_already_have():
    assert units_needed(n_units=20, half_width=0.05, target_half_width=0.10) == 20


def test_units_needed_handles_a_target_of_zero():
    with pytest.raises(ValueError, match="positive"):
        units_needed(n_units=20, half_width=0.2, target_half_width=0.0)
