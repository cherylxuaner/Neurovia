"""Confidence intervals by resampling the unit of independence.

Every threshold in this product is applied to an estimate from a handful of
people. Twenty subjects is a small sample, and a verdict that flips when a
different twenty walk through the door is not a verdict. The intervals here are
what let the report say so.

They cost nothing. Each unit's score is already computed by the evaluation, so
resampling those numbers needs no refitting - ten thousand resamples take
milliseconds, against minutes for even one extra pass of the pipeline.
"""

import numpy as np

DEFAULT_RESAMPLES = 10_000


def bootstrap_ci(values, confidence=0.95, n_resamples=DEFAULT_RESAMPLES, seed=0):
    """Percentile interval for the mean of per-unit scores.

    Resamples units with replacement, which is the right unit because trials
    inside one person are not independent of each other. Returns (None, None)
    when there is nothing to resample: a single number says nothing about how
    it would vary.
    """
    clean = np.array(
        [v for v in values if v is not None and not np.isnan(float(v))], dtype=float
    )
    if len(clean) < 2:
        return (None, None)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clean), size=(n_resamples, len(clean)))
    means = clean[draws].mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [100 * tail, 100 * (1 - tail)])
    return (float(low), float(high))


def units_needed(n_units, half_width, target_half_width):
    """Roughly how many units would shrink an interval to the target width.

    A bootstrap interval narrows with the square root of the sample, so
    quartering the width costs sixteen times the people. This is an
    order-of-magnitude guide for planning a follow-up, not a power calculation.
    """
    if target_half_width <= 0:
        raise ValueError("target_half_width must be positive")
    if half_width <= target_half_width:
        return int(n_units)
    return int(np.ceil(n_units * (half_width / target_half_width) ** 2))
