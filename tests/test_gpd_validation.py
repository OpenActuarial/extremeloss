"""Regression tests for GPD parameter validation.

``gpd_tail_probability`` previously validated only the threshold and scale, so a
negative or above-one ``exceedance_fraction`` produced an out-of-range
"probability" (e.g. a negative value for ``x <= threshold``), while ``gpd_var``
two functions over already required ``(0, 1]``. Both now share one validator.
"""

from __future__ import annotations

import math

import pytest

from extremeloss.evt.gpd import gpd_tail_probability, gpd_var
from extremeloss.utils.validation import validate_gpd_params

VALID = dict(threshold=10.0, xi=0.2, beta=3.0)


def test_tail_probability_rejects_out_of_range_exceedance():
    for bad in (-0.1, -1.0, 1.5, math.nan, math.inf):
        with pytest.raises(ValueError):
            gpd_tail_probability(5.0, exceedance_fraction=bad, **VALID)


def test_tail_probability_allows_zero_exceedance_as_zero():
    # a zero exceedance rate is a meaningful tail probability of zero, not an error
    assert gpd_tail_probability(5.0, exceedance_fraction=0.0, **VALID) == 0.0
    assert gpd_tail_probability(50.0, exceedance_fraction=0.0, **VALID) == 0.0


def test_tail_probability_rejects_nonfinite_params():
    with pytest.raises(ValueError):
        gpd_tail_probability(50.0, threshold=10.0, xi=math.nan, beta=3.0, exceedance_fraction=0.1)
    with pytest.raises(ValueError):
        gpd_tail_probability(50.0, threshold=10.0, xi=0.2, beta=math.inf, exceedance_fraction=0.1)
    with pytest.raises(ValueError):
        gpd_tail_probability(50.0, threshold=10.0, xi=0.2, beta=-1.0, exceedance_fraction=0.1)


def test_var_still_requires_strictly_positive_exceedance():
    with pytest.raises(ValueError):
        gpd_var(0.99, exceedance_fraction=0.0, **VALID)
    with pytest.raises(ValueError):
        gpd_var(0.99, exceedance_fraction=-0.2, **VALID)


def test_tail_probability_valid_inputs_unchanged():
    # a normal fit-driven exceedance fraction is unaffected
    p = gpd_tail_probability(50.0, exceedance_fraction=0.05, **VALID)
    assert 0.0 <= p <= 1.0


def test_shared_validator_zero_flag():
    validate_gpd_params(10.0, 0.2, 3.0, 0.0, allow_zero_exceedance=True)  # no raise
    with pytest.raises(ValueError):
        validate_gpd_params(10.0, 0.2, 3.0, 0.0)  # zero not allowed by default
    with pytest.raises(ValueError):
        validate_gpd_params(10.0, 0.2, 3.0, 1.0000001)
