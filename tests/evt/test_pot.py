from __future__ import annotations

import numpy as np
import pytest

from extremeloss.evt.pot import extract_exceedances, fit_pot


def test_extract_exceedances_returns_shifted_excesses():
    data = np.array([1.0, 5.0, 10.0, 12.0])
    excesses = extract_exceedances(data, threshold=9.0)

    np.testing.assert_allclose(excesses, np.array([1.0, 3.0]))


def test_extract_exceedances_raises_when_none_found():
    with pytest.raises(ValueError):
        extract_exceedances([1.0, 2.0, 3.0], threshold=10.0)


def test_fit_pot_returns_reasonable_exceedance_fraction():
    rng = np.random.default_rng(321)
    data = rng.exponential(scale=5.0, size=5000)
    threshold = 8.0

    fit = fit_pot(data, threshold=threshold)

    assert fit.threshold == threshold
    assert 0.0 < fit.exceedance_fraction < 1.0
    assert fit.n_exceedances > 0
    assert fit.beta > 0.0
