from __future__ import annotations

import numpy as np
import pytest

from extremeloss.evt.gpd import fit_gpd, gpd_tail_probability, gpd_tvar, gpd_var


def test_fit_gpd_returns_fit_object_with_expected_fields():
    rng = np.random.default_rng(123)
    excesses = rng.exponential(scale=2.0, size=2000)

    fit = fit_gpd(excesses)

    assert fit.n_exceedances == 2000
    assert fit.fit_method == 'mle'
    assert fit.beta > 0.0
    assert abs(fit.xi) < 0.15


def test_gpd_tail_probability_decreases_above_threshold():
    p1 = gpd_tail_probability(12.0, threshold=10.0, xi=0.2, beta=5.0, exceedance_fraction=0.1)
    p2 = gpd_tail_probability(20.0, threshold=10.0, xi=0.2, beta=5.0, exceedance_fraction=0.1)

    assert p2 < p1 < 0.1


def test_gpd_var_and_tvar_are_ordered_in_extreme_tail():
    var_99 = gpd_var(0.99, threshold=10.0, xi=0.2, beta=5.0, exceedance_fraction=0.05)
    tvar_99 = gpd_tvar(0.99, threshold=10.0, xi=0.2, beta=5.0, exceedance_fraction=0.05)

    assert tvar_99 > var_99 > 10.0


def test_gpd_var_rejects_non_tail_probability():
    with pytest.raises(ValueError):
        gpd_var(0.9, threshold=10.0, xi=0.2, beta=5.0, exceedance_fraction=0.05)
