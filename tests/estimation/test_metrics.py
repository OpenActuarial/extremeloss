from __future__ import annotations

import numpy as np

from extremeloss.estimation.metrics import (
    empirical_tvar,
    empirical_var,
    exceedance_curve,
    exceedance_probability,
    survival_function,
)


def test_empirical_var_and_tvar_match_known_values():
    losses = np.array([1.0, 2.0, 3.0, 4.0, 100.0])

    var_80 = empirical_var(losses, 0.8)
    tvar_80 = empirical_tvar(losses, 0.8)

    assert var_80 == np.quantile(losses, 0.8)
    assert tvar_80 == 100.0


def test_exceedance_probability_and_curve_are_consistent():
    losses = np.array([1.0, 2.0, 3.0, 4.0])
    thresholds = np.array([0.0, 2.5, 10.0])

    prob = exceedance_probability(losses, 2.5)
    curve = exceedance_curve(losses, thresholds)
    surv = survival_function(losses, thresholds)

    assert prob == 0.5
    np.testing.assert_allclose(curve['thresholds'], thresholds)
    np.testing.assert_allclose(curve['probabilities'], np.array([1.0, 0.5, 0.0]))
    np.testing.assert_allclose(curve['probabilities'], surv['probabilities'])
