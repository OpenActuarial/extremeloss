from __future__ import annotations

import numpy as np
import pytest

from extremeloss.evt.tail_index import hill_curve, hill_estimator, pickands_estimator


def test_hill_estimator_is_positive_for_pareto_sample():
    rng = np.random.default_rng(123)
    data = rng.pareto(a=2.0, size=5000) + 1.0

    estimate = hill_estimator(data, k=200)

    assert estimate > 0.0
    assert estimate == pytest.approx(0.5, abs=0.12)


def test_pickands_estimator_is_finite_for_pareto_sample():
    rng = np.random.default_rng(456)
    data = rng.pareto(a=2.0, size=8000) + 1.0

    estimate = pickands_estimator(data, k=200)

    assert np.isfinite(estimate)
    assert estimate == pytest.approx(0.5, abs=0.2)


def test_hill_curve_returns_matching_arrays():
    rng = np.random.default_rng(1)
    data = rng.pareto(a=3.0, size=2000) + 1.0

    curve = hill_curve(data, k_grid=[10, 20, 30])

    np.testing.assert_array_equal(curve['k'], np.array([10, 20, 30]))
    assert curve['hill'].shape == (3,)
