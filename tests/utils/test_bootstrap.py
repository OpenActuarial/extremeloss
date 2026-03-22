from __future__ import annotations

import numpy as np

from extremeloss.utils.bootstrap import (
    bootstrap_statistic,
    bootstrap_tail_probability,
    bootstrap_tvar,
    bootstrap_var,
)


def test_bootstrap_statistic_returns_expected_shape_and_ci():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    result = bootstrap_statistic(data, np.mean, n_resamples=50, random_state=123)
    assert result.bootstrap_estimates.shape == (50,)
    assert result.ci is not None
    assert result.stderr is not None


def test_bootstrap_var_and_tvar_return_valid_results():
    data = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    var_result = bootstrap_var(data, q=0.8, n_resamples=50, random_state=123)
    tvar_result = bootstrap_tvar(data, q=0.8, n_resamples=50, random_state=123)
    assert tvar_result.estimate >= var_result.estimate
    assert len(var_result.bootstrap_estimates) == 50


def test_bootstrap_tail_probability_is_reasonable():
    data = np.array([1.0, 2.0, 10.0, 20.0])
    result = bootstrap_tail_probability(data, threshold=9.0, n_resamples=40, random_state=123)
    assert 0.0 <= result.estimate <= 1.0
    assert len(result.bootstrap_estimates) == 40
