from __future__ import annotations

import numpy as np

from extremeloss.estimation.rare_event import (
    estimate_tail_probability,
    estimate_tvar,
    estimate_var,
    estimate_var_tvar,
)


class ConstantSampler:
    def sample(self, size: int = 1) -> np.ndarray:
        return np.full(size, 10.0)


def test_estimate_tail_probability_on_array_data():
    losses = np.array([1.0, 5.0, 10.0, 20.0])
    result = estimate_tail_probability(losses, threshold=9.0)

    assert result.method == 'empirical'
    assert result.estimate == 0.5
    assert result.n == 4
    assert result.threshold == 9.0
    assert result.diagnostics['n_exceedances'] == 2


def test_estimate_tail_probability_accepts_sampler_objects():
    result = estimate_tail_probability(ConstantSampler(), threshold=9.0, size=6)

    assert result.estimate == 1.0
    assert result.n == 6


def test_estimate_var_and_tvar_on_simple_tail():
    losses = np.array([1.0, 2.0, 3.0, 4.0, 100.0])

    var_result = estimate_var(losses, 0.8)
    tvar_result = estimate_tvar(losses, 0.8)

    assert var_result.estimate == np.quantile(losses, 0.8)
    assert tvar_result.estimate == 100.0
    assert tvar_result.diagnostics['tail_sample_size'] == 1


def test_estimate_var_tvar_returns_expected_keys():
    losses = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    out = estimate_var_tvar(losses, 0.8)

    assert set(out) == {'var', 'tvar', 'tail_probability'}
    assert out['var'].quantile == 0.8
    assert out['tvar'].quantile == 0.8
    assert out['tail_probability'].estimate == 0.2
