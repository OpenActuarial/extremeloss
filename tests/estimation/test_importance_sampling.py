from __future__ import annotations

import numpy as np
import pytest

from extremeloss.estimation.importance_sampling import (
    effective_sample_size,
    estimate_tail_probability_is,
    estimate_tvar_is,
    estimate_var_is,
    normalized_weights,
)


def test_normalized_weights_and_effective_sample_size():
    weights = np.array([1.0, 1.0, 2.0])
    normed = normalized_weights(weights)

    np.testing.assert_allclose(normed, np.array([0.25, 0.25, 0.5]))
    assert effective_sample_size(weights) == pytest.approx(1.0 / np.sum(normed**2))


def test_importance_sampling_tail_probability_matches_weighted_indicator_mean():
    losses = np.array([1.0, 2.0, 10.0, 20.0])
    weights = np.array([1.0, 1.0, 3.0, 5.0])

    result = estimate_tail_probability_is(losses, weights, threshold=9.0)

    expected = (3.0 + 5.0) / np.sum(weights)
    assert result.method == 'importance_sampling'
    assert result.estimate == pytest.approx(expected)
    assert result.effective_n is not None


def test_importance_sampling_var_and_tvar_are_well_ordered():
    losses = np.array([1.0, 2.0, 10.0, 20.0])
    weights = np.array([1.0, 1.0, 3.0, 5.0])

    var_result = estimate_var_is(losses, weights, q=0.8)
    tvar_result = estimate_tvar_is(losses, weights, q=0.8)

    assert var_result.method == 'importance_sampling'
    assert tvar_result.method == 'importance_sampling'
    assert tvar_result.estimate >= var_result.estimate
    assert 0.0 < tvar_result.diagnostics['tail_weight'] <= 1.0


def test_importance_sampling_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        estimate_var_is([1.0, 2.0], [1.0], q=0.9)
