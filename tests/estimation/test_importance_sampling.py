from __future__ import annotations

import numpy as np
import pytest

from extremeloss.estimation.importance_sampling import (
    effective_sample_size,
    estimate_exceedance_curve_is,
    estimate_mean_is,
    estimate_tail_probability_is,
    estimate_tvar_is,
    estimate_var_is,
    estimate_var_tvar_is,
    importance_sampling_diagnostics,
    log_importance_weights,
    normalized_weights,
    stabilize_weights,
)


def test_normalized_weights_and_effective_sample_size():
    weights = np.array([1.0, 1.0, 2.0])
    normed = normalized_weights(weights)

    np.testing.assert_allclose(normed, np.array([0.25, 0.25, 0.5]))
    assert effective_sample_size(weights) == pytest.approx(1.0 / np.sum(normed**2))


def test_stabilize_weights_clips_and_renormalizes():
    weights = np.array([1.0, 1.0, 100.0])
    stabilized = stabilize_weights(weights, clip_quantile=0.5)
    assert np.isclose(np.sum(stabilized), 1.0)
    assert np.max(stabilized) < normalized_weights(weights)[-1]


def test_log_importance_weights_matches_direct_ratio():
    log_t = np.log(np.array([2.0, 1.0]))
    log_p = np.log(np.array([1.0, 1.0]))
    weights = log_importance_weights(log_t, log_p)
    np.testing.assert_allclose(weights, np.array([2.0/3.0, 1.0/3.0]))


def test_importance_sampling_diagnostics_contains_effective_n():
    diag = importance_sampling_diagnostics([1.0, 2.0, 3.0])
    assert diag["effective_n"] > 0.0
    assert diag["max_weight"] <= 1.0


def test_importance_sampling_tail_probability_matches_weighted_indicator_mean():
    losses = np.array([1.0, 2.0, 10.0, 20.0])
    weights = np.array([1.0, 1.0, 3.0, 5.0])

    result = estimate_tail_probability_is(losses, weights, threshold=9.0)

    expected = (3.0 + 5.0) / np.sum(weights)
    assert result.method == 'importance_sampling'
    assert result.estimate == pytest.approx(expected)
    assert result.effective_n is not None


def test_importance_sampling_mean_matches_weighted_mean():
    losses = np.array([1.0, 2.0, 10.0, 20.0])
    weights = np.array([1.0, 1.0, 3.0, 5.0])
    result = estimate_mean_is(losses, weights)
    expected = np.sum(losses * weights / np.sum(weights))
    assert result.estimate == pytest.approx(expected)


def test_importance_sampling_exceedance_curve_is_monotone():
    losses = np.array([1.0, 2.0, 10.0, 20.0])
    weights = np.array([1.0, 1.0, 3.0, 5.0])
    curve = estimate_exceedance_curve_is(losses, weights, thresholds=[0.0, 5.0, 15.0])
    assert curve["probabilities"][0] >= curve["probabilities"][1] >= curve["probabilities"][2]


def test_importance_sampling_var_and_tvar_are_well_ordered():
    losses = np.array([1.0, 2.0, 10.0, 20.0])
    weights = np.array([1.0, 1.0, 3.0, 5.0])

    var_result = estimate_var_is(losses, weights, q=0.8)
    tvar_result = estimate_tvar_is(losses, weights, q=0.8)
    both = estimate_var_tvar_is(losses, weights, q=0.8)

    assert var_result.method == 'importance_sampling'
    assert tvar_result.method == 'importance_sampling'
    assert tvar_result.estimate >= var_result.estimate
    assert 0.0 < tvar_result.diagnostics['tail_weight'] <= 1.0
    assert both['var'].estimate == pytest.approx(var_result.estimate)
    assert both['tvar'].estimate == pytest.approx(tvar_result.estimate)


def test_importance_sampling_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        estimate_var_is([1.0, 2.0], [1.0], q=0.9)
