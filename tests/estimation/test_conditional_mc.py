from __future__ import annotations

import numpy as np
import pytest

from extremeloss.estimation.conditional_mc import (
    estimate_tail_probability_cmc,
    estimate_tvar_cmc,
)


def test_tail_probability_cmc_is_mean_of_conditional_probabilities():
    probs = np.array([0.1, 0.2, 0.4, 0.3])
    result = estimate_tail_probability_cmc(probs, threshold=10.0)

    assert result.method == "conditional_monte_carlo"
    assert result.estimate == pytest.approx(np.mean(probs))
    assert result.threshold == pytest.approx(10.0)
    assert result.ci is not None


def test_tvar_cmc_is_mean_of_conditional_tail_expectations():
    vals = np.array([12.0, 14.0, 20.0])
    result = estimate_tvar_cmc(vals, q=0.99, threshold=10.0)

    assert result.method == "conditional_monte_carlo"
    assert result.quantile == pytest.approx(0.99)
    assert result.estimate == pytest.approx(np.mean(vals))


def test_tail_probability_cmc_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        estimate_tail_probability_cmc([0.1, 1.2])
