from __future__ import annotations

import numpy as np
import pytest

from extremeloss.analytics.return_periods import exceedance_frequency, return_level, return_period
from extremeloss.results import GPDFit


def test_return_period_is_inverse_of_probability():
    assert return_period(0.02) == 50.0


def test_exceedance_frequency_matches_empirical_probability():
    losses = np.array([1.0, 2.0, 10.0, 20.0])
    assert exceedance_frequency(losses, 9.0) == 0.5


def test_return_level_delegates_to_fit():
    fit = GPDFit(
        threshold=10.0,
        xi=0.1,
        beta=4.0,
        exceedance_fraction=0.05,
        n_exceedances=20,
    )

    assert return_level(50.0, fit) == pytest.approx(fit.return_level(50.0))
