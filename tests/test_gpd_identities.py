"""Closed-form GPD identities for the tail metrics and the conditional tail."""
import numpy as np
import pytest
from scipy.stats import expon

from extremeloss import GPDFit, GPDTail, gpd_tvar, gpd_var, return_level

U, XI, BETA, ZETA = 1000.0, 0.25, 400.0, 0.05
FIT = GPDFit(threshold=U, xi=XI, beta=BETA, exceedance_fraction=ZETA,
             n_exceedances=500, fit_method="mle", covariance=None)


def _closed_var(q):
    return U + (BETA / XI) * (((1 - q) / ZETA) ** (-XI) - 1.0)


def test_gpd_var_matches_the_closed_form():
    for q in (0.96, 0.99, 0.995, 0.999):
        assert gpd_var(q, U, XI, BETA, ZETA) == pytest.approx(_closed_var(q), rel=1e-12)


def test_var_is_only_quoted_strictly_beyond_the_threshold():
    # at q = 1 - zeta the quote is refused by design ...
    with pytest.raises(ValueError):
        gpd_var(1 - ZETA, U, XI, BETA, ZETA)
    # ... and from just inside the tail it converges to the threshold
    assert gpd_var(1 - ZETA + 1e-9, U, XI, BETA, ZETA) == pytest.approx(U, rel=1e-8)


def test_return_level_is_the_var_at_one_minus_one_over_period():
    for period in (50.0, 200.0, 1000.0):
        assert return_level(period, FIT) == pytest.approx(
            gpd_var(1 - 1 / period, U, XI, BETA, ZETA), rel=1e-12)


def test_tvar_is_var_plus_the_gpd_mean_excess():
    for q in (0.96, 0.99, 0.999):
        v = gpd_var(q, U, XI, BETA, ZETA)
        mean_excess = (BETA + XI * (v - U)) / (1 - XI)
        assert gpd_tvar(q, U, XI, BETA, ZETA) == pytest.approx(v + mean_excess, rel=1e-10)


def test_gpd_tail_round_trip_and_moments():
    tail = GPDTail(threshold=U, xi=XI, beta=BETA)
    assert tail.cdf(U) == pytest.approx(0.0, abs=1e-12)
    for q in (0.1, 0.5, 0.9, 0.99):
        assert tail.cdf(tail.quantile(q)) == pytest.approx(q, abs=1e-10)
    assert tail.mean() == pytest.approx(U + BETA / (1 - XI), rel=1e-12)
    assert tail.variance() == pytest.approx(BETA**2 / ((1 - XI) ** 2 * (1 - 2 * XI)), rel=1e-12)


def test_moments_raise_outside_existence():
    with pytest.raises(ValueError):
        GPDTail(U, 1.0, BETA).mean()
    with pytest.raises(ValueError):
        GPDTail(U, 0.5, BETA).variance()


def test_xi_to_zero_limit_is_the_shifted_exponential():
    tail = GPDTail(U, 1e-9, BETA)
    xs = U + np.array([50.0, 400.0, 1200.0])
    assert np.allclose(tail.cdf(xs), expon(loc=U, scale=BETA).cdf(xs), atol=1e-6)
