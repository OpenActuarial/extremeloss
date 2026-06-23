import numpy as np
import pytest
from scipy.stats import genpareto

from extremeloss import GPDTail, fit_pot

U, XI, BETA = 10_000.0, 0.25, 5_000.0


def make_tail(xi=XI):
    return GPDTail(threshold=U, xi=xi, beta=BETA)


def test_cdf_zero_at_threshold_and_monotone():
    t = make_tail()
    assert t.cdf(U) == pytest.approx(0.0, abs=1e-12)
    xs = np.array([U, 2 * U, 5 * U, 10 * U])
    assert np.all(np.diff(t.cdf(xs)) > 0)


def test_matches_scipy_genpareto():
    t = make_tail()
    x = np.array([U, 15_000.0, 50_000.0])
    assert t.cdf(x) == pytest.approx(genpareto.cdf(x, c=XI, loc=U, scale=BETA))
    assert t.pdf(x) == pytest.approx(genpareto.pdf(x, c=XI, loc=U, scale=BETA))
    assert t.quantile(0.99) == pytest.approx(genpareto.ppf(0.99, c=XI, loc=U, scale=BETA))


def test_quantile_roundtrip():
    t = make_tail()
    for p in (0.1, 0.5, 0.99):
        assert t.cdf(t.quantile(p)) == pytest.approx(p, abs=1e-9)


def test_scalar_vs_array_outputs():
    t = make_tail()
    assert isinstance(t.cdf(20_000.0), float)
    assert isinstance(t.pdf(20_000.0), float)
    assert t.cdf(np.array([U, 2 * U])).shape == (2,)


def test_sample_lands_above_threshold():
    np.random.seed(0)
    s = make_tail().sample(10_000)
    assert s.shape == (10_000,)
    assert s.min() >= U


def test_moments_match_closed_form():
    t = make_tail()
    assert t.mean() == pytest.approx(U + BETA / (1.0 - XI))
    assert t.variance() == pytest.approx(BETA ** 2 / ((1.0 - XI) ** 2 * (1.0 - 2.0 * XI)))


def test_moments_raise_for_heavy_tail():
    with pytest.raises(ValueError):
        make_tail(xi=1.2).mean()
    with pytest.raises(ValueError):
        make_tail(xi=0.7).variance()


def test_from_fit_round_trips_parameters():
    np.random.seed(1)
    data = np.random.lognormal(8.0, 1.0, 5000)
    fit = fit_pot(data, threshold=float(np.quantile(data, 0.9)))
    t = GPDTail.from_fit(fit)
    assert t.threshold == pytest.approx(fit.threshold)
    assert t.xi == pytest.approx(fit.xi)
    assert t.beta == pytest.approx(fit.beta)
