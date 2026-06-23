import numpy as np
import pytest

lossmodels = pytest.importorskip("lossmodels")

from lossmodels import SplicedSeverity  # noqa: E402
from extremeloss import fit_spliced_gpd, splice_gpd_tail, fit_pot, GPDTail  # noqa: E402


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.lognormal(9.0, 1.2, 20_000)


def test_fit_spliced_gpd_returns_spliced_with_gpd_tail(data):
    u = float(np.quantile(data, 0.90))
    body = lossmodels.fit_lognormal(data)
    sp = fit_spliced_gpd(body, data, threshold=u)
    assert isinstance(sp, SplicedSeverity)
    assert isinstance(sp.tail, GPDTail)
    assert sp.threshold == pytest.approx(u)
    assert sp.weight == pytest.approx(0.90, abs=0.01)   # body mass = 1 - exceedance_fraction
    assert sp.cdf(sp.quantile(0.99)) == pytest.approx(0.99, abs=1e-6)


def test_splice_gpd_tail_from_existing_fit(data):
    u = float(np.quantile(data, 0.92))
    fit = fit_pot(data, threshold=u)
    body = lossmodels.fit_lognormal(data)
    sp = splice_gpd_tail(body, fit)
    assert isinstance(sp, SplicedSeverity)
    assert sp.weight == pytest.approx(1.0 - fit.exceedance_fraction)
    assert sp.tail.xi == pytest.approx(fit.xi)


def test_weight_override(data):
    u = float(np.quantile(data, 0.90))
    body = lossmodels.fit_lognormal(data)
    sp = fit_spliced_gpd(body, data, threshold=u, weight=0.8)
    assert sp.weight == pytest.approx(0.8)
