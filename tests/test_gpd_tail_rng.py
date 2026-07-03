import numpy as np

import lossmodels
from extremeloss import GPDFit, GPDTail, splice_gpd_tail


def test_gpd_tail_sample_rng_reproducible():
    tail = GPDTail(threshold=100.0, xi=0.2, beta=50.0)
    a = tail.sample(500, rng=42)
    b = tail.sample(500, rng=42)
    assert np.array_equal(a, b)
    assert a.min() >= 100.0
    assert not np.array_equal(a, tail.sample(500, rng=43))


def test_spliced_severity_in_crm_samples_with_rng():
    # regression: raised TypeError before 0.4.1
    body = lossmodels.Lognormal(7.0, 1.0)
    fit = GPDFit(threshold=float(np.exp(8.0)), xi=0.15, beta=800.0,
                 exceedance_fraction=0.05, n_exceedances=100,
                 fit_method="mle", covariance=None)
    sev = splice_gpd_tail(body, fit)
    crm = lossmodels.CollectiveRiskModel(lossmodels.Poisson(20.0), sev)
    x = crm.sample(200, rng=5)
    y = crm.sample(200, rng=5)
    assert np.array_equal(x, y)
