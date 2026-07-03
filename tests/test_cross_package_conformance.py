"""The published claim: VaR/TVaR agree byte for byte across the ecosystem."""
import numpy as np
import pytest

import risksim as rs
from lossmodels.aggregate import tvar as lm_tvar, var as lm_var
from extremeloss import empirical_tvar, empirical_var, tail_summary_from_risksim


def test_var_tvar_byte_identical_across_packages():
    losses = np.random.default_rng(11).lognormal(7.0, 1.1, size=4001)
    for q in (0.5, 0.9, 0.99, 0.995):
        assert lm_var(losses, q) == rs.metrics.var(losses, q) == empirical_var(losses, q)
        assert lm_tvar(losses, q) == rs.metrics.tvar(losses, q) == empirical_tvar(losses, q)


class _Model:
    def sample(self, size=1, rng=None):
        return np.random.default_rng(rng).gamma(2.0, 500.0, size=size)


def test_tail_summary_reports_the_retained_view_with_matching_metrics():
    port = rs.Portfolio([rs.PortfolioItem("a", _Model())])
    res = port.simulate(5000, contract=rs.AggregateLayer(2000.0, 1000.0), rng=3)
    ts = tail_summary_from_risksim(res)
    assert ts["n"] == 5000
    assert ts["mean"] == pytest.approx(res.retained_losses.mean(), rel=1e-12)
    for row in ts["var_tvar"]:
        q = row["quantile"]
        assert row["var"] == rs.metrics.var(res.retained_losses, q)
        assert row["tvar"] == rs.metrics.tvar(res.retained_losses, q)
        assert row["tail_ratio"] == pytest.approx(row["tvar"] / row["var"], rel=1e-12)
