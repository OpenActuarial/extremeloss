"""Experience seam for tail fitting."""
import numpy as np
import pandas as pd
import pytest
from actuarialpy import Experience

from extremeloss import fit_gpd
from extremeloss.integrations.actuarialpy import fit_gpd_from_experience


def _exp():
    rng = np.random.default_rng(11)
    amounts = rng.pareto(2.5, 500) * 10_000.0 + 1_000.0
    return Experience(pd.DataFrame({"claim_id": range(500), "paid": amounts}),
                      expense="paid", exposure_keys="claim_id"), amounts


def test_gpd_fit_matches_array_path():
    exp, amounts = _exp()
    u = float(np.quantile(amounts, 0.9))
    via_exp = fit_gpd_from_experience(exp, threshold=u)
    via_arr = fit_gpd(amounts[amounts > u] - u, threshold=u)
    assert via_exp.xi == pytest.approx(via_arr.xi)
    assert via_exp.beta == pytest.approx(via_arr.beta)


def test_empty_tail_is_a_clear_error():
    exp, amounts = _exp()
    from extremeloss.integrations.actuarialpy import claim_amounts  # noqa: F401
    with pytest.raises(ValueError, match="no claim amounts exceed"):
        fit_gpd_from_experience(exp, threshold=float(amounts.max()) + 1)

def test_experienceset_routes_to_the_named_listing():
    from actuarialpy import ExperienceSet, Source
    exp, amounts = _exp()
    grain = pd.DataFrame({"claim_id": range(500), "month": pd.Timestamp("2025-01-01"), "n": 1.0})
    book = ExperienceSet.from_tables(
        grain, grain=["claim_id"], exposure="n",
        sources=[Source(exp.data, expense="paid", name="claims")])
    via_book = fit_gpd_from_experience(book, threshold=float(np.quantile(amounts, 0.9)))
    via_member = fit_gpd_from_experience(book["claims"], threshold=float(np.quantile(amounts, 0.9)))
    assert via_book.xi == via_member.xi
