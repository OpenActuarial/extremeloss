"""Consume the canonical ``actuarialpy.Experience`` at claim grain.

Import this module explicitly; it is the ecosystem seam and requires
``actuarialpy`` (``pip install actuarialpy`` or ``pip install openactuarial``).
The core package stays array-level and does not depend on it.
"""
from __future__ import annotations

import warnings

import numpy as np

try:
    from actuarialpy import Experience, ExperienceSet, resolve_amount
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "this integration consumes actuarialpy.Experience; install it with "
        "pip install actuarialpy (or pip install openactuarial)"
    ) from _err


def _resolve_listing(exp):
    if isinstance(exp, ExperienceSet):
        if len(exp.listings) != 1:
            raise ValueError(
                "pass the listing member explicitly (book['claims']); the set "
                f"has {sorted(exp.listings) or 'no'} named listings"
            )
        (exp,) = exp.listings.values()
    return exp


def _claim_grain_guard(exp: Experience) -> None:
    if exp.pivots:
        raise ValueError(
            "this Experience carries recorded wide_by pivots -- it is an "
            "aggregated experience tab, and severity/tail fits need one row "
            "per claim. Bind the claims listing instead, e.g. "
            "Experience(claim_lines, expense='paid_amount', date='incurred_date')."
        )
    if exp.exposure:
        warnings.warn(
            "an exposure role is bound, which usually marks an aggregated "
            "experience tab; severity/tail fits expect claim-level amounts "
            "(one row per claim).",
            stacklevel=3,
        )


def claim_amounts(exp: Experience, *, amount_col: str | None = None) -> np.ndarray:
    """Claim-level amounts resolved from the bound expense role."""
    exp = _resolve_listing(exp)
    _claim_grain_guard(exp)
    frame, col = resolve_amount(exp, amount_col)
    values = frame[col].to_numpy(dtype=float)
    return values[~np.isnan(values)]


def fit_gpd_from_experience(exp: Experience, *, threshold: float,
                            amount_col: str | None = None, method: str = "mle"):
    """Fit a GPD tail to the claim amounts above ``threshold``.

    Threshold selection is judgment and stays an argument; extracting the
    excesses is structural.
    """
    from extremeloss import fit_gpd

    amounts = claim_amounts(exp, amount_col=amount_col)
    excesses = amounts[amounts > threshold] - threshold
    if excesses.size == 0:
        raise ValueError(
            f"no claim amounts exceed threshold={threshold}; largest amount "
            f"is {amounts.max() if amounts.size else float('nan')}"
        )
    return fit_gpd(excesses, threshold=threshold, method=method)
