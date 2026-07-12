# Changelog

## 0.7.0

Add `extremeloss.integrations.actuarialpy`: consume the canonical
`actuarialpy.Experience` at claim grain (claims-listing bindings). Resolves
amounts from the bound roles, refuses aggregated experience tabs
(recorded pivots), and warns when an exposure role marks aggregated data.
Soft dependency: the core package stays array-level; install the
`[actuarialpy]` extra or the `openactuarial` meta-package.

## 0.6.4

Bump version number to update PyPI with updated README.

## 0.6.3

### Fixed
- **Worked-example regression test vs actuarialpy 0.44.** `test_worked_example_page_numbers`
  still imported `UnderwritingSummary` from `actuarialpy`, which moved to
  `experiencestudies` in the 0.43/0.44 split — the next push to this repo
  would have failed CI (the green badge predated the move). The test now
  imports from `experiencestudies`, and `experiencestudies>=0.3` joins the
  `dev` extras so the module is actually present in CI rather than the test
  silently skipping.
- `pip install -e ".[dev]" && pytest` now works without the `plot` extra:
  `test_plotting.py` skips via `pytest.importorskip("matplotlib")` instead
  of failing collection at import time.
- Two unused test imports (ruff F401).

### Added
- **Docstrings for all 30 previously undocumented public names** — `fit_pot`,
  `fit_gev`, `fit_block_maxima`, `make_blocks`, `block_return_level`,
  `gpd_var` / `gpd_tvar` / `gpd_tail_probability`, `mean_excess`,
  `threshold_diagnostic_table`, `return_level` / `return_period`,
  `exceedance_probability`, `extreme_loss_summary`, the importance-sampling
  estimators and diagnostics, the empirical `estimate_var` / `estimate_tvar`
  wrappers, and the risksim/lossmodels integration helpers. Sphinx
  `automodule :members:` silently omits objects without a docstring, so all
  of these were absent from the API reference on openactuarial.org.
- A guard test (`tests/test_public_api_docstrings.py`) asserting every
  function and class in `__all__` carries a docstring, turning a silent
  autodoc omission into a CI failure.

## 0.6.2

### Added
- Docstrings in bootstrap.py, pot.py, and tail_index.py. 

## 0.6.1

### Fixed
- **Worked-example regression test robustness.** `test_worked_example_page_numbers`
  previously pinned the seeded 100k-path simulation's 99% VaR and TVaR to a
  0.2% / 0.3% tolerance. Those tolerances are tighter than the Monte Carlo
  standard deviation of the estimators themselves (~0.22% for the 99% VaR at
  100k paths), so the assertion was effectively testing that the RNG draw
  sequence was byte-identical rather than that the VaR was correct. It broke
  when `lossmodels` 0.7.1 landed the negative-binomial fit at a slightly
  different (likelihood-equivalent) parameter set, which re-seeds the draw
  sequence. The test now asserts draw-independent invariants (loss
  conservation, ceded <= limit, TVaR >= VaR, exact retained-TVaR, and
  convergence of the simulated mean to the analytical `crm.mean()`) and keeps
  the documented VaR/TVaR/mean-ceded figures as point checks with tolerances
  set from the measured Monte Carlo variability, so they survive lossless
  upstream numerical changes while still catching gross regressions.

## 0.6.0


Block-maxima parity and fit adequacy: the uncertainty treatment
GPD got in 0.5.0, extended to GEV, plus the tools that ask
whether a fit succeeded at all.

### Added
- **GEV uncertainty.** `fit_gev` now populates the (previously
  always-empty) `GEVFit.covariance` with the observed-information
  covariance of `(xi, loc, scale)` -- the same treatment `fit_pot` got,
  removing the asymmetry where switching from POT to block maxima
  silently lost all error bars. New `GEVFit.se` property and
  `gev_return_level` with delta-method confidence intervals (Coles,
  2001, section 3.3.3); point estimates agree exactly with
  `GEVFit.return_level` / `analytics.block_return_level`.- **Fitted-model diagnostics.** The threshold tools ask where GPD
  behavior starts; these ask whether the fit succeeded. `qq_points` and
  `pp_points` (data functions, GPD and GEV), and
  `parametric_bootstrap_gof` -- KS and Anderson-Darling with p-values
  from a parametric bootstrap that *refits inside every replicate*
  (naive fixed-parameter p-values are anti-conservative when the
  parameters were estimated from the same data). Graphical companions in
  the `plot` extra: `plot_gpd_diagnostics` and `plot_gev_diagnostics`,
  the Coles four-panel exhibits (probability, quantile, return level
  with band and empirical points, density).
## 0.5.0

### Added
- **GPD parameter uncertainty.** `fit_gpd` and `fit_pot` now populate the
  (previously always-empty) `GPDFit.covariance` field with the
  observed-information covariance of `(xi, beta)` -- the numerical Hessian
  of the GPD log-likelihood at the MLE, inverted. When the information
  matrix is not positive definite (the MLE is irregular for `xi <= -1/2`
  and near-boundary fits) the field stays `None` rather than carrying a
  covariance that means nothing. New `GPDFit.se` property.
- **`gpd_return_level`.** Return levels with delta-method confidence
  intervals over `(zeta_u, xi, beta)`, including the binomial variance of
  the exceedance rate (Coles, 2001, section 4.3.3). Vectorized over return
  periods with explicit `observations_per_period` semantics; the point
  estimates agree with the existing `GPDFit.return_level` /
  `analytics.return_level` exactly, which now serve as the scalar
  shorthand.
- **Threshold scan with error bands.** `threshold_diagnostic_table` gains
  `xi_se`, `modified_scale`, and `modified_scale_se` columns. The modified
  scale `beta* = beta - xi * u` is the quantity that is actually constant
  above a valid threshold (raw `beta` drifts linearly in `u` even under a
  perfect GPD), so threshold selection becomes "where do `xi` and `beta*`
  flatten *within their confidence bands*". New `ThresholdScan` fields are
  optional and default to `None`; existing consumers are unaffected.


- **Tail protocol.** `GPDFit.sf(x)` (alias of `tail_probability`) and
  `GPDFit.mean_excess(d)` (closed form, `ValueError` below the threshold,
  `inf` for `xi >= 1`) -- the same two-method protocol `lossmodels`
  severities expose, so a fitted tail plugs into
  `ratingmodels.pooling_charge_from_severity` directly.

## 0.4.1

### Fixed

- `GPDTail.sample` accepts the shared `rng` argument (`None`, seed, or
  `Generator`) per the ecosystem reproducibility convention. Previously,
  passing `rng` through a spliced severity — e.g.
  `CollectiveRiskModel.sample(..., rng=...)` with a `splice_gpd_tail`
  severity — raised `TypeError`.

### Changed

- More descriptive package `description` metadata.

### Added

- Conformance, identity, and integration test suites (scipy/closed-form
  conformance, mathematical identities, cross-package seams). Example
  scripts are now executed by the test suite.

## 0.4.0

### Changed
- **Breaking (numeric):** `empirical_var` and `empirical_tvar` follow the
  ecosystem-wide convention: VaR is the inverted-CDF order statistic and TVaR
  the Acerbi-Tasche average-quantile estimator, so TVaR(q) >= VaR(q) always.
  Conformance is pinned by a test shared byte-for-byte with risksim and
  lossmodels.

### Fixed
- `coerce_losses` now accepts `pandas.Series` and other array-likes that also
  expose a `sample` attribute; array conversion is attempted before the
  model-sampling branch.
- Component/layer summaries in the integration module zip names strictly
  against value columns.

## 0.2.2

### Fixed
- Removed an unused `numpy` import in `plotting.py` (lint hygiene; no behaviour change).

### Tests
- Added `test_risksim_integration.py`, exercising the duck-typed risksim bridge
  (`losses_from_risksim`, `tail_summary_from_risksim`, `component_tail_metrics`,
  `layer_tail_metrics`) against a real `risksim` `SimulationResult` rather than only a
  stand-in object. Skips cleanly when `risksim`/`lossmodels` are not installed, matching
  the existing spliced-severity integration test. Guards against a future rename of a
  risksim loss view silently breaking the integration.

## 0.2.1

### Added (packaging)
- `splice` optional-dependency extra (`pip install extremeloss[splice]`) declaring
  `lossmodels>=0.4.0`, which `fit_spliced_gpd` / `splice_gpd_tail` require. The
  import remains lazy, so the base install is unchanged. Note: resolving the extra
  needs `lossmodels>=0.4.0` available on your index.

### Fixed
- Corrected stale repository links in the README (`michaelabryant` -> `actuarialpy`).

## 0.2.0

### Added
- `GPDTail`: a conditional generalized-Pareto tail distribution on
  `[threshold, inf)` (wrapping `scipy.stats.genpareto`), exposing
  `pdf` / `cdf` / `sample` / `quantile` / `mean` / `variance`. Build one from a
  fit with `GPDTail.from_fit(gpd_fit)`. Moments raise when they do not exist
  (`xi >= 1` for the mean, `xi >= 1/2` for the variance).
- `fit_spliced_gpd(body, data, *, threshold, weight=None)` and
  `splice_gpd_tail(body, fit, *, weight=None)`: fit (or reuse) a peaks-over-
  threshold GPD tail and splice it onto a body severity, returning a
  `lossmodels.SplicedSeverity`. The mixing weight defaults to the body mass
  implied by the fit (`1 - exceedance_fraction`). `lossmodels` is imported
  lazily, so it is only required when these constructors are called.

### Fixed
- Version mismatch between `__init__.__version__` (was `0.1.0`) and
  `pyproject.toml` (was `0.1.1`); both are now `0.2.0`.
