# Changelog

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
