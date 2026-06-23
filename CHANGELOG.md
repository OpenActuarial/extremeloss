# Changelog

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
