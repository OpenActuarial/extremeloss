# API reference

## Top-level exports

The root package currently exports the main estimation, EVT, analytics, integration, and bootstrap helpers.

## Result containers

### `TailEstimateResult`

Fields commonly used:

- `estimate`
- `method`
- `stderr`
- `ci`
- `n`
- `effective_n`
- `threshold`
- `quantile`
- `diagnostics`

### `GPDFit`

Methods:

- `summary()`
- `tail_probability(x)`
- `var(p)`
- `tvar(p)`
- `return_level(period)`

### `GEVFit`

Methods:

- `summary()`
- `cdf(x)`
- `return_level(period)`

### `BootstrapResult`

Fields:

- `estimate`
- `bootstrap_estimates`
- `method`
- `ci`
- `stderr`
- `alpha`

### `ThresholdScan`

Methods:

- `to_dict()`

## Estimation

### Empirical

- `estimate_tail_probability(losses, threshold, size=None, alpha=0.05)`
- `estimate_var(losses, q, size=None)`
- `estimate_tvar(losses, q, size=None)`
- `estimate_var_tvar(losses, q, size=None)`
- `empirical_var(losses, q)`
- `empirical_tvar(losses, q)`
- `exceedance_probability(losses, threshold)`
- `exceedance_curve(losses, thresholds)`
- `survival_function(losses, grid)`

### Importance sampling

- `estimate_mean_is(values, weights, alpha=0.05)`
- `estimate_tail_probability_is(losses, weights, threshold, alpha=0.05)`
- `estimate_exceedance_curve_is(losses, weights, thresholds)`
- `estimate_var_is(losses, weights, q)`
- `estimate_tvar_is(losses, weights, q)`
- `estimate_var_tvar_is(losses, weights, q)`
- `effective_sample_size(weights)`
- `importance_sampling_diagnostics(weights)`
- `log_importance_weights(log_target_density, log_proposal_density, normalize=True)`
- `stabilize_weights(weights, clip_quantile=None, renormalize=True)`

### Conditional Monte Carlo

- `estimate_tail_probability_cmc(conditional_probabilities, threshold=None, alpha=0.05)`
- `estimate_tvar_cmc(conditional_tail_expectations, q, threshold=None, alpha=0.05)`

## EVT

### POT / GPD

- `extract_exceedances(data, threshold)`
- `fit_gpd(excesses, method="mle")`
- `fit_pot(data, threshold, method="mle")`
- `gpd_tail_probability(x, threshold, xi, beta, exceedance_fraction)`
- `gpd_var(p, threshold, xi, beta, exceedance_fraction)`
- `gpd_tvar(p, threshold, xi, beta, exceedance_fraction)`

### Block maxima / GEV

- `make_blocks(data, block_size, drop_last=True)`
- `fit_gev(block_maxima, method="mle", block_size=None)`
- `fit_block_maxima(data, block_size, method="mle", drop_last=True)`
- `block_return_level(period, fit)`

### Tail index and threshold diagnostics

- `hill_estimator(data, k)`
- `pickands_estimator(data, k)`
- `hill_curve(data, k_grid=None)`
- `mean_excess(data, thresholds)`
- `threshold_diagnostic_table(data, thresholds)`

## Analytics

- `extreme_loss_summary(losses, thresholds=None, quantiles=(0.95, 0.99, 0.995))`
- `var_tvar_diagnostic_table(losses, quantiles=(0.95, 0.99, 0.995))`
- `return_period(probability)`
- `exceedance_frequency(losses, threshold)`
- `return_level(period, fit)`

## Bootstrap utilities

- `bootstrap_statistic(data, statistic, n_resamples=1000, alpha=0.05, random_state=None)`
- `bootstrap_tail_probability(losses, threshold, **kwargs)`
- `bootstrap_var(losses, q, **kwargs)`
- `bootstrap_tvar(losses, q, **kwargs)`

## Integration helpers

- `sample_lossmodel(model, size)`
- `fit_pot_from_lossmodel(model, size, threshold)`
- `losses_from_risksim(result, view="losses")`
- `tail_summary_from_risksim(result, view="losses", thresholds=None, quantiles=(0.95, 0.99, 0.995))`
- `component_tail_metrics(result, q=0.99, threshold=None)`
- `layer_tail_metrics(result, q=0.99, threshold=None)`

## Plotting

- `plot_exceedance_curve(losses, thresholds, ax=None)`
- `plot_mean_excess(losses, thresholds, ax=None)`
- `plot_hill_curve(losses, k_grid=None, ax=None)`
