# Package overview

`extremeloss` follows a hybrid structure:

- domain subpackages similar to `lossmodels`
- shared protocol and result modules similar to `risksim`

## estimation/

This subpackage contains empirical and simulation-based tail estimators.

### Empirical estimators

- `estimate_tail_probability`
- `estimate_var`
- `estimate_tvar`
- `estimate_var_tvar`
- `exceedance_probability`
- `exceedance_curve`
- `survival_function`

### Importance sampling

- `estimate_mean_is`
- `estimate_tail_probability_is`
- `estimate_exceedance_curve_is`
- `estimate_var_is`
- `estimate_tvar_is`
- `estimate_var_tvar_is`
- `effective_sample_size`
- `importance_sampling_diagnostics`
- `log_importance_weights`
- `stabilize_weights`

### Conditional Monte Carlo

- `estimate_tail_probability_cmc`
- `estimate_tvar_cmc`

These functions operate on precomputed conditional probabilities or conditional tail expectations. They are generic building blocks rather than contract-specific CMC implementations.

## evt/

This subpackage contains extreme value modeling workflows.

### Peaks over threshold

- `extract_exceedances`
- `fit_gpd`
- `fit_pot`
- `gpd_tail_probability`
- `gpd_var`
- `gpd_tvar`

### Block maxima / GEV

- `make_blocks`
- `fit_gev`
- `fit_block_maxima`
- `block_return_level`

### Tail-index and threshold diagnostics

- `hill_estimator`
- `pickands_estimator`
- `hill_curve`
- `mean_excess`
- `threshold_diagnostic_table`

## analytics/

This subpackage contains light reporting and practical tail summaries.

- `extreme_loss_summary`
- `var_tvar_diagnostic_table`
- `return_period`
- `return_level`
- `exceedance_frequency`

## utils/

This subpackage currently contains:

- validation helpers
- bootstrap wrappers for tail statistics

Bootstrap entry points:

- `bootstrap_statistic`
- `bootstrap_tail_probability`
- `bootstrap_var`
- `bootstrap_tvar`

## integration.py

The integration layer is intentionally duck-typed, not hard-coupled to installed `lossmodels` or `risksim` classes.

Main helpers:

- `sample_lossmodel`
- `fit_pot_from_lossmodel`
- `losses_from_risksim`
- `tail_summary_from_risksim`
- `component_tail_metrics`
- `layer_tail_metrics`

## results.py

Main result containers:

- `TailEstimateResult`
- `GPDFit`
- `GEVFit`
- `BootstrapResult`
- `ThresholdScan`

These result objects make the API easier to inspect and test than returning bare tuples or scalars everywhere.
