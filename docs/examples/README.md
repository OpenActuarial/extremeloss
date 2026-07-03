# Examples overview

The repository includes small runnable scripts that exercise the current API.

## Basic empirical workflows

- `examples/empirical_tail_analysis.py`
- `examples/pot_gpd_workflow.py`
- `examples/importance_sampling_demo.py`
- `examples/diagnostic_plots.py`

## Extended workflows

- `examples/conditional_mc_demo.py`
- `examples/block_maxima_gev_workflow.py`
- `examples/bootstrap_uncertainty_demo.py`
- `examples/integration_helpers_demo.py`

## Suggested order

1. `empirical_tail_analysis.py`
2. `pot_gpd_workflow.py`
3. `importance_sampling_demo.py`
4. `conditional_mc_demo.py`
5. `block_maxima_gev_workflow.py`
6. `bootstrap_uncertainty_demo.py`
7. `integration_helpers_demo.py`
8. `diagnostic_plots.py`

## Running scripts

From the repository root:

```bash
python examples/empirical_tail_analysis.py
python examples/pot_gpd_workflow.py
python examples/importance_sampling_demo.py
python examples/conditional_mc_demo.py
python examples/block_maxima_gev_workflow.py
python examples/bootstrap_uncertainty_demo.py
python examples/integration_helpers_demo.py
python examples/diagnostic_plots.py
```

The plotting script writes PNG files into `examples/output/`.
