from __future__ import annotations

import numpy as np

from extremeloss.results import GPDFit, TailEstimateResult, ThresholdScan


def test_tail_estimate_result_summary_contains_expected_fields():
    result = TailEstimateResult(
        estimate=0.1,
        method='empirical',
        stderr=0.01,
        ci=(0.08, 0.12),
        n=100,
        effective_n=80.0,
        threshold=10.0,
        quantile=0.99,
        diagnostics={'n_exceedances': 10},
    )

    summary = result.summary()

    assert summary['estimate'] == 0.1
    assert summary['method'] == 'empirical'
    assert summary['stderr'] == 0.01
    assert summary['ci'] == (0.08, 0.12)
    assert summary['n'] == 100
    assert summary['effective_n'] == 80.0
    assert summary['threshold'] == 10.0
    assert summary['quantile'] == 0.99
    assert summary['diagnostics'] == {'n_exceedances': 10}


def test_gpd_fit_summary_and_return_level_are_well_formed():
    fit = GPDFit(
        threshold=100.0,
        xi=0.2,
        beta=20.0,
        exceedance_fraction=0.1,
        n_exceedances=50,
    )

    summary = fit.summary()

    assert summary['threshold'] == 100.0
    assert summary['xi'] == 0.2
    assert summary['beta'] == 20.0
    assert summary['exceedance_fraction'] == 0.1
    assert summary['n_exceedances'] == 50
    assert fit.return_level(20.0) > fit.threshold


def test_threshold_scan_to_dict_preserves_shapes():
    scan = ThresholdScan(
        thresholds=np.array([1.0, 2.0, 3.0]),
        mean_excess=np.array([4.0, 5.0, np.nan]),
        xi=np.array([0.1, 0.2, np.nan]),
        beta=np.array([1.0, 1.5, np.nan]),
        n_exceedances=np.array([10, 5, 0]),
    )

    out = scan.to_dict()

    assert set(out) == {'thresholds', 'mean_excess', 'xi', 'beta', 'n_exceedances'}
    assert out['thresholds'].shape == (3,)
    assert out['n_exceedances'].dtype.kind in {'i', 'u'}
