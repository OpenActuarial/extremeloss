from __future__ import annotations

import numpy as np

from extremeloss.evt.thresholds import mean_excess, threshold_diagnostic_table


def test_mean_excess_returns_counts_and_nan_when_no_exceedances():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    thresholds = np.array([2.0, 10.0])

    out = mean_excess(data, thresholds)

    np.testing.assert_array_equal(out['n_exceedances'], np.array([2, 0]))
    assert out['mean_excess'][0] == 1.5
    assert np.isnan(out['mean_excess'][1])


def test_threshold_diagnostic_table_returns_threshold_scan():
    rng = np.random.default_rng(42)
    data = rng.exponential(scale=4.0, size=3000)
    thresholds = np.array([2.0, 4.0, 30.0])

    scan = threshold_diagnostic_table(data, thresholds)
    out = scan.to_dict()

    np.testing.assert_array_equal(out['thresholds'], thresholds)
    np.testing.assert_array_equal(out['n_exceedances'].shape, (3,))
    assert np.isfinite(out['xi'][:2]).all()
    assert np.isnan(out['xi'][2])
