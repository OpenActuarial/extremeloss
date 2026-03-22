from __future__ import annotations

import numpy as np

from extremeloss.analytics.diagnostics import extreme_loss_summary, var_tvar_diagnostic_table


def test_var_tvar_diagnostic_table_builds_rows_for_each_quantile():
    losses = np.array([1.0, 2.0, 3.0, 10.0, 50.0])
    out = var_tvar_diagnostic_table(losses, quantiles=(0.8, 0.9))

    assert out['n'] == 5
    assert len(out['rows']) == 2
    assert out['rows'][1]['tvar'] >= out['rows'][1]['var']


def test_extreme_loss_summary_includes_exceedance_curve_when_requested():
    losses = np.array([1.0, 2.0, 3.0, 10.0, 50.0])
    out = extreme_loss_summary(losses, thresholds=[5.0, 20.0], quantiles=(0.8,))

    assert out['n'] == 5
    assert 'var_tvar' in out
    assert 'exceedance_curve' in out
    np.testing.assert_allclose(out['exceedance_curve']['probabilities'], np.array([0.4, 0.2]))
