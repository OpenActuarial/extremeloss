from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import numpy as np

from extremeloss.plotting import plot_exceedance_curve, plot_hill_curve, plot_mean_excess


def test_plotting_helpers_return_axes_objects():
    losses = np.array([1.0, 2.0, 3.0, 10.0, 50.0, 100.0])

    ax1 = plot_exceedance_curve(losses, thresholds=[2.0, 20.0])
    ax2 = plot_mean_excess(losses, thresholds=[2.0, 20.0])
    ax3 = plot_hill_curve(losses + 1.0, k_grid=[1, 2, 3])

    assert ax1.get_title() == 'Exceedance Curve'
    assert ax2.get_title() == 'Mean Excess Plot'
    assert ax3.get_title() == 'Hill Plot'
