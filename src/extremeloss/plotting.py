from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .estimation.metrics import exceedance_curve
from .evt.tail_index import hill_curve
from .evt.thresholds import mean_excess
from .utils.validation import as_1d_float_array


def plot_exceedance_curve(losses, thresholds, ax=None):
    curve = exceedance_curve(losses, thresholds)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(curve["thresholds"], curve["probabilities"])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("P(X > u)")
    ax.set_title("Exceedance Curve")
    return ax


def plot_mean_excess(losses, thresholds, ax=None):
    curve = mean_excess(losses, thresholds)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(curve["thresholds"], curve["mean_excess"])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean excess")
    ax.set_title("Mean Excess Plot")
    return ax


def plot_hill_curve(losses, k_grid=None, ax=None):
    arr = as_1d_float_array(losses, name="losses")
    curve = hill_curve(arr, k_grid=k_grid)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(curve["k"], curve["hill"])
    ax.set_xlabel("k")
    ax.set_ylabel("Hill estimate")
    ax.set_title("Hill Plot")
    return ax