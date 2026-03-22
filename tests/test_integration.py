from __future__ import annotations

import numpy as np
import pytest

from extremeloss.estimation.rare_event import estimate_tail_probability
from extremeloss.integration import (
    component_tail_metrics,
    fit_pot_from_lossmodel,
    layer_tail_metrics,
    losses_from_risksim,
    sample_lossmodel,
    tail_summary_from_risksim,
)


class DummyLossModel:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def sample(self, size: int = 1):
        if size != self._values.size:
            return self._values[:size]
        return self._values.copy()


class DummySimulationResult:
    def __init__(self):
        self.gross_losses = np.array([10.0, 20.0, 5.0, 30.0])
        self.retained_losses = np.array([8.0, 15.0, 4.0, 20.0])
        self.ceded_losses = np.array([2.0, 5.0, 1.0, 10.0])
        self.component_losses = np.array([
            [6.0, 4.0],
            [10.0, 10.0],
            [2.0, 3.0],
            [15.0, 15.0],
        ])
        self.component_names = ["attritional", "cat"]
        self.layer_losses = np.array([
            [1.0, 1.0],
            [2.0, 3.0],
            [0.0, 1.0],
            [4.0, 6.0],
        ])
        self.layer_names = ["layer_a", "layer_b"]

    @property
    def losses(self):
        return self.retained_losses


def test_sample_lossmodel_uses_sample_protocol():
    model = DummyLossModel([1.0, 2.0, 3.0, 4.0])
    sampled = sample_lossmodel(model, size=4)
    np.testing.assert_allclose(sampled, np.array([1.0, 2.0, 3.0, 4.0]))


def test_fit_pot_from_lossmodel_returns_positive_scale():
    model = DummyLossModel([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    fit = fit_pot_from_lossmodel(model, size=6, threshold=5.0)
    assert fit.beta > 0.0
    assert fit.n_exceedances == 3


def test_losses_from_risksim_extracts_requested_view():
    result = DummySimulationResult()
    np.testing.assert_allclose(losses_from_risksim(result, view="gross"), result.gross_losses)
    np.testing.assert_allclose(losses_from_risksim(result, view="losses"), result.losses)


def test_standard_estimators_accept_simulation_result_like_object():
    result = DummySimulationResult()
    estimate = estimate_tail_probability(result, threshold=10.0)
    assert 0.0 <= estimate.estimate <= 1.0


def test_tail_summary_and_component_layer_metrics_have_expected_keys():
    result = DummySimulationResult()
    summary = tail_summary_from_risksim(result, thresholds=[5.0, 10.0])
    components = component_tail_metrics(result, q=0.75, threshold=5.0)
    layers = layer_tail_metrics(result, q=0.75, threshold=1.0)

    assert "exceedance_curve" in summary
    assert set(components.keys()) == {"attritional", "cat"}
    assert set(layers.keys()) == {"layer_a", "layer_b"}
    assert "exceedance_probability" in components["cat"]


def test_losses_from_risksim_rejects_missing_view():
    result = DummySimulationResult()
    result.retained_losses = None
    with pytest.raises(ValueError):
        losses_from_risksim(result, view="retained")
