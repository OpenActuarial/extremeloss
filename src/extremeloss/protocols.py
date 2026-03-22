from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SupportsSample(Protocol):
    """Minimal protocol for objects that can generate simulated losses."""

    def sample(self, size: int = 1) -> np.ndarray:
        ...


@runtime_checkable
class SupportsLosses(Protocol):
    """Minimal protocol for objects exposing a one-dimensional loss view."""

    @property
    def losses(self) -> np.ndarray:
        ...


@runtime_checkable
class SupportsSimulationResult(SupportsLosses, Protocol):
    """Minimal risksim-like simulation result protocol."""

    gross_losses: np.ndarray
    retained_losses: np.ndarray | None
    ceded_losses: np.ndarray | None
