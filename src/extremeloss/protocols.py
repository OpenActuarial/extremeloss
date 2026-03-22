from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SupportsSample(Protocol):
    """Minimal protocol for objects that can generate simulated losses."""

    def sample(self, size: int = 1) -> np.ndarray:
        ...