from __future__ import annotations

import numpy as np

from extremeloss.protocols import SupportsSample


class DummySampler:
    def sample(self, size: int = 1) -> np.ndarray:
        return np.arange(size, dtype=float)


def test_supports_sample_protocol_runtime_checkable():
    sampler = DummySampler()
    assert isinstance(sampler, SupportsSample)
