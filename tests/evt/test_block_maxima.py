from __future__ import annotations

import numpy as np
import pytest

from extremeloss.evt.block_maxima import (
    block_return_level,
    fit_block_maxima,
    fit_gev,
    make_blocks,
)


def test_make_blocks_returns_block_maxima():
    data = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 8.0])
    blocks = make_blocks(data, block_size=2)
    np.testing.assert_allclose(blocks, np.array([3.0, 5.0, 8.0]))


def test_fit_gev_and_return_level_are_valid():
    maxima = np.array([10.0, 12.0, 13.0, 15.0, 18.0, 20.0, 21.0, 25.0])
    fit = fit_gev(maxima)
    rl10 = fit.return_level(10.0)
    rl20 = block_return_level(20.0, fit)
    assert fit.n_blocks == maxima.size
    assert fit.scale > 0.0
    assert rl20 >= rl10


def test_fit_block_maxima_uses_block_size_metadata():
    rng = np.random.default_rng(123)
    data = rng.normal(size=100)
    fit = fit_block_maxima(data, block_size=10)
    assert fit.block_size == 10
    assert fit.n_blocks == 10


def test_make_blocks_rejects_too_large_block_size():
    with pytest.raises(ValueError):
        make_blocks([1.0, 2.0], block_size=5)
