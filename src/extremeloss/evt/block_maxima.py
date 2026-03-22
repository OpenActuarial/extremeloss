from __future__ import annotations

import numpy as np
from scipy.stats import genextreme

from ..results import GEVFit
from ..utils.validation import as_1d_float_array, validate_positive


def make_blocks(data, block_size: int, *, drop_last: bool = True) -> np.ndarray:
    validate_positive(block_size, name="block_size")
    x = as_1d_float_array(data, name="data")
    block_size = int(block_size)
    n_blocks = x.size // block_size
    if not drop_last and x.size % block_size:
        n_blocks += 1
    if n_blocks < 1:
        raise ValueError("block_size is larger than the data length")
    maxima = []
    for i in range(n_blocks):
        start = i * block_size
        stop = min((i + 1) * block_size, x.size)
        block = x[start:stop]
        if block.size == 0:
            continue
        maxima.append(float(np.max(block)))
    out = np.asarray(maxima, dtype=float)
    if out.size < 2:
        raise ValueError("at least two blocks are required for GEV fitting")
    return out


def fit_gev(block_maxima, method: str = "mle", *, block_size: int | None = None) -> GEVFit:
    if method != "mle":
        raise ValueError("only method='mle' is currently supported")
    x = as_1d_float_array(block_maxima, name="block_maxima")
    if x.size < 2:
        raise ValueError("at least two block maxima are required")
    c_hat, loc_hat, scale_hat = genextreme.fit(x)
    if scale_hat <= 0.0:
        raise RuntimeError("GEV fit returned a nonpositive scale parameter")
    return GEVFit(
        xi=float(-c_hat),
        loc=float(loc_hat),
        scale=float(scale_hat),
        n_blocks=int(x.size),
        block_size=int(block_size) if block_size is not None else None,
        fit_method=method,
    )


def fit_block_maxima(data, block_size: int, method: str = "mle", *, drop_last: bool = True) -> GEVFit:
    maxima = make_blocks(data, block_size=block_size, drop_last=drop_last)
    return fit_gev(maxima, method=method, block_size=block_size)


def block_return_level(period: float, fit: GEVFit) -> float:
    if period <= 1.0:
        raise ValueError("period must exceed 1.0")
    return fit.return_level(period)