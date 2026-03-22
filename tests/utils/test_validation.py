from __future__ import annotations

import numpy as np
import pytest

from extremeloss.utils.validation import as_1d_float_array, coerce_losses, validate_weights


class GoodSampler:
    def sample(self, size: int = 1):
        return np.full(size, 7.0)


class BadSampler:
    def sample(self, size: int = 1):
        return np.full(size - 1, 7.0)


def test_as_1d_float_array_rejects_multidimensional_input():
    with pytest.raises(ValueError):
        as_1d_float_array([[1.0, 2.0], [3.0, 4.0]], name='x')


def test_validate_weights_rejects_negative_values():
    with pytest.raises(ValueError):
        validate_weights([0.2, -0.1, 0.9])


def test_coerce_losses_accepts_arrays_and_sampler_objects():
    arr = coerce_losses([1, 2, 3])
    sampled = coerce_losses(GoodSampler(), size=4)

    np.testing.assert_allclose(arr, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(sampled, np.array([7.0, 7.0, 7.0, 7.0]))


def test_coerce_losses_rejects_bad_sampler_size():
    with pytest.raises(ValueError):
        coerce_losses(BadSampler(), size=4)
