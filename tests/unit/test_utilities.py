"""Tests for small linear-algebra helpers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from scalabath.utilities import ABAd, compose

pytestmark = pytest.mark.unit


def test_compose_matches_numpy_kron() -> None:
    X = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex128)
    Z = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex128)

    result = compose([X, Z])

    np.testing.assert_allclose(np.asarray(result), np.kron(np.asarray(X), np.asarray(Z)))


def test_ABAd_computes_adjoint_sandwich() -> None:
    A = jnp.asarray([[1, 1j], [0, 1]], dtype=jnp.complex128)
    B = jnp.asarray([[2, 0], [0, 3]], dtype=jnp.complex128)

    result = ABAd(A, B)

    np.testing.assert_allclose(np.asarray(result), np.asarray(A @ B @ A.conj().T))
