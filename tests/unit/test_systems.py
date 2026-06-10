"""Tests for state ensemble containers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from scalabath.systems import (
    DensityMatrixEnsemble,
    PureStatesEnsemble,
    TensorProductDensityMatrixEnsemble,
    TensorProductPureStatesEnsemble,
)

pytestmark = pytest.mark.unit


def test_pure_state_ensemble_broadcasts_and_normalizes() -> None:
    ensemble = PureStatesEnsemble(2, batch_size=3)
    ensemble.set_pse(jnp.asarray([3, 4j], dtype=jnp.complex128))

    normalized = ensemble.normalize()

    assert normalized.shape == (3, 2)
    np.testing.assert_allclose(np.asarray(ensemble.norm), np.ones(3))
    np.testing.assert_allclose(np.asarray(normalized[0]), np.asarray([0.6, 0.8j]))


def test_density_matrix_ensemble_broadcasts_and_normalizes_trace() -> None:
    ensemble = DensityMatrixEnsemble(2, batch_size=2)
    ensemble.set_dme(jnp.asarray([[2, 0], [0, 0]], dtype=jnp.complex128))

    normalized = ensemble.normalize()

    assert normalized.shape == (2, 2, 2)
    np.testing.assert_allclose(np.asarray(ensemble.trace), np.ones(2))
    np.testing.assert_allclose(np.asarray(normalized[0]), np.asarray([[1, 0], [0, 0]]))


def test_tensor_product_pure_state_accepts_flat_and_native_layouts() -> None:
    ensemble = TensorProductPureStatesEnsemble((2, 3), batch_size=2, dtype=jnp.complex128)
    flat = jnp.asarray([[1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 2j]], dtype=jnp.complex128)

    ensemble.set_pse(flat)
    normalized = ensemble.normalize()

    assert normalized.shape == (2, 2, 3)
    np.testing.assert_allclose(np.asarray(ensemble.norm), np.ones(2))
    np.testing.assert_allclose(np.asarray(ensemble.get_flat_pse()[1, -1]), np.asarray(1j))


def test_tensor_product_density_matrix_round_trips_flat_layout() -> None:
    ensemble = TensorProductDensityMatrixEnsemble((2, 2), batch_size=1, dtype=jnp.complex128)
    rho = jnp.diag(jnp.asarray([2, 0, 0, 0], dtype=jnp.complex128))

    ensemble.set_dme(rho)
    normalized = ensemble.normalize()

    assert normalized.shape == (1, 2, 2, 2, 2)
    np.testing.assert_allclose(np.asarray(ensemble.trace), np.asarray([1]))
    np.testing.assert_allclose(np.asarray(ensemble.get_flat_dme()[0, 0, 0]), np.asarray(1))


def test_tensor_product_pure_state_sets_product_state_and_reduces_density_matrix() -> None:
    ensemble = TensorProductPureStatesEnsemble((2, 3), dtype=jnp.complex128)
    ensemble.set_product_state(
        (
            jnp.asarray([1, 1j], dtype=jnp.complex128) / jnp.sqrt(2),
            jnp.asarray([0, 1, 0], dtype=jnp.complex128),
        )
    )

    rho_s = ensemble.reduced_density_matrix(0)

    expected = np.asarray([[0.5, -0.5j], [0.5j, 0.5]])
    np.testing.assert_allclose(np.asarray(rho_s[0]), expected, atol=1e-12)


def test_tensor_product_density_matrix_reduces_density_matrix() -> None:
    ensemble = TensorProductDensityMatrixEnsemble((2, 2), dtype=jnp.complex128)
    rho = jnp.zeros((4, 4), dtype=jnp.complex128).at[1, 1].set(1)
    ensemble.set_dme(rho)

    reduced = ensemble.reduced_density_matrix(0)

    expected = np.asarray([[1, 0], [0, 0]])
    np.testing.assert_allclose(np.asarray(reduced[0]), expected, atol=1e-12)
