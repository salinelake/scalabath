"""Tests for state ensemble containers."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from scalabath.systems import DensityMatrixEnsemble, PureStatesEnsemble

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
