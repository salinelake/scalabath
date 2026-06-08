"""Tests for minimal simulation classes."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from scalabath.operators_base import tls
from scalabath.operators_groups import OperatorGroup, SpinOperatorGroup
from scalabath.simulations import LindbladSimulation, UnitarySimulation

pytestmark = pytest.mark.unit


def test_unitary_simulation_steps_and_observes() -> None:
    field = SpinOperatorGroup(1, "field")
    field.add_operator("X")
    simulation = UnitarySimulation(jnp.asarray([1, 0], dtype=jnp.complex128), dt=jnp.pi / 2)
    simulation.add_operator_group_to_hamiltonian(field)

    state = simulation.step()
    projector_one = OperatorGroup("projector-one", 2)
    projector_one.add_operator(jnp.asarray([[0, 0], [0, 1]], dtype=jnp.complex128))

    np.testing.assert_allclose(np.asarray(jnp.abs(state[0]) ** 2), np.asarray([0, 1]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(simulation.observe(projector_one)), np.asarray([1]))


def test_lindblad_simulation_preserves_trace_for_one_step() -> None:
    local = tls()
    rho0 = jnp.asarray([[0, 0], [0, 1]], dtype=jnp.complex128)
    jump = OperatorGroup("jump", 2)
    jump.add_operator(local.sigma_minus)
    simulation = LindbladSimulation(rho0, dt=0.01, jump_operators=[jump])

    rho = simulation.step()

    np.testing.assert_allclose(np.asarray(jnp.trace(rho[0])), np.asarray(1.0 + 0.0j))


def test_lindblad_simulation_observes_density_matrix_expectation() -> None:
    rho0 = jnp.asarray([[1, 0], [0, 0]], dtype=jnp.complex128)
    observable = OperatorGroup("projector-zero", 2)
    observable.add_operator(jnp.asarray([[1, 0], [0, 0]], dtype=jnp.complex128))
    simulation = LindbladSimulation(rho0, dt=0.1)

    np.testing.assert_allclose(np.asarray(simulation.observe(observable)), np.asarray([1]))
