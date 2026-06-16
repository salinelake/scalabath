"""Tests for dense and tensorized simulation classes."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

from scalabath.operators_base import tls
from scalabath.simulations import (
    CoupledLindbladTrajectorySimulation,
    LindbladSimulation,
    SystemBathUnitarySimulation,
    UnitarySimulation,
)
from scalabath.utilities import compose

pytestmark = pytest.mark.unit


def test_unitary_simulation_steps_and_observes_dense_state() -> None:
    local = tls(dtype=jnp.complex128)
    simulation = UnitarySimulation(
        2,
        dt=jnp.pi / 2,
        hamiltonian=local.sigma_x,
        dtype=jnp.complex128,
    )
    simulation.state = jnp.asarray([1, 0], dtype=jnp.complex128)

    state = simulation.step()
    projector_one = jnp.asarray([[0, 0], [0, 1]], dtype=jnp.complex128)

    np.testing.assert_allclose(np.asarray(jnp.abs(state[0]) ** 2), np.asarray([0, 1]), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(simulation.observe(projector_one)),
        np.asarray([1]),
        atol=1e-12,
    )


def test_lindblad_simulation_preserves_trace_for_one_step() -> None:
    local = tls(dtype=jnp.complex128)
    rho0 = jnp.asarray([[0, 0], [0, 1]], dtype=jnp.complex128)
    simulation = LindbladSimulation(
        2,
        dt=0.01,
        jump_operators=[local.sigma_minus],
        dtype=jnp.complex128,
    )
    simulation.density_matrices = rho0

    rho = simulation.step()

    np.testing.assert_allclose(np.asarray(jnp.trace(rho[0])), np.asarray(1.0 + 0.0j), atol=1e-12)


def test_lindblad_simulation_observes_density_matrix_expectation() -> None:
    rho0 = jnp.asarray([[1, 0], [0, 0]], dtype=jnp.complex128)
    observable = jnp.asarray([[1, 0], [0, 0]], dtype=jnp.complex128)
    simulation = LindbladSimulation(2, dt=0.1, dtype=jnp.complex128)
    simulation.density_matrices = rho0

    np.testing.assert_allclose(np.asarray(simulation.observe(observable)), np.asarray([1]))


def test_system_bath_unitary_trotter_matches_dense_local_factorization() -> None:
    sigma_x = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex128)
    number = jnp.diag(jnp.asarray([0, 1], dtype=jnp.complex128))
    identity_s = jnp.eye(2, dtype=jnp.complex128)
    identity_b = jnp.eye(2, dtype=jnp.complex128)
    dt = 0.07
    initial = jnp.zeros((1, 2, 2), dtype=jnp.complex128).at[0, 0, 1].set(1)

    simulation = SystemBathUnitarySimulation(
        2,
        (2,),
        dt,
        system_hamiltonian=sigma_x,
        bath_hamiltonians=[number],
        system_bath_hamiltonians=[compose([sigma_x, number])],
        dtype=jnp.complex128,
    )
    simulation.state = initial

    tensor_state = simulation.step()

    u_s = jsp.linalg.expm(-0.5j * dt * compose([sigma_x, identity_b]))
    u_b = jsp.linalg.expm(-0.5j * dt * compose([identity_s, number]))
    u_sb = jsp.linalg.expm(-1j * dt * compose([sigma_x, number]))
    dense_step = u_b @ u_s @ u_sb @ u_b @ u_s
    expected = (dense_step @ initial[0].reshape(-1)).reshape(2, 2)

    np.testing.assert_allclose(np.asarray(tensor_state[0]), np.asarray(expected), atol=1e-12)


def test_system_bath_thermal_sampling_matches_rubrene_helper() -> None:
    from examples.rubrene.helpers import sample_initial_ensemble

    batch_size = 4
    chain_length = 5
    boson_dims = np.asarray([3, 2], dtype=int)
    omega = np.asarray([0.4, 1.1])
    kbT = 0.7
    dtype = jnp.complex128

    np.random.seed(123)  # noqa: NPY002
    helper_ensemble, helper_levels = sample_initial_ensemble(
        batch_size=batch_size,
        chain_length=chain_length,
        boson_dims=boson_dims,
        omega=omega,
        kbT=kbT,
        dtype=dtype,
    )

    system_state = np.zeros(chain_length, dtype=np.complex128)
    system_state[(chain_length - 1) // 2] = 1.0
    simulation = SystemBathUnitarySimulation(
        chain_length,
        boson_dims.tolist(),
        0.01,
        boson_freqs=omega.tolist(),
        batch_size=batch_size,
        dtype=dtype,
    )

    np.random.seed(123)  # noqa: NPY002
    sampled_ensemble, sampled_levels = simulation.sample_thermal_bath_state(system_state, kbT)

    np.testing.assert_array_equal(sampled_levels, helper_levels)
    np.testing.assert_allclose(
        np.asarray(sampled_ensemble.get_pse()),
        np.asarray(helper_ensemble.get_pse()),
        atol=1e-12,
    )


def test_system_bath_thermal_sampling_accepts_batched_system_states() -> None:
    simulation = SystemBathUnitarySimulation(
        2,
        (1,),
        0.01,
        boson_freqs=(1.0,),
        batch_size=2,
        dtype=jnp.complex128,
    )
    system_states = jnp.asarray([[1, 0], [0, 1]], dtype=jnp.complex128)

    ensemble, chosen_levels = simulation.sample_thermal_bath_state(system_states, kbT=1.0)

    expected = np.zeros((2, 2, 1), dtype=np.complex128)
    expected[0, :, 0] = np.asarray([1, 0], dtype=np.complex128)
    expected[1, :, 0] = np.asarray([0, 1], dtype=np.complex128)
    np.testing.assert_array_equal(chosen_levels, np.zeros((2, 1), dtype=int))
    np.testing.assert_allclose(np.asarray(ensemble.get_pse()), expected, atol=1e-12)


def test_coupled_lindblad_trajectory_accepts_local_bath_hamiltonians() -> None:
    initial = jnp.zeros((1, 2, 2), dtype=jnp.complex128).at[0, 0, 1].set(1)
    simulation = CoupledLindbladTrajectorySimulation(
        2,
        (2,),
        0.01,
        bath_hamiltonians=[jnp.diag(jnp.asarray([0, -0.1j], dtype=jnp.complex128))],
        dtype=jnp.complex128,
    )
    simulation.state = initial

    state = simulation.step()

    assert state.shape == (1, 2, 2)
