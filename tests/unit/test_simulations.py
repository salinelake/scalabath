"""Tests for dense and tensorized simulation classes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from scalabath.operators_base import tls
from scalabath.simulations_lindblad import (
    CoupledLindbladTrajectorySimulation,
    LindbladSimulation,
)
from scalabath.simulations_unitary import (
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
    projector_zero = jnp.asarray([[1, 0], [0, 0]], dtype=jnp.complex128)
    identity_s = jnp.eye(2, dtype=jnp.complex128)
    identity_b = jnp.eye(2, dtype=jnp.complex128)
    zero_b = jnp.zeros_like(number)
    system_bath_blocks = jnp.stack([number, zero_b])
    dt = 0.07
    initial = jnp.zeros((1, 2, 2), dtype=jnp.complex128).at[0, 0, 1].set(1)

    simulation = SystemBathUnitarySimulation(
        2,
        (2,),
        dt,
        system_hamiltonian=sigma_x,
        bath_hamiltonians=[number],
        system_bath_hamiltonians=[system_bath_blocks],
        dtype=jnp.complex128,
    )
    simulation.state = initial

    tensor_state = simulation.step()

    u_s = jsp.linalg.expm(-0.5j * dt * compose([sigma_x, identity_b]))
    u_b = jsp.linalg.expm(-0.5j * dt * compose([identity_s, number]))
    u_sb = jsp.linalg.expm(-1j * dt * compose([projector_zero, number]))
    dense_step = u_b @ u_s @ u_sb @ u_b @ u_s
    expected = (dense_step @ initial[0].reshape(-1)).reshape(2, 2)

    np.testing.assert_allclose(np.asarray(tensor_state[0]), np.asarray(expected), atol=1e-12)


def test_system_bath_unitary_trotter_accepts_batched_system_bath_blocks() -> None:
    mode_x = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex128)
    mode_z = jnp.asarray([[1, 0], [0, -1]], dtype=jnp.complex128)
    zero_mode = jnp.zeros_like(mode_x)
    blocks = jnp.asarray(
        [
            [mode_x, zero_mode],
            [zero_mode, mode_z],
        ],
        dtype=jnp.complex128,
    )
    dt = 0.03
    initial = jnp.zeros((2, 2, 2), dtype=jnp.complex128)
    initial = initial.at[0, 0, 0].set(1)
    initial = initial.at[1, 1, 1].set(1)
    simulation = SystemBathUnitarySimulation(
        2,
        (2,),
        dt,
        batch_size=2,
        system_bath_hamiltonians=[blocks],
        dtype=jnp.complex128,
    )
    simulation.state = initial

    tensor_state = simulation.step()

    expected = []
    for batch_index in range(2):
        dense_hamiltonian = compose(
            [
                jnp.asarray([[1, 0], [0, 0]], dtype=jnp.complex128),
                blocks[batch_index, 0],
            ]
        ) + compose(
            [
                jnp.asarray([[0, 0], [0, 1]], dtype=jnp.complex128),
                blocks[batch_index, 1],
            ]
        )
        dense_step = jsp.linalg.expm(-1j * dt * dense_hamiltonian)
        expected.append((dense_step @ initial[batch_index].reshape(-1)).reshape(2, 2))

    np.testing.assert_allclose(np.asarray(tensor_state), np.asarray(expected), atol=1e-12)


def test_system_bath_unitary_trotter_compacts_dense_block_diagonal_coupling() -> None:
    number = jnp.diag(jnp.asarray([0, 1], dtype=jnp.complex128))
    projector_zero = jnp.asarray([[1, 0], [0, 0]], dtype=jnp.complex128)
    coupling = compose([projector_zero, number])
    dt = 0.05
    initial = jnp.zeros((1, 2, 2), dtype=jnp.complex128).at[0, 0, 1].set(1)
    simulation = SystemBathUnitarySimulation(
        2,
        (2,),
        dt,
        system_bath_hamiltonians=[coupling],
        dtype=jnp.complex128,
    )
    simulation.state = initial

    tensor_state = simulation.step()

    assert simulation._system_bath_full[0].shape == (2, 2, 2)
    assert simulation.system_bath_hamiltonians is None
    expected = (jsp.linalg.expm(-1j * dt * coupling) @ initial[0].reshape(-1)).reshape(2, 2)
    np.testing.assert_allclose(np.asarray(tensor_state[0]), np.asarray(expected), atol=1e-12)


def test_system_bath_unitary_trotter_keeps_non_block_diagonal_dense_coupling() -> None:
    sigma_x = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.complex128)
    number = jnp.diag(jnp.asarray([0, 1], dtype=jnp.complex128))
    coupling = compose([sigma_x, number])
    dt = 0.05
    initial = jnp.zeros((1, 2, 2), dtype=jnp.complex128).at[0, 0, 1].set(1)
    simulation = SystemBathUnitarySimulation(
        2,
        (2,),
        dt,
        system_bath_hamiltonians=[coupling],
        dtype=jnp.complex128,
    )
    simulation.state = initial

    tensor_state = simulation.step()

    assert simulation._system_bath_full[0].shape == (4, 4)
    expected = (jsp.linalg.expm(-1j * dt * coupling) @ initial[0].reshape(-1)).reshape(2, 2)
    np.testing.assert_allclose(np.asarray(tensor_state[0]), np.asarray(expected), atol=1e-12)


def test_system_bath_unitary_replicates_propagators_on_state_mesh() -> None:
    devices = np.asarray(jax.devices())
    mesh = Mesh(devices, ("system",))
    state_sharding = NamedSharding(mesh, P(None, "system", None))
    replicated_sharding = NamedSharding(mesh, P())
    system_dim = 2 * len(devices)
    mode_dim = 2
    system_hamiltonian = jnp.eye(system_dim, dtype=jnp.complex64)
    bath_hamiltonian = jnp.diag(jnp.asarray([0, 1], dtype=jnp.complex64))
    coupling_blocks = jnp.zeros((system_dim, mode_dim, mode_dim), dtype=jnp.complex64)
    initial = jnp.zeros((1, system_dim, mode_dim), dtype=jnp.complex64).at[0, 0, 0].set(1)
    simulation = SystemBathUnitarySimulation(
        system_dim,
        (mode_dim,),
        0.01,
        system_hamiltonian=system_hamiltonian,
        bath_hamiltonians=[bath_hamiltonian],
        system_bath_hamiltonians=[coupling_blocks],
        dtype=jnp.complex64,
    )
    simulation._pse.sharding = state_sharding
    simulation.state = initial

    simulation.step()

    assert simulation.state.sharding.is_equivalent_to(state_sharding, ndim=3)
    assert simulation._system_half.sharding.is_equivalent_to(replicated_sharding, ndim=2)
    assert simulation._bath_half[0].sharding.is_equivalent_to(replicated_sharding, ndim=2)
    assert simulation._system_bath_full[0].sharding.is_equivalent_to(
        replicated_sharding,
        ndim=3,
    )
    assert simulation._matrix_system_full.sharding.is_equivalent_to(
        replicated_sharding,
        ndim=2,
    )
    assert simulation._matrix_bath_full_by_mode[0].sharding.is_equivalent_to(
        replicated_sharding,
        ndim=2,
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


def test_system_bath_thermal_sampling_uses_state_sharding() -> None:
    devices = np.asarray(jax.devices())
    system_dim = 2 * len(devices)
    sharding = NamedSharding(Mesh(devices, ("system",)), P(None, "system", None))
    simulation = SystemBathUnitarySimulation(
        system_dim,
        (1,),
        0.01,
        boson_freqs=(1.0,),
        dtype=jnp.complex64,
    )
    simulation._pse.sharding = sharding
    system_state = jnp.zeros(system_dim, dtype=jnp.complex64).at[0].set(1)

    ensemble, _ = simulation.sample_thermal_bath_state(system_state, kbT=1.0)

    assert ensemble.get_pse().sharding.is_equivalent_to(sharding, ndim=3)


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


def test_coupled_lindblad_trajectory_accepts_full_bath_hamiltonian() -> None:
    initial = jnp.zeros((1, 2, 2, 2), dtype=jnp.complex128).at[0, 0, 1, 0].set(1)
    system_hamiltonian = jnp.asarray([[0.0, 0.2], [0.2, 0.0]], dtype=jnp.complex128)
    bath_hamiltonian = jnp.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.1 - 0.03j, 0.04, 0.0],
            [0.0, 0.04, 0.3 - 0.05j, 0.0],
            [0.0, 0.0, 0.0, 0.4 - 0.08j],
        ],
        dtype=jnp.complex128,
    )
    projector_zero = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
    mode_coupling = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    zero_coupling = jnp.zeros_like(mode_coupling)
    coupling = compose([projector_zero, mode_coupling])
    coupling_blocks = jnp.stack([mode_coupling, zero_coupling])
    zero_blocks = jnp.zeros_like(coupling_blocks)
    simulation = CoupledLindbladTrajectorySimulation(
        2,
        (2, 2),
        0.03,
        system_hamiltonian=system_hamiltonian,
        bath_hamiltonian=bath_hamiltonian,
        system_bath_hamiltonians=[coupling_blocks, zero_blocks],
        dtype=jnp.complex128,
    )
    simulation.state = initial
    simulation._sample_thresholds = lambda: jnp.asarray([-1.0], dtype=jnp.float32)

    state = simulation.step()

    u_s = jsp.linalg.expm(-1j * 0.03 * system_hamiltonian)
    u_b = jsp.linalg.expm(-1j * 0.03 * bath_hamiltonian)
    u_sb = jsp.linalg.expm(-1j * 0.03 * compose([coupling, jnp.eye(2, dtype=jnp.complex128)]))
    dense_step = (
        u_sb
        @ compose([jnp.eye(2, dtype=jnp.complex128), u_b])
        @ compose([u_s, jnp.eye(4, dtype=jnp.complex128)])
    )
    expected = (dense_step @ initial.reshape(-1)).reshape(1, 2, 2, 2)
    expected = expected / jnp.linalg.norm(expected.reshape(1, -1), axis=1).reshape(1, 1, 1, 1)

    np.testing.assert_allclose(np.asarray(state), np.asarray(expected), atol=1e-12)


def test_coupled_lindblad_trajectory_uses_fresh_step_thresholds() -> None:
    initial = jnp.zeros((2, 2, 2), dtype=jnp.complex128)
    initial = initial.at[0, 0, 1].set(1)
    initial = initial.at[1, 1, 0].set(1)
    simulation = CoupledLindbladTrajectorySimulation(
        2,
        (2,),
        0.01,
        batch_size=2,
        key=jax.random.PRNGKey(123),
        dtype=jnp.complex128,
    )
    simulation.state = initial
    thresholds = jnp.asarray([2.0, -1.0], dtype=jnp.float32)
    simulation._sample_thresholds = lambda: thresholds

    state = simulation.step()

    np.testing.assert_allclose(np.asarray(state[0, 0]), np.asarray([1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(state[1, 1]), np.asarray([1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(simulation.thresholds), np.asarray(thresholds))


def test_coupled_lindblad_trajectory_uses_squared_norm_jump_threshold() -> None:
    damping_rate = 0.4
    dt = 0.2
    bath_hamiltonian = jnp.diag(jnp.asarray([0.0, -0.5j * damping_rate], dtype=jnp.complex128))
    simulation = CoupledLindbladTrajectorySimulation(
        1,
        (2,),
        dt,
        bath_hamiltonian=bath_hamiltonian,
        dtype=jnp.complex128,
    )
    simulation.state = jnp.zeros((1, 1, 2), dtype=jnp.complex128).at[0, 0, 1].set(1.0)
    simulation._sample_thresholds = lambda: jnp.asarray([0.94], dtype=jnp.float32)

    state = simulation.step()

    np.testing.assert_allclose(np.asarray(jnp.abs(state[0, 0]) ** 2), [1.0, 0.0], atol=1e-12)


def test_coupled_lindblad_trajectory_normalizes_no_jump_and_samples_fresh_threshold() -> None:
    damping_rate = 0.4
    dt = 0.2
    bath_hamiltonian = jnp.diag(jnp.asarray([0.0, -0.5j * damping_rate], dtype=jnp.complex128))
    simulation = CoupledLindbladTrajectorySimulation(
        1,
        (2,),
        dt,
        bath_hamiltonian=bath_hamiltonian,
        dtype=jnp.complex128,
    )
    simulation.state = jnp.zeros((1, 1, 2), dtype=jnp.complex128).at[0, 0, 1].set(1.0)
    threshold_sequence = iter(
        [
            jnp.asarray([0.88], dtype=jnp.float32),
            jnp.asarray([0.94], dtype=jnp.float32),
        ]
    )
    simulation._sample_thresholds = lambda: next(threshold_sequence)

    first_state = simulation.step()

    np.testing.assert_allclose(
        np.asarray(jnp.abs(first_state[0, 0]) ** 2),
        [0.0, 1.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(np.asarray(simulation.thresholds), np.asarray([0.88]))

    second_state = simulation.step()

    np.testing.assert_allclose(
        np.asarray(jnp.abs(second_state[0, 0]) ** 2),
        [1.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(np.asarray(simulation.thresholds), np.asarray([0.94]))


def test_coupled_lindblad_trajectory_normalizes_tensor_state() -> None:
    simulation = CoupledLindbladTrajectorySimulation(2, (2,), 0.01, dtype=jnp.complex128)
    simulation.state = jnp.zeros((1, 2, 2), dtype=jnp.complex128).at[0, 1, 1].set(2.0)

    simulation.normalize()

    norm = jnp.linalg.norm(simulation.state.reshape(1, -1), axis=1)
    np.testing.assert_allclose(np.asarray(norm), [1.0])
