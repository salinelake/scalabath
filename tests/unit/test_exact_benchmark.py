"""Tests for the explicit-bath exact benchmark helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from scalabath.simulations_unitary import SystemBathUnitarySimulation

EXACT_HELPERS_PATH = (
    Path(__file__).resolve().parents[2] / "examples" / "benchmark" / "exact" / "helpers.py"
)
SPEC = importlib.util.spec_from_file_location("exact_benchmark_helpers", EXACT_HELPERS_PATH)
assert SPEC is not None and SPEC.loader is not None
EXACT_HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXACT_HELPERS)


def test_thermal_product_state_sampling_is_reproducible() -> None:
    system_state = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.complex128)
    arguments = (system_state, [1.0, 2.0], [2, 3], 1.5, 2, jnp.complex128)

    np.random.seed(17)  # noqa: NPY002
    states_a, levels_a = EXACT_HELPERS.sample_thermal_product_states(*arguments)
    np.random.seed(17)  # noqa: NPY002
    states_b, levels_b = EXACT_HELPERS.sample_thermal_product_states(*arguments)

    np.testing.assert_array_equal(levels_a, levels_b)
    np.testing.assert_array_equal(np.asarray(states_a), np.asarray(states_b))
    np.testing.assert_allclose(np.linalg.norm(np.asarray(states_a).reshape(2, -1), axis=1), 1.0)


def test_local_system_bath_hamiltonians_couple_each_mode_to_its_own_site() -> None:
    simulation = SystemBathUnitarySimulation(
        system_dim=3,
        boson_dims=(2, 2, 2),
        dt=0.1,
        dtype=jnp.complex128,
    )

    hamiltonians = EXACT_HELPERS.build_local_system_bath_hamiltonians(
        simulation,
        coupling=0.25,
    )

    expected_local = 0.25 * np.asarray(
        simulation.boson_basis[0].annihilation + simulation.boson_basis[0].creation
    )
    assert len(hamiltonians) == 3
    for mode_index, hamiltonian in enumerate(hamiltonians):
        expected = np.zeros((3, 2, 2), dtype=np.complex128)
        expected[mode_index] = expected_local
        np.testing.assert_allclose(np.asarray(hamiltonian), expected)
