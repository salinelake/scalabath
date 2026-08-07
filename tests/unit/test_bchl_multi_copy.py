"""Tests for the R-copy alternating BCHL-chain example helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from scalabath.operators_base import boson

BCHL_HELPERS_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "BCHL_chain_alternating_multi_copy"
    / "helpers.py"
)
SPEC = importlib.util.spec_from_file_location("bchl_multi_copy_helpers", BCHL_HELPERS_PATH)
assert SPEC is not None and SPEC.loader is not None
BCHL_HELPERS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BCHL_HELPERS)


def test_independent_bath_mode_groups_are_copy_major() -> None:
    groups = BCHL_HELPERS.independent_bath_mode_groups(num_copies=3, modes_per_copy=2)

    assert groups == ((0, 1), (2, 3), (4, 5))


def test_system_bath_couplings_use_copy_phases_and_inverse_sqrt_scaling() -> None:
    local_bases = [boson(1, dtype=jnp.complex128) for _ in range(2)]
    phases = np.asarray([[[0.0, 0.5 * np.pi], [np.pi, -0.5 * np.pi]]])
    coupling = np.asarray([2.0])

    hamiltonians = BCHL_HELPERS.build_system_bath_hamiltonians(
        local_bases,
        coupling,
        phases,
        chain_length=2,
        modes_per_copy=1,
        dtype=jnp.complex128,
    )

    scaled_coupling = coupling[0] / np.sqrt(2)
    for copy_index, hamiltonian in enumerate(hamiltonians):
        expected = np.zeros((1, 2, 2, 2), dtype=np.complex128)
        for site_index in range(2):
            phase = phases[0, copy_index, site_index]
            expected[0, site_index] = scaled_coupling * (
                np.exp(1j * phase) * np.asarray(local_bases[copy_index].creation)
                + np.exp(-1j * phase) * np.asarray(local_bases[copy_index].annihilation)
            )
        np.testing.assert_allclose(np.asarray(hamiltonian), expected, atol=1e-12)


def test_jump_operators_repeat_each_copy_damping_rate() -> None:
    local_bases = [boson(1, dtype=jnp.complex128) for _ in range(4)]
    rates = np.asarray([0.2, 0.3])

    jump_operators = BCHL_HELPERS.build_jump_operators(
        local_bases,
        rates,
        modes_per_copy=2,
    )

    assert len(jump_operators) == 4
    for mode_index, jump_operator in enumerate(jump_operators):
        expected = np.sqrt(2 * rates[mode_index % 2]) * np.asarray(
            local_bases[mode_index].annihilation
        )
        np.testing.assert_allclose(np.asarray(jump_operator), expected, atol=1e-12)
