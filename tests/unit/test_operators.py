"""Tests for base and grouped operators."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from scalabath.operators_base import boson, tight_binding_1d, tight_binding_2d, tls
from scalabath.operators_groups import (
    BosonOperatorGroup,
    ComposedOperatorGroups,
    SpinOperatorGroup,
    TightBindingChainOperatorGroup,
)
from scalabath.simulations_unitary import UnitarySimulation

pytestmark = pytest.mark.unit


def test_boson_operator_matrices_have_expected_cutoff_values() -> None:
    local = boson(2)

    expected_annihilation = np.asarray(
        [[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=np.complex128
    )

    np.testing.assert_allclose(np.asarray(local.annihilation), expected_annihilation)
    np.testing.assert_allclose(np.asarray(local.creation), expected_annihilation.conj().T)
    np.testing.assert_allclose(np.asarray(local.number), np.diag([0, 1, 2]))


def test_tls_pauli_matrices_and_spin_group_sum() -> None:
    local = tls()
    group = SpinOperatorGroup(1, "field-x")
    group.add_operator("X", prefactor=0.5)

    group_sum = group.sum_operators()

    np.testing.assert_allclose(np.asarray(local.sigma_x), np.asarray([[0, 1], [1, 0]]))
    assert group_sum.shape == (1, 2, 2)
    np.testing.assert_allclose(np.asarray(group_sum[0]), 0.5 * np.asarray(local.sigma_x))


def test_tight_binding_1d_sequence_conventions() -> None:
    local = tight_binding_1d(3, periodic=False)

    right_from_middle = local.get_operator("XRX")
    left_from_edge = local.get_operator("LXX")

    expected_right = np.zeros((3, 3), dtype=np.complex128)
    expected_right[2, 1] = 1
    np.testing.assert_allclose(np.asarray(right_from_middle), expected_right)
    np.testing.assert_allclose(np.asarray(left_from_edge), np.zeros((3, 3)))


def test_tight_binding_2d_maps_sites_and_hops() -> None:
    local = tight_binding_2d(2, 3, periodic=False)
    hop = local.hopping((0, 1), (1, 1))

    expected = np.zeros((6, 6), dtype=np.complex128)
    expected[4, 1] = 1
    assert local.site_index((1, 1)) == 4
    np.testing.assert_allclose(np.asarray(hop), expected)


def test_tight_binding_2d_triangular_hopping_open_boundaries() -> None:
    local = tight_binding_2d(3, 3, periodic=False, dtype=jnp.complex128)
    amp_x = 2.0 + 0.5j
    amp_y = -0.75 + 1.25j

    hopping = local.triangular_hopping((amp_x, amp_y))

    expected = np.zeros((9, 9), dtype=np.complex128)

    def add_hopping(source: tuple[int, int], target: tuple[int, int], amplitude: complex) -> None:
        source_idx = local.site_index(source)
        target_idx = local.site_index(target)
        expected[target_idx, source_idx] += amplitude
        expected[source_idx, target_idx] += np.conjugate(amplitude)

    for x in range(3):
        for y in range(3):
            if x + 1 < 3:
                add_hopping((x, y), (x + 1, y), amp_x)
            if y + 1 < 3:
                add_hopping((x, y), (x, y + 1), amp_y)
            if x + 1 < 3 and y + 1 < 3:
                add_hopping((x, y), (x + 1, y + 1), amp_y)

    np.testing.assert_allclose(np.asarray(hopping), expected)
    np.testing.assert_allclose(np.asarray(hopping), np.asarray(hopping).conj().T)


def test_operator_groups_compose_subsystems() -> None:
    spin_group = SpinOperatorGroup(1, "spin-x")
    spin_group.add_operator("X")
    boson_group = BosonOperatorGroup(1, "boson-n", nmax=1)
    boson_group.add_operator("N")

    composed = ComposedOperatorGroups("spin-boson", [spin_group, boson_group])

    spin_sum = spin_group.sum_operators()
    boson_sum = boson_group.sum_operators()
    expected = jnp.kron(spin_sum[0], boson_sum[0])[None, :, :]
    composed_sum = composed.sum_operators()

    assert spin_sum.shape == (1, 2, 2)
    assert boson_sum.shape == (1, 2, 2)
    assert composed_sum.shape == (1, 4, 4)
    np.testing.assert_allclose(np.asarray(composed_sum), np.asarray(expected))


def test_tight_binding_group_sums_static_terms() -> None:
    group = TightBindingChainOperatorGroup(3, "tb", periodic=False)
    group.add_operator("XRX", prefactor=2.0)
    group.add_operator("XXN", prefactor=0.5)

    expected = np.zeros((3, 3), dtype=np.complex128)
    expected[2, 1] = 2
    expected[2, 2] = 0.5
    group_sum = group.sum_operators()

    assert group_sum.shape == (1, 3, 3)
    np.testing.assert_allclose(np.asarray(group_sum[0]), expected)


def test_singleton_operator_group_batch_is_not_broadcast_to_larger_simulation_batch() -> None:
    group = SpinOperatorGroup(1, "field-x", batch_size=1, dtype=jnp.complex128)
    group.add_operator("X")
    simulation = UnitarySimulation(2, batch_size=2, dtype=jnp.complex128)

    with pytest.raises(ValueError, match="operator must have shape"):
        simulation.add_operator_group_to_hamiltonian(group)
