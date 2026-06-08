"""Operator-group helpers for building static Hamiltonian terms."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import mul
from typing import Any

import jax.numpy as jnp
from jax import Array

from scalabath.operators_base import boson, tight_binding_1d, tls
from scalabath.utilities import compose


class OperatorGroup:
    """Base class for a static sum of operators on one subsystem.

    Args:
        id: Human-readable identifier for diagnostics and model construction.
        hilbert_dim: Dimension of the subsystem Hilbert space.
        dtype: Complex dtype used for operator matrices.
    """

    def __init__(
        self,
        id: str,
        hilbert_dim: int,
        *,
        dtype: Any = jnp.complex128,
    ) -> None:
        self.id = id
        self.hilbert_dim = int(hilbert_dim)
        if self.hilbert_dim <= 0:
            raise ValueError("hilbert_dim must be positive")
        self.dtype = jnp.dtype(dtype)
        if not jnp.issubdtype(self.dtype, jnp.complexfloating):
            raise ValueError("OperatorGroup requires a complex dtype")
        self._operators: list[Array] = []
        self._prefactors: list[complex] = []

    @property
    def ns(self) -> int:
        """Alias for ``hilbert_dim``."""

        return self.hilbert_dim

    @property
    def n_terms(self) -> int:
        """Number of terms stored in this group."""

        return len(self._operators)

    def add_operator(self, operator: Any, prefactor: complex = 1.0) -> None:
        """Add an operator matrix to the static group."""

        operator_array = jnp.asarray(operator, dtype=self.dtype)
        expected_shape = (self.hilbert_dim, self.hilbert_dim)
        if operator_array.shape != expected_shape:
            raise ValueError(f"operator must have shape {expected_shape}")
        self._operators.append(operator_array)
        self._prefactors.append(prefactor)

    def sum_operators(self) -> Array:
        """Return the weighted sum of all operators in the group."""

        total = jnp.zeros((self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        for operator, prefactor in zip(self._operators, self._prefactors, strict=True):
            total = total + jnp.asarray(prefactor, dtype=self.dtype) * operator
        return total


class BosonOperatorGroup(OperatorGroup):
    """Static operator group for a multi-mode bosonic subsystem."""

    def __init__(
        self,
        num_modes: int,
        id: str,
        nmax: int,
        *,
        dtype: Any = jnp.complex128,
    ) -> None:
        self.num_modes = int(num_modes)
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")
        self.nmax = int(nmax)
        self.local = boson(nmax, dtype=dtype)
        super().__init__(id, self.local.dim**self.num_modes, dtype=dtype)
        self._descriptions: list[str] = []

    def add_operator(self, boson_sequence: str, prefactor: complex = 1.0) -> None:
        """Add a product operator described by a sequence such as ``"UNI"``."""

        if len(boson_sequence) != self.num_modes:
            raise ValueError("boson_sequence length must equal num_modes")
        operator = self.local.get_composite_ops(boson_sequence)
        super().add_operator(operator, prefactor)
        self._descriptions.append(boson_sequence)


class SpinOperatorGroup(OperatorGroup):
    """Static operator group for a tensor product of two-level systems."""

    def __init__(
        self,
        num_spins: int,
        id: str,
        *,
        dtype: Any = jnp.complex128,
    ) -> None:
        self.num_spins = int(num_spins)
        if self.num_spins <= 0:
            raise ValueError("num_spins must be positive")
        self.local = tls(dtype=dtype)
        super().__init__(id, 2**self.num_spins, dtype=dtype)
        self._descriptions: list[str] = []

    def add_operator(self, spin_sequence: str, prefactor: complex = 1.0) -> None:
        """Add a product operator described by a sequence such as ``"XIZ"``."""

        if len(spin_sequence) != self.num_spins:
            raise ValueError("spin_sequence length must equal num_spins")
        operator = self.local.get_composite_ops(spin_sequence)
        super().add_operator(operator, prefactor)
        self._descriptions.append(spin_sequence)


class TightBindingOperatorGroup(OperatorGroup):
    """Static operator group for a 1D single-particle tight-binding subsystem."""

    def __init__(
        self,
        n_sites: int,
        id: str,
        *,
        periodic: bool = True,
        dtype: Any = jnp.complex128,
    ) -> None:
        self.n_sites = int(n_sites)
        self.local = tight_binding_1d(n_sites, periodic=periodic, dtype=dtype)
        super().__init__(id, self.local.hilbert_dim, dtype=dtype)
        self._descriptions: list[str] = []

    def add_operator(self, tight_binding_sequence: str, prefactor: complex = 1.0) -> None:
        """Add a sequence-defined tight-binding operator such as ``"XRX"``."""

        operator = self.local.get_composite_ops(tight_binding_sequence)
        super().add_operator(operator, prefactor)
        self._descriptions.append(tight_binding_sequence)


class ComposedOperatorGroups(OperatorGroup):
    """Tensor product of subsystem operator groups.

    This class is useful for terms such as ``n_j`` on a lattice subsystem tensored
    with a bosonic coordinate operator on an environment subsystem.
    """

    def __init__(
        self,
        id: str,
        operator_groups: Sequence[OperatorGroup],
        *,
        dtype: Any | None = None,
    ) -> None:
        if len(operator_groups) < 2:
            raise ValueError("ComposedOperatorGroups requires at least two groups")
        self.operator_groups = tuple(operator_groups)
        hilbert_dim = reduce(mul, (group.hilbert_dim for group in self.operator_groups), 1)
        inferred_dtype = dtype or jnp.result_type(*(group.dtype for group in self.operator_groups))
        super().__init__(id, hilbert_dim, dtype=inferred_dtype)

    def add_operator(self, operator: Any, prefactor: complex = 1.0) -> None:
        """Composed groups are defined by their subsystem groups."""

        raise ValueError("cannot add an operator directly to ComposedOperatorGroups")

    def sum_operators(self) -> Array:
        """Return the tensor product of subsystem group sums."""

        return compose([group.sum_operators() for group in self.operator_groups])


__all__ = [
    "BosonOperatorGroup",
    "ComposedOperatorGroups",
    "OperatorGroup",
    "SpinOperatorGroup",
    "TightBindingOperatorGroup",
]
