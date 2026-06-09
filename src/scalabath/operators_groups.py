"""Operator-group helpers for building static Hamiltonian terms."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import mul
from typing import Any

import jax.numpy as jnp
from jax import Array

from scalabath.operators_base import boson, tight_binding_1d, tls
from scalabath.utilities import compose, positive_int, nonnegative_int, complex_dtype


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
        batch_size: int = 1,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.id = id
        self.hilbert_dim = positive_int(hilbert_dim, "hilbert_dim")
        self.batch_size = positive_int(batch_size, "batch_size")
        self.dtype = complex_dtype(dtype)
        self._descriptors: list[str] = []
        self._prefactors: list[complex] = []

    @property
    def ns(self) -> int:
        """Alias for ``hilbert_dim``."""

        return self.hilbert_dim

    @property
    def n_terms(self) -> int:
        """Number of terms stored in this group."""

        return len(self._descriptors)

    def add_operator(self, descriptor: str, prefactor: Array) -> None:
        """Add an operator (with a prefactor) to the group. Description of the operator is stored in self._descriptors. To be implemented in subclasses.
        Args:
            descriptor: str, the descriptor of the operator to be added. Example: "XIII" denotes the operator :math:`\sigma_x \otimes identity \otimes identity \otimes identity` for a spin-1/2 system.
            prefactor: Array, the prefactor of the operator of shape (self.batch_size,).
        """
        raise NotImplementedError("add_operator is not implemented in the base class")

    def sum_operators(self) -> Array:
        """Sum up the operators in the group. To be implemented in subclasses.
        Returns:
            total_ops: Array, the total operator matrix of shape (self.hilbert_dim, self.hilbert_dim).
        """
        raise NotImplementedError("sum_operators is not implemented in the base class")
    


class BosonOperatorGroup(OperatorGroup):
    """Static operator group for a multi-mode bosonic subsystem."""

    def __init__(
        self,
        num_modes: int,
        id: str,
        nmax: tuple[int, ...],
        *,
        batch_size: int = 1,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.num_modes = positive_int(num_modes, "num_modes")
        self.nmax = tuple(nonnegative_int(n, "n") for n in nmax)
        if len(self.nmax) != self.num_modes:
            raise ValueError("nmax must have the same length as num_modes")
        self.local = []
        hilbert_dim = 1
        for n in self.nmax:
            self.local.append(boson(n, dtype=dtype))
            hilbert_dim *= self.local[-1].dim
        super().__init__(id, hilbert_dim, batch_size=batch_size, dtype=dtype)

    def add_operator(self, descriptor: str, prefactor: Array) -> None:
        """Add a product operator described by a descriptor such as ``"UNI"``. This overrides the base class method.
        Args:
            descriptor: str, the descriptor of the operator to be added. Example: "UDI" denotes the operator :math:`b^\dagger_0 \otimes b_1 \otimes I_2` for a boson system.
            prefactor: Array, the prefactor of the operator of shape (self.batch_size,).
        """

        if len(descriptor) != self.num_modes:
            raise ValueError("descriptor length must equal num_modes")
        if prefactor.shape[0] != self.batch_size:
            raise ValueError("prefactor shape must equal batch_size")
        for idx,d in enumerate(descriptor):
            if d not in self.local[idx].descriptors_dict:
                raise ValueError(f"unknown boson operator descriptor {d!r}")
        self._descriptors.append(descriptor)
        self._prefactors.append(prefactor)
        return

    def sum_operators(self) -> Array:
        """Sum up the operators in the group. This overrides the base class method.
        Returns:
            total_ops: Array, the total operator matrix of shape (self.batch_size, self.hilbert_dim, self.hilbert_dim).
        """
        total_ops = jnp.zeros((self.batch_size, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        for descriptor, prefactor in zip(self._descriptors, self._prefactors):
            ops = [self.local[idx].get_operator(d) for idx, d in enumerate(descriptor)]
            total_ops += compose(ops)[None, :, :] * prefactor[:, None, None]
        return total_ops

class SpinOperatorGroup(OperatorGroup):
    """Static operator group for a tensor product of two-level systems."""

    def __init__(
        self,
        num_spins: int,
        id: str,
        *,
        batch_size: int = 1,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.num_spins = positive_int(num_spins, "num_spins")
        self.local = tls(dtype=dtype)
        super().__init__(id, 2**self.num_spins, batch_size=batch_size, dtype=dtype)

    def add_operator(self, descriptor: str, prefactor: Array) -> None:
        """Add a product operator described by a sequence such as ``"XIZ"``. This overrides the base class method.
        Args:
            descriptor: str, the descriptor of the operator to be added. Example: "XIZ" denotes the operator :math:`\sigma_x \otimes \sigma_z \otimes \sigma_z` for a spin-1/2 system.
            prefactor: Array, the prefactor of the operator of shape (self.batch_size,).
        """

        if len(descriptor) != self.num_spins:
            raise ValueError("descriptor length must equal num_spins")
        if prefactor.shape[0] != self.batch_size:
            raise ValueError("prefactor shape must equal batch_size")
        for d in descriptor:
            if d not in self.local.descriptors_dict:
                raise ValueError(f"unknown spin operator descriptor {d!r}")
        self._descriptors.append(descriptor)
        self._prefactors.append(prefactor)
        return

    def sum_operators(self) -> Array:
        """Sum up the operators in the group. This overrides the base class method.
        Returns:
            total_ops: Array, the total operator matrix of shape (self.batch_size, self.hilbert_dim, self.hilbert_dim).
        """
        total_ops = jnp.zeros((self.batch_size, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        for descriptor, prefactor in zip(self._descriptors, self._prefactors):
            ops = [self.local.get_operator(d) for d in descriptor]
            total_ops += compose(ops)[None, :, :] * prefactor[:, None, None]
        return total_ops

class TightBindingChainOperatorGroup(OperatorGroup):
    """Static operator group for a 1D single-particle tight-binding subsystem."""

    def __init__(
        self,
        n_sites: int,
        id: str,
        *,
        batch_size: int = 1,
        periodic: bool = True,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.n_sites = positive_int(n_sites, "n_sites")
        self.local = tight_binding_1d(n_sites, periodic=periodic, dtype=dtype)
        super().__init__(id, self.local.hilbert_dim, batch_size=batch_size, dtype=dtype)

    def add_operator(self, descriptor: str, prefactor: Array) -> None:
        """Add a sequence-defined tight-binding operator such as ``"XRX"``. This overrides the base class method."""
        if len(descriptor) != self.n_sites:
            raise ValueError("descriptor length must equal n_sites")
        if prefactor.shape[0] != self.batch_size:
            raise ValueError("prefactor shape must equal batch_size")
        for d in descriptor:
            if d not in self.local.descriptors_dict:
                raise ValueError(f"unknown tight-binding operator descriptor {d!r}")
        self._descriptors.append(descriptor)
        self._prefactors.append(prefactor)
        return

    def sum_operators(self) -> Array:
        """Sum up the operators in the group. This overrides the base class method.
        Returns:
            total_ops: Array, the total operator matrix of shape (self.batch_size, self.hilbert_dim, self.hilbert_dim).
        """
        total_ops = jnp.zeros((self.batch_size, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        for descriptor, prefactor in zip(self._descriptors, self._prefactors):
            total_ops += self.local.get_operator(descriptor)[None, :, :] * prefactor[:, None, None]
        return total_ops

## TODO: Implement TightBindingSquareOperatorGroup

class ComposedOperatorGroups(OperatorGroup):
    """Tensor product of subsystem operator groups.

    This class is useful for terms such as ``n_j`` on a lattice subsystem tensored
    with a bosonic operator on a bosonic environment subsystem.
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
        inferred_batch_size = self.operator_groups[0].batch_size
        for group in self.operator_groups:
            if group.batch_size != inferred_batch_size:
                raise ValueError("all operator groups must have the same batch size")
        hilbert_dim = reduce(mul, (group.hilbert_dim for group in self.operator_groups), 1)
        inferred_dtype = dtype or jnp.result_type(*(group.dtype for group in self.operator_groups))
        super().__init__(id, hilbert_dim, batch_size=inferred_batch_size, dtype=inferred_dtype)

    def add_operator(self, operator: Any, prefactor: Array) -> None:
        """Composed groups are defined by their subsystem groups."""

        raise ValueError("cannot add an operator directly to ComposedOperatorGroups")

    def sum_operators(self) -> Array:
        """Return the tensor product of subsystem group sums."""
        all_groups = [group.sum_operators() for group in self.operator_groups]
        return compose(all_groups)

__all__ = [
    "BosonOperatorGroup",
    "ComposedOperatorGroups",
    "OperatorGroup",
    "SpinOperatorGroup",
    "TightBindingOperatorGroup",
]
