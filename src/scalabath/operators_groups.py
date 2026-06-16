from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import mul
from typing import Any

import jax.numpy as jnp
from jax import Array

from scalabath.operators_base import boson, tight_binding_1d, tls
from scalabath.utilities import complex_dtype, compose, nonnegative_int, positive_int


def _as_batch_prefactor(prefactor: Any, batch_size: int, dtype: Any, name: str) -> Array:
    values = jnp.asarray(prefactor, dtype=dtype)
    if values.shape == ():
        return jnp.broadcast_to(values, (batch_size,))
    if values.shape == (batch_size,):
        return values
    raise ValueError(f"{name} shape must be scalar or ({batch_size},)")


def _maybe_unbatch_operator(operator: Array, batch_size: int) -> Array:
    if batch_size == 1:
        return operator[0]
    return operator


############################################################################################
###################### Basic classes for building many-body operators ######################
############################################################################################


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

    def add_operator(self, descriptor: str, prefactor: Any = 1.0) -> None:
        """Add an operator to the group.

        Args:
            descriptor: Operator descriptor such as ``"XIII"``.
            prefactor: Scalar or array of shape (self.batch_size,).
        """
        raise NotImplementedError("add_operator is not implemented in the base class")

    def sum_operators(self) -> Array:
        """Sum up the operators in the group. To be implemented in subclasses.

        Returns:
            The total operator matrix.
        """
        raise NotImplementedError("sum_operators is not implemented in the base class")


class BosonOperatorGroup(OperatorGroup):
    """Static operator group for a multi-mode bosonic subsystem."""

    def __init__(
        self,
        num_modes: int,
        id: str,
        nmax: int | Sequence[int],
        *,
        batch_size: int = 1,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.num_modes = positive_int(num_modes, "num_modes")
        nmax_values = (nmax,) if isinstance(nmax, int) else tuple(nmax)
        self.nmax = tuple(nonnegative_int(n, "n") for n in nmax_values)
        if len(self.nmax) != self.num_modes:
            raise ValueError("nmax must have the same length as num_modes")
        self.local = []
        hilbert_dim = 1
        for n in self.nmax:
            self.local.append(boson(n, dtype=dtype))
            hilbert_dim *= self.local[-1].dim
        super().__init__(id, hilbert_dim, batch_size=batch_size, dtype=dtype)

    def add_operator(self, descriptor: str, prefactor: Any = 1.0) -> None:
        """Add a product operator described by a string such as ``"UNI"``.

        Args:
            descriptor: One single-mode descriptor per bosonic mode.
            prefactor: Scalar or array of shape (self.batch_size,).
        """

        if len(descriptor) != self.num_modes:
            raise ValueError("descriptor length must equal num_modes")
        prefactor = _as_batch_prefactor(prefactor, self.batch_size, self.dtype, "prefactor")
        for idx, d in enumerate(descriptor):
            if d not in self.local[idx].descriptors_dict:
                raise ValueError(f"unknown boson operator descriptor {d!r}")
        self._descriptors.append(descriptor)
        self._prefactors.append(prefactor)
        return

    def add_harmonic_operators(self, omega: Array) -> None:
        """Add harmonic energy terms to the group.

        Args:
            omega: Frequencies shaped ``(num_modes,)`` or
                ``(batch_size, num_modes)``.
        """
        omega = jnp.asarray(omega, dtype=self.dtype)
        if omega.shape[-1] != self.num_modes:
            raise ValueError("omega shape must equal (num_modes,) or (batch_size, num_modes)")
        if omega.ndim == 2:
            if omega.shape[0] != self.batch_size:
                raise ValueError("omega shape must equal (batch_size, num_modes)")
            _omega = omega
        elif omega.ndim == 1:
            _omega = omega[None, :].repeat(self.batch_size, axis=0)
        for idx in range(self.num_modes):
            descriptor = ["I"] * self.num_modes
            descriptor[idx] = "N"
            self._descriptors.append("".join(descriptor))
            self._prefactors.append(_omega[:, idx])
        return

    def sum_operators(self) -> Array:
        """Sum up the operators in the group.

        Returns:
            A matrix, or batched matrices when ``batch_size > 1``.
        """
        total_ops = jnp.zeros(
            (self.batch_size, self.hilbert_dim, self.hilbert_dim),
            dtype=self.dtype,
        )
        for descriptor, prefactor in zip(self._descriptors, self._prefactors, strict=True):
            ops = [self.local[idx].get_operator(d) for idx, d in enumerate(descriptor)]
            total_ops += compose(ops)[None, :, :] * prefactor[:, None, None]
        return _maybe_unbatch_operator(total_ops, self.batch_size)


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

    def add_operator(self, descriptor: str, prefactor: Any = 1.0) -> None:
        """Add a product operator described by a sequence such as ``"XIZ"``.

        Args:
            descriptor: One two-level-system descriptor per spin.
            prefactor: Scalar or array of shape (self.batch_size,).
        """

        if len(descriptor) != self.num_spins:
            raise ValueError("descriptor length must equal num_spins")
        prefactor = _as_batch_prefactor(prefactor, self.batch_size, self.dtype, "prefactor")
        for d in descriptor:
            if d not in self.local.descriptors_dict:
                raise ValueError(f"unknown spin operator descriptor {d!r}")
        self._descriptors.append(descriptor)
        self._prefactors.append(prefactor)
        return

    def sum_operators(self) -> Array:
        """Sum up the operators in the group.

        Returns:
            A matrix, or batched matrices when ``batch_size > 1``.
        """
        total_ops = jnp.zeros(
            (self.batch_size, self.hilbert_dim, self.hilbert_dim),
            dtype=self.dtype,
        )
        for descriptor, prefactor in zip(self._descriptors, self._prefactors, strict=True):
            ops = [self.local.get_operator(d) for d in descriptor]
            total_ops += compose(ops)[None, :, :] * prefactor[:, None, None]
        return _maybe_unbatch_operator(total_ops, self.batch_size)


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

    def add_operator(self, descriptor: str, prefactor: Any = 1.0) -> None:
        """Add a sequence-defined tight-binding operator such as ``"XRX"``."""
        if len(descriptor) != self.n_sites:
            raise ValueError("descriptor length must equal n_sites")
        prefactor = _as_batch_prefactor(prefactor, self.batch_size, self.dtype, "prefactor")
        for d in descriptor:
            if d not in self.local.descriptors_dict:
                raise ValueError(f"unknown tight-binding operator descriptor {d!r}")
        self._descriptors.append(descriptor)
        self._prefactors.append(prefactor)
        return

    def add_hopping_operators(self, amplitude: Array) -> None:
        """Add nearest-neighbor hopping operators to the group.

        Args:
            amplitude: Array, the hopping amplitude of shape (self.batch_size,).
        """
        amplitude = _as_batch_prefactor(amplitude, self.batch_size, self.dtype, "amplitude")
        for site in range(self.n_sites):
            descriptor = ["X"] * self.n_sites
            descriptor[site] = "L"
            self._descriptors.append("".join(descriptor))
            self._prefactors.append(amplitude)
            descriptor = ["X"] * self.n_sites
            descriptor[site] = "R"
            self._descriptors.append("".join(descriptor))
            self._prefactors.append(amplitude)
        return

    def add_onsite_operators(self, amplitude: Array) -> None:
        """Add on-site number operators to the group.

        Args:
            amplitude: Array, the onsite amplitude of shape (self.batch_size,).
        """
        amplitude = _as_batch_prefactor(amplitude, self.batch_size, self.dtype, "amplitude")
        for site in range(self.n_sites):
            descriptor = ["X"] * self.n_sites
            descriptor[site] = "N"
            self._descriptors.append("".join(descriptor))
            self._prefactors.append(amplitude)
        return

    def sum_operators(self) -> Array:
        """Sum up the operators in the group.

        Returns:
            A matrix, or batched matrices when ``batch_size > 1``.
        """
        total_ops = jnp.zeros(
            (self.batch_size, self.hilbert_dim, self.hilbert_dim),
            dtype=self.dtype,
        )
        for descriptor, prefactor in zip(self._descriptors, self._prefactors, strict=True):
            total_ops += self.local.get_operator(descriptor)[None, :, :] * prefactor[:, None, None]
        return _maybe_unbatch_operator(total_ops, self.batch_size)


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
    "TightBindingChainOperatorGroup",
]
