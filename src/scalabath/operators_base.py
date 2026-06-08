"""Basic operator matrices for small subsystems."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from warnings import warn

import jax.numpy as jnp
from jax import Array

from scalabath.utilities import adjoint, compose


def _complex_dtype(dtype: Any) -> jnp.dtype:
    dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.complexfloating):
        raise ValueError("operators require a complex dtype")
    return dtype


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


class boson:
    """Single-mode bosonic operators with a finite occupation cutoff.

    Args:
        nmax: Maximum occupation number. The local Hilbert dimension is
            ``nmax + 1``.
        dtype: Complex dtype for all matrices.
    """

    def __init__(self, nmax: int, *, dtype: Any = jnp.complex128) -> None:
        self.nmax = _nonnegative_int(nmax, "nmax")
        self.dim = self.nmax + 1
        self.dtype = _complex_dtype(dtype)

        levels = jnp.arange(self.dim)
        weights = jnp.sqrt(jnp.arange(1, self.dim, dtype=jnp.float64)).astype(self.dtype)
        self.identity = jnp.eye(self.dim, dtype=self.dtype)
        self.annihilation = (
            jnp.zeros((self.dim, self.dim), dtype=self.dtype)
            .at[levels[:-1], levels[1:]]
            .set(weights)
        )
        self.creation = adjoint(self.annihilation)
        self.number = jnp.diag(levels).astype(self.dtype)

        self.I = self.identity
        self.D = self.annihilation
        self.U = self.creation
        self.N = self.number

    def get_operator(self, name: str) -> Array:
        """Return a named single-mode operator.

        Accepted names are ``"I"``, ``"D"``/``"-"`` for annihilation,
        ``"U"``/``"+"`` for creation, and ``"N"`` for number.
        """

        operators = {
            "+": self.creation,
            "-": self.annihilation,
            "D": self.annihilation,
            "I": self.identity,
            "N": self.number,
            "U": self.creation,
        }
        try:
            return operators[name]
        except KeyError as exc:
            raise ValueError(f"unknown boson operator name {name!r}") from exc

    def get_sequence_ops(self, name_sequence: str) -> list[Array]:
        """Return local operators for a sequence such as ``"UDI"``."""

        return [self.get_operator(name) for name in name_sequence]

    def get_composite_ops(self, name_sequence: str) -> Array:
        """Return the Kronecker product for a multi-mode operator sequence."""

        return compose(self.get_sequence_ops(name_sequence))


class tls:
    """Two-level-system operators.

    The class exposes Pauli matrices, identity, and raising/lowering operators
    as JAX arrays with shape ``(2, 2)``.
    """

    def __init__(self, *, dtype: Any = jnp.complex128) -> None:
        self.dtype = _complex_dtype(dtype)

        self.sigma_x = jnp.asarray([[0, 1], [1, 0]], dtype=self.dtype)
        self.sigma_y = jnp.asarray([[0, -1j], [1j, 0]], dtype=self.dtype)
        self.sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=self.dtype)
        self.identity = jnp.eye(2, dtype=self.dtype)
        self.sigma_plus = jnp.asarray([[0, 1], [0, 0]], dtype=self.dtype)
        self.sigma_minus = jnp.asarray([[0, 0], [1, 0]], dtype=self.dtype)
        self.number = self.sigma_plus @ self.sigma_minus

        self.X = self.sigma_x
        self.Y = self.sigma_y
        self.Z = self.sigma_z
        self.I = self.identity
        self.U = self.sigma_plus
        self.D = self.sigma_minus
        self.N = self.number

    def get_operator(self, name: str) -> Array:
        """Return a named two-level-system operator.

        Accepted names are ``"X"``, ``"Y"``, ``"Z"``, ``"I"``,
        ``"U"``/``"+"``/``"P"`` for raising, ``"D"``/``"-"``/``"M"``
        for lowering, and ``"N"`` for ``sigma_plus @ sigma_minus``.
        """

        operators = {
            "+": self.sigma_plus,
            "-": self.sigma_minus,
            "D": self.sigma_minus,
            "I": self.identity,
            "M": self.sigma_minus,
            "N": self.number,
            "P": self.sigma_plus,
            "U": self.sigma_plus,
            "X": self.sigma_x,
            "Y": self.sigma_y,
            "Z": self.sigma_z,
        }
        try:
            return operators[name]
        except KeyError as exc:
            raise ValueError(f"unknown two-level operator name {name!r}") from exc

    def get_sequence_ops(self, name_sequence: str) -> list[Array]:
        """Return local operators for a sequence such as ``"XIY"``."""

        return [self.get_operator(name) for name in name_sequence]

    def get_composite_ops(self, name_sequence: str) -> Array:
        """Return the Kronecker product for a multi-spin operator sequence."""

        return compose(self.get_sequence_ops(name_sequence))


class tight_binding_1d:
    """Single-particle tight-binding operators on a 1D lattice.

    Args:
        n_sites: Number of lattice sites.
        periodic: Whether nearest-neighbor boundary hops wrap around.
        dtype: Complex dtype for all matrices.
    """

    def __init__(
        self,
        n_sites: int,
        *,
        periodic: bool = True,
        dtype: Any = jnp.complex128,
    ) -> None:
        self.n_sites = _positive_int(n_sites, "n_sites")
        if self.n_sites == 1:
            raise ValueError("n_sites can not be 1")
        if self.n_sites == 2:
            raise ValueError("n_sites has to be greater than 2. Use tls instead of you only need a two-level-system.")
        self.periodic = bool(periodic)
        self.dtype = _complex_dtype(dtype)
        self.hilbert_dim = self.n_sites
        self.identity = jnp.eye(self.hilbert_dim, dtype=self.dtype)

    def _validate_site(self, site: int) -> int:
        site = int(site)
        if site < 0 or site >= self.n_sites:
            raise ValueError(f"site index {site} is outside [0, {self.n_sites})")
        return site

    def on_site(self, site: int, amplitude: complex = 1.0) -> Array:
        """Return ``amplitude * |site><site|``."""

        site = self._validate_site(site)
        return (
            jnp.zeros((self.n_sites, self.n_sites), dtype=self.dtype)
            .at[site, site]
            .set(jnp.asarray(amplitude, dtype=self.dtype))
        )

    def onsite_potential(self, values: Sequence[complex]) -> Array:
        """Return a diagonal on-site potential matrix.

        Args:
            values: Sequence shaped ``(n_sites,)``.
        """

        values_array = jnp.asarray(values, dtype=self.dtype)
        if values_array.shape != (self.n_sites,):
            raise ValueError("onsite potential values must have shape (n_sites,)")
        return jnp.diag(values_array)

    def hopping(self, from_site: int, to_site: int, amplitude: complex = 1.0) -> Array:
        """Return ``amplitude * |to_site><from_site|``."""

        from_site = self._validate_site(from_site)
        to_site = self._validate_site(to_site)
        return (
            jnp.zeros((self.n_sites, self.n_sites), dtype=self.dtype)
            .at[to_site, from_site]
            .set(jnp.asarray(amplitude, dtype=self.dtype))
        )

    def left(self, site: int, amplitude: complex = 1.0) -> Array:
        """Return an operator moving a particle from ``site`` to ``site - 1``."""

        site = self._validate_site(site)
        if site == 0 and not self.periodic:
            return jnp.zeros((self.n_sites, self.n_sites), dtype=self.dtype)
        return self.hopping(site, (site - 1) % self.n_sites, amplitude)

    def right(self, site: int, amplitude: complex = 1.0) -> Array:
        """Return an operator moving a particle from ``site`` to ``site + 1``."""

        site = self._validate_site(site)
        if site == self.n_sites - 1 and not self.periodic:
            return jnp.zeros((self.n_sites, self.n_sites), dtype=self.dtype)
        return self.hopping(site, (site + 1) % self.n_sites, amplitude)

    def nearest_neighbor_hopping(self, amplitude: complex = 1.0) -> Array:
        """Return Hermitian nearest-neighbor hopping for the 1D lattice."""

        total = jnp.zeros((self.n_sites, self.n_sites), dtype=self.dtype)
        stop = self.n_sites if self.periodic else self.n_sites - 1
        for site in range(stop):
            neighbor = (site + 1) % self.n_sites
            total = total + self.hopping(site, neighbor, amplitude)
            total = total + self.hopping(neighbor, site, jnp.conjugate(amplitude))
        return total

    # def get_composite_ops(self, name_sequence: str) -> Array:
    #     """Return a sequence-defined one-particle tight-binding operator.

    #     ``"X"`` denotes identity on a site, ``"N"`` an on-site projector,
    #     ``"L"`` a hop from the marked site to the left, and ``"R"`` a hop from
    #     the marked site to the right. The sequence must contain zero or one
    #     non-``"X"`` character.
    #     """

    #     if len(name_sequence) != self.n_sites:
    #         raise ValueError("name_sequence length must equal n_sites")
    #     if not all(name in {"L", "N", "R", "X"} for name in name_sequence):
    #         raise ValueError("only L, N, R, and X are allowed in name_sequence")

    #     marked_sites = [idx for idx, name in enumerate(name_sequence) if name != "X"]
    #     if not marked_sites:
    #         return self.identity
    #     if len(marked_sites) > 1:
    #         raise ValueError("name_sequence may contain at most one non-X operator")

    #     site = marked_sites[0]
    #     name = name_sequence[site]
    #     if name == "L":
    #         return self.left(site)
    #     if name == "R":
    #         return self.right(site)
    #     return self.on_site(site)


class tight_binding_2d:
    """Single-particle tight-binding operators on a rectangular 2D lattice."""

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        periodic: bool = True,
        dtype: Any = jnp.complex128,
    ) -> None:
        self.nx = _positive_int(nx, "nx")
        self.ny = _positive_int(ny, "ny")
        if self.nx == 1 or self.ny == 1:
            raise ValueError("nx and ny have to be greater than 1. Use tight_binding_1d instead if you only need a 1D lattice.")
        if self.nx == 2 or self.ny == 2:
            warn("nx or ny are 2. Do not use `nearest_neighbor_hopping` to construct the Hamiltonian because it will double count the hopping. Use `hopping` to construct the Hamiltonian one term by one term instead.")
        self.periodic = bool(periodic)
        self.dtype = _complex_dtype(dtype)
        self.hilbert_dim = self.nx * self.ny
        self.identity = jnp.eye(self.hilbert_dim, dtype=self.dtype)

    def site_index(self, site: tuple[int, int]) -> int:
        """Map a lattice coordinate ``(x, y)`` to a flat Hilbert-space index."""

        x, y = int(site[0]), int(site[1])
        if x < 0 or x >= self.nx or y < 0 or y >= self.ny:
            raise ValueError("2D site is outside the lattice")
        return x * self.ny + y

    def on_site(self, site: tuple[int, int], amplitude: complex = 1.0) -> Array:
        """Return ``amplitude * |site><site|``."""

        idx = self.site_index(site)
        return (
            jnp.zeros((self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
            .at[idx, idx]
            .set(jnp.asarray(amplitude, dtype=self.dtype))
        )

    def hopping(
        self,
        from_site: tuple[int, int],
        to_site: tuple[int, int],
        amplitude: complex = 1.0,
    ) -> Array:
        """Return ``amplitude * |to_site><from_site|``."""

        source = self.site_index(from_site)
        target = self.site_index(to_site)
        return (
            jnp.zeros((self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
            .at[target, source]
            .set(jnp.asarray(amplitude, dtype=self.dtype))
        )

    def onsite_potential(self, values: Array) -> Array:
        """Return a diagonal on-site potential matrix.

        Args:
            values: Array shaped ``(nx, ny)``.
        """

        values_array = jnp.asarray(values, dtype=self.dtype)
        if values_array.shape != (self.nx, self.ny):
            raise ValueError("onsite potential values must have shape (nx, ny)")
        return jnp.diag(values_array.reshape(-1))

    def nearest_neighbor_hopping(self, amplitude: complex = 1.0) -> Array:
        """Return Hermitian nearest-neighbor hopping on the 2D lattice."""

        total = jnp.zeros((self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        directions = ((1, 0), (0, 1))
        for x in range(self.nx):
            for y in range(self.ny):
                for dx, dy in directions:
                    nx = x + dx
                    ny = y + dy
                    if self.periodic:
                        nx %= self.nx
                        ny %= self.ny
                    elif nx >= self.nx or ny >= self.ny:
                        continue
                    total = total + self.hopping((x, y), (nx, ny), amplitude)
                    total = total + self.hopping((nx, ny), (x, y), jnp.conjugate(amplitude))
        return total


__all__ = ["boson", "tight_binding_1d", "tight_binding_2d", "tls"]
