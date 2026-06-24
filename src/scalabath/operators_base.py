"""Basic operator matrices for bosons, two-level systems, and tight-binding lattices."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from jax import Array

from scalabath.utilities import adjoint, complex_dtype, nonnegative_int, positive_int


class boson:
    """Single-mode bosonic operators with a finite occupation cutoff.

    Args:
        nmax: Maximum occupation number. The local Hilbert dimension is
            ``nmax + 1``.
        dtype: Complex dtype for all matrices.
    """

    def __init__(self, nmax: int, *, dtype: Any = jnp.complex64) -> None:
        self.nmax = nonnegative_int(nmax, "nmax")
        self.dim = self.nmax + 1
        self.dtype = complex_dtype(dtype)

        levels = jnp.arange(self.dim)
        real_dtype = jnp.float64 if self.dtype == jnp.dtype(jnp.complex128) else jnp.float32
        weights = jnp.sqrt(jnp.arange(1, self.dim).astype(real_dtype)).astype(self.dtype)
        self.identity = jnp.eye(self.dim, dtype=self.dtype)
        self.annihilation = (
            jnp.zeros((self.dim, self.dim), dtype=self.dtype)
            .at[levels[:-1], levels[1:]]
            .set(weights)
        )
        self.creation = adjoint(self.annihilation)
        self.number = jnp.diag(levels).astype(self.dtype)

        self.descriptors_dict = {
            "+": self.creation,
            "-": self.annihilation,
            "D": self.annihilation,
            "I": self.identity,
            "N": self.number,
            "U": self.creation,
        }

    def get_operator(self, descriptor: str) -> Array:
        """Return a named single-mode operator.

        Accepted descriptors are ``"I"``, ``"D"``/``"-"`` for annihilation,
        ``"U"``/``"+"`` for creation, and ``"N"`` for number.
        """

        try:
            return self.descriptors_dict[descriptor]
        except KeyError as exc:
            raise ValueError(f"unknown boson operator descriptor {descriptor!r}") from exc


class tls:
    """Two-level-system operators.

    The class exposes Pauli matrices, identity, and raising/lowering operators
    as JAX arrays with shape ``(2, 2)``.
    """

    def __init__(self, *, dtype: Any = jnp.complex64) -> None:
        self.dtype = complex_dtype(dtype)

        self.sigma_x = jnp.asarray([[0, 1], [1, 0]], dtype=self.dtype)
        self.sigma_y = jnp.asarray([[0, -1j], [1j, 0]], dtype=self.dtype)
        self.sigma_z = jnp.asarray([[1, 0], [0, -1]], dtype=self.dtype)
        self.identity = jnp.eye(2, dtype=self.dtype)
        self.sigma_plus = jnp.asarray([[0, 1], [0, 0]], dtype=self.dtype)
        self.sigma_minus = jnp.asarray([[0, 0], [1, 0]], dtype=self.dtype)
        self.number = self.sigma_plus @ self.sigma_minus

        self.descriptors_dict = {
            "X": self.sigma_x,
            "Y": self.sigma_y,
            "Z": self.sigma_z,
            "I": self.identity,
            "N": self.number,
            "U": self.sigma_plus,
            "D": self.sigma_minus,
            "+": self.sigma_plus,
            "-": self.sigma_minus,
        }

    def get_operator(self, descriptor: str) -> Array:
        """Return a named two-level-system operator.

        Accepted descriptors are ``"X"``, ``"Y"``, ``"Z"``, ``"I"``,
        ``"U"``/``"+"``/``"P"`` for raising, ``"D"``/``"-"``/``"M"``
        for lowering, and ``"N"`` for ``sigma_plus @ sigma_minus``.
        """
        try:
            return self.descriptors_dict[descriptor]
        except KeyError as exc:
            raise ValueError(f"unknown two-level operator descriptor {descriptor!r}") from exc


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
        dtype: Any = jnp.complex64,
    ) -> None:
        self.n_sites = positive_int(n_sites, "n_sites")
        if self.n_sites == 1:
            raise ValueError("n_sites can not be 1")
        if self.n_sites == 2:
            raise ValueError(
                "n_sites has to be greater than 2. Use tls instead if you only need "
                "a two-level-system."
            )
        self.periodic = bool(periodic)
        self.dtype = complex_dtype(dtype)
        self.hilbert_dim = self.n_sites
        self.identity = jnp.eye(self.hilbert_dim, dtype=self.dtype)
        self.descriptors_dict = {
            "X": None,
            "N": None,
            "L": None,
            "R": None,
        }

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

        source = self._validate_site(from_site)
        target = self._validate_site(to_site)
        return (
            jnp.zeros((self.n_sites, self.n_sites), dtype=self.dtype)
            .at[target, source]
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

    def get_operator(self, descriptor: str) -> Array:
        """Return a sequence-defined one-particle tight-binding operator.

        ``"X"`` denotes identity on a site, ``"N"`` an on-site projector,
        ``"L"`` a hop from the marked site to the left, and ``"R"`` a hop from
        the marked site to the right. The sequence must contain zero or one
        non-``"X"`` character.
        Example:
            >>> tb = tight_binding_1d(3)
            >>> tb.get_operator("XRX")
            Array([[0.+0.j, 1.+0.j, 0.+0.j],
                   [0.+0.j, 0.+0.j, 0.+0.j],
                   [0.+0.j, 0.+0.j, 0.+0.j]], dtype=complex64)
        """

        if len(descriptor) != self.n_sites:
            raise ValueError("descriptor length must equal n_sites")

        marked_sites = [idx for idx, mark in enumerate(descriptor) if mark != "X"]
        if not marked_sites:
            return self.identity
        if len(marked_sites) > 1:
            raise ValueError("descriptor may contain at most one non-X operator")

        site = marked_sites[0]
        mark = descriptor[site]
        if mark == "L":
            return self.left(site)
        elif mark == "R":
            return self.right(site)
        elif mark == "N":
            return self.on_site(site)
        else:
            raise ValueError(f"unknown tight-binding operator descriptor {mark!r}")


class tight_binding_2d:
    """Single-particle tight-binding operators on a rectangular 2D lattice."""

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        periodic: bool = True,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.nx = positive_int(nx, "nx")
        self.ny = positive_int(ny, "ny")
        if self.nx == 1 or self.ny == 1:
            raise ValueError(
                "nx and ny have to be greater than 1. Use tight_binding_1d instead "
                "if you only need a 1D lattice."
            )
        self.periodic = bool(periodic)
        self.dtype = complex_dtype(dtype)
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

    def onsite_potential(self, values: Array) -> Array:
        """Return a diagonal on-site potential matrix.

        Args:
            values: Array shaped ``(nx, ny)``.
        """

        values_array = jnp.asarray(values, dtype=self.dtype)
        if values_array.shape != (self.nx, self.ny):
            raise ValueError("onsite potential values must have shape (nx, ny)")
        return jnp.diag(values_array.reshape(-1))

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

    def nearest_neighbor_hopping(self, amplitude: tuple[complex, complex] = (1.0, 1.0)) -> Array:
        """Return Hermitian nearest-neighbor hopping on the 2D lattice.

        The x-direction bonds connect ``(x, y)`` to ``(x + 1, y)`` with
        ``amplitude[0]``. The y-direction bonds connect ``(x, y)`` to
        ``(x, y + 1)`` with ``amplitude[1]``.
        """
        if self.nx == 2 or self.ny == 2:
            raise ValueError(
                "nearest_neighbor_hopping will double-count hopping amplitudes for 2x2 lattices"
            )
        total = jnp.zeros((self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        directions = ((1, 0), (0, 1))
        for x in range(self.nx):
            for y in range(self.ny):
                for direction, amp in zip(directions, amplitude, strict=True):
                    dx, dy = direction
                    nn_x = x + dx
                    nn_y = y + dy
                    if self.periodic:
                        nn_x %= self.nx
                        nn_y %= self.ny
                    elif nn_x >= self.nx or nn_y >= self.ny:
                        continue
                    total = total + self.hopping((x, y), (nn_x, nn_y), amp)
                    total = total + self.hopping((nn_x, nn_y), (x, y), jnp.conjugate(amp))
        return total

    def triangular_hopping(self, amplitude: tuple[complex, complex] = (1.0, 1.0)) -> Array:
        """Return Hermitian triangular hopping on the 2D lattice.

        The x-direction bonds connect ``(x, y)`` to ``(x + 1, y)`` with
        ``amplitude[0]``. The y-direction and diagonal bonds connect
        ``(x, y)`` to ``(x, y + 1)`` and ``(x + 1, y + 1)`` with
        ``amplitude[1]``.
        """
        if self.nx == 2 or self.ny == 2:
            raise ValueError(
                "triangular_hopping will double-count hopping amplitudes for 2x2 lattices"
            )
        amp_x, amp_y = amplitude
        total = jnp.zeros((self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        directions = (((1, 0), amp_x), ((0, 1), amp_y), ((1, 1), amp_y))
        for x in range(self.nx):
            for y in range(self.ny):
                for direction, amp in directions:
                    dx, dy = direction
                    nn_x = x + dx
                    nn_y = y + dy
                    if self.periodic:
                        nn_x %= self.nx
                        nn_y %= self.ny
                    elif nn_x >= self.nx or nn_y >= self.ny:
                        continue
                    total = total + self.hopping((x, y), (nn_x, nn_y), amp)
                    total = total + self.hopping((nn_x, nn_y), (x, y), jnp.conjugate(amp))
        return total

    def get_operator(self, descriptor_sequence: str) -> Array:
        """Return a sequence-defined one-particle tight-binding operator.

        ``"X"`` denotes identity on a site, ``"N"`` an on-site projector,
        ``"L"`` a hop from the marked site to the left, and ``"R"`` a hop from
        the marked site to the right. The sequence must contain zero or one
        non-``"X"`` character.
        """
        raise NotImplementedError("get_operator has not been implemented for tight_binding_2d")


__all__ = ["boson", "tight_binding_1d", "tight_binding_2d", "tls"]
