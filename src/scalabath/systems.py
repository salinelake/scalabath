"""State containers for flat and tensor-product quantum ensembles."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import mul
from typing import Any

import jax.numpy as jnp
from jax import Array

from scalabath.utilities import batch_trace, complex_dtype, positive_int


def _subsystem_dims(subsystem_dims: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(positive_int(dim, "subsystem_dim") for dim in subsystem_dims)
    if not dims:
        raise ValueError("subsystem_dims must contain at least one dimension")
    return dims


def _prod(values: Sequence[int]) -> int:
    return reduce(mul, values, 1)


def _normalize_keep_axes(keep: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(keep, int):
        axes = (keep,)
    else:
        axes = tuple(int(axis) for axis in keep)
    if not axes:
        raise ValueError("keep must contain at least one subsystem axis")

    normalized = []
    for axis in axes:
        axis = axis + ndim if axis < 0 else axis
        if axis < 0 or axis >= ndim:
            raise ValueError(f"subsystem axis {axis} is outside [0, {ndim})")
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        raise ValueError("keep must not contain duplicate subsystem axes")
    return tuple(normalized)


class PureStatesEnsemble:
    """Container for a batch of flat pure states.

    Args:
        hilbert_dim: Dimension of the Hilbert space.
        batch_size: Number of states in the ensemble.
        dtype: Complex dtype used when storing states.

    Stored states have shape ``(batch_size, hilbert_dim)``. Passing a single
    state shaped ``(hilbert_dim,)`` to :meth:`set_pse` broadcasts it across the
    ensemble batch.
    """

    def __init__(
        self,
        hilbert_dim: int,
        batch_size: int = 1,
        *,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.hilbert_dim = positive_int(hilbert_dim, "hilbert_dim")
        self.batch_size = positive_int(batch_size, "batch_size")
        self.dtype = complex_dtype(dtype)
        self._pse: Array | None = None

    @property
    def ns(self) -> int:
        """Alias for ``hilbert_dim`` used by operator-group code."""

        return self.hilbert_dim

    @property
    def nb(self) -> int:
        """Alias for ``batch_size`` used by operator-group code."""

        return self.batch_size

    def get_pse(self) -> Array:
        """Return pure states with shape ``(batch, hilbert_dim)``."""

        if self._pse is None:
            raise ValueError("pure states have not been set")
        return self._pse

    def set_pse(self, pse: Any) -> None:
        """Set pure states from ``(hilbert_dim,)`` or ``(batch, hilbert_dim)`` arrays."""

        states = jnp.asarray(pse, dtype=self.dtype)
        if states.shape == (self.hilbert_dim,):
            states = jnp.broadcast_to(states, (self.batch_size, self.hilbert_dim))
        elif states.shape != (self.batch_size, self.hilbert_dim):
            raise ValueError(
                "pure states must have shape (hilbert_dim,) or (batch_size, hilbert_dim)"
            )
        self._pse = states

    @property
    def norm(self) -> Array:
        """Return the L2 norm of each pure state as shape ``(batch,)``."""

        return jnp.linalg.norm(self.get_pse(), axis=1)

    def normalize(self) -> Array:
        """Normalize the stored pure-state ensemble in place."""

        states = self.get_pse()
        norms = jnp.linalg.norm(states, axis=1)
        if bool(jnp.any(norms == 0)):
            raise ValueError("cannot normalize a zero pure state")
        self._pse = states / norms[:, None]
        return self._pse


class DensityMatrixEnsemble:
    """Container for a batch of flat density matrices.

    Stored density matrices have shape ``(batch_size, hilbert_dim, hilbert_dim)``.
    Passing one matrix shaped ``(hilbert_dim, hilbert_dim)`` broadcasts it across
    the ensemble batch.
    """

    def __init__(
        self,
        hilbert_dim: int,
        batch_size: int = 1,
        *,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.hilbert_dim = positive_int(hilbert_dim, "hilbert_dim")
        self.batch_size = positive_int(batch_size, "batch_size")
        self.dtype = complex_dtype(dtype)
        self._dme: Array | None = None

    @property
    def ns(self) -> int:
        """Alias for ``hilbert_dim`` used by operator-group code."""

        return self.hilbert_dim

    @property
    def nb(self) -> int:
        """Alias for ``batch_size`` used by operator-group code."""

        return self.batch_size

    def get_dme(self) -> Array:
        """Return density matrices shaped ``(batch, hilbert_dim, hilbert_dim)``."""

        if self._dme is None:
            raise ValueError("density matrices have not been set")
        return self._dme

    def set_dme(self, dme: Any) -> None:
        """Set density matrices from flat matrix or batched flat matrices."""

        density_matrices = jnp.asarray(dme, dtype=self.dtype)
        matrix_shape = (self.hilbert_dim, self.hilbert_dim)
        batch_shape = (self.batch_size, self.hilbert_dim, self.hilbert_dim)
        if density_matrices.shape == matrix_shape:
            density_matrices = jnp.broadcast_to(density_matrices, batch_shape)
        elif density_matrices.shape != batch_shape:
            raise ValueError(
                "density matrices must have shape (hilbert_dim, hilbert_dim) "
                "or (batch_size, hilbert_dim, hilbert_dim)"
            )
        self._dme = density_matrices

    def get_rho(self) -> Array:
        """Alias for :meth:`get_dme`."""

        return self.get_dme()

    def set_rho(self, rho: Any) -> None:
        """Alias for :meth:`set_dme`."""

        self.set_dme(rho)

    @property
    def trace(self) -> Array:
        """Return the trace of each density matrix as shape ``(batch,)``."""

        return batch_trace(self.get_dme())

    def normalize(self) -> Array:
        """Normalize the stored density matrices to unit trace in place."""

        density_matrices = self.get_dme()
        traces = batch_trace(density_matrices)
        if bool(jnp.any(traces == 0)):
            raise ValueError("cannot normalize a zero-trace density matrix")
        self._dme = density_matrices / traces[:, None, None]
        return self._dme


class TensorProductPureStatesEnsemble:
    """Pure states on a tensor-product Hilbert space.

    Args:
        subsystem_dims: Dimensions ``(n_0, n_1, ...)`` of each tensor factor.
        batch_size: Number of states in the ensemble.
        dtype: Complex dtype used when storing states.

    The native layout is batch-first, with shape
    ``(batch_size, *subsystem_dims)``. For a system coupled to bosonic modes this
    is ``(batch_size, n_s, n_1, ..., n_N)``.
    """

    def __init__(
        self,
        subsystem_dims: Sequence[int],
        batch_size: int = 1,
        *,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.subsystem_dims = _subsystem_dims(subsystem_dims)
        self.hilbert_dim = _prod(self.subsystem_dims)
        self.batch_size = positive_int(batch_size, "batch_size")
        self.dtype = complex_dtype(dtype)
        self._pse: Array | None = None

    @property
    def ns(self) -> int:
        """Alias for the flattened Hilbert-space dimension."""

        return self.hilbert_dim

    @property
    def nb(self) -> int:
        """Alias for ``batch_size``."""

        return self.batch_size

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        """Native tensor shape ``(batch_size, *subsystem_dims)``."""

        return (self.batch_size, *self.subsystem_dims)

    def get_pse(self) -> Array:
        """Return pure states with native shape ``(batch, *subsystem_dims)``."""

        if self._pse is None:
            raise ValueError("pure states have not been set")
        return self._pse

    def get_flat_pse(self) -> Array:
        """Return pure states reshaped as ``(batch, hilbert_dim)``."""

        states = self.get_pse()
        return states.reshape(self.batch_size, self.hilbert_dim)

    def set_pse(self, pse: Any) -> None:
        """Set states from native tensor or flat layouts.

        Accepted shapes are ``subsystem_dims``, ``(batch, *subsystem_dims)``,
        ``(hilbert_dim,)``, and ``(batch, hilbert_dim)``.
        """

        states = jnp.asarray(pse, dtype=self.dtype)
        if states.shape == self.subsystem_dims:
            states = jnp.broadcast_to(states[None, ...], self.tensor_shape)
        elif states.shape == self.tensor_shape:
            pass
        elif states.shape == (self.hilbert_dim,):
            states = jnp.broadcast_to(
                states.reshape((1, *self.subsystem_dims)),
                self.tensor_shape,
            )
        elif states.shape == (self.batch_size, self.hilbert_dim):
            states = states.reshape(self.tensor_shape)
        else:
            raise ValueError(
                "pure states must have shape subsystem_dims, (batch_size, *subsystem_dims), "
                "(hilbert_dim,), or (batch_size, hilbert_dim)"
            )
        self._pse = states

    @property
    def norm(self) -> Array:
        """Return one L2 norm per batch element as shape ``(batch,)``."""

        return jnp.linalg.norm(self.get_flat_pse(), axis=1)

    def normalize(self) -> Array:
        """Normalize each batch element in place and return native-layout states."""

        states = self.get_pse()
        norms = self.norm
        if bool(jnp.any(norms == 0)):
            raise ValueError("cannot normalize a zero pure state")
        denom = norms.reshape((self.batch_size, *([1] * len(self.subsystem_dims))))
        self._pse = states / denom
        return self._pse

    def set_product_state(self, factors: Sequence[Any]) -> None:
        """Set a product state over all tensor factors.

        Args:
            factors: One one-dimensional state vector per subsystem. Each factor
                must have shape ``(subsystem_dims[i],)``. The resulting tensor
                product is broadcast across the ensemble batch.
        """

        if len(factors) != len(self.subsystem_dims):
            raise ValueError("factors must contain one state per subsystem")
        product = jnp.asarray(factors[0], dtype=self.dtype)
        if product.shape != (self.subsystem_dims[0],):
            raise ValueError("each product-state factor must match its subsystem dimension")
        for factor, dim in zip(factors[1:], self.subsystem_dims[1:], strict=True):
            factor = jnp.asarray(factor, dtype=self.dtype)
            if factor.shape != (dim,):
                raise ValueError("each product-state factor must match its subsystem dimension")
            product = product[..., None] * factor.reshape((1,) * product.ndim + (dim,))
        self.set_pse(product)

    def reduced_density_matrix(self, keep: int | Sequence[int] = 0) -> Array:
        """Trace out all subsystems except ``keep``.

        Args:
            keep: Subsystem axis or axes to keep, indexed relative to
                ``subsystem_dims``. The kept axes are flattened in the provided
                order.

        Returns:
            Reduced density matrices shaped ``(batch, kept_dim, kept_dim)``.
        """

        keep_axes = _normalize_keep_axes(keep, len(self.subsystem_dims))
        trace_axes = tuple(
            axis for axis in range(len(self.subsystem_dims)) if axis not in keep_axes
        )
        kept_dim = _prod(tuple(self.subsystem_dims[axis] for axis in keep_axes))
        traced_dim = _prod(tuple(self.subsystem_dims[axis] for axis in trace_axes))
        permutation = (0, *(axis + 1 for axis in keep_axes), *(axis + 1 for axis in trace_axes))
        states = jnp.transpose(self.get_pse(), permutation).reshape(
            self.batch_size,
            kept_dim,
            traced_dim,
        )
        return jnp.einsum("bia,bja->bij", states, jnp.conjugate(states))


class TensorProductDensityMatrixEnsemble:
    """Density matrices on a tensor-product Hilbert space.

    The native layout is ``(batch_size, *subsystem_dims, *subsystem_dims)``. The
    first group of subsystem axes indexes rows and the second group indexes
    columns.
    """

    def __init__(
        self,
        subsystem_dims: Sequence[int],
        batch_size: int = 1,
        *,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.subsystem_dims = _subsystem_dims(subsystem_dims)
        self.hilbert_dim = _prod(self.subsystem_dims)
        self.batch_size = positive_int(batch_size, "batch_size")
        self.dtype = complex_dtype(dtype)
        self._dme: Array | None = None

    @property
    def ns(self) -> int:
        """Alias for the flattened Hilbert-space dimension."""

        return self.hilbert_dim

    @property
    def nb(self) -> int:
        """Alias for ``batch_size``."""

        return self.batch_size

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        """Native tensor shape ``(batch_size, *dims, *dims)``."""

        return (self.batch_size, *self.subsystem_dims, *self.subsystem_dims)

    def get_dme(self) -> Array:
        """Return density matrices with native tensor-product layout."""

        if self._dme is None:
            raise ValueError("density matrices have not been set")
        return self._dme

    def get_flat_dme(self) -> Array:
        """Return density matrices as ``(batch, hilbert_dim, hilbert_dim)``."""

        density_matrices = self.get_dme()
        return density_matrices.reshape(self.batch_size, self.hilbert_dim, self.hilbert_dim)

    def set_dme(self, dme: Any) -> None:
        """Set density matrices from native tensor or flat layouts."""

        density_matrices = jnp.asarray(dme, dtype=self.dtype)
        matrix_shape = (self.hilbert_dim, self.hilbert_dim)
        batch_shape = (self.batch_size, self.hilbert_dim, self.hilbert_dim)
        tensor_no_batch = (*self.subsystem_dims, *self.subsystem_dims)
        if density_matrices.shape == tensor_no_batch:
            density_matrices = jnp.broadcast_to(density_matrices[None, ...], self.tensor_shape)
        elif density_matrices.shape == self.tensor_shape:
            pass
        elif density_matrices.shape == matrix_shape:
            density_matrices = density_matrices.reshape(tensor_no_batch)
            density_matrices = jnp.broadcast_to(density_matrices[None, ...], self.tensor_shape)
        elif density_matrices.shape == batch_shape:
            density_matrices = density_matrices.reshape((self.batch_size, *tensor_no_batch))
        else:
            raise ValueError(
                "density matrices must have shape (*dims, *dims), (batch, *dims, *dims), "
                "(hilbert_dim, hilbert_dim), or (batch, hilbert_dim, hilbert_dim)"
            )
        self._dme = density_matrices

    def get_rho(self) -> Array:
        """Alias for :meth:`get_dme`."""

        return self.get_dme()

    def set_rho(self, rho: Any) -> None:
        """Alias for :meth:`set_dme`."""

        self.set_dme(rho)

    @property
    def trace(self) -> Array:
        """Return one trace per batch element as shape ``(batch,)``."""

        return batch_trace(self.get_flat_dme())

    def normalize(self) -> Array:
        """Normalize each tensor-product density matrix to unit trace in place."""

        density_matrices = self.get_dme()
        traces = self.trace
        if bool(jnp.any(traces == 0)):
            raise ValueError("cannot normalize a zero-trace density matrix")
        denom = traces.reshape((self.batch_size, *([1] * (2 * len(self.subsystem_dims)))))
        self._dme = density_matrices / denom
        return self._dme

    def reduced_density_matrix(self, keep: int | Sequence[int] = 0) -> Array:
        """Trace out all subsystems except ``keep``.

        Args:
            keep: Subsystem axis or axes to keep, indexed relative to
                ``subsystem_dims``. The kept axes are flattened in the provided
                order.

        Returns:
            Reduced density matrices shaped ``(batch, kept_dim, kept_dim)``.
        """

        keep_axes = _normalize_keep_axes(keep, len(self.subsystem_dims))
        trace_axes = tuple(
            axis for axis in range(len(self.subsystem_dims)) if axis not in keep_axes
        )
        kept_dim = _prod(tuple(self.subsystem_dims[axis] for axis in keep_axes))
        traced_dim = _prod(tuple(self.subsystem_dims[axis] for axis in trace_axes))
        n_subsystems = len(self.subsystem_dims)
        permutation = (
            0,
            *(axis + 1 for axis in keep_axes),
            *(axis + 1 for axis in trace_axes),
            *(axis + 1 + n_subsystems for axis in keep_axes),
            *(axis + 1 + n_subsystems for axis in trace_axes),
        )
        density_matrices = jnp.transpose(self.get_dme(), permutation).reshape(
            self.batch_size,
            kept_dim,
            traced_dim,
            kept_dim,
            traced_dim,
        )
        return jnp.einsum("bikjk->bij", density_matrices)


__all__ = [
    "DensityMatrixEnsemble",
    "PureStatesEnsemble",
    "TensorProductDensityMatrixEnsemble",
    "TensorProductPureStatesEnsemble",
]
