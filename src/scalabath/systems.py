"""State containers for pure-state and density-matrix ensembles."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from scalabath.utilities import batch_trace, positive_int, complex_dtype


class PureStatesEnsemble:
    """Container for a batch of pure states.

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
        self._pse: Array | None = None  ## shape: (batch_size, hilbert_dim)

    @property
    def ns(self) -> int:
        """Alias for ``hilbert_dim`` used by operator-group code."""

        return self.hilbert_dim

    @property
    def nb(self) -> int:
        """Alias for ``batch_size`` used by operator-group code."""

        return self.batch_size

    def get_pse(self) -> Array:
        """Return the stored pure states with shape ``(batch, hilbert_dim)``."""

        if self._pse is None:
            raise ValueError("pure states have not been set")
        return self._pse

    def set_pse(self, pse: Any) -> None:
        """Set pure states.

        Args:
            pse: Complex array shaped ``(hilbert_dim,)`` or
                ``(batch_size, hilbert_dim)``.

        Raises:
            ValueError: If the shape is incompatible.
        """

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
        """Normalize the stored pure-state ensemble in place.

        Returns:
            The normalized states shaped ``(batch_size, hilbert_dim)``.

        Raises:
            ValueError: If any state has zero norm.
        """

        states = self.get_pse()
        norms = jnp.linalg.norm(states, axis=1)
        if bool(jnp.any(norms == 0)):
            raise ValueError("cannot normalize a zero pure state")
        self._pse = states / norms[:, None]
        return self._pse


class DensityMatrixEnsemble:
    """Container for a batch of density matrices.

    Args:
        hilbert_dim: Dimension of the Hilbert space.
        batch_size: Number of density matrices in the ensemble.
        dtype: Complex dtype used when storing density matrices.

    Stored density matrices have shape
    ``(batch_size, hilbert_dim, hilbert_dim)``. Passing a single density matrix
    shaped ``(hilbert_dim, hilbert_dim)`` to :meth:`set_dme` broadcasts it across
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
        self._dme: Array | None = None  ## shape: (batch_size, hilbert_dim, hilbert_dim)

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
        """Set density matrices.

        Args:
            dme: Complex array shaped ``(hilbert_dim, hilbert_dim)`` or
                ``(batch_size, hilbert_dim, hilbert_dim)``.

        Raises:
            ValueError: If the shape is incompatible.
        """

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
        """Normalize the stored density matrices to unit trace in place.

        Returns:
            The normalized density matrices shaped
            ``(batch_size, hilbert_dim, hilbert_dim)``.

        Raises:
            ValueError: If any density matrix has zero trace.
        """

        density_matrices = self.get_dme()
        traces = batch_trace(density_matrices)
        if bool(jnp.any(traces == 0)):
            raise ValueError("cannot normalize a zero-trace density matrix")
        self._dme = density_matrices / traces[:, None, None]
        return self._dme


__all__ = ["DensityMatrixEnsemble", "PureStatesEnsemble"]
