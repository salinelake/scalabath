"""JAX-based unitary time-evolution classes for dense and tensorized dynamics."""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from operator import mul
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding, Sharding
from jax.sharding import PartitionSpec as P

from scalabath.operators_base import boson
from scalabath.operators_groups import OperatorGroup
from scalabath.systems import PureStatesEnsemble, TensorProductPureStatesEnsemble
from scalabath.utilities import positive_int


def _prod(values: Sequence[int]) -> int:
    return reduce(mul, values, 1)


def _place_on_sharding(array: Array, sharding: Sharding | None) -> Array:
    if sharding is None:
        return array
    return jax.device_put(array, sharding)


def _replicated_sharding_from(sharding: Sharding | None) -> Sharding | None:
    if isinstance(sharding, NamedSharding):
        return NamedSharding(sharding.mesh, P())
    return None


def _as_operator_matrix(operator: Any, hilbert_dim: int, batch_size: int, dtype: Any) -> Array:
    """Return a dense operator shaped ``(batch, hilbert_dim, hilbert_dim)``."""

    if hasattr(operator, "sum_operators"):
        matrix = operator.sum_operators()
    else:
        matrix = operator
    matrix = jnp.asarray(matrix, dtype=dtype)
    matrix_shape = (hilbert_dim, hilbert_dim)
    batch_shape = (batch_size, hilbert_dim, hilbert_dim)
    if matrix.shape == matrix_shape:
        return jnp.broadcast_to(matrix, batch_shape)
    if matrix.shape == batch_shape:
        return matrix
    raise ValueError(
        "operator must have shape (hilbert_dim, hilbert_dim) or "
        f"(batch_size, hilbert_dim, hilbert_dim), got {matrix.shape}"
    )


def _as_local_matrix(matrix: Any, dim: int, batch_size: int, dtype: Any, name: str) -> Array:
    """Return a local matrix with optional leading batch axis."""

    arr = jnp.asarray(matrix, dtype=dtype)
    if arr.shape == (dim, dim):
        return arr
    if arr.shape == (batch_size, dim, dim):
        return arr
    raise ValueError(f"{name} must have shape ({dim}, {dim}) or ({batch_size}, {dim}, {dim})")


def _as_system_bath_hamiltonian(
    matrix: Any,
    system_dim: int,
    mode_dim: int,
    batch_size: int,
    dtype: Any,
    name: str,
) -> Array:
    """Return a system-bath Hamiltonian in compact block or dense layout."""

    arr = jnp.asarray(matrix, dtype=dtype)
    dense_dim = system_dim * mode_dim
    block_shape = (system_dim, mode_dim, mode_dim)
    batch_block_shape = (batch_size, system_dim, mode_dim, mode_dim)
    dense_shape = (dense_dim, dense_dim)
    batch_dense_shape = (batch_size, dense_dim, dense_dim)
    if arr.shape == block_shape:
        return arr
    if arr.shape == batch_block_shape:
        return arr
    if arr.shape == dense_shape:
        return arr
    if arr.shape == batch_dense_shape:
        return arr
    raise ValueError(
        f"{name} must have shape ({system_dim}, {mode_dim}, {mode_dim}) or "
        f"({batch_size}, {system_dim}, {mode_dim}, {mode_dim}) or "
        f"({dense_dim}, {dense_dim}) or ({batch_size}, {dense_dim}, {dense_dim})"
    )


def _zeros_local(dim: int, dtype: Any) -> Array:
    return jnp.zeros((dim, dim), dtype=dtype)


def _zeros_system_bath_blocks(system_dim: int, mode_dim: int, dtype: Any) -> Array:
    return jnp.zeros((system_dim, mode_dim, mode_dim), dtype=dtype)


def _is_system_bath_block_array(matrix: Array, system_dim: int, mode_dim: int) -> bool:
    return (matrix.ndim == 3 and matrix.shape == (system_dim, mode_dim, mode_dim)) or (
        matrix.ndim == 4 and matrix.shape[1:] == (system_dim, mode_dim, mode_dim)
    )


def _is_site_block_diagonal(matrix: Array, system_dim: int, mode_dim: int) -> bool:
    if matrix.ndim == 2:
        grid = matrix.reshape(system_dim, mode_dim, system_dim, mode_dim)
        mask = jnp.eye(system_dim, dtype=bool).reshape(system_dim, 1, system_dim, 1)
    else:
        grid = matrix.reshape(matrix.shape[0], system_dim, mode_dim, system_dim, mode_dim)
        mask = jnp.eye(system_dim, dtype=bool).reshape(1, system_dim, 1, system_dim, 1)
    off_diagonal = jnp.where(mask, jnp.zeros_like(grid), grid)
    return bool(jnp.allclose(off_diagonal, 0))


def _dense_system_bath_to_blocks(matrix: Array, system_dim: int, mode_dim: int) -> Array:
    if matrix.ndim == 2:
        grid = matrix.reshape(system_dim, mode_dim, system_dim, mode_dim)
        diagonal = jnp.diagonal(grid, axis1=0, axis2=2)
        return jnp.moveaxis(diagonal, -1, 0)

    grid = matrix.reshape(matrix.shape[0], system_dim, mode_dim, system_dim, mode_dim)
    diagonal = jnp.diagonal(grid, axis1=1, axis2=3)
    return jnp.moveaxis(diagonal, -1, 1)


def _system_bath_propagator(
    hamiltonian: Array,
    coefficient: Array,
    system_dim: int,
    mode_dim: int,
) -> Array:
    if _is_system_bath_block_array(hamiltonian, system_dim, mode_dim):
        return _matrix_exponential(hamiltonian, coefficient)
    if _is_site_block_diagonal(hamiltonian, system_dim, mode_dim):
        blocks = _dense_system_bath_to_blocks(hamiltonian, system_dim, mode_dim)
        return _matrix_exponential(blocks, coefficient)
    return _matrix_exponential(hamiltonian, coefficient)


def _matrix_exponential(matrix: Array, coefficient: Array) -> Array:
    """Compute ``expm(coefficient * matrix)`` with optional leading batch axis."""

    if matrix.ndim == 2:
        return jsp.linalg.expm(coefficient * matrix)
    if matrix.ndim == 3:
        return jax.vmap(lambda local: jsp.linalg.expm(coefficient * local))(matrix)
    if matrix.ndim == 4:
        leading_shape = matrix.shape[:-2]
        local_dim = matrix.shape[-1]
        flat_matrix = matrix.reshape((-1, local_dim, local_dim))
        flat_exponential = jax.vmap(lambda local: jsp.linalg.expm(coefficient * local))(flat_matrix)
        return flat_exponential.reshape((*leading_shape, local_dim, local_dim))
    raise ValueError("matrix exponential expects a rank-2, rank-3, or rank-4 matrix")


@jax.jit
def _apply_dense_evolution_operator(states: Array, evolution_operator: Array) -> Array:
    return (evolution_operator @ states[:, :, None])[:, :, 0]


@jax.jit
def _batch_expectation_pure_dense(states: Array, operator: Array) -> Array:
    """Compute ``<psi|O|psi>`` for batched dense operators."""

    operated = operator @ states[..., None]
    return (jnp.conjugate(states)[:, None, :] @ operated)[:, 0, 0]


def _apply_axis_operator(state: Array, operator: Array, axis: int) -> Array:
    moved = jnp.moveaxis(state, axis, 1)
    batch_size, dim = moved.shape[0], moved.shape[1]
    mat = moved.reshape(batch_size, dim, -1)
    if operator.ndim == 2:
        out = operator @ mat
    else:
        out = operator @ mat
    return jnp.moveaxis(out.reshape(moved.shape), 1, axis)


def _apply_axis_single(state: Array, operator: Array, axis: int) -> Array:
    moved = jnp.moveaxis(state, axis, 0)
    dim = moved.shape[0]
    mat = moved.reshape(dim, -1)
    out = (operator @ mat).reshape(moved.shape)
    return jnp.moveaxis(out, 0, axis)


def _apply_system_mode_operator(state: Array, operator: Array, mode_index: int) -> Array:
    mode_axis = mode_index + 2
    moved = jnp.moveaxis(state, mode_axis, 2)
    batch_size, system_dim, mode_dim = moved.shape[:3]
    mat = moved.reshape(batch_size, system_dim * mode_dim, -1)
    if operator.ndim == 2:
        out = operator @ mat
    else:
        out = operator @ mat
    return jnp.moveaxis(out.reshape(moved.shape), 2, mode_axis)


def _apply_system_mode_blocks(state: Array, operator: Array, mode_index: int) -> Array:
    mode_axis = mode_index + 2
    moved = jnp.moveaxis(state, mode_axis, 2)
    batch_size, system_dim, mode_dim = moved.shape[:3]
    mat = moved.reshape(batch_size, system_dim, mode_dim, -1)
    if operator.ndim == 3:
        out = operator[None, ...] @ mat
    else:
        out = operator @ mat
    return jnp.moveaxis(out.reshape(moved.shape), 2, mode_axis)


def _apply_system_mode_coupling(state: Array, operator: Array, mode_index: int) -> Array:
    mode_axis = mode_index + 2
    moved = jnp.moveaxis(state, mode_axis, 2)
    system_dim, mode_dim = moved.shape[1], moved.shape[2]
    if _is_system_bath_block_array(operator, system_dim, mode_dim):
        return _apply_system_mode_blocks(state, operator, mode_index)
    return _apply_system_mode_operator(state, operator, mode_index)


def _apply_bath_operator(state: Array, operator: Array) -> Array:
    batch_size, system_dim = state.shape[:2]
    bath_dim = _prod(state.shape[2:])
    matrix = state.reshape(batch_size, system_dim, bath_dim)
    out = matrix @ jnp.swapaxes(operator, -1, -2)
    return out.reshape(state.shape)


@jax.jit
def _normalize_state(state: Array) -> Array:
    norms = jnp.linalg.norm(state, axis=1)
    denom = jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype))
    return state / denom[:, None]

@jax.jit
def _norm_tensor_state(state: Array) -> Array:
    batch_size = state.shape[0]
    flat = state.reshape(batch_size, -1)
    norms = jnp.linalg.norm(flat, axis=1)
    return norms

@jax.jit
def _normalize_tensor_state(state: Array) -> Array:
    batch_size = state.shape[0]
    flat = state.reshape(batch_size, -1)
    norms = jnp.linalg.norm(flat, axis=1)
    denom = jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype))
    return (flat / denom[:, None]).reshape(state.shape)


@jax.jit
def _system_bath_unitary_trotter_step(
    state: Array,
    system_half: Array,
    bath_half: tuple[Array, ...],
    system_bath_full: tuple[Array, ...],
) -> Array:
    state = _apply_axis_operator(state, system_half, 1)
    for mode_index, operator in enumerate(bath_half):
        state = _apply_axis_operator(state, operator, mode_index + 2)
    for mode_index, operator in enumerate(system_bath_full):
        state = _apply_system_mode_coupling(state, operator, mode_index)
    state = _apply_axis_operator(state, system_half, 1)
    for mode_index, operator in enumerate(bath_half):
        state = _apply_axis_operator(state, operator, mode_index + 2)
    return state


@jax.jit
def _system_bath_deterministic_trajectory_step(
    state: Array,
    system_full: Array,
    bath_full_by_mode: tuple[Array, ...],
    system_bath_full: tuple[Array, ...],
) -> Array:
    state = _apply_axis_operator(state, system_full, 1)
    for mode_index, operator in enumerate(bath_full_by_mode):
        state = _apply_axis_operator(state, operator, mode_index + 2)
    for mode_index, operator in enumerate(system_bath_full):
        state = _apply_system_mode_coupling(state, operator, mode_index)
    return state


@jax.jit
def _system_bath_deterministic_trajectory_step_with_full_bath(
    state: Array,
    system_full: Array,
    bath_full: Array,
    system_bath_full: tuple[Array, ...],
) -> Array:
    state = _apply_axis_operator(state, system_full, 1)
    state = _apply_bath_operator(state, bath_full)
    for mode_index, operator in enumerate(system_bath_full):
        state = _apply_system_mode_coupling(state, operator, mode_index)
    return state


class UnitarySimulation:
    """Dense unitary Schrodinger-equation evolution for pure-state ensembles.
    This class is for simulating small systems and does not support multi-GPU parallelization."""

    def __init__(
        self,
        hilbert_dim: int,
        batch_size: int = 1,
        dt: float = 1.0,
        *,
        hamiltonian: Any | None = None,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.dtype = jnp.dtype(dtype)
        self.hilbert_dim = positive_int(hilbert_dim, "hilbert_dim")
        self.batch_size = positive_int(batch_size, "batch_size")
        self._pse = PureStatesEnsemble(
            hilbert_dim=self.hilbert_dim,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )
        self.dt = jnp.asarray(dt)
        if hamiltonian is None:
            self.hamiltonian = jnp.zeros(
                (self.batch_size, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype
            )
        else:
            self.hamiltonian = _as_operator_matrix(
                hamiltonian, self.hilbert_dim, self.batch_size, self.dtype
            )
        self._evo_exact: Array | None = None

    @property
    def state(self) -> Array:
        """Return the pure-state ensemble with shape ``(batch, hilbert_dim)``."""

        return self._pse.get_pse()

    @state.setter
    def state(self, state: Any) -> None:
        self._pse.set_pse(state)

    def add_operator_group_to_hamiltonian(self, operator_group: OperatorGroup | Array) -> None:
        """Add a static dense operator to the Hamiltonian."""

        self.hamiltonian = self.hamiltonian + _as_operator_matrix(
            operator_group, self.hilbert_dim, self.batch_size, self.dtype
        )
        self._evo_exact = None

    def _compute_exact_evolution_operator(self) -> None:
        self._evo_exact = _matrix_exponential(self.hamiltonian, -1j * self.dt)

    def step(self, n_steps: int = 1) -> Array:
        """Advance by ``n_steps`` using dense matrix exponentials."""

        n_steps = positive_int(n_steps, "n_steps")
        if self._evo_exact is None:
            self._compute_exact_evolution_operator()
        for _ in range(n_steps):
            self.state = _apply_dense_evolution_operator(self.state, self._evo_exact)
        return self.state

    def normalize(self) -> None:
        """Normalize the state in place."""
        self.state = _normalize_state(self.state)

    def observe(self, operator: Any) -> Array:
        """Return ``<psi|O|psi>`` for each state in the dense ensemble."""

        matrix = _as_operator_matrix(operator, self.hilbert_dim, self.batch_size, self.dtype)
        return _batch_expectation_pure_dense(self.state, matrix)


class SystemBathUnitarySimulation:
    """Efficient split-step unitary evolution for a system coupled to bosonic modes.
    This class is for simulating large systems and supports multi-GPU parallelization.
    States use the tensor layout ``(batch_size, system_dim, *boson_dims)``.
    Operators are local matrices:

    - ``system_hamiltonian``: ``(system_dim, system_dim)`` or batched.
    - ``bath_hamiltonians``: one ``(n_i, n_i)`` matrix per mode.
    - ``system_bath_hamiltonians``: one dense ``(system_dim * n_i, system_dim * n_i)``
      matrix, batched dense matrix, compact ``(system_dim, n_i, n_i)`` block
      array, or batched compact block array per mode.

    The step order is the second-order Trotter pattern used in the Hamiltonian
    scratch code: half system, half bath, full system-mode couplings, then the
    same half factors again.
    """

    def __init__(
        self,
        system_dim: int,
        boson_dims: Sequence[int],
        dt: float,
        *,
        boson_freqs: Sequence[float] | None = None,
        batch_size: int = 1,
        system_hamiltonian: Any | None = None,
        bath_hamiltonians: Sequence[Any] | None = None,
        system_bath_hamiltonians: Sequence[Any] | None = None,
        sharding: Sharding | None = None,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.dtype = jnp.dtype(dtype)
        self.sharding = sharding
        self.system_dim = positive_int(system_dim, "system_dim")
        ## initialize boson-bath frequencies and basis
        self.boson_dims = tuple(positive_int(dim, "boson_dim") for dim in boson_dims)
        self.nmodes = len(self.boson_dims)
        if boson_freqs is None:
            self.boson_freqs: tuple[float, ...] | None = None
        else:
            self.boson_freqs = tuple(float(freq) for freq in boson_freqs)
            if self.nmodes != len(self.boson_freqs):
                raise ValueError("boson_dims and boson_freqs must have the same length")
        self.boson_basis = tuple(boson(dim - 1, dtype=self.dtype) for dim in self.boson_dims)
        self.bath_dim = _prod(self.boson_dims)
        ## initialize batch size and time step
        self.batch_size = positive_int(batch_size, "batch_size")
        self.dt = jnp.asarray(dt)
        ## initialize tensor-product pure-state ensemble
        self._pse = TensorProductPureStatesEnsemble(
            (self.system_dim, *self.boson_dims),
            self.batch_size,
            dtype=self.dtype,
            sharding=sharding,
        )
        ## initialize the Hamiltonian if provided
        self.system_hamiltonian = _zeros_local(self.system_dim, self.dtype)
        self.bath_hamiltonians: tuple[Array, ...] = tuple(
            _zeros_local(dim, self.dtype) for dim in self.boson_dims
        )
        self.system_bath_hamiltonians: tuple[Array, ...] | None = tuple(
            _zeros_system_bath_blocks(self.system_dim, dim, self.dtype) for dim in self.boson_dims
        )
        self._propagators_ready = False
        if system_hamiltonian is not None:
            self.set_system_hamiltonian(system_hamiltonian)
        if bath_hamiltonians is not None:
            self.set_bath_hamiltonians(bath_hamiltonians)
        if system_bath_hamiltonians is not None:
            self.set_system_bath_hamiltonians(system_bath_hamiltonians)

    @property
    def state(self) -> Array:
        """Return states with shape ``(batch, system_dim, *boson_dims)``."""

        return self._pse.get_pse()

    @state.setter
    def state(self, state: Any) -> None:
        self._pse.set_pse(state)

    def set_product_state(self, system_state: Any, boson_states: Sequence[Any]) -> None:
        """Set a product initial state and broadcast it across the batch."""

        if len(boson_states) != len(self.boson_dims):
            raise ValueError("boson_states must contain one state per bosonic mode")
        self._pse.set_product_state((system_state, *boson_states))

    def normalize(self) -> None:
        """Normalize every batch state in place."""

        self.state = _normalize_tensor_state(self.state)

    def set_system_hamiltonian(self, hamiltonian: Any) -> None:
        self.system_hamiltonian = _as_local_matrix(
            hamiltonian, self.system_dim, self.batch_size, self.dtype, "system_hamiltonian"
        )
        self._propagators_ready = False

    def set_bath_hamiltonians(self, hamiltonians: Sequence[Any]) -> None:
        if len(hamiltonians) != len(self.boson_dims):
            raise ValueError("bath_hamiltonians must contain one matrix per bosonic mode")
        self.bath_hamiltonians = tuple(
            _as_local_matrix(matrix, dim, self.batch_size, self.dtype, "bath_hamiltonian")
            for matrix, dim in zip(hamiltonians, self.boson_dims, strict=True)
        )
        self._propagators_ready = False

    def set_bath_harmonic_hamiltonians(self) -> None:
        if self.boson_freqs is None:
            raise ValueError("boson_freqs must be provided to build harmonic bath Hamiltonians")
        hamiltonians = []
        for freq, basis in zip(self.boson_freqs, self.boson_basis, strict=True):
            mode_omega = jnp.asarray(freq, dtype=self.dtype)
            hamiltonians.append(mode_omega * basis.number)
        self.set_bath_hamiltonians(hamiltonians)

    def sample_thermal_bath_state(
        self,
        system_state: Any,
        kbT: float,
    ) -> tuple[TensorProductPureStatesEnsemble, np.ndarray]:
        """Sample product states with thermally occupied bath basis states.

        Args:
            system_state: System state shaped ``(system_dim,)`` or
                ``(batch_size, system_dim)``. A one-dimensional state is
                broadcast across the batch.
            kbT: Thermal energy in the same units as ``boson_freqs``.

        Returns:
            A tensor-product pure-state ensemble with native shape
            ``(batch_size, system_dim, *boson_dims)`` and an integer array of
            chosen bath occupation levels shaped ``(batch_size, nmodes)``.
        """
        if self.boson_freqs is None:
            raise ValueError("boson_freqs must be provided to sample thermal bath states")
        kbT_value = float(kbT)
        if kbT_value <= 0:
            raise ValueError("kbT must be positive")

        system_states = jnp.asarray(system_state, dtype=self.dtype)
        if system_states.shape == (self.system_dim,):
            system_states = jnp.broadcast_to(system_states, (self.batch_size, self.system_dim))
        elif system_states.shape != (self.batch_size, self.system_dim):
            raise ValueError(
                "system_state must have shape (system_dim,) or (batch_size, system_dim)"
            )

        states_shape = (self.batch_size, self.system_dim, *self.boson_dims)
        states = np.zeros(states_shape, dtype=np.dtype(self.dtype))
        system_states_np = np.asarray(system_states)
        chosen_levels = np.zeros((self.batch_size, self.nmodes), dtype=int)

        for batch_index in range(self.batch_size):
            levels = []
            for mode_index, mode_dim in enumerate(self.boson_dims):
                occupations = np.arange(mode_dim)
                weights = np.exp(-(occupations * self.boson_freqs[mode_index]) / kbT_value)
                probabilities = weights / weights.sum()
                level = int(np.random.choice(mode_dim, p=probabilities))  # noqa: NPY002
                levels.append(level)
            chosen_levels[batch_index] = levels
            states[(batch_index, slice(None), *levels)] = system_states_np[batch_index]

        flat_states = states.reshape(self.batch_size, -1)
        norms = np.linalg.norm(flat_states, axis=1)
        if np.any(norms == 0):
            raise ValueError("cannot normalize a zero pure state")
        states = (flat_states / norms[:, None]).reshape(states_shape)
        subsystem_dims = (self.system_dim, *self.boson_dims)
        state_sharding = self._pse.sharding if self._pse.sharding is not None else self.sharding
        ensemble = TensorProductPureStatesEnsemble(
            subsystem_dims,
            batch_size=self.batch_size,
            dtype=self.dtype,
            sharding=state_sharding,
        )
        ensemble.set_pse(states)
        return ensemble, chosen_levels

    def set_system_bath_hamiltonians(self, hamiltonians: Sequence[Any]) -> None:
        if len(hamiltonians) != len(self.boson_dims):
            raise ValueError("system_bath_hamiltonians must contain one operator per mode")
        self.system_bath_hamiltonians = tuple(
            _as_system_bath_hamiltonian(
                matrix,
                self.system_dim,
                dim,
                self.batch_size,
                self.dtype,
                "system_bath_hamiltonian",
            )
            for matrix, dim in zip(hamiltonians, self.boson_dims, strict=True)
        )
        self._propagators_ready = False

    def _prepare_propagators(self) -> None:
        half = -0.5j * self.dt
        full = -1j * self.dt
        state_sharding = self._pse.sharding if self._pse.sharding is not None else self.sharding
        propagator_sharding = _replicated_sharding_from(state_sharding)
        self._system_half = _place_on_sharding(
            _matrix_exponential(self.system_hamiltonian, half),
            propagator_sharding,
        )
        self._bath_half = tuple(
            _place_on_sharding(_matrix_exponential(matrix, half), propagator_sharding)
            for matrix in self.bath_hamiltonians
        )
        if self.system_bath_hamiltonians is not None:
            self._system_bath_full = tuple(
                _place_on_sharding(
                    _system_bath_propagator(matrix, full, self.system_dim, dim),
                    propagator_sharding,
                )
                for matrix, dim in zip(
                    self.system_bath_hamiltonians,
                    self.boson_dims,
                    strict=True,
                )
            )
            self.system_bath_hamiltonians = None
        elif not hasattr(self, "_system_bath_full"):
            raise ValueError("system-bath propagators have not been initialized")
        self._matrix_system_full = _place_on_sharding(
            _matrix_exponential(self.system_hamiltonian, full),
            propagator_sharding,
        )
        self._matrix_bath_full_by_mode = tuple(
            _place_on_sharding(_matrix_exponential(matrix, full), propagator_sharding)
            for matrix in self.bath_hamiltonians
        )
        self._propagators_ready = True

    def step(self, n_steps: int = 1) -> Array:
        """Advance by ``n_steps`` tensorized Trotter steps."""

        n_steps = positive_int(n_steps, "n_steps")
        if not self._propagators_ready:
            self._prepare_propagators()
        state = self.state
        for _ in range(n_steps):
            state = _system_bath_unitary_trotter_step(
                state,
                self._system_half,
                self._bath_half,
                self._system_bath_full,
            )
        self.state = state
        return self.state

    def reduced_system_density_matrix(self) -> Array:
        """Return ``rho_S`` with shape ``(batch, system_dim, system_dim)``."""

        return self._pse.reduced_density_matrix(0)

    def observe_system(self, operator: Any) -> Array:
        """Return system-only expectations from the reduced density matrix."""

        matrix = _as_local_matrix(
            operator,
            self.system_dim,
            self.batch_size,
            self.dtype,
            "operator",
        )
        rho_s = self.reduced_system_density_matrix()
        return jnp.trace(rho_s @ matrix, axis1=-2, axis2=-1)




__all__ = [
    "SystemBathUnitarySimulation",
    "UnitarySimulation",
]
