"""JAX-based time-evolution classes for dense and tensorized quantum dynamics."""

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

from scalabath.operators_base import boson
from scalabath.operators_groups import OperatorGroup
from scalabath.systems import (
    DensityMatrixEnsemble,
    PureStatesEnsemble,
    TensorProductPureStatesEnsemble,
)
from scalabath.utilities import adjoint, positive_int


def _prod(values: Sequence[int]) -> int:
    return reduce(mul, values, 1)


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


def _zeros_local(dim: int, dtype: Any) -> Array:
    return jnp.zeros((dim, dim), dtype=dtype)


def _matrix_exponential(matrix: Array, coefficient: Array) -> Array:
    """Compute ``expm(coefficient * matrix)`` with optional leading batch axis."""

    if matrix.ndim == 2:
        return jsp.linalg.expm(coefficient * matrix)
    if matrix.ndim == 3:
        return jax.vmap(lambda local: jsp.linalg.expm(coefficient * local))(matrix)
    raise ValueError("matrix exponential expects a rank-2 or rank-3 matrix")


@jax.jit
def _apply_dense_evolution_operator(states: Array, evolution_operator: Array) -> Array:
    return (evolution_operator @ states[:, :, None])[:, :, 0]


@jax.jit
def _dense_lindblad_step(
    density_matrices: Array,
    hamiltonian: Array,
    jump_operators: Array,
    dt: Array,
) -> Array:
    left = hamiltonian @ density_matrices
    right = density_matrices @ hamiltonian
    coherent = -1j * (left - right)

    def one_jump(jump_operator: Array) -> Array:
        jump_dagger = adjoint(jump_operator)
        jump_dagger_jump = jump_dagger @ jump_operator
        gain = jump_operator @ density_matrices @ jump_dagger
        loss_left = jump_dagger_jump @ density_matrices
        loss_right = density_matrices @ jump_dagger_jump
        return gain - 0.5 * (loss_left + loss_right)

    dissipative = jax.vmap(one_jump)(jump_operators).sum(axis=0)
    return density_matrices + dt * (coherent + dissipative)


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


def _apply_system_mode_single(state: Array, operator: Array, mode_index: int) -> Array:
    mode_axis = mode_index + 1
    moved = jnp.moveaxis(state, mode_axis, 1)
    system_dim, mode_dim = moved.shape[0], moved.shape[1]
    mat = moved.reshape(system_dim * mode_dim, -1)
    out = (operator @ mat).reshape(moved.shape)
    return jnp.moveaxis(out, 1, mode_axis)


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


@jax.jit
def _normalize_state(state: Array) -> Array:
    norms = jnp.linalg.norm(state, axis=1)
    denom = jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype))
    return state / denom[:, None]


@jax.jit
def _normalize_tensor_state(state: Array) -> Array:
    batch_size = state.shape[0]
    flat = state.reshape(batch_size, -1)
    norms = jnp.linalg.norm(flat, axis=1)
    denom = jnp.maximum(norms, jnp.asarray(1e-30, dtype=norms.dtype))
    return (flat / denom[:, None]).reshape(state.shape)


@jax.jit
def _normalize_density_matrix(density_matrix: Array) -> Array:
    trace = jnp.trace(density_matrix, axis1=-2, axis2=-1)
    denom = jnp.maximum(trace, jnp.asarray(1e-30, dtype=trace.dtype))
    return density_matrix / denom[:, None, None]


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
        state = _apply_system_mode_operator(state, operator, mode_index)
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
        state = _apply_system_mode_operator(state, operator, mode_index)
    return state


class UnitarySimulation:
    """Dense unitary Schrodinger-equation evolution for pure-state ensembles."""

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
        self._evo_ab: Array | None = None

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
        self._evo_ab = None

    def _compute_exact_evolution_operator(self) -> None:
        self._evo_exact = _matrix_exponential(self.hamiltonian, -1j * self.dt)

    def _compute_ab_evolution_operator(self) -> None:
        diagonal = jnp.diagonal(self.hamiltonian, axis1=-2, axis2=-1)
        diagonal_factor = jnp.exp(-1j * self.dt * diagonal)
        identity = jnp.eye(self.hilbert_dim, dtype=self.dtype)[None, :, :]
        off_diagonal_mask = 1 - identity
        off_diagonal = self.hamiltonian * off_diagonal_mask
        off_diagonal_factor = identity - 1j * self.dt * off_diagonal
        self._evo_ab = off_diagonal_factor * diagonal_factor[:, None, :]

    def step_exact(self, n_steps: int = 1) -> Array:
        """Advance by ``n_steps`` using dense matrix exponentials."""

        n_steps = positive_int(n_steps, "n_steps")
        if self._evo_exact is None:
            self._compute_exact_evolution_operator()
        for _ in range(n_steps):
            self.state = _apply_dense_evolution_operator(self.state, self._evo_exact)
        return self.state

    def step(self, n_steps: int = 1) -> Array:
        """Alias for :meth:`step_exact`."""

        return self.step_exact(n_steps)

    def step_AB_scheme(self, n_steps: int = 1) -> Array:
        """Advance with the dense diagonal/off-diagonal first-order AB scheme."""

        n_steps = positive_int(n_steps, "n_steps")
        if self._evo_ab is None:
            self._compute_ab_evolution_operator()
        for _ in range(n_steps):
            self.state = _apply_dense_evolution_operator(self.state, self._evo_ab)
        return self.state

    def normalize(self) -> None:
        """Normalize the state in place."""
        self.state = _normalize_state(self.state)

    def observe(self, operator: Any) -> Array:
        """Return ``<psi|O|psi>`` for each state in the dense ensemble."""

        matrix = _as_operator_matrix(operator, self.hilbert_dim, self.batch_size, self.dtype)
        return _batch_expectation_pure_dense(self.state, matrix)


class LindbladSimulation:
    """Dense Lindblad master-equation evolution for density-matrix ensembles."""

    def __init__(
        self,
        hilbert_dim: int,
        batch_size: int = 1,
        dt: float = 1.0,
        *,
        hamiltonian: Any | None = None,
        jump_operators: Sequence[Any] | None = None,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.dtype = jnp.dtype(dtype)
        self.hilbert_dim = positive_int(hilbert_dim, "hilbert_dim")
        self.batch_size = positive_int(batch_size, "batch_size")
        self._dme = DensityMatrixEnsemble(
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
        self.jump_operators = jnp.zeros(
            (0, self.batch_size, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype
        )
        if jump_operators is not None:
            for jump_operator in jump_operators:
                self.add_operator_group_to_jumping(jump_operator)

    @property
    def density_matrices(self) -> Array:
        """Return density matrices with shape ``(batch, hilbert_dim, hilbert_dim)``."""

        return self._dme.get_dme()

    @density_matrices.setter
    def density_matrices(self, density_matrices: Any) -> None:
        self._dme.set_dme(density_matrices)

    def add_operator_group_to_hamiltonian(self, operator_group: OperatorGroup | Array) -> None:
        """Add a static dense operator to the Hamiltonian."""

        self.hamiltonian = self.hamiltonian + _as_operator_matrix(
            operator_group, self.hilbert_dim, self.batch_size, self.dtype
        )

    def add_operator_group_to_jumping(self, operator_group: OperatorGroup | Array) -> None:
        """Add a static dense Lindblad jump operator."""

        jump_operator = _as_operator_matrix(
            operator_group,
            self.hilbert_dim,
            self.batch_size,
            self.dtype,
        )
        self.jump_operators = jnp.concatenate(
            [self.jump_operators, jump_operator[None, ...]],
            axis=0,
        )

    def step(self, n_steps: int = 1) -> Array:
        """Advance by ``n_steps`` explicit Euler Lindblad steps."""

        n_steps = positive_int(n_steps, "n_steps")
        for _ in range(n_steps):
            self.density_matrices = _dense_lindblad_step(
                self.density_matrices,
                self.hamiltonian,
                self.jump_operators,
                self.dt,
            )
        return self.density_matrices

    def normalize(self) -> None:
        """Normalize the state in place."""
        self.density_matrices = _normalize_density_matrix(self.density_matrices)

    def observe(self, operator: Any) -> Array:
        """Return ``Tr(rho O)`` for each density matrix."""

        matrix = _as_operator_matrix(operator, self.hilbert_dim, self.batch_size, self.dtype)
        return jnp.trace(self.density_matrices @ matrix, axis1=-2, axis2=-1)


class SystemBathUnitarySimulation:
    """Efficient split-step unitary evolution for a system coupled to bosonic modes.

    States use the tensor layout ``(batch_size, system_dim, *boson_dims)``.
    Operators are local matrices:

    - ``system_hamiltonian``: ``(system_dim, system_dim)`` or batched.
    - ``bath_hamiltonians``: one ``(n_i, n_i)`` matrix per mode.
    - ``system_bath_hamiltonians``: one ``(system_dim * n_i, system_dim * n_i)``
      matrix per mode.

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
        dtype: Any = jnp.complex64,
    ) -> None:
        self.dtype = jnp.dtype(dtype)
        self.system_dim = positive_int(system_dim, "system_dim")
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
        self.batch_size = positive_int(batch_size, "batch_size")
        self.dt = jnp.asarray(dt)
        self._pse = TensorProductPureStatesEnsemble(
            (self.system_dim, *self.boson_dims),
            self.batch_size,
            dtype=self.dtype,
        )
        self.system_hamiltonian = _zeros_local(self.system_dim, self.dtype)
        self.bath_hamiltonians: tuple[Array, ...] = tuple(
            _zeros_local(dim, self.dtype) for dim in self.boson_dims
        )
        self.system_bath_hamiltonians: tuple[Array, ...] = tuple(
            _zeros_local(self.system_dim * dim, self.dtype) for dim in self.boson_dims
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

        subsystem_dims = (self.system_dim, *self.boson_dims)
        ensemble = TensorProductPureStatesEnsemble(
            subsystem_dims,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )
        ensemble.set_pse(states)
        ensemble.normalize()
        return ensemble, chosen_levels

    def set_system_bath_hamiltonians(self, hamiltonians: Sequence[Any]) -> None:
        if len(hamiltonians) != len(self.boson_dims):
            raise ValueError("system_bath_hamiltonians must contain one matrix per bosonic mode")
        self.system_bath_hamiltonians = tuple(
            _as_local_matrix(
                matrix,
                self.system_dim * dim,
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
        self._system_half = _matrix_exponential(self.system_hamiltonian, half)
        self._bath_half = tuple(
            _matrix_exponential(matrix, half) for matrix in self.bath_hamiltonians
        )
        self._system_bath_full = tuple(
            _matrix_exponential(matrix, full) for matrix in self.system_bath_hamiltonians
        )
        self._matrix_system_full = _matrix_exponential(self.system_hamiltonian, full)
        self._matrix_bath_full_by_mode = tuple(
            _matrix_exponential(matrix, full) for matrix in self.bath_hamiltonians
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


class CoupledLindbladTrajectorySimulation:
    """Tensorized quantum-trajectory evolution for coupled Lindbladian baths.

    This class follows the structure of the scratch coupled-Lindbladian code:
    apply non-Hermitian local propagators to a pure state tensor, compare the
    resulting norm with a per-trajectory random threshold, and apply one bath
    jump operator when the threshold is crossed.
    """

    def __init__(
        self,
        system_dim: int,
        boson_dims: Sequence[int],
        dt: float,
        *,
        batch_size: int = 1,
        system_hamiltonian: Any | None = None,
        bath_hamiltonians: Sequence[Any] | None = None,
        system_bath_hamiltonians: Sequence[Any] | None = None,
        jump_operators: Sequence[Any] | None = None,
        key: Array | None = None,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.unitary_part = SystemBathUnitarySimulation(
            system_dim,
            boson_dims,
            dt,
            batch_size=batch_size,
            system_hamiltonian=system_hamiltonian,
            bath_hamiltonians=bath_hamiltonians,
            system_bath_hamiltonians=system_bath_hamiltonians,
            dtype=dtype,
        )
        self.dtype = self.unitary_part.dtype
        self.system_dim = self.unitary_part.system_dim
        self.boson_dims = self.unitary_part.boson_dims
        self.batch_size = self.unitary_part.batch_size
        self.dt = self.unitary_part.dt
        self.key = jax.random.PRNGKey(0) if key is None else key
        self.thresholds = self._sample_thresholds()
        if jump_operators is None:
            self.jump_operators = tuple(self._default_annihilation(dim) for dim in self.boson_dims)
        else:
            if len(jump_operators) != len(self.boson_dims):
                raise ValueError("jump_operators must contain one matrix per bosonic mode")
            self.jump_operators = tuple(
                _as_local_matrix(matrix, dim, self.batch_size, self.dtype, "jump_operator")
                for matrix, dim in zip(jump_operators, self.boson_dims, strict=True)
            )

    @property
    def state(self) -> Array:
        """Return states with shape ``(batch, system_dim, *boson_dims)``."""

        return self.unitary_part.state

    @state.setter
    def state(self, state: Any) -> None:
        self.unitary_part.state = state

    def _sample_thresholds(self) -> Array:
        self.key, subkey = jax.random.split(self.key)
        return jax.random.uniform(subkey, (self.batch_size,), dtype=jnp.float32)

    def _default_annihilation(self, dim: int) -> Array:
        weights = jnp.sqrt(jnp.arange(1, dim, dtype=jnp.float32)).astype(self.dtype)
        return (
            jnp.zeros((dim, dim), dtype=self.dtype)
            .at[jnp.arange(dim - 1), jnp.arange(1, dim)]
            .set(weights)
        )

    def _apply_jump_single_batch(self, state: Array, batch_index: int) -> tuple[Array, bool]:
        one_state = state[batch_index]
        norm = jnp.linalg.norm(one_state.reshape(-1))
        if bool(norm >= self.thresholds[batch_index]):
            return state, False

        weights = []
        jumped_states = []
        for mode_index, jump_operator in enumerate(self.jump_operators):
            local_jump = jump_operator if jump_operator.ndim == 2 else jump_operator[batch_index]
            jumped = _apply_axis_single(one_state, local_jump, mode_index + 1)
            jumped_states.append(jumped)
            weights.append(jnp.linalg.norm(jumped.reshape(-1)))
        weight_array = jnp.asarray(weights, dtype=jnp.float32)
        total = jnp.sum(weight_array)
        self.key, subkey = jax.random.split(self.key)
        if bool(total > 0):
            channel = int(jax.random.choice(subkey, len(jumped_states), p=weight_array / total))
        else:
            channel = int(jax.random.randint(subkey, (), 0, len(jumped_states)))
        jumped = jumped_states[channel]
        jumped_norm = jnp.linalg.norm(jumped.reshape(-1))
        jumped = jumped / jnp.maximum(jumped_norm, jnp.asarray(1e-30, dtype=jumped_norm.dtype))
        return state.at[batch_index].set(jumped), True

    def step(self, n_steps: int = 1) -> Array:
        """Advance by ``n_steps`` non-Hermitian trajectory steps."""

        n_steps = positive_int(n_steps, "n_steps")
        if not self.unitary_part._propagators_ready:
            self.unitary_part._prepare_propagators()
        state = self.state
        for _ in range(n_steps):
            state = _system_bath_deterministic_trajectory_step(
                state,
                self.unitary_part._matrix_system_full,
                self.unitary_part._matrix_bath_full_by_mode,
                self.unitary_part._system_bath_full,
            )
            jumped_any = False
            for batch_index in range(self.batch_size):
                state, jumped = self._apply_jump_single_batch(state, batch_index)
                jumped_any = jumped_any or jumped
            if jumped_any:
                self.thresholds = self._sample_thresholds()
        self.state = state
        return self.state

    def normalize(self) -> None:
        """Normalize the state in place."""
        ## TODO: implement this
        raise NotImplementedError(
            "Normalization for coupled Lindblad trajectory simulation is not implemented yet."
        )

    def reduced_system_density_matrix(self) -> Array:
        """Return ``rho_S`` with shape ``(batch, system_dim, system_dim)``."""

        return self.unitary_part.reduced_system_density_matrix()

    def observe_system(self, operator: Any) -> Array:
        """Return system-only expectations from the reduced density matrix."""

        return self.unitary_part.observe_system(operator)


__all__ = [
    "CoupledLindbladTrajectorySimulation",
    "LindbladSimulation",
    "SystemBathUnitarySimulation",
    "UnitarySimulation",
]
