"""JAX-based Lindblad and stochastic-trajectory simulation classes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from scalabath.operators_groups import OperatorGroup
from scalabath.simulations_unitary import (
    SystemBathUnitarySimulation,
    _apply_axis_single,
    _as_local_matrix,
    _as_operator_matrix,
    _matrix_exponential,
    _norm_tensor_state,
    _system_bath_deterministic_trajectory_step,
    _system_bath_deterministic_trajectory_step_with_full_bath,
)
from scalabath.systems import DensityMatrixEnsemble
from scalabath.utilities import ABAd, adjoint, positive_int


@jax.jit
def _dense_lindblad_euler_step(
    density_matrices: Array,
    hamiltonian: Array,
    jump_operators: Array,
    dt: Array,
) -> Array:
    """
    Solve one Lindblad step with naive first-order Euler integration.

    This method should not be used for long-time evolution.
    Args:
        density_matrices: Density matrices with shape (batch, hilbert_dim, hilbert_dim).
        hamiltonian: Hamiltonian with shape (batch, hilbert_dim, hilbert_dim).
        jump_operators: Jump operators with shape (n_jumps, batch, hilbert_dim, hilbert_dim).
        dt: Time step.
    Returns:
        Density matrices with shape (batch, hilbert_dim, hilbert_dim).
    """
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
def _dense_lindblad_step(
    density_matrices: Array,
    hamiltonian: Array,
    jump_operators: Array,
    dt: Array,
) -> Array:
    """
    Solve one Lindblad step with a structure-preserving integrator.

    Positivity is preserved by construction. Trace is not exactly conserved.
    Args:
        density_matrices: Density matrices with shape (batch, hilbert_dim, hilbert_dim).
        hamiltonian: Hamiltonian with shape (batch, hilbert_dim, hilbert_dim).
        jump_operators: Jump operators with shape (n_jumps, batch, hilbert_dim, hilbert_dim).
        dt: Time step.
    Returns:
        Density matrices with shape (batch, hilbert_dim, hilbert_dim).
    """
    ## get the effective Hamiltonian H_eff = H - 1/2 * i * \sum_k L_k^\dagger L_k
    ham_eff = hamiltonian * 1.0
    for jump_operator in jump_operators:
        ham_eff = ham_eff - 1j * 0.5 * adjoint(jump_operator) @ jump_operator
    ## first step: rho -> (1-i*dt*H_eff)rho(1+i*dt*H_eff)
    identity = jnp.eye(density_matrices.shape[-1], dtype=density_matrices.dtype)
    rho = ABAd(identity[None, :, :] - 1j * dt * ham_eff, density_matrices)
    ## second step: rho -> rho + dt * \sum_k L_k \rho L_k^\dagger
    for jump_operator in jump_operators:
        rho += dt * ABAd(jump_operator, rho)
    return rho


@jax.jit
def _normalize_density_matrix(density_matrix: Array) -> Array:
    trace = jnp.trace(density_matrix, axis1=-2, axis2=-1)
    denom = jnp.maximum(trace, jnp.asarray(1e-30, dtype=trace.dtype))
    return density_matrix / denom[:, None, None]


class LindbladSimulation:
    """Dense Lindblad master-equation evolution for density-matrix ensembles.
    This class is for simulating small systems and does not support multi-GPU parallelization."""

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

    def step_euler(self, n_steps: int = 1) -> Array:
        """Advance by ``n_steps`` explicit Euler Lindblad steps."""

        n_steps = positive_int(n_steps, "n_steps")
        for _ in range(n_steps):
            self.density_matrices = _dense_lindblad_euler_step(
                self.density_matrices,
                self.hamiltonian,
                self.jump_operators,
                self.dt,
            )
        return self.density_matrices

    def step(self, n_steps: int = 1) -> Array:
        """Advance by ``n_steps`` structure preserving Lindblad steps."""

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


class CoupledLindbladTrajectorySimulation:
    """Stochastic Schrödinger evolution for a system coupled to bosonic modes.

    After averaging, the reduced system density matrix is equivalent to the
    Lindblad master-equation evolution.

    The simulation steps are:
    1. Apply non-Hermitian local propagators to a pure state tensor.
    2. Compare the resulting norm with a per-trajectory random threshold.
    3. Apply one bath jump operator when the threshold is crossed.
    4. Repeat the above steps for a given number of steps.
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
        bath_hamiltonian: Any | None = None,
        bath_hamiltonians: Sequence[Any] | None = None,
        system_bath_hamiltonians: Sequence[Any] | None = None,
        jump_operators: Sequence[Any] | None = None,
        key: Array | None = None,
        dtype: Any = jnp.complex64,
    ) -> None:
        """
        Initialize the coupled Lindblad trajectory simulation.
        Args:
            system_dim: The dimension of the system.
            boson_dims: The dimensions of the bosonic modes.
            dt: The time step.
            boson_freqs: The frequencies of the bosonic modes.
            batch_size: The batch size.
            system_hamiltonian: The system Hamiltonian.
            bath_hamiltonian: The total bath Hamiltonian, for all modes.
            bath_hamiltonians: A list of bath Hamiltonians, one for each mode.
                This is used when the bath Hamiltonian is diagonal.
            system_bath_hamiltonians: The system-bath Hamiltonians, one for each mode.
            jump_operators: The jump operators, one for each mode.
            key: The random key.
            dtype: The data type.
        """

        if bath_hamiltonian is not None and bath_hamiltonians is not None:
            raise ValueError("provide either bath_hamiltonian or bath_hamiltonians, not both")
        ## initialize the unitary part. If the bath Hamiltonian is not diagonal,
        ## we set the bath hamiltonian to be zero in the unitary part.
        self.unitary_part = SystemBathUnitarySimulation(
            system_dim,
            boson_dims,
            dt,
            boson_freqs=boson_freqs,
            batch_size=batch_size,
            system_hamiltonian=system_hamiltonian,
            bath_hamiltonians=bath_hamiltonians if bath_hamiltonian is None else None,
            system_bath_hamiltonians=system_bath_hamiltonians,
            dtype=dtype,
        )
        ## initialize the Lindblad part.
        self.dtype = self.unitary_part.dtype
        self.system_dim = self.unitary_part.system_dim
        self.boson_dims = self.unitary_part.boson_dims
        self.nmodes = self.unitary_part.nmodes
        self.boson_freqs = self.unitary_part.boson_freqs
        self.bath_dim = self.unitary_part.bath_dim
        self.batch_size = self.unitary_part.batch_size
        self.dt = self.unitary_part.dt
        if bath_hamiltonian is None:
            self.bath_hamiltonian = None
            self._bath_full = None
        else:
            self.bath_hamiltonian = _as_local_matrix(
                bath_hamiltonian,
                self.bath_dim,
                self.batch_size,
                self.dtype,
                "bath_hamiltonian",
            )
            self._bath_full = None
        self.key = jax.random.PRNGKey(0) if key is None else key
        self.thresholds = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        if jump_operators is None:
            self.jump_operators = tuple(self._default_annihilation(dim) for dim in self.boson_dims)
        else:
            if len(jump_operators) != len(self.boson_dims):
                raise ValueError("jump_operators must contain one matrix per bosonic mode")
            self.jump_operators = tuple(
                _as_local_matrix(matrix, dim, self.batch_size, self.dtype, "jump_operator")
                for matrix, dim in zip(jump_operators, self.boson_dims, strict=True)
            )

    def _prepare_trajectory_propagators(self) -> None:
        if not self.unitary_part._propagators_ready:
            self.unitary_part._prepare_propagators()
        ## we need to handle the non-diagonal bath Hamiltonian here because the
        ## unitary part does not support it.
        if self.bath_hamiltonian is not None and self._bath_full is None:
            self._bath_full = _matrix_exponential(self.bath_hamiltonian, -1j * self.dt)

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

    def _apply_jump_single_batch(self, state: Array, batch_index: int) -> Array:
        """Apply a jump to one already-selected batch element."""
        one_state = state[batch_index]  # (system_dim, *boson_dims)
        weights = []
        jumped_states = []
        ## calculate <psi|L_k^\dagger L_k|psi> for each jump operator.
        for mode_index, jump_operator in enumerate(self.jump_operators):
            local_jump = jump_operator if jump_operator.ndim == 2 else jump_operator[batch_index]
            jumped = _apply_axis_single(one_state, local_jump, mode_index + 1)
            jumped_states.append(jumped)
            jumped_norm = jnp.linalg.norm(jumped.reshape(-1))
            weights.append(jumped_norm**2)
        weight_array = jnp.asarray(weights, dtype=jnp.float32)
        total = jnp.sum(weight_array)
        if bool(total <= 0):
            raise ValueError(
                "total weight of possible jumping events is zero. check the jump operators."
            )
        self.key, subkey = jax.random.split(self.key)
        channel = int(jax.random.choice(subkey, len(jumped_states), p=weight_array / total))
        jumped = jumped_states[channel]
        jumped_norm = weights[channel] ** 0.5  ## this is the norm of the jumped state.
        jumped = jumped / jnp.maximum(
            jumped_norm,
            jnp.asarray(1e-30, dtype=jumped_norm.dtype),
        )
        return state.at[batch_index].set(jumped)

    def step(self, n_steps: int = 1) -> Array:
        """Advance by ``n_steps`` non-Hermitian trajectory steps."""

        n_steps = positive_int(n_steps, "n_steps")
        self._prepare_trajectory_propagators()
        state = self.state
        for _ in range(n_steps):
            if self.bath_hamiltonian is None:
                state = _system_bath_deterministic_trajectory_step(
                    state,
                    self.unitary_part._matrix_system_full,
                    self.unitary_part._matrix_bath_full_by_mode,
                    self.unitary_part._system_bath_full,
                )
            else:
                state = _system_bath_deterministic_trajectory_step_with_full_bath(
                    state,
                    self.unitary_part._matrix_system_full,
                    self._bath_full,
                    self.unitary_part._system_bath_full,
                )

            flat_state = state.reshape(self.batch_size, -1)
            norm = _norm_tensor_state(flat_state)  # (batch_size,)
            self.thresholds = self._sample_thresholds()
            jump_mask = (norm**2) < self.thresholds  # these batches will jump
            state = (flat_state / norm[:, None]).reshape(state.shape)
            ## for those batches that will jump, apply the jump update.
            for batch_index in np.flatnonzero(np.asarray(jump_mask)):
                state = self._apply_jump_single_batch(state, batch_index)
        self.state = state
        return self.state

    def normalize(self) -> None:
        """Normalize the state in place."""
        self.unitary_part.normalize()

    def reduced_system_density_matrix(self) -> Array:
        """Return ``rho_S`` with shape ``(batch, system_dim, system_dim)``."""

        return self.unitary_part.reduced_system_density_matrix()

    def observe_system(self, operator: Any) -> Array:
        """Return system-only expectations from the reduced density matrix."""

        return self.unitary_part.observe_system(operator)


__all__ = [
    "CoupledLindbladTrajectorySimulation",
    "LindbladSimulation",
]
