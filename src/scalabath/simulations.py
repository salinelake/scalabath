"""JAX-based time-evolution skeletons for quantum dynamics."""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from scalabath.systems import DensityMatrixEnsemble, PureStatesEnsemble
from scalabath.utilities import batch_expectation_density, batch_expectation_pure, positive_int


class _OperatorGroupLike(Protocol):
    hilbert_dim: int

    def sum_operators(self) -> Array: ...


def _as_pure_state_batch(initial_state: Any, dtype: Any) -> Array:
    if isinstance(initial_state, PureStatesEnsemble):
        state = initial_state.get_pse()
    else:
        state = jnp.asarray(initial_state, dtype=dtype)
    if state.ndim == 1:
        state = state[None, :]
    if state.ndim != 2:
        raise ValueError("pure states must have shape (hilbert_dim,) or (batch, hilbert_dim)")
    return jnp.asarray(state, dtype=dtype)

def _as_density_batch(initial_density_matrix: Any, dtype: Any) -> Array:
    if isinstance(initial_density_matrix, DensityMatrixEnsemble):
        density_matrix = initial_density_matrix.get_dme()
    else:
        density_matrix = jnp.asarray(initial_density_matrix, dtype=dtype)
    if density_matrix.ndim == 2:
        density_matrix = density_matrix[None, :, :]
    if density_matrix.ndim != 3 or density_matrix.shape[-1] != density_matrix.shape[-2]:
        raise ValueError(
            "density matrices must have shape (hilbert_dim, hilbert_dim) "
            "or (batch, hilbert_dim, hilbert_dim)"
        )
    return jnp.asarray(density_matrix, dtype=dtype)

def _as_operator_matrix(operator: Any, hilbert_dim: int, batch_size: int, dtype: Any) -> Array:
    if hasattr(operator, "sum_operators"):
        matrix = operator.sum_operators()
    else:
        matrix = operator
    matrix = jnp.asarray(matrix, dtype=dtype)
    expected_shape = (batch_size, hilbert_dim, hilbert_dim)
    if matrix.shape != expected_shape:
        raise ValueError(f"operator must have shape (batch_size, hilbert_dim, hilbert_dim), but got {matrix.shape}")
    return matrix


@jax.jit
def _apply_evolution_operator(states: Array, evolution_operator: Array) -> Array:
    return jnp.matmul(evolution_operator, states[..., None])[..., 0]
 
@jax.jit
def _lindblad_step_exact(
    density_matrices: Array,
    hamiltonian: Array,
    jump_operators: Array,
    dt: Array,
    ) -> Array:
    left = jnp.einsum("ij,bjk->bik", hamiltonian, density_matrices)
    right = jnp.einsum("bij,jk->bik", density_matrices, hamiltonian)
    coherent = -1j * (left - right)

    def one_jump(jump_operator: Array) -> Array:
        jump_dagger_jump = jnp.conjugate(jump_operator).T @ jump_operator
        gain = jnp.einsum(
            "ij,bjk,lk->bil",
            jump_operator,
            density_matrices,
            jnp.conjugate(jump_operator),
        )
        loss_left = jnp.einsum("ij,bjk->bik", jump_dagger_jump, density_matrices)
        loss_right = jnp.einsum("bij,jk->bik", density_matrices, jump_dagger_jump)
        return gain - 0.5 * (loss_left + loss_right)

    dissipative = jax.vmap(one_jump)(jump_operators).sum(axis=0)
    return density_matrices + dt * (coherent + dissipative)


class UnitarySimulation:
    """Unitary Schrodinger-equation evolution for pure-state ensembles.

    Args:
        initial_state: Complex array shaped ``(hilbert_dim,)`` or
            ``(batch, hilbert_dim)``. A :class:`PureStatesEnsemble` is also
            accepted.
        dt: Time-step size.
        hamiltonian: Optional initial Hamiltonian shaped
            ``(hilbert_dim, hilbert_dim)``.
        dtype: Complex dtype used to store state and operators.
    """

    def __init__(
        self,
        initial_state: Any,
        dt: float,
        *,
        hamiltonian: Any | None = None,
        dtype: Any = jnp.complex128,
    ) -> None:
        self.dtype = jnp.dtype(dtype)
        self.state = _as_pure_state_batch(initial_state, self.dtype)
        self.batch_size, self.hilbert_dim = self.state.shape
        self.dt = jnp.asarray(dt)
        if hamiltonian is not None:
            self.hamiltonian = _as_operator_matrix(hamiltonian, self.hilbert_dim, self.batch_size, self.dtype)
        else:
            self.hamiltonian = jnp.zeros((self.batch_size, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        self._evo_exact = None ## only computed when the system is small enough to allow for exact integration. shape: (batch_size, hilbert_dim, hilbert_dim)
        self._evo_AB = None ## shape: (batch_size, hilbert_dim, hilbert_dim)

    def add_operator_group_to_hamiltonian(self, operator_group: _OperatorGroupLike) -> None:
        """Add a static operator group to the Hamiltonian."""

        self.hamiltonian = self.hamiltonian + _as_operator_matrix(
            operator_group, self.hilbert_dim, self.batch_size, self.dtype
        )
    
    def _compute_exact_evolution_operator(self) -> None:
        """Save the exact evolution operator. This is only practical for small systems. For large systems, use approximate integration methods such as Trotter decomposition."""

        self._evo_exact = jsp.linalg.expm(-1j * self.dt * self.hamiltonian) ## shape: (batch_size, hilbert_dim, hilbert_dim)
    
    def _compute_AB_evolution_operator(self) -> None:
        """Compute the AB evolution operator."""

        ## get exp(-1j * dt * A) first
        A = jnp.diagonal(self.hamiltonian, axis1=-2, axis2=-1)
        evo_A = jnp.exp(-1j * self.dt * A) ## shape: (batch_size, hilbert_dim)
        
        ## get (1 - 1j * dt * B) next
        mask = 1 - jnp.eye(self.hilbert_dim, dtype=self.hamiltonian.dtype).reshape(1, self.hilbert_dim, self.hilbert_dim)
        B = self.hamiltonian * mask
        identity = jnp.eye(self.hilbert_dim, dtype=self.hamiltonian.dtype).reshape(1, self.hilbert_dim, self.hilbert_dim)
        evo_B = identity - 1j * self.dt * B ## shape: (batch_size, hilbert_dim, hilbert_dim)

        ## compute the AB evolution operator
        self._evo_AB = evo_B * evo_A[:,None,:]

    def step_exact(self, n_steps: int = 1) -> Array:
        """Advance the pure-state ensemble by ``n_steps`` time steps."""

        n_steps = positive_int(n_steps, "n_steps")
        if self._evo_exact is None:
            self._compute_exact_evolution_operator()
        for _ in range(n_steps):
            self.state = _apply_evolution_operator(self.state, self._evo_exact)
        return self.state

    def step_AB_scheme(self, n_steps: int = 1) -> Array:
        """Advance the system by ``n_steps`` time steps using the AB scheme.
        A is the diagonal part of the Hamiltonian, and B is the off-diagonal part. The step is given by ``pse(t+dt) = (1 - 1j * dt * B) * exp(-1j * dt * A) * pse(t)``.
        """
        n_steps = positive_int(n_steps, "n_steps")
        if self._evo_AB is None:
            self._compute_AB_evolution_operator()
        for _ in range(n_steps):
            self.state = _apply_evolution_operator(self.state, self._evo_AB)
        return self.state

    def observe(self, operator: Any) -> Array:
        """Return ``<psi|O|psi>`` for each state in the ensemble."""

        matrix = _as_operator_matrix(operator, self.hilbert_dim, self.batch_size, self.dtype)
        return batch_expectation_pure(self.state, matrix)

    def get_state(self) -> Array:
        """Return the current pure-state ensemble."""

        return self.state


class LindbladSimulation:
    """Lindblad master-equation evolution for density-matrix ensembles.

    The initial walking skeleton uses an explicit Euler step for
    ``d rho / dt = -i[H, rho] + sum_k D[L_k](rho)``. This keeps the evolution
    kernel simple and JAX-transformable while leaving room for higher-order
    integrators in later work.
    """

    def __init__(
        self,
        initial_density_matrix: Any,
        dt: float,
        *,
        hamiltonian: Any | None = None,
        jump_operators: Any | None = None,
        dtype: Any = jnp.complex128,
    ) -> None:
        self.dtype = jnp.dtype(dtype)
        self.density_matrices = _as_density_batch(initial_density_matrix, self.dtype)
        self.batch_size = self.density_matrices.shape[0]
        self.hilbert_dim = self.density_matrices.shape[1]
        self.dt = jnp.asarray(dt)
        if hamiltonian is not None:
            self.hamiltonian = _as_operator_matrix(hamiltonian, self.hilbert_dim, self.batch_size, self.dtype)
        else:
            self.hamiltonian = jnp.zeros((self.batch_size, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        self.jump_operators = jnp.zeros((0, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        if jump_operators is not None:
            for jump_operator in jump_operators:
                self.add_operator_group_to_jumping(jump_operator)

    def add_operator_group_to_hamiltonian(self, operator_group: _OperatorGroupLike) -> None:
        """Add a static operator group to the Hamiltonian."""

        self.hamiltonian = self.hamiltonian + _as_operator_matrix(
            operator_group, self.hilbert_dim, self.batch_size, self.dtype
        )

    def add_operator_group_to_jumping(self, operator_group: _OperatorGroupLike) -> None:
        """Add a static operator group to the Lindblad jump operators."""

        jump_operator = _as_operator_matrix(operator_group, self.hilbert_dim, self.batch_size, self.dtype)
        self.jump_operators = jnp.concatenate(
            [self.jump_operators, jump_operator[None, :, :]], axis=0
        )

    def step(self, n_steps: int = 1) -> Array:
        """Advance the density-matrix ensemble by ``n_steps`` time steps."""

        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        for _ in range(n_steps):
            self.density_matrices = _lindblad_step(
                self.density_matrices,
                self.hamiltonian,
                self.jump_operators,
                self.dt,
            )
        return self.density_matrices

    def observe(self, operator: Any) -> Array:
        """Return ``Tr(rho O)`` for each density matrix in the ensemble."""

        matrix = _as_operator_matrix(operator, self.hilbert_dim, self.batch_size, self.dtype)
        return batch_expectation_density(self.density_matrices, matrix)

    def get_state(self) -> Array:
        """Return the current density-matrix ensemble."""

        return self.density_matrices


__all__ = ["LindbladSimulation", "UnitarySimulation"]
