"""JAX-based time-evolution skeletons for quantum dynamics."""

from __future__ import annotations

from typing import Any, Protocol
import logging
import time

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array

from scalabath.systems import DensityMatrixEnsemble, PureStatesEnsemble
from scalabath.operators_groups import OperatorGroup
from scalabath.utilities import batch_expectation_density, positive_int

def _as_operator_matrix(operator: Any, hilbert_dim: int, batch_size: int, dtype: Any) -> Array:
    if hasattr(operator, "sum_operators"):
        matrix = operator.sum_operators()
    elif isinstance(operator, Array):
        matrix = operator
    else:
        raise ValueError(f"operator must be an OperatorGroup or an Array, but got {type(operator)}")
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

@jax.jit
def _batch_expectation_pure(states: Array, operator: Array) -> Array:
    """Compute ``<psi|O|psi>`` for a pure-state ensemble.
    Args:
        states: Pure states shaped ``(batch_size, hilbert_dim)``.
        operator: Operator shaped ``(batch_size, hilbert_dim, hilbert_dim)``.
    Returns:
        Expectation values shaped ``(batch_size,)``.
    """
    result = (operator * states[:,None,:]).sum(axis=-1)
    result = (result * jnp.conjugate(states)).sum(axis=-1)
    return result

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
        hilbert_dim: int,
        batch_size: int,
        dt: float,
        *,
        hamiltonian: Any | None = None,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.dtype = jnp.dtype(dtype)
        self.hilbert_dim = positive_int(hilbert_dim, "hilbert_dim")
        self.batch_size = positive_int(batch_size, "batch_size")
        self._pse = PureStatesEnsemble(hilbert_dim=self.hilbert_dim, batch_size=self.batch_size, dtype=self.dtype)
        self.dt = jnp.asarray(dt)
        if hamiltonian is not None:
            self.hamiltonian = _as_operator_matrix(hamiltonian, self.hilbert_dim, self.batch_size, self.dtype)
        else:
            self.hamiltonian = jnp.zeros((self.batch_size, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        self._evo_exact = None ## only computed when the system is small enough to allow for exact integration. shape: (batch_size, hilbert_dim, hilbert_dim)
        self._evo_AB = None ## shape: (batch_size, hilbert_dim, hilbert_dim)
    
    @property
    def state(self) -> Array:
        """Return the pure state ensemble with shape (batch_size, hilbert_dim)."""
        return self._pse.get_pse()
    
    @state.setter
    def state(self, state: Array) -> None:
        """Set the pure state ensemble with shape (batch_size, hilbert_dim)."""
        self._pse.set_pse(state)
    
    def add_operator_group_to_hamiltonian(self, operator_group: OperatorGroup) -> None:
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

    def step_exact(self, n_steps: int = 1, debug: bool = False) -> Array:
        """Advance the pure-state ensemble by ``n_steps`` time steps."""

        n_steps = positive_int(n_steps, "n_steps")
        if self._evo_exact is None:
            if debug:
                time_start = time.time()
            self._compute_exact_evolution_operator()
            if debug:
                print(f'Computing exact evolution operator. Time taken: {time.time() - time_start} seconds')
        for step_idx in range(n_steps):
            if debug:
                if step_idx == 0:
                    print('Stepping exact evolution operator...')
                    time_start = time.time()
                if step_idx > 0:
                    print(f'Time taken for step {step_idx-1}: {time.time() - time_start} seconds')
                    time_start = time.time()
            self.state = _apply_evolution_operator(self.state, self._evo_exact)
        return self.state

    def step_AB_scheme(self, n_steps: int = 1, debug: bool = False) -> Array:
        """Advance the system by ``n_steps`` time steps using the AB scheme.
        A is the diagonal part of the Hamiltonian, and B is the off-diagonal part. The step is given by ``pse(t+dt) = (1 - 1j * dt * B) * exp(-1j * dt * A) * pse(t)``.
        """
        n_steps = positive_int(n_steps, "n_steps")
        if self._evo_AB is None:
            if debug:
                print('Computing AB evolution operator...')
                time_start = time.time()
            self._compute_AB_evolution_operator()
            if debug:
                print('Computing AB evolution operator...Done')
                print(f'Time taken: {time.time() - time_start} seconds')
        for _ in range(n_steps):
            if debug:
                if step_idx == 0:
                    print('Stepping AB evolution operator...')
                    time_start = time.time()
                if step_idx > 0 and step_idx < 5:
                    print(f'Step {step_idx} done')
                    print(f'Time taken: {time.time() - time_start} seconds')
            self.state = _apply_evolution_operator(self.state, self._evo_AB)
        return self.state

    def observe(self, operator: Any, debug: bool = False) -> Array:
        """Return ``<psi|O|psi>`` for each state in the ensemble."""

        matrix = _as_operator_matrix(operator, self.hilbert_dim, self.batch_size, self.dtype)
        if debug:
            time_start = time.time()
        result = _batch_expectation_pure(self.state, matrix)
        if debug:
            print(f'Observation time taken: {time.time() - time_start} seconds')
        return result



class LindbladSimulation:
    """Lindblad master-equation evolution for density-matrix ensembles.

    The initial walking skeleton uses an explicit Euler step for
    ``d rho / dt = -i[H, rho] + sum_k D[L_k](rho)``. This keeps the evolution
    kernel simple and JAX-transformable while leaving room for higher-order
    integrators in later work.
    """

    def __init__(
        self,
        hilbert_dim: int,
        batch_size: int,
        dt: float,
        *,
        hamiltonian: Any | None = None,
        jump_operators: Any | None = None,
        dtype: Any = jnp.complex64,
    ) -> None:
        self.dtype = jnp.dtype(dtype)
        self.hilbert_dim = positive_int(hilbert_dim, "hilbert_dim")
        self.batch_size = positive_int(batch_size, "batch_size")
        self._dme = DensityMatrixEnsemble(hilbert_dim=self.hilbert_dim, batch_size=self.batch_size, dtype=self.dtype)
        self.dt = jnp.asarray(dt)
        if hamiltonian is not None:
            self.hamiltonian = _as_operator_matrix(hamiltonian, self.hilbert_dim, self.batch_size, self.dtype)
        else:
            self.hamiltonian = jnp.zeros((self.batch_size, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        self.jump_operators = jnp.zeros((0, self.hilbert_dim, self.hilbert_dim), dtype=self.dtype)
        if jump_operators is not None:
            for jump_operator in jump_operators:
                self.add_operator_group_to_jumping(jump_operator)

    @property
    def density_matrices(self) -> Array:
        """Return the density-matrix ensemble with shape (batch_size, hilbert_dim, hilbert_dim)."""
        return self._dme.get_dme()
    
    @density_matrices.setter
    def density_matrices(self, density_matrices: Array) -> None:
        """Set the density-matrix ensemble with shape (batch_size, hilbert_dim, hilbert_dim)."""
        self._dme.set_dme(density_matrices)

    def add_operator_group_to_hamiltonian(self, operator_group: OperatorGroup) -> None:
        """Add a static operator group to the Hamiltonian."""

        self.hamiltonian = self.hamiltonian + _as_operator_matrix(
            operator_group, self.hilbert_dim, self.batch_size, self.dtype
        )

    def add_operator_group_to_jumping(self, operator_group: OperatorGroup) -> None:
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


__all__ = ["LindbladSimulation", "UnitarySimulation"]
