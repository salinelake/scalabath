"""Second-order time-convolutionless dynamics for reduced density matrices.

The implementation follows the finite-time, nonsecular TCL2 construction used
for local harmonic environments in J. H. Fetherolf and T. C. Berkelbach,
J. Chem. Phys. 147, 244109 (2017), https://doi.org/10.1063/1.5006824.
The package convention is :math:`\\hbar=1`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from scalabath.utilities import adjoint, complex_dtype, positive_int


def _require_finite(array: Array, name: str) -> None:
    """Validate a setup-time array without adding checks to JIT kernels."""

    if not bool(np.all(np.isfinite(np.asarray(array)))):
        raise ValueError(f"{name} must contain only finite values")


def _require_hermitian(array: Array, name: str, dtype: jnp.dtype) -> None:
    """Validate Hermiticity using a tolerance appropriate for ``dtype``."""

    tolerance = 1e-12 if dtype == jnp.dtype(jnp.complex128) else 1e-5
    host_array = np.asarray(array)
    if not np.allclose(
        host_array,
        np.swapaxes(np.conjugate(host_array), -1, -2),
        rtol=tolerance,
        atol=tolerance,
    ):
        raise ValueError(f"{name} must be Hermitian")


def _as_hamiltonian(hamiltonian: Any, dtype: jnp.dtype) -> Array:
    matrix = jnp.asarray(hamiltonian, dtype=dtype)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must have shape (system_dim, system_dim)")
    _require_finite(matrix, "hamiltonian")
    _require_hermitian(matrix, "hamiltonian", dtype)
    return matrix


def _as_coupling_operators(
    coupling_operators: Any,
    system_dim: int,
    dtype: jnp.dtype,
) -> Array:
    operators = jnp.asarray(coupling_operators, dtype=dtype)
    if operators.ndim == 2:
        operators = operators[None, ...]
    expected_matrix_shape = (system_dim, system_dim)
    if (
        operators.ndim != 3
        or operators.shape[0] == 0
        or operators.shape[1:] != expected_matrix_shape
    ):
        raise ValueError(
            "coupling_operators must have shape (system_dim, system_dim) "
            "or (n_channels, system_dim, system_dim)"
        )
    _require_finite(operators, "coupling_operators")
    _require_hermitian(operators, "coupling_operators", dtype)
    return operators


def _as_correlation_terms(
    values: Any,
    n_channels: int,
    dtype: jnp.dtype,
    name: str,
) -> Array:
    terms = jnp.asarray(values, dtype=dtype)
    if terms.ndim == 0:
        terms = terms.reshape(1)
    if terms.ndim == 1:
        if terms.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one term")
        terms = jnp.broadcast_to(terms[None, :], (n_channels, terms.shape[0]))
    elif terms.ndim != 2 or terms.shape[0] != n_channels or terms.shape[1] == 0:
        raise ValueError(f"{name} must have shape (n_terms,) or (n_channels, n_terms)")
    _require_finite(terms, name)
    return terms


def _integrated_exponential(rate: Array, time: Array) -> Array:
    r"""Return :math:`\\int_0^t \\exp(-z\\tau)d\\tau` stably.

    ``rate`` may be complex and may contain exact or near-zero values. The
    returned array has the broadcast shape of ``rate`` and ``time``.
    """

    rate = jnp.asarray(rate)
    real_dtype = jnp.real(rate).dtype
    time = jnp.asarray(time, dtype=real_dtype)
    scaled_rate = rate * time
    threshold = jnp.sqrt(jnp.asarray(jnp.finfo(real_dtype).eps, dtype=real_dtype))
    safe_rate = jnp.where(jnp.abs(rate) == 0, jnp.ones_like(rate), rate)
    exact = -jnp.expm1(-scaled_rate) / safe_rate
    series = time * (1.0 - scaled_rate / 2.0 + scaled_rate**2 / 6.0 - scaled_rate**3 / 24.0)
    return jnp.where(jnp.abs(scaled_rate) < threshold, series, exact)


@jax.jit
def _integrated_correlation_operators(
    time: Array,
    energy_gaps: Array,
    coupling_operators: Array,
    correlation_coefficients: Array,
    correlation_exponents: Array,
) -> Array:
    r"""Build the finite-time TCL2 operators in the system energy basis.

    The bath correlations are represented as

    .. math::

        C_a(t) = \\sum_r c_{ar}\\exp(-\\nu_{ar}t).

    Args:
        time: Scalar time.
        energy_gaps: Bohr-frequency matrix shaped ``(system_dim, system_dim)``,
            with ``energy_gaps[m, n] = E_m - E_n``.
        coupling_operators: Energy-basis system operators shaped
            ``(n_channels, system_dim, system_dim)``.
        correlation_coefficients: Complex coefficients shaped
            ``(n_channels, n_terms)``.
        correlation_exponents: Complex exponents shaped
            ``(n_channels, n_terms)``.

    Returns:
        Integrated operators shaped
        ``(n_channels, system_dim, system_dim)``.
    """

    rates = correlation_exponents[:, :, None, None] + 1j * energy_gaps[None, None, :, :]
    integrals = _integrated_exponential(rates, time)
    weights = correlation_coefficients[:, :, None, None] * integrals
    return coupling_operators * jnp.sum(weights, axis=1)


@jax.jit
def _tcl2_rhs(
    density_matrices: Array,
    time: Array,
    energy_gaps: Array,
    coupling_operators: Array,
    correlation_coefficients: Array,
    correlation_exponents: Array,
) -> Array:
    r"""Evaluate the nonsecular TCL2 equation in the system energy basis.

    The implemented Schrödinger-picture equation is

    .. math::

        \\dot\\rho = -i[H_S,\\rho]
        -\\sum_a\\left([S_a,\\Phi_a(t)\\rho]
        +[\\rho\\Phi_a^\\dagger(t),S_a]\\right).

    Args:
        density_matrices: Density matrices shaped
            ``(batch, system_dim, system_dim)``.
        time: Scalar time.
        energy_gaps: Matrix shaped ``(system_dim, system_dim)``.
        coupling_operators: Energy-basis operators shaped
            ``(n_channels, system_dim, system_dim)``.
        correlation_coefficients: Array shaped
            ``(n_channels, n_terms)``.
        correlation_exponents: Array shaped
            ``(n_channels, n_terms)``.

    Returns:
        Density-matrix derivatives with the same shape as ``density_matrices``.
    """

    integrated_operators = _integrated_correlation_operators(
        time,
        energy_gaps,
        coupling_operators,
        correlation_coefficients,
        correlation_exponents,
    )
    derivative = -1j * energy_gaps[None, :, :] * density_matrices
    for channel_index in range(coupling_operators.shape[0]):
        system_operator = coupling_operators[channel_index]
        integrated_operator = integrated_operators[channel_index]

        integrated_rho = integrated_operator @ density_matrices
        first_commutator = system_operator @ integrated_rho - integrated_rho @ system_operator

        rho_integrated_dagger = density_matrices @ adjoint(integrated_operator)
        second_commutator = (
            rho_integrated_dagger @ system_operator - system_operator @ rho_integrated_dagger
        )
        derivative = derivative - first_commutator - second_commutator
    return derivative


@jax.jit
def _tcl2_rk4_step(
    density_matrices: Array,
    time: Array,
    dt: Array,
    energy_gaps: Array,
    coupling_operators: Array,
    correlation_coefficients: Array,
    correlation_exponents: Array,
) -> Array:
    """Advance the explicitly time-dependent TCL2 equation by one RK4 step."""

    half_dt = 0.5 * dt
    k1 = _tcl2_rhs(
        density_matrices,
        time,
        energy_gaps,
        coupling_operators,
        correlation_coefficients,
        correlation_exponents,
    )
    k2 = _tcl2_rhs(
        density_matrices + half_dt * k1,
        time + half_dt,
        energy_gaps,
        coupling_operators,
        correlation_coefficients,
        correlation_exponents,
    )
    k3 = _tcl2_rhs(
        density_matrices + half_dt * k2,
        time + half_dt,
        energy_gaps,
        coupling_operators,
        correlation_coefficients,
        correlation_exponents,
    )
    k4 = _tcl2_rhs(
        density_matrices + dt * k3,
        time + dt,
        energy_gaps,
        coupling_operators,
        correlation_coefficients,
        correlation_exponents,
    )
    return density_matrices + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class TCL2Simulation:
    r"""Dense finite-time TCL2 evolution for reduced density matrices.

    The static system Hamiltonian is diagonalized once. Evolution is stored in
    that energy basis, while public density matrices and observables use the
    original input basis.

    Args:
        hamiltonian: Hermitian system Hamiltonian shaped
            ``(system_dim, system_dim)``.
        coupling_operators: Hermitian system coupling operators shaped
            ``(n_channels, system_dim, system_dim)``. A single rank-2 operator
            is accepted.
        correlation_coefficients: Coefficients :math:`c_{ar}` shaped
            ``(n_channels, n_terms)``. A shared ``(n_terms,)`` array is
            broadcast over channels.
        correlation_exponents: Exponents :math:`\\nu_{ar}` with the same shape
            convention as ``correlation_coefficients``. Correlations are
            :math:`C_a(t)=\\sum_r c_{ar}\\exp(-\\nu_{ar}t)`.
        dt: Positive integration time step in the inverse units of the
            Hamiltonian.
        batch_size: Number of independent density matrices.
        dtype: Complex JAX dtype.
        density_matrices: Optional initial matrix shaped
            ``(system_dim, system_dim)`` or batch shaped
            ``(batch_size, system_dim, system_dim)``.

    Notes:
        The Schrödinger-picture convention is
        :math:`\Phi_a(t)=\int_0^t C_a(\tau)e^{-iH_S\tau}S_a
        e^{iH_S\tau}d\tau`; an energy-basis element with gap
        :math:`E_m-E_n` therefore uses the rate
        :math:`\nu_{ar}+i(E_m-E_n)`. No Hamiltonian counterterm is added.

        TCL2 is generally not in GKSL form and is not guaranteed to preserve
        positivity. This class therefore does not normalize, symmetrize, clip,
        or project the state during evolution.
    """

    def __init__(
        self,
        hamiltonian: Any,
        coupling_operators: Any,
        correlation_coefficients: Any,
        correlation_exponents: Any,
        dt: float,
        *,
        batch_size: int = 1,
        dtype: Any = jnp.complex64,
        density_matrices: Any | None = None,
    ) -> None:
        self.dtype = complex_dtype(dtype)
        self.real_dtype = jnp.float64 if self.dtype == jnp.dtype(jnp.complex128) else jnp.float32
        self.batch_size = positive_int(batch_size, "batch_size")

        dt_value = float(dt)
        if not np.isfinite(dt_value) or dt_value <= 0:
            raise ValueError("dt must be a positive finite value")
        self.dt = jnp.asarray(dt_value, dtype=self.real_dtype)

        self.hamiltonian = _as_hamiltonian(hamiltonian, self.dtype)
        self.system_dim = self.hamiltonian.shape[0]
        self.hilbert_dim = self.system_dim
        self.coupling_operators = _as_coupling_operators(
            coupling_operators,
            self.system_dim,
            self.dtype,
        )
        self.n_channels = self.coupling_operators.shape[0]
        self.correlation_coefficients = _as_correlation_terms(
            correlation_coefficients,
            self.n_channels,
            self.dtype,
            "correlation_coefficients",
        )
        self.correlation_exponents = _as_correlation_terms(
            correlation_exponents,
            self.n_channels,
            self.dtype,
            "correlation_exponents",
        )
        if self.correlation_coefficients.shape != self.correlation_exponents.shape:
            raise ValueError(
                "correlation_coefficients and correlation_exponents must have matching shapes"
            )

        self._eigenvalues, self._eigenvectors = jnp.linalg.eigh(self.hamiltonian)
        self._energy_gaps = self._eigenvalues[:, None] - self._eigenvalues[None, :]
        self._coupling_operators_energy = (
            adjoint(self._eigenvectors) @ self.coupling_operators @ self._eigenvectors
        )
        self._density_matrices_energy: Array | None = None
        self._time = jnp.asarray(0.0, dtype=self.real_dtype)
        if density_matrices is not None:
            self.density_matrices = density_matrices

    @property
    def ns(self) -> int:
        """Alias for ``system_dim``."""

        return self.system_dim

    @property
    def nb(self) -> int:
        """Alias for ``batch_size``."""

        return self.batch_size

    @property
    def time(self) -> Array:
        """Current simulation time as a scalar JAX array."""

        return self._time

    def _get_energy_density_matrices(self) -> Array:
        if self._density_matrices_energy is None:
            raise ValueError("density matrices have not been set")
        return self._density_matrices_energy

    @property
    def density_matrices(self) -> Array:
        """Return density matrices shaped ``(batch, system_dim, system_dim)``."""

        density_matrices = self._get_energy_density_matrices()
        return self._eigenvectors @ density_matrices @ adjoint(self._eigenvectors)

    @density_matrices.setter
    def density_matrices(self, density_matrices: Any) -> None:
        matrices = jnp.asarray(density_matrices, dtype=self.dtype)
        matrix_shape = (self.system_dim, self.system_dim)
        batch_shape = (self.batch_size, self.system_dim, self.system_dim)
        if matrices.shape == matrix_shape:
            matrices = jnp.broadcast_to(matrices, batch_shape)
        elif matrices.shape != batch_shape:
            raise ValueError(
                "density_matrices must have shape (system_dim, system_dim) "
                "or (batch_size, system_dim, system_dim)"
            )
        _require_finite(matrices, "density_matrices")
        self._density_matrices_energy = adjoint(self._eigenvectors) @ matrices @ self._eigenvectors

    @property
    def rho(self) -> Array:
        """Alias for :attr:`density_matrices`."""

        return self.density_matrices

    @rho.setter
    def rho(self, density_matrices: Any) -> None:
        self.density_matrices = density_matrices

    def step(self, n_steps: int = 1) -> Array:
        """Advance the TCL2 equation by ``n_steps`` compiled RK4 steps."""

        n_steps = positive_int(n_steps, "n_steps")
        density_matrices = self._get_energy_density_matrices()
        time = self._time
        for _ in range(n_steps):
            density_matrices = _tcl2_rk4_step(
                density_matrices,
                time,
                self.dt,
                self._energy_gaps,
                self._coupling_operators_energy,
                self.correlation_coefficients,
                self.correlation_exponents,
            )
            time = time + self.dt
        self._density_matrices_energy = density_matrices
        self._time = time
        return self.density_matrices

    def observe(self, operator: Any) -> Array:
        """Return ``Tr(rho O)`` for each density matrix.

        Args:
            operator: Original-basis operator shaped
                ``(system_dim, system_dim)``.

        Returns:
            Expectation values shaped ``(batch,)``.
        """

        matrix = jnp.asarray(operator, dtype=self.dtype)
        if matrix.shape != (self.system_dim, self.system_dim):
            raise ValueError("operator must have shape (system_dim, system_dim)")
        _require_finite(matrix, "operator")
        matrix_energy = adjoint(self._eigenvectors) @ matrix @ self._eigenvectors
        return jnp.trace(
            self._get_energy_density_matrices() @ matrix_energy,
            axis1=-2,
            axis2=-1,
        )

    def diagnostics(self) -> dict[str, Array]:
        """Return invariant and positivity diagnostics without changing state.

        Returns:
            Dictionary containing arrays shaped ``(batch,)``:
            ``trace`` is complex, ``hermiticity_error`` is the Frobenius norm
            of :math:`\\rho-\\rho^\\dagger`, and ``minimum_eigenvalue`` is
            computed from the Hermitian part of the density matrix.
        """

        density_matrices = self._get_energy_density_matrices()
        density_dagger = adjoint(density_matrices)
        hermiticity_difference = density_matrices - density_dagger
        hermitian_part = 0.5 * (density_matrices + density_dagger)
        return {
            "trace": jnp.trace(density_matrices, axis1=-2, axis2=-1),
            "hermiticity_error": jnp.sqrt(
                jnp.sum(
                    jnp.abs(hermiticity_difference) ** 2,
                    axis=(-2, -1),
                )
            ),
            "minimum_eigenvalue": jnp.min(
                jnp.linalg.eigvalsh(hermitian_part),
                axis=-1,
            ),
        }


__all__ = ["TCL2Simulation"]
