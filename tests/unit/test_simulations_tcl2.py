"""Tests for finite-time second-order time-convolutionless dynamics."""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import pytest

from scalabath import TCL2Simulation
from scalabath.simulations_tcl2 import (
    _integrated_correlation_operators,
    _integrated_exponential,
    _tcl2_rhs,
    _tcl2_rk4_step,
)

pytestmark = pytest.mark.unit


def _trapezoid(values: np.ndarray, spacing: float) -> np.ndarray:
    """Integrate an array over its first axis with the trapezoid rule."""

    return spacing * (0.5 * values[0] + values[1:-1].sum(axis=0) + 0.5 * values[-1])


def test_integrated_exponential_handles_zero_and_near_zero_rates() -> None:
    rates = jnp.asarray([0.0, 1e-14, 0.2 + 0.7j], dtype=jnp.complex128)
    time = jnp.asarray(0.4, dtype=jnp.float64)

    integrals = np.asarray(_integrated_exponential(rates, time))

    expected = np.asarray(
        [
            0.4,
            0.4 - 0.5 * 1e-14 * 0.4**2,
            -np.expm1(-(0.2 + 0.7j) * 0.4) / (0.2 + 0.7j),
        ],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(integrals, expected, rtol=1e-13, atol=1e-13)


def test_integrated_correlation_operator_matches_numpy_quadrature() -> None:
    energies = np.asarray([-0.3, 0.4])
    energy_gaps = energies[:, None] - energies[None, :]
    coupling_operator = np.asarray(
        [[0.2, 0.3 + 0.1j], [0.3 - 0.1j, -0.1]],
        dtype=np.complex128,
    )
    coefficients = np.asarray([[0.7, 0.2]], dtype=np.complex128)
    exponents = np.asarray([[0.15 + 0.8j, 0.05 - 0.4j]], dtype=np.complex128)
    time = 0.73

    analytic = np.asarray(
        _integrated_correlation_operators(
            jnp.asarray(time),
            jnp.asarray(energy_gaps),
            jnp.asarray(coupling_operator[None, ...]),
            jnp.asarray(coefficients),
            jnp.asarray(exponents),
        )
    )[0]

    times = np.linspace(0.0, time, 20_001)
    correlation = np.sum(
        coefficients[0, :, None] * np.exp(-exponents[0, :, None] * times[None, :]),
        axis=0,
    )
    phase = np.exp(-1j * energy_gaps[:, :, None] * times[None, None, :])
    integrand = coupling_operator[:, :, None] * phase * correlation[None, None, :]
    expected = _trapezoid(np.moveaxis(integrand, -1, 0), times[1] - times[0])

    np.testing.assert_allclose(analytic, expected, rtol=2e-9, atol=2e-9)


def test_tcl2_rhs_matches_direct_double_commutator_quadrature() -> None:
    energies = np.asarray([-0.35, 0.45])
    energy_gaps = energies[:, None] - energies[None, :]
    system_operator = np.asarray([[0.8, 0.25], [0.25, 0.2]], dtype=np.complex128)
    density_matrix = np.asarray(
        [[0.6, 0.15 + 0.1j], [0.15 - 0.1j, 0.4]],
        dtype=np.complex128,
    )
    coefficients = np.asarray([[0.7, 0.3]], dtype=np.complex128)
    omega = 1.2
    exponents = np.asarray([[1j * omega, -1j * omega]], dtype=np.complex128)
    time = 0.37

    derivative = np.asarray(
        _tcl2_rhs(
            jnp.asarray(density_matrix[None, ...]),
            jnp.asarray(time),
            jnp.asarray(energy_gaps),
            jnp.asarray(system_operator[None, ...]),
            jnp.asarray(coefficients),
            jnp.asarray(exponents),
        )
    )[0]

    times = np.linspace(0.0, time, 30_001)
    phase = np.exp(-1j * times[:, None, None] * energy_gaps[None, :, :])
    delayed_operators = phase * system_operator[None, :, :]
    correlation = np.sum(
        coefficients[0, :, None] * np.exp(-exponents[0, :, None] * times[None, :]),
        axis=0,
    )
    delayed_rho = delayed_operators @ density_matrix
    rho_delayed = density_matrix @ delayed_operators
    first = correlation[:, None, None] * (
        system_operator @ delayed_rho - delayed_rho @ system_operator
    )
    second = np.conjugate(correlation)[:, None, None] * (
        rho_delayed @ system_operator - system_operator @ rho_delayed
    )
    dissipative_integral = _trapezoid(first + second, times[1] - times[0])
    expected = -1j * energy_gaps * density_matrix - dissipative_integral

    np.testing.assert_allclose(derivative, expected, rtol=2e-9, atol=2e-9)


def test_tcl2_rhs_preserves_trace_and_hermiticity() -> None:
    energies = jnp.asarray([-0.4, 0.1, 0.7], dtype=jnp.float64)
    energy_gaps = energies[:, None] - energies[None, :]
    coupling_operators = jnp.asarray(
        [
            [[1.0, 0.2j, 0.0], [-0.2j, 0.0, 0.1], [0.0, 0.1, 0.3]],
            [[0.0, 0.1, 0.0], [0.1, 0.5, -0.2j], [0.0, 0.2j, 0.5]],
        ],
        dtype=jnp.complex128,
    )
    density_matrix = jnp.asarray(
        [[0.5, 0.1j, 0.0], [-0.1j, 0.3, 0.05], [0.0, 0.05, 0.2]],
        dtype=jnp.complex128,
    )
    coefficients = jnp.asarray([0.6, 0.2], dtype=jnp.complex128)
    exponents = jnp.asarray([0.1 + 0.9j, 0.1 - 0.9j], dtype=jnp.complex128)

    derivative = _tcl2_rhs(
        density_matrix[None, ...],
        jnp.asarray(0.4),
        energy_gaps,
        coupling_operators,
        jnp.broadcast_to(coefficients, (2, 2)),
        jnp.broadcast_to(exponents, (2, 2)),
    )[0]

    np.testing.assert_allclose(np.asarray(jnp.trace(derivative)), 0.0, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(derivative),
        np.asarray(jnp.conjugate(derivative.T)),
        atol=1e-13,
    )


def test_tcl2_rhs_is_covariant_inside_degenerate_eigenspace() -> None:
    energies = jnp.asarray([0.0, 0.0, 0.8], dtype=jnp.float64)
    energy_gaps = energies[:, None] - energies[None, :]
    system_operator = jnp.asarray(
        [[0.8, 0.2 + 0.1j, 0.1], [0.2 - 0.1j, 0.1, -0.2j], [0.1, 0.2j, 0.3]],
        dtype=jnp.complex128,
    )
    density_matrix = jnp.asarray(
        [[0.5, 0.05j, 0.1], [-0.05j, 0.2, 0.0], [0.1, 0.0, 0.3]],
        dtype=jnp.complex128,
    )
    theta = 0.43
    rotation = jnp.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.complex128,
    )
    coefficients = jnp.asarray([[0.6, 0.15]], dtype=jnp.complex128)
    exponents = jnp.asarray([[0.05 + 1.1j, 0.05 - 1.1j]], dtype=jnp.complex128)

    derivative = _tcl2_rhs(
        density_matrix[None, ...],
        jnp.asarray(0.6),
        energy_gaps,
        system_operator[None, ...],
        coefficients,
        exponents,
    )[0]
    rotated_operator = jnp.conjugate(rotation.T) @ system_operator @ rotation
    rotated_density = jnp.conjugate(rotation.T) @ density_matrix @ rotation
    rotated_derivative = _tcl2_rhs(
        rotated_density[None, ...],
        jnp.asarray(0.6),
        energy_gaps,
        rotated_operator[None, ...],
        coefficients,
        exponents,
    )[0]
    expected = jnp.conjugate(rotation.T) @ derivative @ rotation

    np.testing.assert_allclose(
        np.asarray(rotated_derivative),
        np.asarray(expected),
        rtol=1e-12,
        atol=1e-12,
    )


def test_tcl2_simulation_broadcasts_state_and_observes_batch() -> None:
    hamiltonian = jnp.asarray([[0.0, 0.3], [0.3, 0.0]], dtype=jnp.complex128)
    projector = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
    density_matrices = jnp.asarray(
        [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]],
        dtype=jnp.complex128,
    )
    simulation = TCL2Simulation(
        hamiltonian,
        projector,
        [0.0],
        [0.0],
        0.01,
        batch_size=2,
        dtype=jnp.complex128,
        density_matrices=density_matrices,
    )

    np.testing.assert_allclose(np.asarray(simulation.density_matrices), density_matrices)
    np.testing.assert_allclose(np.asarray(simulation.observe(projector)), [1.0, 0.0])


def test_zero_coupling_rk4_matches_exact_unitary_density_matrix() -> None:
    hamiltonian = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    projector = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
    simulation = TCL2Simulation(
        hamiltonian,
        projector,
        [0.0],
        [0.0],
        0.002,
        dtype=jnp.complex128,
        density_matrices=projector,
    )

    density_matrix = simulation.step(n_steps=50)[0]

    propagator = jsp.linalg.expm(-0.1j * hamiltonian)
    expected = propagator @ projector @ jnp.conjugate(propagator.T)
    np.testing.assert_allclose(
        np.asarray(density_matrix),
        np.asarray(expected),
        rtol=2e-12,
        atol=2e-12,
    )


def test_multistep_call_matches_repeated_steps_and_advances_time() -> None:
    hamiltonian = jnp.asarray([[0.1, 0.2], [0.2, -0.1]], dtype=jnp.complex128)
    projector = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
    inputs = (hamiltonian, projector, [0.4, 0.1], [0.2 + 0.7j, 0.2 - 0.7j], 0.01)
    multi = TCL2Simulation(*inputs, dtype=jnp.complex128, density_matrices=projector)
    repeated = TCL2Simulation(*inputs, dtype=jnp.complex128, density_matrices=projector)

    multi.step(n_steps=3)
    for _ in range(3):
        repeated.step()

    np.testing.assert_allclose(
        np.asarray(multi.density_matrices),
        np.asarray(repeated.density_matrices),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(np.asarray(multi.time), 0.03, atol=1e-15)


def test_rk4_step_exhibits_fourth_order_refinement() -> None:
    energies = jnp.asarray([-0.4, 0.7], dtype=jnp.float64)
    energy_gaps = energies[:, None] - energies[None, :]
    operator = jnp.asarray([[0.8, 0.25], [0.25, 0.2]], dtype=jnp.complex128)[None]
    coefficients = jnp.asarray([[0.7, 0.2]], dtype=jnp.complex128)
    exponents = jnp.asarray([[0.15 + 0.9j, 0.15 - 0.9j]], dtype=jnp.complex128)
    initial = jnp.asarray(
        [[[0.5, 0.5], [0.5, 0.5]]],
        dtype=jnp.complex128,
    )

    def evolve(dt: float, n_steps: int) -> np.ndarray:
        density_matrix = initial
        time = jnp.asarray(0.0, dtype=jnp.float64)
        step_size = jnp.asarray(dt, dtype=jnp.float64)
        for _ in range(n_steps):
            density_matrix = _tcl2_rk4_step(
                density_matrix,
                time,
                step_size,
                energy_gaps,
                operator,
                coefficients,
                exponents,
            )
            time = time + step_size
        return np.asarray(density_matrix)

    coarse = evolve(0.1, 10)
    fine = evolve(0.05, 20)
    reference = evolve(0.00625, 160)
    coarse_error = np.linalg.norm(coarse - reference)
    fine_error = np.linalg.norm(fine - reference)

    assert coarse_error > 10.0 * fine_error


def test_diagnostics_report_trace_hermiticity_and_minimum_eigenvalue() -> None:
    hamiltonian = jnp.diag(jnp.asarray([0.0, 0.4], dtype=jnp.complex128))
    projector = jnp.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=jnp.complex128)
    simulation = TCL2Simulation(
        hamiltonian,
        projector,
        [0.0],
        [0.0],
        0.01,
        dtype=jnp.complex128,
        density_matrices=projector,
    )

    diagnostics = simulation.diagnostics()

    np.testing.assert_allclose(np.asarray(diagnostics["trace"]), [1.0])
    np.testing.assert_allclose(np.asarray(diagnostics["hermiticity_error"]), [0.0])
    np.testing.assert_allclose(np.asarray(diagnostics["minimum_eigenvalue"]), [0.0])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0.0}, "dt must be"),
        ({"dtype": jnp.float64}, "complex dtype"),
        (
            {"hamiltonian": jnp.asarray([[0.0, 1.0], [0.0, 0.0]])},
            "hamiltonian must be Hermitian",
        ),
        ({"correlation_exponents": [0.0, 1.0]}, "matching shapes"),
    ],
)
def test_tcl2_simulation_validates_inputs(kwargs: dict[str, object], message: str) -> None:
    inputs: dict[str, object] = {
        "hamiltonian": jnp.eye(2, dtype=jnp.complex128),
        "coupling_operators": jnp.eye(2, dtype=jnp.complex128),
        "correlation_coefficients": [0.1],
        "correlation_exponents": [0.0],
        "dt": 0.01,
        "dtype": jnp.complex128,
    }
    inputs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        TCL2Simulation(**inputs)


def test_tcl2_simulation_validates_density_matrix_and_observable_shapes() -> None:
    simulation = TCL2Simulation(
        jnp.eye(2, dtype=jnp.complex128),
        jnp.eye(2, dtype=jnp.complex128),
        [0.1],
        [0.0],
        0.01,
        dtype=jnp.complex128,
    )

    with pytest.raises(ValueError, match="density_matrices must have shape"):
        simulation.density_matrices = jnp.ones((3, 3), dtype=jnp.complex128)
    simulation.density_matrices = jnp.eye(2, dtype=jnp.complex128)
    with pytest.raises(ValueError, match="operator must have shape"):
        simulation.observe(jnp.eye(3, dtype=jnp.complex128))
