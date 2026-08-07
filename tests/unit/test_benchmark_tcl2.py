"""Tests for the cutoff-matched 2D TCL2 benchmark helpers and CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from examples.benchmark_2D.TCL2.helpers import (
    build_site_projectors,
    center_site_density_matrix,
    site_populations_from_density_matrices,
    truncated_oscillator_correlation_terms,
    truncated_thermal_factors,
)

from scalabath.operators_base import boson

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "benchmark_2D" / "TCL2" / "01.main.py"


@pytest.mark.parametrize(
    ("boson_dim", "omega", "kbT"),
    [(1, 1.2, 0.7), (2, 0.8, 1.1), (5, 1.4, 0.5)],
)
def test_truncated_thermal_factors_match_explicit_matrix_trace(
    boson_dim: int,
    omega: float,
    kbT: float,
) -> None:
    probabilities, emission_factor, occupation_factor = truncated_thermal_factors(
        omega,
        kbT,
        boson_dim,
    )
    basis = boson(boson_dim - 1, dtype=jnp.complex128)
    thermal_density = np.diag(probabilities).astype(np.complex128)
    annihilation = np.asarray(basis.annihilation)
    creation = np.asarray(basis.creation)
    expected_emission = np.trace(annihilation @ creation @ thermal_density).real
    expected_occupation = np.trace(creation @ annihilation @ thermal_density).real

    np.testing.assert_allclose(emission_factor, expected_emission, atol=1e-14)
    np.testing.assert_allclose(occupation_factor, expected_occupation, atol=1e-14)


def test_truncated_thermal_factors_converge_to_infinite_oscillator() -> None:
    omega = 0.7
    kbT = 1.0
    _, emission_factor, occupation_factor = truncated_thermal_factors(omega, kbT, 80)
    infinite_occupation = 1.0 / np.expm1(omega / kbT)

    np.testing.assert_allclose(occupation_factor, infinite_occupation, atol=1e-13)
    np.testing.assert_allclose(emission_factor, infinite_occupation + 1.0, atol=1e-13)


def test_correlation_terms_match_explicit_truncated_bath_trace() -> None:
    boson_dim = 4
    omega = 1.1
    kbT = 0.9
    coupling = 0.35
    time = 0.63
    probabilities, emission_factor, occupation_factor = truncated_thermal_factors(
        omega,
        kbT,
        boson_dim,
    )
    coefficients, exponents = truncated_oscillator_correlation_terms(
        coupling,
        omega,
        emission_factor,
        occupation_factor,
        jnp.complex128,
    )
    correlation = np.sum(np.asarray(coefficients) * np.exp(-np.asarray(exponents) * time))

    basis = boson(boson_dim - 1, dtype=jnp.complex128)
    annihilation = np.asarray(basis.annihilation)
    creation = np.asarray(basis.creation)
    bath_operator_zero = coupling * (annihilation + creation)
    bath_operator_time = coupling * (
        np.exp(-1j * omega * time) * annihilation + np.exp(1j * omega * time) * creation
    )
    thermal_density = np.diag(probabilities).astype(np.complex128)
    expected = np.trace(bath_operator_time @ bath_operator_zero @ thermal_density)

    np.testing.assert_allclose(correlation, expected, rtol=1e-13, atol=1e-13)


def test_site_projectors_initial_state_and_populations_have_documented_shapes() -> None:
    projectors = build_site_projectors(3, jnp.complex128)
    density_matrix = center_site_density_matrix(3, 1, jnp.complex128)
    populations = site_populations_from_density_matrices(density_matrix[None, ...])

    assert projectors.shape == (3, 3, 3)
    np.testing.assert_allclose(np.asarray(projectors.sum(axis=0)), np.eye(3))
    np.testing.assert_allclose(np.asarray(populations), [[0.0, 1.0, 0.0]])


def test_tcl2_benchmark_cli_writes_reproducible_smoke_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "result"
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, environment.get("PYTHONPATH", "")))
    )
    command = [
        sys.executable,
        str(SCRIPT),
        "--eta",
        "0.1",
        "--dt-fs",
        "0.1",
        "--sample-period-fs",
        "0.1",
        "--sample-time-fs",
        "0.2",
        "--dtype",
        "complex128",
        "--output-dir",
        str(output_dir),
    ]

    subprocess.run(
        command,
        cwd=SCRIPT.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    data_path = output_dir / "batch1_run0.npz"
    metadata_path = output_dir / "batch1_run0.json"
    with np.load(data_path, allow_pickle=False) as data:
        np.testing.assert_allclose(data["time_fs"], [0.0, 0.1, 0.2])
        assert data["site_populations"].shape == (3, 1, 9)
        assert data["trace"].shape == (3, 1)
        assert data["hermiticity_error"].shape == (3, 1)
        assert data["minimum_eigenvalue"].shape == (3, 1)
        np.testing.assert_allclose(data["site_populations"][0, 0, 4], 1.0, atol=1e-13)
        np.testing.assert_allclose(data["trace"], 1.0, atol=1e-12)
        np.testing.assert_allclose(data["hermiticity_error"], 0.0, atol=1e-12)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["method"] == "TCL2"
    assert metadata["bath_correlation"] == "truncated_oscillator"
    assert metadata["integrator"] == "RK4"
    assert metadata["lattice"] == [3, 3]
    assert metadata["dt_fs"] == 0.1
    assert metadata["sample_time_fs"] == 0.2
