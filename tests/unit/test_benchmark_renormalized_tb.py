"""Tests for the Lang--Firsov-renormalized tight-binding benchmark."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from examples.benchmark_2D.compare_center_msd import load_data
from examples.benchmark_2D.RTB.helpers import (
    build_effective_hamiltonian,
    center_site_state,
    finite_cutoff_narrowing_factor,
    lang_firsov_narrowing_factor,
    site_populations_from_state,
    state_norm_from_state,
    thermal_occupation,
)

from scalabath.constants import Constants

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "benchmark_2D" / "RTB" / "01.main.py"
POSTPROCESS = SCRIPT.with_name("02.postprocess.py")


@pytest.mark.parametrize(
    ("omega", "kbT", "g_factor"),
    [(1.0, 0.2, 0.0), (1.4, 0.7, 0.35), (0.8, 1.3, 1.1)],
)
def test_narrowing_factor_matches_bose_and_coth_forms(
    omega: float,
    kbT: float,
    g_factor: float,
) -> None:
    occupation = thermal_occupation(omega, kbT)
    factor = lang_firsov_narrowing_factor(omega, kbT, g_factor)
    expected_from_occupation = np.exp(-(g_factor**2) * (2.0 * occupation + 1.0))
    expected_from_coth = np.exp(-(g_factor**2) / np.tanh(omega / (2.0 * kbT)))

    np.testing.assert_allclose(factor, expected_from_occupation, rtol=1e-14)
    np.testing.assert_allclose(factor, expected_from_coth, rtol=1e-14)


def test_narrowing_factor_zero_coupling_and_zero_temperature_limits() -> None:
    assert lang_firsov_narrowing_factor(1.0, 0.7, 0.0) == 1.0
    factor = lang_firsov_narrowing_factor(1.0, 1e-6, 0.8)
    np.testing.assert_allclose(factor, np.exp(-(0.8**2)), rtol=1e-14)


@pytest.mark.parametrize("eta", [0.5, 1.0, 2.0, 4.0, 8.0])
def test_dimension_six_cutoff_factor_matches_canonical_benchmark_factor(
    eta: float,
) -> None:
    omega = 1000.0 * Constants.cm_inverse_energy
    kbT = Constants.kb * 300.0
    g_factor = np.sqrt(eta * 100.0 / 1000.0)
    canonical = lang_firsov_narrowing_factor(omega, kbT, g_factor)
    cutoff = finite_cutoff_narrowing_factor(omega, kbT, g_factor, boson_dim=6)

    np.testing.assert_allclose(cutoff, canonical, rtol=1e-6, atol=1e-12)


def test_effective_hamiltonian_renormalizes_hopping_and_adds_uniform_shift() -> None:
    hopping = np.array(
        [[0.0, 1.0 + 0.5j], [1.0 - 0.5j, 0.0]],
        dtype=np.complex128,
    )
    narrowing_factor = 0.42
    polaron_shift = 1.7
    effective = build_effective_hamiltonian(
        hopping,
        narrowing_factor,
        polaron_shift,
        jnp.complex128,
    )

    expected = narrowing_factor * hopping - polaron_shift * np.eye(2)
    np.testing.assert_allclose(np.asarray(effective), expected, atol=1e-14)


def _evolved_populations(
    hamiltonian: np.ndarray,
    initial_state: np.ndarray,
    time: float,
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    coefficients = eigenvectors.conj().T @ initial_state
    state = eigenvectors @ (np.exp(-1j * eigenvalues * time) * coefficients)
    return np.abs(state) ** 2


def test_lf_populations_equal_bare_populations_at_rescaled_time() -> None:
    bare_hamiltonian = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.6], [0.0, 0.6, 0.0]],
        dtype=np.complex128,
    )
    narrowing_factor = 0.37
    polaron_shift = 1.2
    time = 2.3
    initial_state = np.array([0.0, 1.0, 0.0], dtype=np.complex128)
    effective_hamiltonian = np.asarray(
        build_effective_hamiltonian(
            bare_hamiltonian,
            narrowing_factor,
            polaron_shift,
            jnp.complex128,
        )
    )

    lf_populations = _evolved_populations(effective_hamiltonian, initial_state, time)
    bare_populations = _evolved_populations(
        bare_hamiltonian,
        initial_state,
        narrowing_factor * time,
    )
    np.testing.assert_allclose(lf_populations, bare_populations, atol=1e-13)


def test_center_state_population_and_norm_helpers_have_documented_shapes() -> None:
    state = center_site_state(5, 2, jnp.complex128)[None, :]
    populations = site_populations_from_state(state)
    norms = state_norm_from_state(state)

    assert state.shape == (1, 5)
    assert populations.shape == (1, 5)
    assert norms.shape == (1,)
    np.testing.assert_allclose(np.asarray(populations), [[0.0, 0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(norms), [1.0])


def test_postprocess_selects_center_neighbor_and_corner() -> None:
    postprocess = runpy.run_path(str(POSTPROCESS))
    selected_sites = postprocess["_selected_sites"]((3, 3), (1, 1))

    assert selected_sites == ((1, 1), (1, 2), (2, 2))


def test_shared_comparison_loads_raw_deterministic_populations(tmp_path: Path) -> None:
    time_fs = np.array([0.0, 1.0])
    populations = np.array([[[0.8, 0.1]], [[0.4, 0.2]]])
    np.savez_compressed(
        tmp_path / "batch1_run0.npz",
        time_fs=time_fs,
        site_populations=populations,
    )

    loaded_time, population_mean, population_sem = load_data(
        tmp_path,
        batch_size=1,
        normalize=False,
    )

    np.testing.assert_array_equal(loaded_time, time_fs)
    np.testing.assert_array_equal(population_mean, populations[:, 0, :])
    np.testing.assert_array_equal(population_sem, np.zeros_like(population_mean))


def test_lf_benchmark_cli_writes_reproducible_smoke_output(tmp_path: Path) -> None:
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
        assert data["state_norm"].shape == (3, 1)
        np.testing.assert_allclose(
            data["site_populations"][0, 0],
            np.eye(9)[4],
            atol=1e-13,
        )
        np.testing.assert_allclose(data["state_norm"], 1.0, atol=1e-12)
        np.testing.assert_allclose(
            data["site_populations"].sum(axis=-1),
            1.0,
            atol=1e-12,
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["method"] == "LF-TB"
    assert metadata["approximation"] == "coherent thermal Lang-Firsov band narrowing"
    assert metadata["integrator"] == "dense matrix exponential"
    assert metadata["lattice"] == [3, 3]
    assert metadata["dt_fs"] == 0.1
    assert metadata["eta"] == 0.1
    assert metadata["sample_time_fs"] == 0.2
    np.testing.assert_allclose(
        metadata["effective_hopping_x_cm"],
        metadata["narrowing_factor"] * metadata["hopping_x_cm"],
        rtol=1e-14,
    )
