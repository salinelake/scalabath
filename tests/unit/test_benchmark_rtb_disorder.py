"""Tests for the renormalized tight-binding disorder benchmark."""

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
from examples.benchmark_2D.RTB_disorder.helpers import (
    build_effective_hamiltonian,
    center_site_state,
    lang_firsov_narrowing_factor,
    sample_site_disorder,
    state_norm_from_state,
    thermal_occupation,
)

from scalabath.simulations_unitary import UnitarySimulation

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "examples" / "benchmark_2D" / "RTB_disorder"
DRIVER = SCRIPT_DIR / "01.main.py"
POSTPROCESS = SCRIPT_DIR / "02.postprocess.py"


def test_narrowing_factor_matches_bose_and_coth_forms() -> None:
    omega = 1.4
    kbT = 0.7
    g_factor = 0.35
    occupation = thermal_occupation(omega, kbT)
    factor = lang_firsov_narrowing_factor(omega, kbT, g_factor)

    expected_from_occupation = np.exp(-(g_factor**2) * (2.0 * occupation + 1.0))
    expected_from_coth = np.exp(-(g_factor**2) / np.tanh(omega / (2.0 * kbT)))
    np.testing.assert_allclose(factor, expected_from_occupation, rtol=1e-14)
    np.testing.assert_allclose(factor, expected_from_coth, rtol=1e-14)


def test_disorder_sampler_matches_spa_first_draw() -> None:
    lattice = (4, 5)
    width = 200.0
    seed = 42
    random_state = np.random.RandomState(seed)  # noqa: NPY002
    expected_first = random_state.uniform(-width, width, size=lattice)

    disorder = sample_site_disorder(lattice, batch_size=3, disorder_width=width, seed=seed)

    assert disorder.shape == (3, *lattice)
    np.testing.assert_array_equal(disorder[0], expected_first)
    assert not np.array_equal(disorder[0], disorder[1])


def test_zero_disorder_sampler_returns_zero_realizations() -> None:
    disorder = sample_site_disorder((3, 4), batch_size=2, disorder_width=0.0, seed=7)

    np.testing.assert_array_equal(disorder, np.zeros((2, 3, 4)))


def test_effective_hamiltonian_renormalizes_hopping_and_adds_disorder() -> None:
    hopping = np.array(
        [
            [0.0, 1.0 + 0.5j, 0.0],
            [1.0 - 0.5j, 0.0, -0.3],
            [0.0, -0.3, 0.0],
        ],
        dtype=np.complex128,
    )
    disorder = np.array([[0.1, -0.2, 0.4], [-0.5, 0.7, 0.2]])
    narrowing_factor = 0.42
    polaron_shift = 1.7

    effective = build_effective_hamiltonian(
        hopping,
        disorder,
        narrowing_factor,
        polaron_shift,
        jnp.complex128,
    )

    expected = np.stack(
        [
            narrowing_factor * hopping + np.diag(realization) - polaron_shift * np.eye(3)
            for realization in disorder
        ]
    )
    assert effective.shape == (2, 3, 3)
    np.testing.assert_allclose(np.asarray(effective), expected, atol=1e-14)


def test_effective_hamiltonian_uses_existing_unitary_simulation() -> None:
    hopping = np.array(
        [[0.0, 0.8, 0.0], [0.8, 0.0, 0.5], [0.0, 0.5, 0.0]],
        dtype=np.complex128,
    )
    disorder = np.array([[0.0, 0.3, -0.2], [0.4, -0.1, 0.2]])
    hamiltonian = build_effective_hamiltonian(
        hopping,
        disorder,
        narrowing_factor=0.6,
        polaron_shift=0.7,
        dtype=jnp.complex128,
    )
    simulation = UnitarySimulation(
        hilbert_dim=3,
        batch_size=2,
        dt=0.03,
        hamiltonian=hamiltonian,
        dtype=jnp.complex128,
    )
    initial_state = center_site_state(3, 1, jnp.complex128)
    simulation.state = jnp.broadcast_to(initial_state, (2, 3))

    final_state = simulation.step(n_steps=100)

    np.testing.assert_allclose(
        np.asarray(state_norm_from_state(final_state)),
        1.0,
        atol=2e-13,
    )
    assert not np.allclose(np.asarray(final_state[0]), np.asarray(final_state[1]))


def test_band_narrowing_rescales_population_evolution_without_disorder() -> None:
    hopping = np.array(
        [[0.0, 0.8, 0.0], [0.8, 0.0, 0.6], [0.0, 0.6, 0.0]],
        dtype=np.complex128,
    )
    narrowing_factor = 0.43
    dt = 0.07
    effective_hamiltonian = build_effective_hamiltonian(
        hopping,
        np.zeros((1, 3)),
        narrowing_factor,
        polaron_shift=1.2,
        dtype=jnp.complex128,
    )
    effective_simulation = UnitarySimulation(
        hilbert_dim=3,
        dt=dt,
        hamiltonian=effective_hamiltonian,
        dtype=jnp.complex128,
    )
    bare_simulation = UnitarySimulation(
        hilbert_dim=3,
        dt=narrowing_factor * dt,
        hamiltonian=hopping,
        dtype=jnp.complex128,
    )
    initial_state = center_site_state(3, 1, jnp.complex128)[None, :]
    effective_simulation.state = initial_state
    bare_simulation.state = initial_state

    effective_state = effective_simulation.step(n_steps=20)
    bare_state = bare_simulation.step(n_steps=20)

    np.testing.assert_allclose(
        np.abs(np.asarray(effective_state)) ** 2,
        np.abs(np.asarray(bare_state)) ** 2,
        atol=1e-13,
    )


def test_postprocess_msd_matches_spatial_variance_convention() -> None:
    postprocess = runpy.run_path(str(POSTPROCESS))
    calculate_msd = postprocess["calculate_msd"]
    populations = np.zeros((1, 1, 9))
    populations[0, 0, 4] = 0.5
    populations[0, 0, 5] = 0.5

    msd = calculate_msd(populations, (3, 3), (1, 1))

    np.testing.assert_allclose(msd, [[0.25]])


def test_rtb_disorder_cli_and_postprocess_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "result"
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (source_path, environment.get("PYTHONPATH", "")))
    )
    driver_command = [
        sys.executable,
        str(DRIVER),
        "--lattice-lx",
        "4",
        "--lattice-ly",
        "5",
        "--eta",
        "0.5",
        "--disorder-strength",
        "50",
        "--dt-fs",
        "0.1",
        "--sample-period-fs",
        "0.1",
        "--sample-time-fs",
        "0.2",
        "--batch-size",
        "2",
        "--seed",
        "17",
        "--dtype",
        "complex128",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(
        driver_command,
        cwd=SCRIPT_DIR,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    data_path = output_dir / "batch2_run0.npz"
    metadata_path = output_dir / "batch2_run0.json"
    with np.load(data_path, allow_pickle=False) as data:
        np.testing.assert_allclose(data["time_fs"], [0.0, 0.1, 0.2])
        assert data["site_populations"].shape == (3, 2, 20)
        assert data["state_norm"].shape == (3, 2)
        assert data["site_disorder_cm_inverse"].shape == (2, 4, 5)
        np.testing.assert_allclose(data["site_populations"][0, :, 7], 1.0)
        np.testing.assert_allclose(data["site_populations"].sum(axis=-1), 1.0, atol=1e-13)
        np.testing.assert_allclose(data["state_norm"], 1.0, atol=1e-13)
        expected_disorder = np.random.RandomState(17).uniform(  # noqa: NPY002
            -50.0,
            50.0,
            size=(4, 5),
        )
        np.testing.assert_array_equal(data["site_disorder_cm_inverse"][0], expected_disorder)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["method"] == "RTB-disorder"
    assert metadata["batch_size_semantics"] == "independent static-disorder realizations"
    assert metadata["disorder_strength_cm_inverse"] == 50.0
    assert metadata["disorder_seed"] == 17
    assert metadata["integrator"] == "dense matrix exponential"
    assert metadata["simulation_class"] == ("scalabath.simulations_unitary.UnitarySimulation")
    assert metadata["polaron_shift_treatment"] == (
        "included uniform identity term; global phase only"
    )

    output_prefix = tmp_path / "analysis"
    postprocess_command = [
        sys.executable,
        str(POSTPROCESS),
        "--lattice-lx",
        "4",
        "--lattice-ly",
        "5",
        "--boson-dim",
        "6",
        "--eta",
        "0.5",
        "--sample-time-fs",
        "0.2",
        "--disorder-strength",
        "50",
        "--batch-size",
        "2",
        "--data-folder",
        str(output_dir),
        "--output-prefix",
        str(output_prefix),
    ]
    subprocess.run(
        postprocess_command,
        cwd=SCRIPT_DIR,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert Path(f"{output_prefix}_msd.png").is_file()
    assert Path(f"{output_prefix}_snapshots.png").is_file()
    summary = json.loads(Path(f"{output_prefix}_summary.json").read_text(encoding="utf-8"))
    assert summary["run_count"] == 1
    assert summary["trajectory_count"] == 2
    assert summary["maximum_state_norm_error"] < 1e-12
