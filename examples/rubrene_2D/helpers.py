from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from scalabath.operators_base import boson
from scalabath.simulations_unitary import SystemBathUnitarySimulation


def state_sharding_from_gpus(num_bath_modes: int) -> NamedSharding | None:
    gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    if len(gpu_devices) <= 1:
        print(f"Detected {len(gpu_devices)} GPU(s); state sharding disabled.")
        return None

    mesh = Mesh(np.asarray(gpu_devices), ("system",))
    sharding = NamedSharding(mesh, P(None, "system", *([None] * num_bath_modes)))
    print(f"Detected {len(gpu_devices)} GPUs; sharding state over the system axis.")
    return sharding


def save_metadata(args, boson_dims: np.ndarray, center_site_index: int, lambda_cm: np.ndarray, omega_cm: np.ndarray, output_path: Path) -> Path:
    metadata = {
        "batch_size": args.batch_size,
        "num_modes": args.num_modes,
        "boson_dims": boson_dims.tolist(),
        "center_site_index": center_site_index,
        "lattice_l1": args.lattice_l1,
        "lattice_l2": args.lattice_l2,
        "dtype": args.dtype,
        "dt_fs": args.dt_fs,
        "hopping_l1_mev": args.hopping_l1_mev,
        "hopping_l2_mev": args.hopping_l2_mev,
        "lambda_cm_inverse": lambda_cm.tolist(),
        "omega_cm_inverse": omega_cm.tolist(),
        "periodic": args.periodic,
        "run_id": args.run_id,
        "sample_period_fs": args.sample_period_fs,
        "sample_time_fs": args.sample_time_fs,
        "temperature_K": args.temperature,
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

def build_system_bath_hamiltonians(
    simulation: SystemBathUnitarySimulation,
    *,
    coupling: np.ndarray,
    phases: np.ndarray,
) -> list[jnp.ndarray]:
    hamiltonians = []
    batch_size = simulation.batch_size
    system_dim = simulation.system_dim
    boson_dims = simulation.boson_dims
    dtype = simulation.dtype
    np_dtype = np.complex128 if jnp.dtype(dtype) == jnp.complex128 else np.complex64

    for mode_index, mode_dim in enumerate(boson_dims):
        local_boson = simulation.boson_basis[mode_index]
        annihilation = np.asarray(local_boson.annihilation)
        creation = np.asarray(local_boson.creation)
        blocks = np.zeros(
            (batch_size, system_dim, mode_dim, mode_dim),
            dtype=np_dtype,
        )

        for batch_index in range(batch_size):
            for site in range(system_dim):
                phase = phases[batch_index, site]
                local_coupling = coupling[mode_index] * (
                    np.exp(1j * phase) * annihilation + np.exp(-1j * phase) * creation
                )
                blocks[batch_index, site] = local_coupling

        hamiltonians.append(jnp.asarray(blocks, dtype=dtype))
    return hamiltonians

@jax.jit
def site_populations_from_state(state: jnp.ndarray) -> jnp.ndarray:
    batch_size, system_dim = state.shape[:2]
    flattened_bath = state.reshape(batch_size, system_dim, -1)
    populations = jnp.sum(jnp.abs(flattened_bath) ** 2, axis=-1).real
    return populations


def normalized_site_populations(site_populations: np.ndarray) -> np.ndarray:
    populations = np.asarray(site_populations, dtype=float)
    if populations.ndim != 3:
        raise ValueError("site_populations must have shape (time, batch, n_sites)")

    populations = np.where((populations < 0.0) & (populations > -1e-10), 0.0, populations)
    if np.any(populations < -1e-10):
        raise ValueError("site_populations contains significantly negative entries")

    norms = populations.sum(axis=-1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("site_populations contains invalid normalization")
    return populations / norms
