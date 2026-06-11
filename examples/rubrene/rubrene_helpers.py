from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from scalabath.operators_base import boson
from scalabath.systems import TensorProductPureStatesEnsemble

DEFAULT_BOSON_DIMS = np.asarray([9, 4, 2, 2, 2, 2, 2, 2, 2], dtype=int)
DTYPES = {
    "complex64": jnp.complex64,
    "complex128": jnp.complex128,
}


def parse_boson_dims(value: str | None, num_modes: int) -> np.ndarray:
    if value is None:
        if num_modes > len(DEFAULT_BOSON_DIMS):
            raise ValueError(f"num_modes must be at most {len(DEFAULT_BOSON_DIMS)} by default")
        return DEFAULT_BOSON_DIMS[:num_modes].copy()

    dims = np.asarray([int(part.strip()) for part in value.split(",") if part.strip()], dtype=int)
    if dims.shape != (num_modes,):
        raise ValueError("boson_dims must contain exactly num_modes entries")
    if np.any(dims <= 0):
        raise ValueError("all boson dimensions must be positive")
    return dims


def numpy_complex_dtype(dtype: jnp.dtype) -> type[np.complex64] | type[np.complex128]:
    return np.complex128 if jnp.dtype(dtype) == jnp.complex128 else np.complex64


def load_coupling_data(path: Path, num_modes: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path.expanduser(), delimiter=",", comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("coupling CSV must have at least two columns")
    if num_modes > data.shape[0]:
        raise ValueError(f"requested {num_modes} modes, but {path} only has {data.shape[0]}")
    return data[:num_modes, 0], data[:num_modes, 1]


def tight_binding_hamiltonian(
    chain_length: int,
    hopping: float,
    *,
    periodic: bool,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    if chain_length < 1:
        raise ValueError("chain_length must be positive")

    matrix = np.zeros((chain_length, chain_length), dtype=numpy_complex_dtype(dtype))
    for site in range(chain_length - 1):
        matrix[site, site + 1] = hopping
        matrix[site + 1, site] = hopping
    if periodic and chain_length > 2:
        matrix[0, chain_length - 1] = hopping
        matrix[chain_length - 1, 0] = hopping
    return jnp.asarray(matrix, dtype=dtype)


def sample_initial_ensemble(
    *,
    batch_size: int,
    chain_length: int,
    boson_dims: np.ndarray,
    omega: np.ndarray,
    kbT: float,
    seed: int,
    dtype: jnp.dtype,
) -> tuple[TensorProductPureStatesEnsemble, np.ndarray]:
    rng = np.random.default_rng(seed)
    states_shape = (batch_size, chain_length, *boson_dims.tolist())
    states = np.zeros(states_shape, dtype=numpy_complex_dtype(dtype))
    chosen_levels = np.zeros((batch_size, boson_dims.size), dtype=int)
    center_site = (chain_length - 1) // 2

    for batch_index in range(batch_size):
        levels = []
        for mode_index, mode_dim in enumerate(boson_dims):
            occupations = np.arange(mode_dim)
            weights = np.exp(-(occupations * omega[mode_index]) / kbT)
            probabilities = weights / weights.sum()
            level = int(rng.choice(mode_dim, p=probabilities))
            levels.append(level)
        chosen_levels[batch_index] = levels
        states[(batch_index, center_site, *levels)] = 1.0

    subsystem_dims = (chain_length, *boson_dims.tolist())
    ensemble = TensorProductPureStatesEnsemble(subsystem_dims, batch_size=batch_size, dtype=dtype)
    ensemble.set_pse(jnp.asarray(states, dtype=dtype))
    ensemble.normalize()
    return ensemble, chosen_levels


def build_bath_hamiltonians(
    *,
    boson_dims: np.ndarray,
    omega: np.ndarray,
    dtype: jnp.dtype,
) -> list[jnp.ndarray]:
    hamiltonians = []
    for mode_dim, mode_omega in zip(boson_dims, omega, strict=True):
        local_boson = boson(int(mode_dim) - 1, dtype=dtype)
        hamiltonians.append(jnp.asarray(mode_omega, dtype=dtype) * local_boson.number)
    return hamiltonians


def build_system_bath_hamiltonians(
    *,
    batch_size: int,
    chain_length: int,
    boson_dims: np.ndarray,
    coupling: np.ndarray,
    phases: np.ndarray,
    dtype: jnp.dtype,
) -> list[jnp.ndarray]:
    hamiltonians = []
    np_dtype = numpy_complex_dtype(dtype)

    for mode_index, mode_dim in enumerate(boson_dims):
        local_boson = boson(int(mode_dim) - 1, dtype=dtype)
        annihilation = np.asarray(local_boson.annihilation)
        creation = np.asarray(local_boson.creation)
        matrix = np.zeros(
            (batch_size, chain_length * mode_dim, chain_length * mode_dim),
            dtype=np_dtype,
        )

        for batch_index in range(batch_size):
            for site in range(chain_length):
                phase = phases[batch_index, mode_index, site]
                local_coupling = coupling[mode_index] * (
                    np.exp(1j * phase) * annihilation + np.exp(-1j * phase) * creation
                )
                start = site * mode_dim
                stop = start + mode_dim
                matrix[batch_index, start:stop, start:stop] = local_coupling

        hamiltonians.append(jnp.asarray(matrix, dtype=dtype))
    return hamiltonians


def site_populations_from_state(state: jnp.ndarray) -> np.ndarray:
    batch_size, chain_length = state.shape[:2]
    flattened_bath = state.reshape(batch_size, chain_length, -1)
    return np.asarray(jnp.sum(jnp.abs(flattened_bath) ** 2, axis=-1).real)
