from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scalabath.simulations_unitary import SystemBathUnitarySimulation
 

def build_upper_level_system_bath_hamiltonians(
    simulation: SystemBathUnitarySimulation,
    *,
    coupling: np.ndarray,
    upper_level: int,
) -> list[jnp.ndarray]:
    """Build mode-local coupling blocks for ``|upper><upper| (b + b^\dagger)``."""

    if upper_level < 0 or upper_level >= simulation.system_dim:
        raise ValueError("upper_level must be a valid system level index")

    hamiltonians = []
    system_dim = simulation.system_dim
    boson_dims = simulation.boson_dims
    dtype = simulation.dtype
    np_dtype = np.complex128 if jnp.dtype(dtype) == jnp.complex128 else np.complex64

    for mode_index, mode_dim in enumerate(boson_dims):
        local_boson = simulation.boson_basis[mode_index]
        annihilation = np.asarray(local_boson.annihilation)
        creation = np.asarray(local_boson.creation)
        matrix = np.zeros((system_dim, mode_dim, mode_dim), dtype=np_dtype)
        matrix[upper_level] = coupling[mode_index] * (annihilation + creation)

        hamiltonians.append(jnp.asarray(matrix, dtype=dtype))
    return hamiltonians
