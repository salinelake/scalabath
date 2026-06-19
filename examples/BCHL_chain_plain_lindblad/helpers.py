from __future__ import annotations

from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

def site_populations(density_matrices: jnp.ndarray, batch_size, chain_length, bath_dim) -> jnp.ndarray:
    diagonal = jnp.diagonal(density_matrices, axis1=-2, axis2=-1).real
    diagonal = diagonal.reshape(batch_size, chain_length, bath_dim)
    return diagonal.sum(axis=-1)

def save_metadata(args, boson_dims: np.ndarray, center_site: int, gamma_cm_inverse: np.ndarray, output_path: Path) -> Path:
    metadata = {
        "batch_size": args.batch_size,
        "boson_dims": boson_dims.tolist(),
        "center_site": center_site,
        "dtype": args.dtype,
        "dt_fs": args.dt_fs,
        "gamma_cm_inverse": gamma_cm_inverse.tolist(),
        "run_id": args.run_id,
        "sample_period_fs": args.sample_period_fs,
        "sample_time_fs": args.sample_time_fs,
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
