from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def site_populations(state: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.abs(state) ** 2, axis=tuple(range(2, state.ndim)))

def save_metadata(
    args,
    boson_dims: np.ndarray,
    center_site: int,
    gamma_cm_inverse: np.ndarray,
    output_path: Path,
) -> Path:
    metadata = {
        "batch_size": args.batch_size,
        "boson_dims": boson_dims.tolist(),
        "center_site": center_site,
        "dtype": args.dtype,
        "dt_fs": args.dt_fs,
        "gamma_cm_inverse": gamma_cm_inverse.tolist(),
        "method": "CoupledLindbladTrajectorySimulation",
        "run_id": args.run_id,
        "sample_period_fs": args.sample_period_fs,
        "sample_time_fs": args.sample_time_fs,
        "site_populations": "normalized over sites",
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output_path
