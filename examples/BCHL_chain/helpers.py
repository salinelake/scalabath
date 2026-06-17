from __future__ import annotations

from pathlib import Path
import json

import jax
import jax.numpy as jnp
import numpy as np

def save_metadata(args, boson_dims: np.ndarray, center_site: int, omega_cm: np.ndarray, output_path: Path) -> Path:
    metadata = {
        "batch_size": args.batch_size,
        "boson_dims": boson_dims.tolist(),
        "center_site": center_site,
        "chain_length": args.chain_length,
        "dtype": args.dtype,
        "dt_fs": args.dt_fs,
        "gamma_cm_inverse": gamma_cm.tolist(),
        "hopping_cm_inverse": args.hopping_cm_inverse,
        "huang_rhys": huang_rhys.tolist(),
        "num_modes": args.num_modes,
        "periodic": args.periodic,
        "run_id": args.run_id,
        "sample_period_fs": args.sample_period_fs,
        "sample_time_fs": args.sample_time_fs,
        "seed": args.seed,
        "omega_cm_inverse": omega_cm.tolist(),
    }
    output_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
