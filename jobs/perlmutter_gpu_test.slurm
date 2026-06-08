#!/bin/bash
#SBATCH --account=m1027
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -J scalabath-gpu-test
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

RUN_DIR="$SCRATCH/scalabath-runs/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python - <<'PY' | tee "$RUN_DIR/jax_devices.txt"
import jax

print(jax.devices())
PY

python -m pytest tests/gpu -m "gpu" -ra --tb=short \
  --junitxml="$RUN_DIR/pytest-gpu.xml" \
  2>&1 | tee "$RUN_DIR/pytest-gpu.log"

python - <<PY
from pathlib import Path
import json

run_dir = Path("$RUN_DIR")
summary = {
    "job_id": "$SLURM_JOB_ID",
    "submit_dir": "$SLURM_SUBMIT_DIR",
    "run_dir": str(run_dir),
}
(run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
PY

echo "Results written to $RUN_DIR"
