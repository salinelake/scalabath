# AGENTS.md

## Project overview

`scalabath` is a scientific Python package for simulating quantum dynamics
on lattice systems. The intended model families are:

- Schrodinger-equation dynamics with bosonic environments.
- Lindblad master-equation dynamics with pseudomodes.
- Lattice Hamiltonians, baths, couplings, observables, and time-evolution
  solvers.

The implementation should be based on JAX so that core kernels can use `jit` 
and GPU execution (including multi-GPU via NamedSharding). NumPy is allowed for host-side setup, plotting,
fixtures, and reference calculations, but JIT-critical paths should use
`jax.numpy`.

## Expected repository layout

Keep the repository organized like this unless there is a strong reason to
change it:

```text
scalabath/
  pyproject.toml
  AGENTS.md
  README.md
  LICENSE
  .gitignore
  src/
    scalabath/
      __init__.py
      models.py
      operators.py
      simulations.py
      utilities.py
  tests/
    unit/
    cpu/
    gpu/
    conftest.py
  examples/
  docs/
    plans/
  jobs/
    perlmutter_gpu_test.slurm
```

Do not commit large simulation outputs, checkpoints, logs, virtual
environments, `.jax_cache`, or machine-specific files. Commit small JSON/CSV
summaries only when they are useful for reproducibility.

## Local development commands

Use these commands on a Linux machine with only CPU support:

```bash
conda create -n scalabath python=3.11 -y
conda activate scalabath
pip install -U jax
pip install -e .
```

Use these commands on a Linux machine with CUDA support:

```bash
conda create -n scalabath python=3.11 -y
conda activate scalabath
pip install -U "jax[cuda13]"
pip install -e .
```

For fast local validation:

```bash
JAX_ENABLE_X64=1 python -m pytest -m "not gpu and not cpu"
python -m ruff check src tests
python -m ruff format --check src tests
```

For formatting:

```bash
python -m ruff format src tests
python -m ruff check --fix src tests
```
 

## Coding guidelines

- Prefer small, composable functions over large solver scripts.
- Format docstrings for future readthedocs documentation.
- Public APIs should have docstrings and type hints where practical.
- Keep array shape conventions explicit in docstrings. For example, document
  whether a state is shaped as `(n_sites,)`, `(hilbert_dim,)`, or
  `(batch, hilbert_dim)`.
- Use complex dtypes deliberately. For quantum dynamics, tests should include
  complex-valued states/operators.
- Keep JAX-transformable code pure: avoid mutation, file I/O, Python iterators,
  and data-dependent Python control flow inside functions expected to be `jit`ed.
  Use `jax.lax.scan`, `jax.lax.cond`, or `jax.lax.while_loop` when needed.
- Use NumPy for reference implementations in tests when that makes correctness
  easier to verify.
- Add tests with every behavior change.
- Do not add new runtime dependencies without a short explanation in the commit
  or PR description.

## Testing tiers

Use pytest markers consistently:

- `unit`: small CPU-only tests.
- `cpu`: longer CPU tests.
- `gpu`: requires a CUDA GPU; normally run on Perlmutter.

Examples:

```bash
# Fast checks on a Mac
JAX_ENABLE_X64=1 python -m pytest -m "not gpu and not cpu"

# All CPU checks on a Mac or Perlmutter CPU node
JAX_ENABLE_X64=1 python -m pytest -m "not gpu"

# GPU checks, normally inside a Slurm job
JAX_ENABLE_X64=1 python -m pytest -m "gpu"
```

## Perlmutter principles

Perlmutter is for GPU/HPC validation, not primary editing. Make source changes
on a Mac or in a normal development checkout, commit them, push them, then pull
on Perlmutter.

Never run heavy CPU/GPU simulations on Perlmutter login nodes. On login nodes,
only do lightweight actions such as `git pull`, editing small files, inspecting
logs, or submitting/monitoring Slurm jobs.

Use `$SCRATCH` for temporary high-throughput job outputs. Important results must
be copied back or archived because scratch is temporary and purgeable.

## Perlmutter scratch space

```bash
ssh perlmutter
cd ~/scratch/jobs
```


## Perlmutter job script

Keep the default GPU test script at `jobs/perlmutter_gpu_test.slurm`. If it does
not exist, create it from this template and replace `<NERSC_ACCOUNT>` before
first use:

```bash
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
```

For larger benchmark jobs, create a separate script such as
`jobs/perlmutter_benchmark.slurm` rather than changing the default debug test
script.


## Monitor a Perlmutter job

Use these from Perlmutter:

```bash
JOBID=$(cat .last_perlmutter_job)
squeue -j "$JOBID"
squeue --me
sacct -j "$JOBID" --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```

To inspect logs:

```bash
JOBID=$(cat .last_perlmutter_job)
tail -n 80 "logs/scalabath-gpu-test-${JOBID}.out"
tail -n 80 "logs/scalabath-gpu-test-${JOBID}.err"
```

Avoid aggressive polling. Do not run frequent `watch squeue`. If `watch` is
needed, use an interval of at least 60 seconds and terminate it when done:

```bash
watch -n 60 squeue --me
```

To cancel a job:

```bash
scancel "$JOBID"
```

## Pull results back to a Mac

Small or medium results can be copied with `rsync`. From the Mac checkout:

```bash
JOBID=<JOB_ID>
mkdir -p "results/perlmutter/$JOBID"
rsync -avP perlmutter.nersc.gov:'$SCRATCH/scalabath-runs/'"$JOBID"'/' \
  "results/perlmutter/$JOBID/"
```

For larger transfers, use a NERSC data transfer node instead of a login node:

```bash
JOBID=<JOB_ID>
mkdir -p "results/perlmutter/$JOBID"
rsync -avP dtn01.nersc.gov:'$SCRATCH/scalabath-runs/'"$JOBID"'/' \
  "results/perlmutter/$JOBID/"
```

After pulling results back:

```bash
cat "results/perlmutter/$JOBID/summary.json"
cat "results/perlmutter/$JOBID/pytest-gpu.log"
```

Commit only small summaries or benchmark metadata. Do not commit raw arrays,
large trajectories, caches, or cluster logs unless they are intentionally small
regression fixtures.

## Codex working rules

- Start by reading `AGENTS.md`, `pyproject.toml`, and relevant tests.
- Keep diffs small and reviewable.
- For nontrivial solver changes, write or update a short plan in `docs/plans/`.
- Run the fastest relevant local tests after edits.
- If GPU validation is required, update the Perlmutter instructions or job script
  but do not claim GPU validation passed unless a Perlmutter job actually ran.
- Do not invent physical formulas. If adding a method, include a reference in
  comments, docs, or the PR description when appropriate.
- Preserve reproducibility: record random seeds, dtype assumptions, lattice size,
  time step, and solver tolerances in tests and examples.
- Never print, commit, or request secrets, tokens, private keys, or credentials.

## Done criteria

A change is done when:

- The package imports successfully.
- Fast CPU tests pass with `JAX_ENABLE_X64=1 python -m pytest -m "not gpu and not cpu"`.
- Ruff checks pass.
- New or changed behavior has tests.
- GPU-sensitive changes either have a passing Perlmutter job ID recorded in the
  PR/commit notes or clearly state that GPU validation remains to be run.