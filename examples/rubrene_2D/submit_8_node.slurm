#!/bin/bash
#SBATCH --account=m5218
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 4:00:00
#SBATCH -N 32
#SBATCH --gpus-per-node=4
#SBATCH -J 400K
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

module load python
conda activate scalabath
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export TEMPERATURE=400.0
export LATTICE_L1=128
export LATTICE_L2=36
export SAMPLE_TIME_FS=350.0

NUM_RUNS=32
GPUS_PER_NODE=4
CPUS_PER_TASK=64
PYTHON_SCRIPT="01.main.py"

# -----------------------------------------------------------------------------
# Log directory
#
# Slurm writes the batch-level stdout/stderr to logs/%x-%j.out/.err. Each Python
# run also gets its own stdout/stderr file so outputs do not interleave.
# -----------------------------------------------------------------------------
mkdir -p logs

# -----------------------------------------------------------------------------
# Node discovery
#
# Convert Slurm's compressed node list into one hostname per array entry. This
# script expects exactly 8 nodes because it launches one independent run per node.
# -----------------------------------------------------------------------------
mapfile -t JOB_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#JOB_NODES[@]}" -ne "${NUM_RUNS}" ]]; then
  echo "Expected 8 allocated nodes, got ${#JOB_NODES[@]}: ${JOB_NODES[*]}" >&2
  exit 1
fi

echo "job_id=${SLURM_JOB_ID}"
echo "nodes=${JOB_NODES[*]}"

# -----------------------------------------------------------------------------
# Launch one run per node
#
# Each srun step:
#   - uses exactly one allocated node,
#   - uses all 4 GPUs on that node,
#   - is marked --exclusive so two runs cannot land on the same resources,
#   - receives a unique --run-id matching the loop index.
# -----------------------------------------------------------------------------
pids=()
for run_id in $(seq 0 $((NUM_RUNS - 1))); do
  node_name="${JOB_NODES[$run_id]}"
  stdout_log="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_run${run_id}.out"
  stderr_log="logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}_run${run_id}.err"

  echo "Launching run_id=${run_id} on node=${node_name}"
  srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --exclusive \
    --nodelist="${node_name}" \
    python "${PYTHON_SCRIPT}" \
      --temperature "${TEMPERATURE}" \
      --lattice-l1 "${LATTICE_L1}" \
      --lattice-l2 "${LATTICE_L2}" \
      --sample-time-fs "${SAMPLE_TIME_FS}" \
      --run-id "${run_id}" \
      >"${stdout_log}" 2>"${stderr_log}" &
  pids+=("$!")
done

# -----------------------------------------------------------------------------
# Completion handling
#
# Wait for every backgrounded srun. Preserve a nonzero final exit code if any
# node-local Python run fails.
# -----------------------------------------------------------------------------
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

exit "${status}"
