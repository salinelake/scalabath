#!/bin/bash
#SBATCH --account=m5025
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH -J 300K
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --array=0-19

module load python
conda activate scalabath

export TEMPERATURE=300.0
export CHAIN_LENGTH=200
export SAMPLE_TIME_FS=450
export BATCH_SIZE=5


python 01.main.py \
  --temperature "${TEMPERATURE}" \
  --chain-length "${CHAIN_LENGTH}" \
  --sample-time-fs "${SAMPLE_TIME_FS}" \
  --batch-size "${BATCH_SIZE}" \
  --run-id ${SLURM_ARRAY_TASK_ID}
