#!/bin/bash
#SBATCH --account=m5218
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -J spinboson
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load python
conda activate scalabath
export JAX_ENABLE_X64=true

export CUDA_VISIBLE_DEVICES=0
python 01.main.py --temperature 200.0 --run-id 0 &

export CUDA_VISIBLE_DEVICES=1
python 01.main.py --temperature 250.0 --run-id 0 &

export CUDA_VISIBLE_DEVICES=2
python 01.main.py --temperature 350.0 --run-id 0 &

export CUDA_VISIBLE_DEVICES=3
python 01.main.py --temperature 400.0 --run-id 0 &

wait
