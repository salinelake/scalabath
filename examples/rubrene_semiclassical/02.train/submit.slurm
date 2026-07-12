#!/bin/bash
#SBATCH --account=m5218
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -J train
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load python
conda activate qe_dev

export CUDA_VISIBLE_DEVICES=0
python 01.train.py --temperature 200.0 &

export CUDA_VISIBLE_DEVICES=1
python 01.train.py --temperature 250.0 &

export CUDA_VISIBLE_DEVICES=2
python 01.train.py --temperature 350.0 &

export CUDA_VISIBLE_DEVICES=3
python 01.train.py --temperature 400.0 &

wait
