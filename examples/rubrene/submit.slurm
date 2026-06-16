#!/bin/bash
#SBATCH --account=m5025
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 7:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -J scalabath
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load python
conda activate scalabath

export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TEMPERATURE=400.0

# export CUDA_VISIBLE_DEVICES=0
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 0 &

# export CUDA_VISIBLE_DEVICES=1
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 1 &

# export CUDA_VISIBLE_DEVICES=2
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 2 &

# export CUDA_VISIBLE_DEVICES=3
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 3 &

# wait

# export CUDA_VISIBLE_DEVICES=0
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 4 &

# export CUDA_VISIBLE_DEVICES=1
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 5 &

# export CUDA_VISIBLE_DEVICES=2
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 6 &

# export CUDA_VISIBLE_DEVICES=3
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 7 &

# wait

# export CUDA_VISIBLE_DEVICES=0
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 8 &

# export CUDA_VISIBLE_DEVICES=1
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 9 &

# export CUDA_VISIBLE_DEVICES=2
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 10 &

# export CUDA_VISIBLE_DEVICES=3
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 11 &

# wait

# export CUDA_VISIBLE_DEVICES=0
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 12 &

# export CUDA_VISIBLE_DEVICES=1
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 13 &

# export CUDA_VISIBLE_DEVICES=2
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 14 &

# export CUDA_VISIBLE_DEVICES=3
# python 01.main.py --temperature "${TEMPERATURE}" --run-id 15 &

# wait


export CUDA_VISIBLE_DEVICES=0
python 01.main.py --temperature "${TEMPERATURE}" --run-id 16 &

export CUDA_VISIBLE_DEVICES=1
python 01.main.py --temperature "${TEMPERATURE}" --run-id 17 &

export CUDA_VISIBLE_DEVICES=2
python 01.main.py --temperature "${TEMPERATURE}" --run-id 18 &

export CUDA_VISIBLE_DEVICES=3
python 01.main.py --temperature "${TEMPERATURE}" --run-id 19 &

wait

export CUDA_VISIBLE_DEVICES=0
python 01.main.py --temperature "${TEMPERATURE}" --run-id 20 &

export CUDA_VISIBLE_DEVICES=1
python 01.main.py --temperature "${TEMPERATURE}" --run-id 21 &

export CUDA_VISIBLE_DEVICES=2
python 01.main.py --temperature "${TEMPERATURE}" --run-id 22 &

export CUDA_VISIBLE_DEVICES=3
python 01.main.py --temperature "${TEMPERATURE}" --run-id 23 &

wait


export CUDA_VISIBLE_DEVICES=0
python 01.main.py --temperature "${TEMPERATURE}" --run-id 24 &

export CUDA_VISIBLE_DEVICES=1
python 01.main.py --temperature "${TEMPERATURE}" --run-id 25 &

export CUDA_VISIBLE_DEVICES=2
python 01.main.py --temperature "${TEMPERATURE}" --run-id 26 &

export CUDA_VISIBLE_DEVICES=3
python 01.main.py --temperature "${TEMPERATURE}" --run-id 27 &

wait


export CUDA_VISIBLE_DEVICES=0
python 01.main.py --temperature "${TEMPERATURE}" --run-id 28 &

export CUDA_VISIBLE_DEVICES=1
python 01.main.py --temperature "${TEMPERATURE}" --run-id 29 &

export CUDA_VISIBLE_DEVICES=2
python 01.main.py --temperature "${TEMPERATURE}" --run-id 30 &

export CUDA_VISIBLE_DEVICES=3
python 01.main.py --temperature "${TEMPERATURE}" --run-id 31 &

wait