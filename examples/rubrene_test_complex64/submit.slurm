#!/bin/bash
#SBATCH --account=m5025
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 9:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH -J 300K
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

module load python
conda activate scalabath

export TEMPERATURE=300.0
export CHAIN_LENGTH=200
export SAMPLE_TIME_FS=450
export BATCH_SIZE=5

export CUDA_VISIBLE_DEVICES=0
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 0 &

export CUDA_VISIBLE_DEVICES=1
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 1 &

export CUDA_VISIBLE_DEVICES=2
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 2 &

export CUDA_VISIBLE_DEVICES=3
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 3 &

wait

export CUDA_VISIBLE_DEVICES=0
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 4 &

export CUDA_VISIBLE_DEVICES=1
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 5 &

export CUDA_VISIBLE_DEVICES=2
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 6 &

export CUDA_VISIBLE_DEVICES=3
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 7 &

wait

export CUDA_VISIBLE_DEVICES=0
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 8 &

export CUDA_VISIBLE_DEVICES=1
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 9 &

export CUDA_VISIBLE_DEVICES=2
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 10 &

export CUDA_VISIBLE_DEVICES=3
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 11 &

wait

export CUDA_VISIBLE_DEVICES=0
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 12 &

export CUDA_VISIBLE_DEVICES=1
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 13 &

export CUDA_VISIBLE_DEVICES=2
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 14 &

export CUDA_VISIBLE_DEVICES=3
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 15 &

wait


export CUDA_VISIBLE_DEVICES=0
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 16 &

export CUDA_VISIBLE_DEVICES=1
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 17 &

export CUDA_VISIBLE_DEVICES=2
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 18 &

export CUDA_VISIBLE_DEVICES=3
python 01.main.py --temperature "${TEMPERATURE}" --chain-length "${CHAIN_LENGTH}" --sample-time-fs "${SAMPLE_TIME_FS}" --batch-size "${BATCH_SIZE}" --run-id 19 &

wait
