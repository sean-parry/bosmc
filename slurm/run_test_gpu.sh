#!/bin/bash

#SBATCH -D /users/seanpar/scratch/bosmc
#SBATCH --export=ALL
#SBATCH -J bosmc_gpu
#SBATCH -p gpu-a100-dacdt
#SBATCH -t 2-0:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1

module load miniforge3/25.3.0-python3.12.10
source activate ./env

python -m benchmarks.api