#!/bin/bash

#SBATCH -D /users/seanpar/scratch/bosmc
#SBATCH --export=ALL
#SBATCH -J bosmc_run
#SBATCH -p nodes
#SBATCH -t 1-0:00:00
#SBATCH -N 1
#SBATCH -n 32

module load miniforge3/25.3.0-python3.12.10
source activate ./env

python -m benchmarks.api