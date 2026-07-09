#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --gpus=1
#SBATCH --cpus-per-gpus=60
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=24:00:00  

module load cuda/12.6

source ~/miniforge3/bin/activate
conda activate .planetWarsVenv

python agents/ppo.py