#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=selfplay.out
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=60
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=24:00:00

source ~/miniforge3/bin/activate
conda activate .planetWarsVenv
module load cuda/12.6

python agents/ppo.py