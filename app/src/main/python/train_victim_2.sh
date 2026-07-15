#!/bin/bash

#SBATCH --job-name=train2
#SBATCH --output=selfplay.out
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=60
#SBATCH --ntasks-per-gpu=1
#SBATCH --time=00:10:00

source ~/miniforge3/bin/activate
conda activate .planetWarsVenv
module load cuda/12.6

python agents/ppo.py \
  --total_timesteps 1500 \
  --exp_name "train_GNN_target_2" \
  --model_weights "models/train_GNN_target_final.pt"