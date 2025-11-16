#!/bin/bash
#SBATCH --job-name=rf_walker2d_medium
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_walker2d_medium_%j.out
#SBATCH --error=logs/rf_walker2d_medium_%j.err

set -e

cd "$SLURM_SUBMIT_DIR/.."

echo "Running Reinformer Walker2d-medium experiment on $(hostname)"
date

python main.py \
  --env walker2d \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/ \
  --tau 0.99 \
  --context_len 5 \
  --lr 1e-4 \
  --max_train_iters 3 \
  --num_updates_per_iter 5000 \
  --n_blocks 4

date
