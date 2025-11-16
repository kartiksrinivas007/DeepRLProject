#!/bin/bash
#SBATCH --job-name=rf_kitchen_mixed
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_kitchen_mixed_%j.out
#SBATCH --error=logs/rf_kitchen_mixed_%j.err

set -e

cd "$SLURM_SUBMIT_DIR/.."

echo "Running Reinformer Kitchen-mixed experiment on $(hostname)"
date

python main.py \
  --env kitchen \
  --dataset mixed \
  --dataset_dir data/d4rl_dataset/ \
  --context_len 20 \
  --tau 0.9 \
  --n_blocks 5 \
  --max_train_iters 10 \
  --num_updates_per_iter 5000

date
