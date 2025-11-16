#!/bin/bash
#SBATCH --job-name=rf_hopper_medium
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_hopper_medium_%j.out
#SBATCH --error=logs/rf_hopper_medium_%j.err

set -e

cd "$SLURM_SUBMIT_DIR/.."

echo "Running Reinformer Hopper-medium experiment on $(hostname)"
date

python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/ \
  --num_eval_ep 10 \
  --max_eval_ep_len 1000 \
  --tau 0.999 \
  --max_train_iters 6

date
