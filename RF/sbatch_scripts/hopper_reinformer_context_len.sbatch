#!/bin/bash
#SBATCH --job-name=rf_hopper_ctx
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_hopper_ctx_%j.out
#SBATCH --error=logs/rf_hopper_ctx_%j.err

set -e

cd "$SLURM_SUBMIT_DIR/.."

CONTEXT_LEN="${CONTEXT_LEN:-5}"

echo "Running Reinformer Hopper-medium with context_len=${CONTEXT_LEN} on $(hostname)"
date

python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/ \
  --num_eval_ep 15 \
  --max_eval_ep_len 1000 \
  --tau 0.999 \
  --max_train_iters 15 \
  --context_len "${CONTEXT_LEN}"

date

