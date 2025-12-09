#!/bin/bash
# Online-only Reinformer finetuning for Hopper-medium (very gentle: beta_rl=0.01, adv_scale=3.0)

#SBATCH --job-name=rf_hopper_online_gentle
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_hopper_online_gentle_%j.out
#SBATCH --error=logs/rf_hopper_online_gentle_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.."
mkdir -p logs models

BETA_RL=0.01
ADV_SCALE=3.0

echo "Running online Reinformer (Hopper-medium) very gentle on $(hostname)"
echo "beta_rl=${BETA_RL}, adv_scale=${ADV_SCALE}"
date

export PLOT_RUN_DIR="plots/online/hopper_medium/gentle_beta_${BETA_RL}_adv_${ADV_SCALE}/job_${SLURM_JOB_ID}"

python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/ \
  --online_training \
  --beta_rl "${BETA_RL}" \
  --adv_scale "${ADV_SCALE}" \
  --online_buffer_size 1000 \
  --pretrained_model "models/reinformer_hopper_medium_offline.pt" \
  --max_train_iters 50 \
  --num_updates_per_iter 300 \
  --num_eval_ep 10 \
  --max_eval_ep_len 1000 \
  --tau 0.999 \
  --target_entropy 3

date
echo "Finished online-only Reinformer very gentle run."

