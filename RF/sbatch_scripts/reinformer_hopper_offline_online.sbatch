#!/bin/bash
# Offline + online Reinformer for Hopper-medium in one job
# Offline: 10 iters * 5k updates = 50k steps
# Online:  10 iters * 5k updates = 50k steps (starts from fresh model unless you add checkpoint saving to main.py)

#SBATCH --job-name=rf_hopper_off_on
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_hopper_off_on_%j.out
#SBATCH --error=logs/rf_hopper_off_on_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.."
mkdir -p logs

echo "Running offline Reinformer (Hopper-medium) on $(hostname)"
date

# Offline training (50k updates: 10 iters * 5k)
python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/ \
  --max_train_iters 10 \
  --num_updates_per_iter 5000 \
  --num_eval_ep 10 \
  --max_eval_ep_len 1000 \
  --tau 0.999 \
  --save_model_path "models/reinformer_hopper_medium_offline.pt"

echo "Running online finetuning (50k updates) starting from fresh model (no checkpoint save in main.py)"
date

python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/ \
  --online_training \
  --beta_rl 0.5 \
  --adv_scale 1.0 \
  --online_buffer_size 1000 \
  --pretrained_model "models/reinformer_hopper_medium_offline.pt" \
  --max_train_iters 10 \
  --num_updates_per_iter 5000 \
  --num_eval_ep 10 \
  --max_eval_ep_len 1000 \
  --tau 0.999

date
echo "Finished offline + online Reinformer run."
