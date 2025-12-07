#!/bin/bash
# Offline + online Decision Transformer for Hopper-medium in one job
# Offline: 5 iters * 10k steps = 50k updates
# Online:  5 iters * 10k steps = 50k updates (starting from offline checkpoint)

#SBATCH --job-name=odt_hopper_off_on
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/odt_hopper_off_on_%j.out
#SBATCH --error=logs/odt_hopper_off_on_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.."
mkdir -p logs

cd online-decision-transformer/gym

echo "Running offline DT (Hopper-medium) on $(hostname)"
date

# Offline pretraining (50k steps: 5 iters * 10k)
python experiment.py \
  --env hopper \
  --dataset medium \
  --model_type dt \
  --max_iters 5 \
  --num_steps_per_iter 10000 \
  --num_eval_episodes 50 \
  --device cuda

# Grab latest offline checkpoint
latest_ckpt=$(ls -t models/hopper/dt_gym-experiment-hopper-medium-*.pt 2>/dev/null | head -n 1 || true)
if [[ -z "${latest_ckpt}" ]]; then
  echo "No offline checkpoint found under models/hopper/. Exiting." >&2
  exit 1
fi

echo "Offline checkpoint: ${latest_ckpt}"
echo "Running online finetuning (50k steps) starting from offline checkpoint"
date

python experiment.py \
  --env hopper \
  --dataset medium \
  --model_type dt \
  --online_training \
  --pretrained_model "${latest_ckpt}" \
  --max_iters 5 \
  --num_steps_per_iter 10000 \
  --num_eval_episodes 50 \
  --device cuda

date
echo "Finished offline + online DT run."
