#!/bin/bash
#SBATCH --job-name=dt_hopper_medium
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dt_hopper_medium_%j.out
#SBATCH --error=logs/dt_hopper_medium_%j.err

set -e

cd "$SLURM_SUBMIT_DIR/.."

echo "Running Decision Transformer (DT) Hopper-medium experiment on $(hostname)"
date

python experiment.py \
  --env hopper \
  --dataset medium \
  --max_iters 10 \
  --num_steps_per_iter 10000 \
  --warmup_steps 100000 \
  --model_type dt

date
