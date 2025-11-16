#!/bin/bash
#SBATCH --job-name=bc_hopper_medium
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bc_hopper_medium_%j.out
#SBATCH --error=logs/bc_hopper_medium_%j.err

set -e

cd "$SLURM_SUBMIT_DIR/.."

echo "Running Behavior Cloning (BC) Hopper-medium experiment on $(hostname)"
date

python experiment.py \
  --env hopper \
  --dataset medium \
  --model_type bc \
  --embed_dim 256 \
  --n_layer 3 \
  --batch_size 64 \
  --activation_function relu \
  --dropout 0.1 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --warmup_steps 100000 \
  --max_iters 10 \
  --num_steps_per_iter 10000

date
