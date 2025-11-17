#!/bin/bash
#SBATCH --job-name=td3_hopper_cql5_s2
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/td3_hopper_cql5_s2_%j.out
#SBATCH --error=logs/td3_hopper_cql5_s2_%j.err

set -e

cd "$SLURM_SUBMIT_DIR/.."

echo "Running TD3 Hopper-medium with CQL alpha 5 (seed 2) on $(hostname)"
date

python runner.py \
  --agent td3 \
  --env_id mujoco/hopper/medium-v0 \
  --total_steps 100000 \
  --cql_alpha 5.0 \
  --seed 42001

date

