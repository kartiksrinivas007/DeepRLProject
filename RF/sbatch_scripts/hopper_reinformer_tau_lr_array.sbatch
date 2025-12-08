#!/bin/bash
#SBATCH --job-name=rf_hopper_tau_lr
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_hopper_tau_lr_%A_%a.out
#SBATCH --error=logs/rf_hopper_tau_lr_%A_%a.err
#SBATCH --array=0-8

set -e

cd "$SLURM_SUBMIT_DIR/.."

# Fixed best-performing settings from your tests
CONTEXT_LEN=20
N_BLOCKS=6
TARGET_ENTROPY=-3

# Sweep over tau and lr
# TAUS=(0.8 0.9 0.99)
TAUS=(0.99)
# LRS=("2e-4" "3e-4" "4e-4")
LRS=("4e-4")

NUM_TAU=${#TAUS[@]}  # 3
NUM_LR=${#LRS[@]}    # 3

IDX=${SLURM_ARRAY_TASK_ID}

# Map array index -> (tau, lr)
tau_idx=$(( IDX / NUM_LR ))
lr_idx=$(( IDX % NUM_LR ))

TAU=${TAUS[$tau_idx]}
LR=${LRS[$lr_idx]}

JOBID="${SLURM_ARRAY_JOB_ID:-manual}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
LOG_ROOT="sbatch_scripts/logs/tau_lr"
RUN_DIR="${LOG_ROOT}/K_${CONTEXT_LEN}/enc_${N_BLOCKS}/tau_${TAU}/lr_${LR}"
mkdir -p "${RUN_DIR}"
PLOT_DIR="${RUN_DIR}/plots"
mkdir -p "${PLOT_DIR}"
LOG_BASE="${RUN_DIR}/run_${JOBID}_${TASK}"

echo "------------------------------------------------------------"
echo "Hopper-medium sweep | context_len=${CONTEXT_LEN} | n_blocks=${N_BLOCKS} | tau=${TAU} | lr=${LR} | target_entropy=${TARGET_ENTROPY}"
echo "logs: ${LOG_BASE}.out / ${LOG_BASE}.err"
echo "plots dir: ${PLOT_DIR}"
echo "Array task: ${TASK} of ${SLURM_ARRAY_TASK_COUNT:-9}"
echo "------------------------------------------------------------"
date

PLOT_RUN_DIR="${PLOT_DIR}" python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/ \
  --num_eval_ep 15 \
  --max_eval_ep_len 1000 \
  --tau "${TAU}" \
  --max_train_iters 15 \
  --context_len "${CONTEXT_LEN}" \
  --n_blocks "${N_BLOCKS}" \
  --lr "${LR}" \
  --target_entropy "${TARGET_ENTROPY}" \
  > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

date
