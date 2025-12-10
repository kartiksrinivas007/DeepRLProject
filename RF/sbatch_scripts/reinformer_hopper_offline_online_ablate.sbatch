#!/bin/bash
# Two ablation runs of the offline_online_test config submitted as a Slurm array (runs in parallel if resources allow).
#  Task 0: online_buffer_size=100, num_online_rollouts_per_iter=1
#  Task 1: online_buffer_size=100, num_online_rollouts_per_iter=200

#SBATCH --job-name=rf_hopper_rlft_ablate
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rf_hopper_rlft_ablate_%A_%a.out
#SBATCH --error=logs/rf_hopper_rlft_ablate_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.."

# Common settings
ENV_ID=hopper
DATASET=medium
CONTEXT_LEN=20
N_BLOCKS=6
MAX_TRAIN_ITERS=20
NUM_UPDATES_PER_ITER=1000
TAU=0.999
TARGET_ENTROPY=-3
PRETRAINED_MODEL="sbatch_scripts/logs/tau_lr/K_20/enc_6/tau_0.99/lr_4e-4/model.pt"
BETA_RL=0.66
ADV_SCALE=8

BUFFER_SIZES=(1000)
ROLLOUTS=(200)

IDX=${SLURM_ARRAY_TASK_ID}
BUFFER_SIZE=${BUFFER_SIZES[$IDX]}
ROLLOUTS_PER_ITER=${ROLLOUTS[$IDX]}

LOG_ROOT="sbatch_scripts/logs/RLFT_ablate"
RUN_DIR="${LOG_ROOT}/buffer_${BUFFER_SIZE}_rollouts_${ROLLOUTS_PER_ITER}"
PLOT_DIR="${RUN_DIR}/plots"
MODEL_DIR="models/RLFT_ablate/buffer_${BUFFER_SIZE}_rollouts_${ROLLOUTS_PER_ITER}"

mkdir -p "${RUN_DIR}" "${PLOT_DIR}" "${MODEL_DIR}"

LOG_BASE="${RUN_DIR}/run_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MODEL_PATH="${MODEL_DIR}/hopper-medium_rlft.pt"

echo "------------------------------------------------------------"
echo "Array task ${SLURM_ARRAY_TASK_ID}: buffer_size=${BUFFER_SIZE}, rollouts_per_iter=${ROLLOUTS_PER_ITER}"
echo "pretrained: ${PRETRAINED_MODEL}"
echo "logs: ${LOG_BASE}.out / ${LOG_BASE}.err"
echo "plots: ${PLOT_DIR}"
echo "model: ${MODEL_PATH}"
echo "------------------------------------------------------------"
date

PLOT_RUN_DIR="${PLOT_DIR}" python main.py \
  --env "${ENV_ID}" \
  --dataset "${DATASET}" \
  --dataset_dir data/d4rl_dataset/ \
  --num_eval_ep 10 \
  --max_eval_ep_len 1000 \
  --tau "${TAU}" \
  --max_train_iters "${MAX_TRAIN_ITERS}" \
  --num_updates_per_iter "${NUM_UPDATES_PER_ITER}" \
  --context_len "${CONTEXT_LEN}" \
  --n_blocks "${N_BLOCKS}" \
  --target_entropy "${TARGET_ENTROPY}" \
  --beta_rl "${BETA_RL}" \
  --adv_scale "${ADV_SCALE}" \
  --online_training \
  --online_buffer_size "${BUFFER_SIZE}" \
  --num_online_rollouts_per_iter "${ROLLOUTS_PER_ITER}" \
  --pretrained_model "${PRETRAINED_MODEL}" \
  --save_model_path "${MODEL_PATH}" \
  > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

date
echo "Ablation task ${SLURM_ARRAY_TASK_ID} completed."
