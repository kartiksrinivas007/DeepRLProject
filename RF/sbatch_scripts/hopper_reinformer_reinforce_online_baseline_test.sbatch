#!/bin/bash
# REINFORCE online finetuning with a learned baseline (expectile critic).
# Two jobs: buffered replay vs. no-buffer (fresh rollouts only).

#SBATCH --job-name=rf_hopper_reinforce_baseline
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_hopper_reinforce_baseline_%A_%a.out
#SBATCH --error=logs/rf_hopper_reinforce_baseline_%A_%a.err
#SBATCH --array=0-5

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.."

PRETRAINED_MODEL="${PRETRAINED_MODEL:-sbatch_scripts/logs/tau_lr/K_20/enc_6/tau_0.99/lr_4e-4/model.pt}"
CONTEXT_LEN=20
N_BLOCKS=6
ONLINE_BUFFER_SIZE=500
NUM_ROLLOUTS_PER_ITER=5
NUM_UPDATES_PER_ITER=1
MAX_TRAIN_ITERS=10
RTG_GAMMA=0.99

# Tau sweep (for expectile baseline), paired with mode (buffered/no_buffer).
TAU_VALUES=(0.5)
TAU_INDEX=$(( (${SLURM_ARRAY_TASK_ID:-0}) / 2 ))
MODE_INDEX=$(( (${SLURM_ARRAY_TASK_ID:-0}) % 2 ))
TAU="${TAU_VALUES[$TAU_INDEX]}"

NO_BUFFER_FLAG=""
RUN_MODE="buffered"
if [[ "${MODE_INDEX}" -eq 1 ]]; then
  NO_BUFFER_FLAG="--no_buffer"
  RUN_MODE="no_buffer"
fi

LOG_ROOT="sbatch_scripts/logs/reinforce_online_baseline"
RUN_DIR="${LOG_ROOT}/tau_${TAU}/${RUN_MODE}/rollouts_${NUM_ROLLOUTS_PER_ITER}_updates_${NUM_UPDATES_PER_ITER}"
PLOT_DIR="${RUN_DIR}/plots"
MODEL_DIR="models/reinforce_online_baseline/tau_${TAU}/${RUN_MODE}/rollouts_${NUM_ROLLOUTS_PER_ITER}_updates_${NUM_UPDATES_PER_ITER}"

mkdir -p "${RUN_DIR}" "${PLOT_DIR}" "${MODEL_DIR}"

LOG_BASE="${RUN_DIR}/run_${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
MODEL_PATH="${MODEL_DIR}/hopper-medium_reinforce_baseline.pt"

echo "------------------------------------------------------------"
echo "REINFORCE (with baseline) online finetuning on Hopper-medium"
echo "pretrained: ${PRETRAINED_MODEL}"
echo "buffer_size=${ONLINE_BUFFER_SIZE}, rollouts/iter=${NUM_ROLLOUTS_PER_ITER}, updates/iter=${NUM_UPDATES_PER_ITER}"
echo "mode: ${RUN_MODE}, tau=${TAU}"
echo "logs: ${LOG_BASE}.out / ${LOG_BASE}.err"
echo "plots: ${PLOT_DIR}"
echo "model: ${MODEL_PATH}"
echo "------------------------------------------------------------"
date

PLOT_RUN_DIR="${PLOT_DIR}/${RUN_MODE}" python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/ \
  --num_eval_ep 10 \
  --max_eval_ep_len 1000 \
  --tau "${TAU}" \
  --baseline \
  --max_train_iters "${MAX_TRAIN_ITERS}" \
  --num_updates_per_iter "${NUM_UPDATES_PER_ITER}" \
  --rtg_gamma "${RTG_GAMMA}" \
  --num_online_rollouts_per_iter "${NUM_ROLLOUTS_PER_ITER}" \
  --online_buffer_size "${ONLINE_BUFFER_SIZE}" \
  --context_len "${CONTEXT_LEN}" \
  --n_blocks "${N_BLOCKS}" \
  --target_entropy -3 \
  --reinforce_online \
  ${NO_BUFFER_FLAG} \
  --pretrained_model "${PRETRAINED_MODEL}" \
  --save_model_path "${MODEL_PATH}" \
  --online_training \
  > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

date
echo "REINFORCE baseline run completed."
