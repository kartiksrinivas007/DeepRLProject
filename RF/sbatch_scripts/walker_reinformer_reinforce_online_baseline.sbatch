#!/bin/bash
# REINFORCE online finetuning with baseline for Walker2d using pretrained Reinformer checkpoints.
# One job per pretrained model; no-buffer mode to focus on fresh rollouts.

#SBATCH --job-name=rf_walker_reinforce_baseline
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rf_walker_reinforce_baseline_%A_%a.out
#SBATCH --error=logs/rf_walker_reinforce_baseline_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.."

# Pretrained checkpoints and hyperparameters inferred from filenames.
MODEL_NAMES=(
  "reinformer_walker_medium_best_K5_enc4_lr1e-4_iter26.pt"
  "reinformer_walker_medium_best_K5_enc4_lr4e-4_iter15.pt"
  "reinformer_walker_medium_best_K5_enc6_lr4e-4_iter25.pt"
)
CONTEXT_LENS=(5 5 5)
N_BLOCKS=(4 4 6)
LRS=(1e-4 4e-4 4e-4)

IDX=${SLURM_ARRAY_TASK_ID:-0}
PRETRAINED_MODEL="models/walker/${MODEL_NAMES[$IDX]}"
CONTEXT_LEN=${CONTEXT_LENS[$IDX]}
N_BLOCK=${N_BLOCKS[$IDX]}
LR=${LRS[$IDX]}

ONLINE_BUFFER_SIZE=500
NUM_ROLLOUTS_PER_ITER=5
NUM_UPDATES_PER_ITER=1
MAX_TRAIN_ITERS=150
RTG_GAMMA=0.99
CRITIC_COEF=1.0
TAU=0.99

# Force no-buffer mode for REINFORCE with baseline.
NO_BUFFER_FLAG="--no_buffer"
RUN_MODE="no_buffer"

LOG_ROOT="sbatch_scripts/logs/reinforce_online_baseline_walker"
RUN_DIR="${LOG_ROOT}/critic_${CRITIC_COEF}/tau_${TAU}/${RUN_MODE}/model_${IDX}/rollouts_${NUM_ROLLOUTS_PER_ITER}_updates_${NUM_UPDATES_PER_ITER}/iters_${MAX_TRAIN_ITERS}"
PLOT_DIR="${RUN_DIR}/plots"
MODEL_DIR="models/reinforce_online_baseline_walker/critic_${CRITIC_COEF}/tau_${TAU}/${RUN_MODE}/model_${IDX}/rollouts_${NUM_ROLLOUTS_PER_ITER}_updates_${NUM_UPDATES_PER_ITER}/iters_${MAX_TRAIN_ITERS}"

mkdir -p "${RUN_DIR}" "${PLOT_DIR}" "${MODEL_DIR}"
mkdir -p "${PLOT_DIR}/${RUN_MODE}"

LOG_BASE="${RUN_DIR}/run_${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
SAVE_MODEL_PATH="${MODEL_DIR}/walker-medium_reinforce_baseline.pt"

echo "------------------------------------------------------------"
echo "Walker REINFORCE (baseline) online finetuning"
echo "checkpoint: ${PRETRAINED_MODEL}"
echo "context_len=${CONTEXT_LEN}, n_blocks=${N_BLOCK}, lr=${LR}"
echo "buffer_size=${ONLINE_BUFFER_SIZE}, rollouts/iter=${NUM_ROLLOUTS_PER_ITER}, updates/iter=${NUM_UPDATES_PER_ITER}"
echo "mode: ${RUN_MODE}, tau=${TAU}, critic_coef=${CRITIC_COEF}"
echo "logs: ${LOG_BASE}.out / ${LOG_BASE}.err"
echo "plots: ${PLOT_DIR}"
echo "model: ${SAVE_MODEL_PATH}"
echo "------------------------------------------------------------"
date

PLOT_RUN_DIR="${PLOT_DIR}/${RUN_MODE}" python main.py \
  --env walker2d \
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
  --n_blocks "${N_BLOCK}" \
  --lr "${LR}" \
  --target_entropy -3 \
  --critic_coef "${CRITIC_COEF}" \
  --reinforce_online \
  ${NO_BUFFER_FLAG} \
  --pretrained_model "${PRETRAINED_MODEL}" \
  --save_model_path "${SAVE_MODEL_PATH}" \
  --online_training \
  > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

date
echo "Walker REINFORCE baseline run completed."
