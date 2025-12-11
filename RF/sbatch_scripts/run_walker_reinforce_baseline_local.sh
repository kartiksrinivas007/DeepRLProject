#!/bin/bash
# Run the full walker REINFORCE-baseline configs locally (no sbatch), backgrounded on one GPU.

set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_NAMES=(
  "reinformer_walker_medium_best_K5_enc4_lr1e-4_iter26.pt"
  "reinformer_walker_medium_best_K5_enc4_lr4e-4_iter15.pt"
  "reinformer_walker_medium_best_K5_enc6_lr4e-4_iter25.pt"
)
CONTEXT_LENS=(5 5 5)
N_BLOCKS=(4 4 6)
LRS=(1e-4 4e-4 4e-4)

ONLINE_BUFFER_SIZE=500
NUM_ROLLOUTS_PER_ITER=5
NUM_UPDATES_PER_ITER=1
MAX_TRAIN_ITERS=150
RTG_GAMMA=0.99
CRITIC_COEF=1.0
TAU=0.99

pids=()

for IDX in "${!MODEL_NAMES[@]}"; do
  (
    PRETRAINED_MODEL="models/walker/${MODEL_NAMES[$IDX]}"
    CONTEXT_LEN=${CONTEXT_LENS[$IDX]}
    N_BLOCK=${N_BLOCKS[$IDX]}
    LR=${LRS[$IDX]}

    RUN_DIR="sbatch_scripts/logs/reinforce_online_baseline_walker_local/critic_${CRITIC_COEF}/tau_${TAU}/no_buffer/model_${IDX}/rollouts_${NUM_ROLLOUTS_PER_ITER}_updates_${NUM_UPDATES_PER_ITER}/iters_${MAX_TRAIN_ITERS}"
    PLOT_DIR="${RUN_DIR}/plots"
    MODEL_DIR="models/reinforce_online_baseline_walker_local/critic_${CRITIC_COEF}/tau_${TAU}/no_buffer/model_${IDX}/rollouts_${NUM_ROLLOUTS_PER_ITER}_updates_${NUM_UPDATES_PER_ITER}/iters_${MAX_TRAIN_ITERS}"
    mkdir -p "${RUN_DIR}" "${PLOT_DIR}/no_buffer" "${MODEL_DIR}"

    LOG_BASE="${RUN_DIR}/run_local_${IDX}"
    SAVE_MODEL_PATH="${MODEL_DIR}/walker-medium_reinforce_baseline.pt"

    echo "------------------------------------------------------------"
    echo "Walker REINFORCE baseline (local, background) idx=${IDX}"
    echo "checkpoint: ${PRETRAINED_MODEL}"
    echo "context_len=${CONTEXT_LEN}, n_blocks=${N_BLOCK}, lr=${LR}"
    echo "mode: no_buffer, tau=${TAU}, critic_coef=${CRITIC_COEF}"
    echo "logs: ${LOG_BASE}.out / ${LOG_BASE}.err"
    echo "plots: ${PLOT_DIR}"
    echo "model: ${SAVE_MODEL_PATH}"
    echo "------------------------------------------------------------"

    PLOT_RUN_DIR="${PLOT_DIR}/no_buffer" python main.py \
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
      --no_buffer \
      --pretrained_model "${PRETRAINED_MODEL}" \
      --save_model_path "${SAVE_MODEL_PATH}" \
      --online_training \
      > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"
  ) &
  pids+=($!)
done

echo "Launched ${#pids[@]} walker REINFORCE baseline LOCAL jobs in background on the same GPU."
echo "PIDs: ${pids[*]}"
echo "Use 'wait' to block until all finish; logs are under sbatch_scripts/logs/reinforce_online_baseline_walker_local/..."
