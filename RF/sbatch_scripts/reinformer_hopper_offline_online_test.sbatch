#!/bin/bash
# Tiny RL finetuning sweep (2 configs) for quick testing
# Uses the same structure as reinformer_hopper_offline_online.sbatch

#SBATCH --job-name=rf_hopper_rlft_test
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/rf_hopper_rlft_test_%A_%a.out
#SBATCH --error=logs/rf_hopper_rlft_test_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.."

# Two test configs
BETAS=(0.0 0.66)
ADV_SCALES=(1 8)

IDX=${SLURM_ARRAY_TASK_ID}
beta=${BETAS[$IDX]}
adv=${ADV_SCALES[$IDX]}

PRETRAINED_MODEL="${PRETRAINED_MODEL:-sbatch_scripts/logs/tau_lr/K_20/enc_6/tau_0.99/lr_4e-4/model.pt}"
CONTEXT_LEN=20
N_BLOCKS=6

LOG_ROOT="sbatch_scripts/logs/RLFT_array_test"
MODEL_ROOT="models/RLFT_array_test"
RUN_DIR="${LOG_ROOT}/beta_${beta}/adv_${adv}"
PLOT_DIR="${RUN_DIR}/plots"
MODEL_DIR="${MODEL_ROOT}/beta_${beta}/adv_${adv}"

mkdir -p "${RUN_DIR}" "${PLOT_DIR}" "${MODEL_DIR}"

JOBID="${SLURM_ARRAY_JOB_ID:-manual}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
LOG_BASE="${RUN_DIR}/run_${JOBID}_${TASK}"
MODEL_PATH="${MODEL_DIR}/hopper-medium_rlft.pt"

echo "------------------------------------------------------------"
echo "TEST Array task ${TASK}: beta_rl=${beta} | adv_scale=${adv}"
echo "pretrained: ${PRETRAINED_MODEL}"
echo "logs: ${LOG_BASE}.out / ${LOG_BASE}.err"
echo "plots: ${PLOT_DIR}"
echo "model: ${MODEL_PATH}"
echo "------------------------------------------------------------"
date

PLOT_RUN_DIR="${PLOT_DIR}" python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/ \
  --num_eval_ep 10 \
  --max_eval_ep_len 1000 \
  --tau 0.999 \
  --max_train_iters 3 \
  --num_updates_per_iter 1000 \
  --context_len "${CONTEXT_LEN}" \
  --n_blocks "${N_BLOCKS}" \
  --target_entropy -3 \
  --beta_rl "${beta}" \
  --adv_scale "${adv}" \
  --pretrained_model "${PRETRAINED_MODEL}" \
  --num_online_rollouts_per_iter 1 \
  --save_model_path "${MODEL_PATH}" \
  --online_training \
  > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

date
