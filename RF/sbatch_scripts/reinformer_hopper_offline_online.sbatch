#!/bin/bash
# RL finetuning sweep for Hopper-medium using a Slurm array
# beta_rl in {0.0, 0.33, 0.66, 0.99, 1.0}
# adv_scale in {1, 2, 4, 8, 16}
# num_updates_per_iter in {1000, 3000, 5000}

#SBATCH --job-name=rf_hopper_rlft_array
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rf_hopper_rlft_%A_%a.out
#SBATCH --error=logs/rf_hopper_rlft_%A_%a.err
#SBATCH --array=0-35

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.."

BETAS=(0.0 0.66 1.0)
ADV_SCALES=(1 4 8 16)
UPDATES=(300 1000 3000)

NUM_B=${#BETAS[@]}
NUM_A=${#ADV_SCALES[@]}
NUM_U=${#UPDATES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
beta_idx=$(( IDX / (NUM_A * NUM_U) ))
rem=$(( IDX % (NUM_A * NUM_U) ))
adv_idx=$(( rem / NUM_U ))
upd_idx=$(( rem % NUM_U ))

beta=${BETAS[$beta_idx]}
adv=${ADV_SCALES[$adv_idx]}
updates=${UPDATES[$upd_idx]}

PRETRAINED_MODEL="${PRETRAINED_MODEL:-sbatch_scripts/logs/tau_lr/K_20/enc_6/tau_0.99/lr_4e-4/model.pt}"
CONTEXT_LEN=20
N_BLOCKS=6

LOG_ROOT="sbatch_scripts/logs/RLFT_array"
MODEL_ROOT="models/RLFT_array"
RUN_DIR="${LOG_ROOT}/beta_${beta}/adv_${adv}/updates_${updates}"
PLOT_DIR="${RUN_DIR}/plots"
MODEL_DIR="${MODEL_ROOT}/beta_${beta}/adv_${adv}"

mkdir -p "${RUN_DIR}" "${PLOT_DIR}" "${MODEL_DIR}"

JOBID="${SLURM_ARRAY_JOB_ID:-manual}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
LOG_BASE="${RUN_DIR}/run_${JOBID}_${TASK}"
MODEL_PATH="${MODEL_DIR}/hopper-medium_rlft.pt"

echo "------------------------------------------------------------"
echo "Array task ${TASK}: beta_rl=${beta} | adv_scale=${adv} | updates_per_iter=${updates}"
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
  --num_eval_ep 15 \
  --max_eval_ep_len 1000 \
  --tau 0.999 \
  --max_train_iters 30 \
  --num_updates_per_iter "${updates}" \
  --context_len "${CONTEXT_LEN}" \
  --n_blocks "${N_BLOCKS}" \
  --target_entropy -3 \
  --beta_rl "${beta}" \
  --adv_scale "${adv}" \
  --pretrained_model "${PRETRAINED_MODEL}" \
  --save_model_path "${MODEL_PATH}" \
  --online_training \
  > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

date
