#!/bin/bash
# RL finetuning sweep for Hopper-medium (offline + online in one run per config)
# beta_rl: 0.0, 0.33, 0.66, 0.99, 1.0
# adv_scale: 1, 2, 4, 8, 16

#SBATCH --job-name=rf_hopper_rlft
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_hopper_rlft_%j.out
#SBATCH --error=logs/rf_hopper_rlft_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR/.."

BETAS=(0.0 0.33 0.66 0.99 1.0)
ADV_SCALES=(1 2 4 8 16)
PRETRAINED_MODEL="${PRETRAINED_MODEL:-sbatch_scripts/logs/tau_lr/K_20/enc_6/tau_0.99/lr_4e-4/model.pt}"

JOBID="${SLURM_JOB_ID:-manual}"
LOG_ROOT="sbatch_scripts/logs/RLFT"
MODEL_ROOT="models/RLFT"

echo "Running Reinformer Hopper-medium RL finetuning sweep on $(hostname)"
echo "beta_rl values: ${BETAS[*]}"
echo "adv_scale values: ${ADV_SCALES[*]}"
date

for beta in "${BETAS[@]}"; do
  for adv in "${ADV_SCALES[@]}"; do
    RUN_DIR="${LOG_ROOT}/beta_${beta}/adv_${adv}"
    mkdir -p "${RUN_DIR}"
    PLOT_DIR="${RUN_DIR}/plots"
    mkdir -p "${PLOT_DIR}"
    MODEL_DIR="${MODEL_ROOT}/beta_${beta}/adv_${adv}"
    mkdir -p "${MODEL_DIR}"
    LOG_BASE="${RUN_DIR}/run_${JOBID}"
    MODEL_PATH="${MODEL_DIR}/hopper-medium_rlft.pt"

    echo "------------------------------------------------------------"
    echo "beta_rl=${beta} | adv_scale=${adv}"
    echo "logs: ${LOG_BASE}.out / ${LOG_BASE}.err"
    echo "plots: ${PLOT_DIR}"
    echo "model: ${MODEL_PATH}"
    echo "pretrained: ${PRETRAINED_MODEL}"
    echo "------------------------------------------------------------"

    PLOT_RUN_DIR="${PLOT_DIR}" python main.py \
      --env hopper \
      --dataset medium \
      --dataset_dir data/d4rl_dataset/ \
      --num_eval_ep 15 \
      --max_eval_ep_len 1000 \
      --tau 0.999 \
      --max_train_iters 15 \
      --num_updates_per_iter 5000 \
      --target_entropy -3 \
      --beta_rl "${beta}" \
      --adv_scale "${adv}" \
      --pretrained_model "${PRETRAINED_MODEL}" \
      --save_model_path "${MODEL_PATH}" \
      > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"
  done
done

echo "Completed RL finetuning sweep."
date
