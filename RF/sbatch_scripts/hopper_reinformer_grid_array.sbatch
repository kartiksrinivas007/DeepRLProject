#!/bin/bash
#SBATCH --job-name=rf_hopper_grid_array
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_hopper_grid_array_%A_%a.out
#SBATCH --error=logs/rf_hopper_grid_array_%A_%a.err
#SBATCH --array=0-39

set -e

cd "$SLURM_SUBMIT_DIR/.."

# Grid definitions
CONTEXT_LENS=(5 20)
ENCODER_LAYERS=(4 6)      # n_blocks in main.py
LRS=("1e-4" "4e-4")
TARGET_ENTS=(-3 -1 0 1 3)

NUM_K=${#CONTEXT_LENS[@]}          # 2
NUM_ENC=${#ENCODER_LAYERS[@]}      # 2
NUM_LR=${#LRS[@]}                  # 2
NUM_TE=${#TARGET_ENTS[@]}          # 5

IDX=${SLURM_ARRAY_TASK_ID}

# Map array index -> grid combo
k_idx=$(( IDX / (NUM_ENC * NUM_LR * NUM_TE) ))
rem=$(( IDX % (NUM_ENC * NUM_LR * NUM_TE) ))
enc_idx=$(( rem / (NUM_LR * NUM_TE) ))
rem=$(( rem % (NUM_LR * NUM_TE) ))
lr_idx=$(( rem / NUM_TE ))
te_idx=$(( rem % NUM_TE ))

K=${CONTEXT_LENS[$k_idx]}
ENC=${ENCODER_LAYERS[$enc_idx]}
LR=${LRS[$lr_idx]}
TE=${TARGET_ENTS[$te_idx]}

JOBID="${SLURM_ARRAY_JOB_ID:-manual}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"
LOG_ROOT="sbatch_scripts/logs/grid_array"
RUN_DIR="${LOG_ROOT}/K_${K}/enc_${ENC}/lr_${LR}/te_${TE}"
mkdir -p "${RUN_DIR}"
PLOT_DIR="${RUN_DIR}/plots"
mkdir -p "${PLOT_DIR}"
LOG_BASE="${RUN_DIR}/run_${JOBID}_${TASK}"
MODEL_PATH="${RUN_DIR}/model.pt"

echo "------------------------------------------------------------"
echo "Running Hopper-medium | context_len=${K} | n_blocks=${ENC} | lr=${LR} | target_entropy=${TE}"
echo "logs: ${LOG_BASE}.out / ${LOG_BASE}.err"
echo "plots dir: ${PLOT_DIR}"
echo "Array task: ${TASK} of ${SLURM_ARRAY_TASK_COUNT:-40}"
echo "------------------------------------------------------------"
date

PLOT_RUN_DIR="${PLOT_DIR}" python main.py \
  --env hopper \
  --dataset medium \
  --dataset_dir data/d4rl_dataset/ \
  --num_eval_ep 15 \
  --max_eval_ep_len 1000 \
  --tau 0.999 \
  --max_train_iters 20 \
  --context_len "${K}" \
  --n_blocks "${ENC}" \
  --lr "${LR}" \
  --target_entropy "${TE}" \
  --save_model_path "${MODEL_PATH}" \
  > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"

date
