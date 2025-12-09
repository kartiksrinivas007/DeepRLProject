#!/bin/bash
#SBATCH --job-name=rf_hopper_grid
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rf_hopper_grid_%j.out
#SBATCH --error=logs/rf_hopper_grid_%j.err

set -e

cd "$SLURM_SUBMIT_DIR/.."

# Sweep configuration
CONTEXT_LENS=(5 20)
ENCODER_LAYERS=(4 6)      # n_blocks in main.py
LRS=("1e-4" "4e-4")
TARGET_ENTS=(-3 -1 0 1 3)

JOBID="${SLURM_JOB_ID:-manual}"
LOG_ROOT="sbatch_scripts/logs/grid"

echo "Running Reinformer Hopper-medium grid on $(hostname)"
echo "context_len: ${CONTEXT_LENS[*]}"
echo "encoder layers (n_blocks): ${ENCODER_LAYERS[*]}"
echo "learning rates: ${LRS[*]}"
echo "target_entropy values: ${TARGET_ENTS[*]}"
date

for k in "${CONTEXT_LENS[@]}"; do
  for enc in "${ENCODER_LAYERS[@]}"; do
    for lr in "${LRS[@]}"; do
      for te in "${TARGET_ENTS[@]}"; do
        RUN_DIR="${LOG_ROOT}/K_${k}/enc_${enc}/lr_${lr}/te_${te}"
        mkdir -p "${RUN_DIR}"
        LOG_BASE="${RUN_DIR}/run_${JOBID}"
        MODEL_PATH="${RUN_DIR}/model.pt"

        echo "------------------------------------------------------------"
        echo "context_len=${k} | n_blocks=${enc} | lr=${lr} | target_entropy=${te}"
        echo "logs: ${LOG_BASE}.out / ${LOG_BASE}.err"
        echo "------------------------------------------------------------"

        python main.py \
          --env hopper \
          --dataset medium \
          --dataset_dir data/d4rl_dataset/ \
          --num_eval_ep 15 \
          --max_eval_ep_len 1000 \
          --tau 0.999 \
          --max_train_iters 15 \
          --context_len "${k}" \
          --n_blocks "${enc}" \
          --lr "${lr}" \
          --target_entropy "${te}" \
          --save_model_path "${MODEL_PATH}" \
          > "${LOG_BASE}.out" 2> "${LOG_BASE}.err"
      done
    done
  done
done

echo "Completed Reinformer Hopper-medium grid sweep."
date
