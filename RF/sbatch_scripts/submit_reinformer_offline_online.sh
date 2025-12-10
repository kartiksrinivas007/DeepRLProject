#!/bin/bash
# Submit the RL finetuning sweep array for Hopper-medium
# The array definitions live in reinformer_hopper_offline_online.sbatch

set -euo pipefail

cd "$(dirname "$0")"

echo "Submitting Reinformer Hopper offline+online sweep array..."
JOB_ID=$(sbatch reinformer_hopper_offline_online.sbatch | awk '{print $4}')

if [[ -n "${JOB_ID:-}" ]]; then
  echo "Submitted array job with ID: ${JOB_ID}"
  echo "Logs will appear under RF/sbatch_scripts/logs/RLFT_array/"
else
  echo "Warning: could not parse job ID from sbatch output."
fi
