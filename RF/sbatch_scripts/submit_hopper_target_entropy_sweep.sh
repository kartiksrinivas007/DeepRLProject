#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sweep target_entropy values from -3 to 3 (inclusive) and submit one job per value.
VALUES=(-3 -1 0 1 3)

echo "Submitting Hopper target_entropy sweep: ${VALUES[*]}"
for te in "${VALUES[@]}"; do
  echo "  submitting target_entropy=${te}"
  sbatch --export=TARGET_ENTROPY="${te}" hopper_reinformer_target_entropy.sbatch
done

echo "All target_entropy jobs submitted."

