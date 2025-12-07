#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting Reinformer sbatch jobs..."
sbatch hopper_reinformer.sbatch
sbatch hopper_reinformer_target_entropy.sbatch
# sbatch maze2d_reinformer.sbatch
# sbatch walker2d_reinformer.sbatch
# sbatch kitchen_reinformer.sbatch

echo "All Reinformer jobs submitted."

