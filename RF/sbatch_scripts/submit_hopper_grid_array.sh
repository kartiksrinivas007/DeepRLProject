#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting Reinformer Hopper grid as a Slurm array (context_len x n_blocks x lr x target_entropy)..."
sbatch hopper_reinformer_grid_array.sbatch
echo "Submitted."

