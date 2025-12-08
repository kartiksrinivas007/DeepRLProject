#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting Reinformer Hopper tau/lr array sweep (context_len=20, n_blocks=6, target_entropy=-3)..."
sbatch hopper_reinformer_tau_lr_array.sbatch
echo "Submitted."

