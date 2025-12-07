#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting Reinformer Hopper grid sweep (context_len, encoder layers, lr)..."
sbatch hopper_reinformer_grid.sbatch
echo "Submitted."

