#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting TEST array (2 combos) for Reinformer Hopper grid..."
sbatch hopper_reinformer_grid_array_test.sbatch
echo "Submitted."

