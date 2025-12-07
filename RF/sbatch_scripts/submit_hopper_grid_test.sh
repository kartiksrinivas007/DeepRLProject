#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting Reinformer Hopper TEST grid (tiny subset)..."
sbatch hopper_reinformer_grid_test.sbatch
echo "Submitted."

