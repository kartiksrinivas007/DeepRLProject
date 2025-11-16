#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting DT + BC Hopper-medium jobs..."
sbatch dt_hopper_medium.sbatch
sbatch bc_hopper_medium.sbatch
echo "All DT/BC Hopper jobs submitted."

