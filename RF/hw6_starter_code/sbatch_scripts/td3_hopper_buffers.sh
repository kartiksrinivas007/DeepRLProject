#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Submitting TD3 Hopper-medium buffer experiments..."

sbatch td3_hopper_bc100.sbatch
sbatch td3_hopper_bc1000.sbatch
sbatch td3_hopper_cql5.sbatch

echo "Submitted all TD3 Hopper jobs."

