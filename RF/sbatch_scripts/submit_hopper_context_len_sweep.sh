#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sweep context_len values and submit one job per value.
VALUES=(5 10 20)

echo "Submitting Hopper context_len sweep: ${VALUES[*]}"
for ctx in "${VALUES[@]}"; do
  echo "  submitting context_len=${ctx}"
  sbatch --export=CONTEXT_LEN="${ctx}" hopper_reinformer_context_len.sbatch
done

echo "All context_len jobs submitted."

