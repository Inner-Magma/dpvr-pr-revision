#!/usr/bin/env bash
# P2-T7: sweep split_layer ∈ {8,12,16,20,24}
set -euo pipefail
PYTHON=${PYTHON:-python}
CONFIG=configs/method/token_diversion.yaml
for L in 8 12 16 20 24; do
    echo "=== split_layer=$L ==="
    CUDA_VISIBLE_DEVICES=${CUDA:-0} $PYTHON scripts/train.py --config $CONFIG \
        method.split_layer=$L \
        training.output_subdir=token_diversion/split_layer_ablation/split${L} \
        "$@"
done
