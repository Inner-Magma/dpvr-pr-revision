#!/usr/bin/env bash
# P2-T9: sweep data scale ∈ {10%, 30%, 60%, 100%}
set -euo pipefail
PYTHON=${PYTHON:-python}
METHOD_CONFIG=configs/method/token_diversion.yaml
for PCT in 10 30 60 100; do
    echo "=== data_scale=${PCT}pct ==="
    CUDA_VISIBLE_DEVICES=${CUDA:-0} $PYTHON scripts/train.py \
        --config $METHOD_CONFIG \
        --config configs/data/llava665k_${PCT}pct.yaml \
        training.output_subdir=token_diversion/data_scale_ablation/pct${PCT} \
        "$@"
done
