#!/usr/bin/env bash
# P2-T8: sweep vision_depth ∈ {1,2,3,4}
set -euo pipefail
PYTHON=${PYTHON:-python}
CONFIG=configs/method/token_diversion.yaml
for D in 1 2 3 4; do
    echo "=== vision_depth=$D ==="
    CUDA_VISIBLE_DEVICES=${CUDA:-0} $PYTHON scripts/train.py --config $CONFIG \
        method.vision_depth=$D \
        training.output_subdir=token_diversion/vision_depth_ablation/depth${D} \
        "$@"
done
