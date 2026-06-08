#!/usr/bin/env bash
# Run the full method matrix: token_diversion (primary) + all comparison baselines.
# Pass training.max_steps=1 to dry-run; omit for full training.
#
# Usage:
#   bash sweeps/run_all_baselines.sh                          # full training, 7B
#   bash sweeps/run_all_baselines.sh training.max_steps=1     # dry-run, 7B
#   MODEL_SIZE=13b bash sweeps/run_all_baselines.sh           # full training, 13B
set -euo pipefail

PYTHON=${PYTHON:-python}
MODEL_SIZE=${MODEL_SIZE:-7b}

if [[ "$MODEL_SIZE" == "13b" ]]; then
    CONFIGS=(
        configs/method/token_diversion_13b.yaml
        configs/method/lora.yaml
        configs/method/fastv.yaml
        configs/method/full_ft.yaml
    )
else
    CONFIGS=(
        configs/method/token_diversion.yaml
        configs/method/lora.yaml
        configs/method/fastv.yaml
        configs/method/full_ft.yaml
        configs/method/td_plus_fastv.yaml
    )
fi

for CFG in "${CONFIGS[@]}"; do
    METHOD=$(basename $CFG .yaml)
    echo "=== method=${METHOD} model_size=${MODEL_SIZE} ==="
    CUDA_VISIBLE_DEVICES=${CUDA:-0} $PYTHON scripts/train.py --config $CFG "$@"
done
