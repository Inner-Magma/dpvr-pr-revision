#!/usr/bin/env bash
# E0: build the locked environment on the GPU server (H20 or A800; CUDA 12.8 wheels
# cover both sm_90 and sm_80).
#
# Usage:
#   bash scripts/revision/setup_env.sh            # conda env "dpvr"
#   ENV_NAME=dpvr2 bash scripts/revision/setup_env.sh
#
# Mirrors (servers in mainland China): set PIP_INDEX_URL for PyPI and TORCH_INDEX for
# the PyTorch wheels, e.g.
#   PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#   TORCH_INDEX=https://mirrors.aliyun.com/pytorch-wheels/cu128 bash scripts/revision/setup_env.sh
# (with the Aliyun torch mirror pip needs `-f` instead of `--index-url`; set TORCH_FIND_LINKS=1).
set -euo pipefail

ENV_NAME=${ENV_NAME:-dpvr}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOCK="$REPO/requirements-lock.txt"
OUT="$REPO/results/revision/E0"
mkdir -p "$OUT"

if ! command -v conda >/dev/null; then
    echo "conda not found; install Miniconda first" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
python -m pip install -U "pip<25"

if [[ "${TORCH_FIND_LINKS:-0}" == "1" ]]; then
    TORCH_SRC=(-f "$TORCH_INDEX")
else
    TORCH_SRC=(--index-url "$TORCH_INDEX")
fi
python -m pip install torch==2.9.0 torchvision==0.24.0 "${TORCH_SRC[@]}"
python - <<'PY'
import torch
assert torch.version.cuda and torch.version.cuda.startswith("12.8"), torch.version.cuda
print("torch", torch.__version__, "CUDA", torch.version.cuda)
PY

# Everything else, pinned by the lock file (also used as constraints for lmms-eval's deps).
python -m pip install -r "$LOCK" -c "$LOCK"
python -m pip install -c "$LOCK" -e "$REPO[analysis,profile,dev]"

python -m pip freeze > "$OUT/pip-freeze.txt"
nvidia-smi > "$OUT/nvidia-smi.txt" || true
echo
echo "Environment '$ENV_NAME' ready. Next:"
echo "  cp .env.example .env   # fill DATA_ROOT, MODEL_ROOT, CKPT_ROOT"
echo "  python scripts/revision/check_env.py --out results/revision/E0/env_check.json"
