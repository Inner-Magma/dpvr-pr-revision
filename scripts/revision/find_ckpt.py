#!/usr/bin/env python
"""Print the checkpoint directory to evaluate under ROOT (exit 1 if none).

Checkpoints relayed from the old server keep their run layout, which is not the
same for every method: ``full_ft/`` and ``lora/`` hold one more level of run
directories, and a run may keep its final weights only in the largest
``checkpoint-N/``. A directory counts as a checkpoint when it has weights
(``*.safetensors`` or ``*.bin``, optimizer/scheduler/training-args files aside)
and ``training_config.json`` or ``config.json``. Preference: a directory not
named ``checkpoint-*`` (the final save; the shallowest, then the first by name),
else the ``checkpoint-N`` with the largest N.

    python scripts/revision/find_ckpt.py ${CKPT_ROOT}/hsf/full_ft
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

NOT_WEIGHTS = {"optimizer.bin", "scheduler.bin", "training_args.bin", "rng_state.bin"}
STEP = re.compile(r"^checkpoint-(\d+)$")


def is_checkpoint(d: Path) -> bool:
    weights = [f for f in d.iterdir() if f.is_file() and f.suffix in (".safetensors", ".bin")
               and f.name not in NOT_WEIGHTS and not f.name.startswith("rng_state")]
    return bool(weights) and ((d / "training_config.json").exists() or (d / "config.json").exists())


def find_checkpoint(root: Path, max_depth: int = 4) -> Path | None:
    root = Path(root)
    if not root.is_dir():
        return None
    found: list[Path] = []
    frontier = [root]
    for _ in range(max_depth + 1):
        nxt = []
        for d in frontier:
            if is_checkpoint(d):
                found.append(d)
            nxt += sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
        frontier = nxt
    final = [d for d in found if not STEP.match(d.name)]
    if final:
        return min(final, key=lambda d: (len(d.relative_to(root).parts), str(d)))
    steps = [d for d in found if STEP.match(d.name)]
    return max(steps, key=lambda d: int(STEP.match(d.name).group(1))) if steps else None


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    hit = find_checkpoint(Path(sys.argv[1]))
    if hit is None:
        print(f"no checkpoint under {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)
    print(hit)


if __name__ == "__main__":
    main()
