#!/usr/bin/env python
"""Diagnose whether LlavaDataCollator labels land on assistant-response tokens
in the PRESENCE of <image> expansion (LlavaProcessor expands <image> to 576
tokens, raw tokenizer counts it as 1).
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from transformers import AutoProcessor
from dpvr.data import LlavaDataCollator
from dpvr.utils import load_config, load_paths, resolve_paths

cfg = load_config(str(REPO_ROOT / "configs/method/token_diversion.yaml"))
cfg = resolve_paths(cfg, load_paths())
model_path = cfg["model"]["base_model_path"]

processor = AutoProcessor.from_pretrained(model_path)
collator = LlavaDataCollator(processor, image_base_path="/tmp")

sample = {
    "image": None,  # 走灰图 fallback
    "conversations": [
        {"from": "human", "value": "<image>\nWhat color is dominant?"},
        {"from": "gpt",   "value": "Red."},
    ],
}
out = collator([sample])
labels = out["labels"][0]
input_ids = out["input_ids"][0]

print(f"input_ids length: {input_ids.shape[0]}")
print(f"unmasked positions count: {(labels != -100).sum().item()}")

unmasked_positions = (labels != -100).nonzero().flatten().tolist()
print(f"first 20 unmasked positions: {unmasked_positions[:20]}")
print(f"last 20 unmasked positions: {unmasked_positions[-20:]}")

unmasked_ids = input_ids[labels != -100].tolist()
decoded = processor.tokenizer.decode(unmasked_ids)
print(f"decoded unmasked tokens: {decoded!r}")

# What we WANT to see (if collator is correct):
#   decoded ≈ "Red.</s>" or "Red." or similar — the assistant response only
#
# What we EXPECT to see if bug exists:
#   decoded contains image-token placeholder IDs, fragments of "<image>" or
#   the user question, with maybe "Red." at the very end
