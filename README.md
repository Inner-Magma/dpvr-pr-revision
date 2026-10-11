# DPVR: Dual-Path Vision Token Routing (`dpvr`)

**English** | [简体中文](README.zh-CN.md)

> In LLaVA-1.5, vision-token hidden states change little across the deep layers
> (adjacent-layer cosine ≥ 0.92; text-to-image attention mass ≈ 0.04 after layer
> 18), while text tokens keep changing. DPVR-LF routes vision tokens at layer
> *s* through a single trainable side-branch block, runs the deep layers on text
> tokens only, and fuses the two streams again in the last layer. With about 3 %
> trainable parameters it stays competitive with full fine-tuning on standard
> benchmarks (per-benchmark numbers and gaps in the paper's Table 2) and reduces
> measured prefill latency of LLaVA-1.5-7B on an A800 by about 28 %.

Reference implementation for the **Dual-Path Vision Token Routing (DPVR)** method
and its core instantiation **DPVR-LF (Late-Layer Fusion)** — a *Pattern
Recognition* journal submission. Base model: LLaVA-1.5 (Vicuna + CLIP-ViT-L/14-336);
only one decoder block (~3 % of parameters, 202 M for the 7B model) is trained.

---

## Method

### DPVR-LF (Late-Layer Fusion) — the proposed method ★

```
input ─► layers 0…17 (frozen, shared)  ─► split @ s=18
            │
            ├─► single_transformer  (1 trainable side-branch block)
            │        └─ produces the image representation
            │
            └─► layers 18…30  (frozen): TEXT-ONLY forward
            │     image positions skipped — no Q/K/V/O/FFN computed for image
            │     tokens in these 13 layers
            │
            └─► layer 31  (frozen): single IMAGE-TEXT FUSION
                  image positions reassembled from the side-branch output,
                  then one full-attention block fuses the two streams ─► lm_head
```

- Image tokens are skipped in the 13 deep layers and re-fused only at the last
  layer, so unlike the per-layer routing variants below these layers do no work
  for image tokens. The vision tower, the shallow layers and the fusion layer
  still process them, so the end-to-end saving is smaller than 13/32 of the
  compute (about 28 % measured prefill latency for 7B).
- For the 7B model (32 layers) fusion happens at the final layer, index **31**;
  for the 13B model (40 layers) the main model uses `s = 28`, with `s ∈ {20, 24, 28, 34}`
  in the split sweep.
- In the fusion-count ablation, a second fusion layer (K = 2) adds about 0.18 pp
  on average over K = 1, so the released configuration uses one fusion layer.
- Decoding note: the DPVR-LF generation path in this release recomputes the
  full sequence at every step (no KV-cache reuse), so per-token decoding is
  slower than vanilla LLaVA-1.5. The latency numbers above are for the prefill
  forward pass.
- Implementation: [`src/dpvr/models/token_diversion_x3_fusion.py`](src/dpvr/models/token_diversion_x3_fusion.py)
- Config: [`configs/method/token_diversion_x3_fusion.yaml`](configs/method/token_diversion_x3_fusion.yaml)

All DPVR variants share the frozen shallow stack (`L0`–`L17`) and the one-layer
trainable side branch; they differ only in how image positions are handled in the
deep stack. DPVR-LF is the configuration used for the headline results.

### Routing baselines / ablations

These are the two predecessor routing strategies, kept for the architectural
ablation (Figure 3 in the paper):

| Variant | Deep-stack handling of image tokens | Compute saving | Code |
|---------|--------------------------------------|----------------|------|
| **DPVR-PC** (per-layer copy) | image positions reset to the side-branch output at **every** deep layer; full attention still runs | none | [`token_diversion.py`](src/dpvr/models/token_diversion.py) |
| **DPVR-KV** (cached-KV substitution) | image positions contribute **only K/V** in the deep stack | partial (~25–35 %) | [`token_diversion_substitution.py`](src/dpvr/models/token_diversion_substitution.py) |
| **DPVR-LF** (late-layer fusion) ★ | image positions skipped for 13 layers, fused only at the last layer | largest of the three | [`token_diversion_x3_fusion.py`](src/dpvr/models/token_diversion_x3_fusion.py) |

### Shared setup

- Base: LLaVA-1.5 (Vicuna + CLIP-ViT-L/14-336)
- Trainable: one `LlamaDecoderLayer` side branch (~3 % of parameters; 202 M for 7B)
- Precision: bf16; labels mask user-turn and image-patch tokens
- Data: LLaVA-1.5-mix-665k (identical dataset across variants for a fair comparison)

---

## Installation

```bash
git clone https://github.com/Inner-Magma/Dual-Path-Token-Routing.git
cd Dual-Path-Token-Routing
pip install -e .            # add [eval,analysis,profile] for optional extras
```

Configure paths for your machine (all absolute paths in the code resolve from these roots):

```bash
cp .env.example .env                                # or:
cp configs/paths.yaml.example configs/paths.yaml    # fill DATA_ROOT / MODEL_ROOT / CKPT_ROOT
```

## Data

LLaVA-1.5-mix-665k instructions + images. Download images with:

```bash
bash scripts/download_llava_images.sh /path/to/llava-1.5-665k-instructions
```

## Training

```bash
# DPVR-LF (proposed, late-layer fusion) — the main model:
python scripts/train.py --config configs/method/token_diversion_x3_fusion.yaml

# Routing baselines / ablations:
python scripts/train.py --config configs/method/token_diversion.yaml                # DPVR-PC
python scripts/train.py --config configs/method/token_diversion_substitution.yaml   # DPVR-KV
```

Override any config key on the command line (resume / ablation example):

```bash
python scripts/train.py --config configs/method/token_diversion_x3_fusion.yaml \
    method.split_layer=24 \
    training.output_subdir=token_diversion_x3_fusion/ablate_split_24 \
    training.resume_from_checkpoint=/path/to/checkpoint-NNNNN
```

13B variants and the fusion-count / split-layer ablations live alongside in
[`configs/method/`](configs/method/) (e.g. `token_diversion_x3_fusion_13b_split24_d1.yaml`,
`token_diversion_x3_fusion_last2.yaml`). Other baselines (`full_ft`, `lora`,
`fastv`) are there too. Ablation sweeps are in [`sweeps/`](sweeps/).

## Evaluation

Evaluation uses [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) 0.5.0
from PyPI (pinned in `requirements-lock.txt`). Task configs are under
[`configs/lmms_tasks/`](configs/lmms_tasks/); benchmarks are read from
`${DATA_ROOT}/benchmarks` (or `BENCH_ROOT`), and
`scripts/revision/download_assets.py benchmarks` fetches them there.

```bash
python scripts/revision/eval_lmms.py --checkpoint /path/to/checkpoints/token_diversion_x3_fusion/... \
    --tasks eight --out /path/to/output
```

> Note: pass `use_cache=False` for all lmms-eval runs. The ScienceQA metric key
> is `mc_accuracy,none`.

---

## Repository layout

```
src/dpvr/
  models/        token_diversion_x3_fusion (DPVR-LF, proposed),
                 token_diversion (DPVR-PC), token_diversion_substitution (DPVR-KV),
                 td_fastv, and baselines (full_ft, lora, fastv)
  training/      trainer, freezing, callbacks
  evaluation/    lmms-eval adapter, native runner
  analysis/      KL-divergence, CKA, hidden-state probes
  profiling/     FLOPs, params, VRAM, GPU-hours
  visualization/ attention maps, t-SNE
configs/         _base + data / method / eval / lmms_tasks ablations
scripts/         train.py, evaluate.py, prepare_data.py, profiling & analysis entry points
sweeps/          split-layer / vision-depth / data-scale ablation drivers
tests/           forward-parity, collator, checkpoint round-trip
```

---

## Citation

If you use this code, please cite the paper (citation to be updated on acceptance):

```bibtex
@article{liu2026dpvr,
  title   = {Late-Layer Fusion is Enough: Dual-Path Vision Token Routing for
             Multimodal Large Language Models under Visual Saturation},
  author  = {Liu, Siyuan and Wu, Jinyang},
  journal = {Pattern Recognition (under review)},
  year    = {2026}
}
```

## License

[MIT](LICENSE).
