#!/usr/bin/env bash
# Download all image sets required by the LLaVA-1.5 665K instruction dataset.
#
# Reference: https://github.com/haotian-liu/LLaVA#visual-instruction-tuning
#
# Image distribution in llava_v1_5_mix665k.json:
#   coco     364,100  54.7%
#   vg        86,417  13.0%
#   ocr_vqa   80,000  12.0%
#   gqa       72,140  10.8%
#   textvqa   21,953   3.3%
#   (text-only 40,688  filtered out at training time)
#
# Usage:
#   bash scripts/download_llava_images.sh [DATA_ROOT]
#
# DATA_ROOT defaults to /path/to/llava-1.5-665k-instructions

set -euo pipefail

DATA_ROOT="${1:-/path/to/llava-1.5-665k-instructions}"
echo "[llava-images] DATA_ROOT = $DATA_ROOT"
mkdir -p "$DATA_ROOT"

# ── 1. COCO train2017 (~18 GB) ───────────────────────────────────────────────
# Target: $DATA_ROOT/coco/train2017/*.jpg
download_coco() {
    local dest="$DATA_ROOT/coco"
    mkdir -p "$dest"
    if [ -d "$dest/train2017" ] && [ "$(ls -A "$dest/train2017" 2>/dev/null | wc -l)" -gt 100000 ]; then
        echo "[coco] Already extracted, skipping."
        return
    fi
    local zip="$dest/train2017.zip"
    if [ ! -f "$zip" ] || [ "$(stat -c%s "$zip" 2>/dev/null || echo 0)" -lt 1000000 ]; then
        echo "[coco] Downloading train2017.zip …"
        wget -c "http://images.cocodataset.org/zips/train2017.zip" \
             -O "$zip" --progress=dot:giga
    fi
    echo "[coco] Extracting …"
    unzip -q "$zip" -d "$dest"
    echo "[coco] Done. $(ls "$dest/train2017" | wc -l) images."
}

# ── 2. GQA images (~20 GB) ───────────────────────────────────────────────────
# Target: $DATA_ROOT/gqa/images/*.jpg
download_gqa() {
    local dest="$DATA_ROOT/gqa"
    mkdir -p "$dest"
    if [ -d "$dest/images" ] && [ "$(ls -A "$dest/images" 2>/dev/null | wc -l)" -gt 100000 ]; then
        echo "[gqa] Already present, skipping."
        return
    fi
    local zip="$dest/images.zip"
    echo "[gqa] Downloading images.zip (Stanford) …"
    wget -c "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip" \
         -O "$zip" --progress=dot:giga
    echo "[gqa] Extracting …"
    unzip -q "$zip" -d "$dest"
    echo "[gqa] Done."
}

# ── 3. OCR-VQA images (~13 GB) ───────────────────────────────────────────────
# Target: $DATA_ROOT/ocr_vqa/images/*.jpg
# Source: HuggingFace mirror (lmms-lab/OCR-VQA or direct script)
download_ocr_vqa() {
    local dest="$DATA_ROOT/ocr_vqa"
    mkdir -p "$dest"
    if [ -d "$dest/images" ] && [ "$(ls -A "$dest/images" 2>/dev/null | wc -l)" -gt 50000 ]; then
        echo "[ocr_vqa] Already present, skipping."
        return
    fi
    echo "[ocr_vqa] Downloading via HuggingFace mirror …"
    HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
        howard-hou/OCR-VQA --repo-type dataset \
        --local-dir "$dest/hf_raw"
    # Images are typically in images/ subdir of the dataset
    if [ -d "$dest/hf_raw/images" ]; then
        ln -sfn "$dest/hf_raw/images" "$dest/images"
    fi
    echo "[ocr_vqa] Done."
}

# ── 4. TextVQA images (~6 GB) ────────────────────────────────────────────────
# Target: $DATA_ROOT/textvqa/train_images/*.jpg
download_textvqa() {
    local dest="$DATA_ROOT/textvqa"
    mkdir -p "$dest"
    if [ -d "$dest/train_images" ] && [ "$(ls -A "$dest/train_images" 2>/dev/null | wc -l)" -gt 20000 ]; then
        echo "[textvqa] Already present, skipping."
        return
    fi
    local zip="$dest/train_images.zip"
    echo "[textvqa] Downloading train_images.zip …"
    wget -c "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" \
         -O "$zip" --progress=dot:giga
    echo "[textvqa] Extracting …"
    unzip -q "$zip" -d "$dest"
    echo "[textvqa] Done."
}

# ── 5. Visual Genome (VG) images (~15 GB, 2 parts) ───────────────────────────
# Target: $DATA_ROOT/vg/VG_100K/*.jpg and $DATA_ROOT/vg/VG_100K_2/*.jpg
download_vg() {
    local dest="$DATA_ROOT/vg"
    mkdir -p "$dest"
    if [ -d "$dest/VG_100K" ] && [ "$(ls -A "$dest/VG_100K" 2>/dev/null | wc -l)" -gt 50000 ]; then
        echo "[vg] Already present, skipping."
        return
    fi
    for part in 1 2; do
        local zip="$dest/images_part${part}.zip"
        echo "[vg] Downloading part ${part}/2 …"
        wget -c "https://cs.stanford.edu/people/rak248/VG_100K_2/images${part}.zip" \
             -O "$zip" --progress=dot:giga
        echo "[vg] Extracting part ${part} …"
        unzip -q "$zip" -d "$dest"
    done
    echo "[vg] Done."
}

# ── Main ─────────────────────────────────────────────────────────────────────
echo "===== LLaVA-1.5 665K image download ====="
download_coco
download_gqa
download_ocr_vqa
download_textvqa
download_vg
echo "===== All downloads attempted. Run scripts/prepare_data.py to verify. ====="
