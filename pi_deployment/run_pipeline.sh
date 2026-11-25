#!/usr/bin/env bash
# Automated test script for Raspberry Pi 5 with Hailo-8 & ExecuTorch LLaMA
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "Edge QC Inspection Test Suite (Hailo-8 + LLaMA INT4)"
echo "=========================================================="

LLAMA_DIR="/home/raspberrypi5/executorch-new"
if [ ! -d "$LLAMA_DIR" ]; then
    LLAMA_DIR="$HOME/executorch-new"
fi
if [ ! -d "$LLAMA_DIR" ]; then
    LLAMA_DIR="$SCRIPT_DIR"
fi

LLAMA_RUNNER="$LLAMA_DIR/llama_main"
chmod +x "$LLAMA_RUNNER" 2>/dev/null || true

PTE_FILE=$(find "$LLAMA_DIR" -maxdepth 1 -name "*.pte" 2>/dev/null | head -n 1)
TOK_FILE=$(find "$LLAMA_DIR" -maxdepth 1 -name "*tokenizer*" 2>/dev/null | head -n 1)

EXTRA_LLAMA_ARGS=()
if [ -f "$LLAMA_RUNNER" ]; then
    EXTRA_LLAMA_ARGS+=(--llama-runner "$LLAMA_RUNNER")
fi
if [ -n "$PTE_FILE" ]; then
    EXTRA_LLAMA_ARGS+=(--llama-model "$PTE_FILE")
fi
if [ -n "$TOK_FILE" ]; then
    EXTRA_LLAMA_ARGS+=(--llama-tokenizer "$TOK_FILE")
fi

echo -e "\n[1/3] Testing TILE (Oil Defect)..."
python3 inference_llama_optimized.py \
    -m models/tile_hailo8.hef \
    -s sample_images/tile/defective_oil.png \
    --llama-max-tokens 60 \
    "${EXTRA_LLAMA_ARGS[@]}"

echo -e "\n[2/3] Testing BOTTLE (Contamination Defect)..."
python3 inference_llama_optimized.py \
    -m models/bottle_hailo8.hef \
    -s sample_images/bottle/defective_contamination.png \
    --llama-max-tokens 60 \
    "${EXTRA_LLAMA_ARGS[@]}"

echo -e "\n[3/3] Testing STEEL DEFECTS (Scratches)..."
python3 inference_llama_optimized.py \
    -m models/steel_defects_hailo8.hef \
    -s sample_images/steel_defects/scratches_3.jpg \
    --llama-max-tokens 60 \
    "${EXTRA_LLAMA_ARGS[@]}"

echo -e "\n=========================================================="
echo "All tests completed! Annotated images saved to ./results/"
echo "=========================================================="
