#!/usr/bin/env bash
# Quick automated test script for Raspberry Pi 5 with Hailo-8 & ExecuTorch LLaMA
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "🚀 Edge QC Inspection Test Suite (Hailo-8 + LLaMA INT4)"
echo "=========================================================="

chmod +x llama_main 2>/dev/null || true

echo -e "\n[1/3] Testing TILE (Oil Defect)..."
python3 inference_llama_optimized.py \
    -m models/tile_hailo8.hef \
    -s sample_images/tile/defective_oil.png \
    --llama-model ./model_int4.pte \
    --llama-runner ./llama_main \
    --llama-tokenizer ./tokenizer.json \
    --llama-max-tokens 60

echo -e "\n[2/3] Testing BOTTLE (Contamination Defect)..."
python3 inference_llama_optimized.py \
    -m models/bottle_hailo8.hef \
    -s sample_images/bottle/defective_contamination.png \
    --llama-model ./model_int4.pte \
    --llama-runner ./llama_main \
    --llama-tokenizer ./tokenizer.json \
    --llama-max-tokens 60

echo -e "\n[3/3] Testing STEEL DEFECTS (Scratches)..."
python3 inference_llama_optimized.py \
    -m models/steel_defects_hailo8.hef \
    -s sample_images/steel_defects/scratches_3.jpg \
    --llama-model ./model_int4.pte \
    --llama-runner ./llama_main \
    --llama-tokenizer ./tokenizer.json \
    --llama-max-tokens 60

echo -e "\n=========================================================="
echo "✅ All tests completed! Annotated images saved to ./results/"
echo "=========================================================="
