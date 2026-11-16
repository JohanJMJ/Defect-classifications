#!/usr/bin/env bash
# ==============================================================================
# One-Click Accuracy & mAP Evaluation for Raspberry Pi AI HAT+ (Hailo-8)
# Evaluates Box & Mask mAP across the complete validation dataset
# ZERO PyTorch / ZERO Ultralytics required.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Find Python executable (prefer local venv if exists)
if [ -f "${SCRIPT_DIR}/venv/bin/python" ]; then
    PYTHON_EXEC="${SCRIPT_DIR}/venv/bin/python"
elif [ -f "/home/$(whoami)/hailo_venv/bin/python" ]; then
    PYTHON_EXEC="/home/$(whoami)/hailo_venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "================================================================"
echo "  Hailo-8 mAP & Accuracy Evaluation (Raspberry Pi AI HAT+)"
echo "  Runtime: Pure Native HailoRT"
echo "================================================================"

mkdir -p "${SCRIPT_DIR}/results"

"${PYTHON_EXEC}" evaluate_hef.py \
    --model "models/tile_hailo8.hef" \
    --images "dataset/images" \
    --labels "dataset/labels" \
    --output "results/evaluation_report.json" \
    "$@"
