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

# Parse arguments
ARGS=()
FIRST_ARG=true

while [ $# -gt 0 ]; do
    case "$1" in
        --list-models|-l)
            ARGS+=("--list-models")
            shift
            ;;
        -m|--model|-c|--category)
            ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            if [ "$FIRST_ARG" = true ] && [[ ! "$1" =~ ^- ]]; then
                ARGS+=("--model" "$1")
            else
                ARGS+=("$1")
            fi
            shift
            ;;
    esac
    FIRST_ARG=false
done

"${PYTHON_EXEC}" evaluate_hef.py \
    --images "dataset/images" \
    --labels "dataset/labels" \
    --output "results/evaluation_report.json" \
    "${ARGS[@]}"
