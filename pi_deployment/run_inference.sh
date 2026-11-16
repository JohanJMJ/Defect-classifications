#!/usr/bin/env bash
# ==============================================================================
# One-Click Inference Launcher for Raspberry Pi AI HAT+ (Hailo-8)
# Pure Native HailoRT (ZERO PyTorch / ZERO Ultralytics)
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
echo "  Hailo-8 Inspection System — Raspberry Pi AI HAT+ (26 TOPS)"
echo "  Runtime: Pure Native HailoRT (Zero PyTorch / Zero Ultralytics)"
echo "================================================================"

# Check if hailortcli is available
if command -v hailortcli &> /dev/null; then
    echo "🔍 Checking Hailo NPU Hardware Status..."
    if hailortcli fw-control identify 2>&1 | grep -q "HAILO8"; then
        echo "✅ Hailo-8 NPU Detected and Ready!"
    else
        echo "⚠️  Warning: Hailo-8 not responding to hailortcli identify. Checking driver..."
        dmesg | grep -i hailo | tail -n 5 || true
    fi
else
    echo "ℹ️  'hailortcli' command not found. Proceeding with Python HailoRT..."
fi

# Ensure output directory exists
mkdir -p "${SCRIPT_DIR}/results"

# Parse command line flags
ARGS=()
for arg in "$@"; do
    case "$arg" in
        --camera)
            ARGS+=("--source" "0" "--show")
            ;;
        *)
            ARGS+=("$arg")
            ;;
    esac
done

# If no arguments provided, default to inspecting sample_images
if [ ${#ARGS[@]} -eq 0 ]; then
    echo "🚀 Running defect inspection on 'sample_images'..."
    "${PYTHON_EXEC}" inference.py \
        --model "models/tile_hailo8.hef" \
        --source "sample_images" \
        --output-dir "results"
else
    "${PYTHON_EXEC}" inference.py \
        --model "models/tile_hailo8.hef" \
        "${ARGS[@]}"
fi
