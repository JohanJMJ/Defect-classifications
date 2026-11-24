#!/usr/bin/env bash
# Inference Launcher for Raspberry Pi AI HAT+ (Hailo-8)

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
    echo "Checking Hailo NPU Hardware Status..."
    if hailortcli fw-control identify 2>&1 | grep -q "HAILO8"; then
        echo "Hailo-8 NPU Detected and Ready!"
    else
        echo "Warning: Hailo-8 not responding to hailortcli identify. Checking driver..."
        dmesg | grep -i hailo | tail -n 5 || true
    fi
else
    echo "'hailortcli' command not found. Proceeding with Python HailoRT..."
fi

# Ensure output directory exists
mkdir -p "${SCRIPT_DIR}/results"

# Parse command line flags
ARGS=()
FIRST_ARG=true

while [ $# -gt 0 ]; do
    case "$1" in
        --camera)
            ARGS+=("--source" "0" "--show")
            shift
            ;;
        --list-models|-l)
            ARGS+=("--list-models")
            shift
            ;;
        -m|--model|-c|--category)
            ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            # If the first argument is not a flag (e.g. "bottle" or "bottle_hailo8.hef"), treat it as --model
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

# Run inference engine with arguments
"${PYTHON_EXEC}" inference.py "${ARGS[@]}"
