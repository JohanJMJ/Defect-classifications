#!/usr/bin/env bash
# ==============================================================================
# One-Click Dashboard Launcher for Raspberry Pi 5 + Hailo-8 AI HAT+
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 1. Activate virtual environment if present
if [ -f "${SCRIPT_DIR}/venv/bin/activate" ]; then
    echo "Activating virtual environment: ${SCRIPT_DIR}/venv"
    source "${SCRIPT_DIR}/venv/bin/activate"
fi

# 2. Check and ensure Flask is installed
if ! python3 -c "import flask" &> /dev/null; then
    echo "Installing dashboard dependencies..."
    pip install flask flask-cors pillow psutil pyyaml
fi

# 3. Fast-Path: Pre-stage ExecuTorch Llama model to /dev/shm (RAM disk) for zero cold load
if [ -d "/dev/shm" ]; then
    mkdir -p /dev/shm/executorch
    for candidate_dir in ~/executorch-deployment /home/raspberrypi5/executorch-deployment ~/Desktop/executorch-deployment ~/Desktop/executorch-1b ~/executorch-1b; do
        if [ -d "$candidate_dir" ]; then
            PTE_FILE=$(find "$candidate_dir" -maxdepth 1 -name "*.pte" 2>/dev/null | head -n 1)
            if [ -n "$PTE_FILE" ]; then
                PTE_NAME=$(basename "$PTE_FILE")
                if [ ! -f "/dev/shm/executorch/${PTE_NAME}" ]; then
                    echo "⚡ Pre-staging ExecuTorch Llama model (${PTE_NAME}) into RAM disk (/dev/shm)..."
                    cp "$PTE_FILE" "/dev/shm/executorch/${PTE_NAME}"
                    echo "✅ ExecuTorch model resident in RAM (/dev/shm/executorch/${PTE_NAME})!"
                else
                    echo "⚡ ExecuTorch model already warm in RAM (/dev/shm/executorch/${PTE_NAME})"
                fi
            fi
            TOK_FILE=$(find "$candidate_dir" -maxdepth 1 -name "*tokenizer*" 2>/dev/null | head -n 1)
            if [ -n "$TOK_FILE" ]; then
                TOK_NAME=$(basename "$TOK_FILE")
                if [ ! -f "/dev/shm/executorch/${TOK_NAME}" ]; then
                    cp "$TOK_FILE" "/dev/shm/executorch/${TOK_NAME}"
                fi
            fi
            break
        fi
    done
fi

# 4. Start Dashboard Server
echo "=================================================================="
echo "  Starting Edge Quality Control Dashboard on Raspberry Pi 5..."
echo "  Runtime: Hailo-8 NPU (Vision) + Arm ExecuTorch (Llama 3.2 1B)"
echo "  Open in browser: http://localhost:5000 or http://$(hostname -I | awk '{print $1}'):5000"
echo "=================================================================="

PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/dashboard" python3 -m dashboard.app
