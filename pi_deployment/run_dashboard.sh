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

# 3. Check for In-Memory Persistent Daemon (0s cold load, ~2.4s generation)
DAEMON_ACTIVE=false
if curl -s -m 1 "http://127.0.0.1:8080/health" &>/dev/null || curl -s -m 1 "http://127.0.0.1:8080/" &>/dev/null; then
    echo "⚡ [Llama Daemon] Connected to active In-Memory Daemon at http://127.0.0.1:8080 (0s startup latency, ~2.4s generation)!"
    DAEMON_ACTIVE=true
elif [ -f "${SCRIPT_DIR}/start_llama_daemon.sh" ]; then
    echo "🔍 Checking for Llama In-Memory Persistent Daemon..."
    if bash "${SCRIPT_DIR}/start_llama_daemon.sh" --auto-check; then
        echo "⚡ [Llama Daemon] Automatically launched Persistent Daemon on port 8080!"
        DAEMON_ACTIVE=true
    fi
fi

if [ "$DAEMON_ACTIVE" = false ]; then
    echo "ℹ️ [Llama] Persistent daemon not active on port 8080. Using ExecuTorch fallback (/dev/shm RAM disk)."
    echo "   Tip: Run './setup_llama_daemon.sh' to enable the ~2.4s In-Memory Daemon!"
fi

# 4. Fast-Path: Pre-stage ExecuTorch Llama model to /dev/shm (RAM disk) for zero cold load
if [ -d "/dev/shm" ]; then
    mkdir -p /dev/shm/executorch
    for candidate_dir in ~/executorch-deployment /home/raspberrypi5/executorch-deployment ~/Desktop/executorch-deployment ~/Desktop/executorch-1b ~/executorch-1b; do
        if [ -d "$candidate_dir" ]; then
            PTE_FILE=$(find "$candidate_dir" -maxdepth 1 -name "*.pte" 2>/dev/null | head -n 1)
            if [ -n "$PTE_FILE" ]; then
                PTE_NAME=$(basename "$PTE_FILE")
                if [ ! -f "/dev/shm/executorch/${PTE_NAME}" ]; then
                    echo "⚡ Pre-staging Llama model (${PTE_NAME}) into RAM disk (/dev/shm)..."
                    cp "$PTE_FILE" "/dev/shm/executorch/${PTE_NAME}"
                    echo "✅ Llama model resident in RAM (/dev/shm/executorch/${PTE_NAME})!"
                else
                    echo "⚡ Llama model already warm in RAM (/dev/shm/executorch/${PTE_NAME})"
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

# 5. Start Dashboard Server
echo "=================================================================="
echo "  Starting Edge Quality Control Dashboard on Raspberry Pi 5..."
echo "  Open in browser: http://localhost:5000 or http://$(hostname -I | awk '{print $1}'):5000"
echo "=================================================================="

PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/dashboard" python3 -m dashboard.app
