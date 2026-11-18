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

# 3. Start Dashboard Server
echo "=================================================================="
echo "  Starting Edge Quality Control Dashboard on Raspberry Pi 5..."
echo "  Open in browser: http://localhost:5000 or http://$(hostname -I | awk '{print $1}'):5000"
echo "=================================================================="

PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/dashboard" python3 -m dashboard.app
