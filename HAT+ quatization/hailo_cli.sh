#!/usr/bin/env bash
# ==============================================================================
# Hailo Dataflow Compiler CLI Wrapper
# Executes Hailo CLI commands inside the compatible container environment
# ==============================================================================

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIFORGE_DIR="/home/acer/miniforge3"

# Check if TTY is available
INTERACTIVE_FLAG=""
if [ -t 0 ]; then
    INTERACTIVE_FLAG="-it"
fi

docker run --rm ${INTERACTIVE_FLAG} \
    -u "$(id -u):$(id -g)" \
    -v "${MINIFORGE_DIR}:${MINIFORGE_DIR}:ro" \
    -v "${WORKSPACE_DIR}:${WORKSPACE_DIR}" \
    -w "${WORKSPACE_DIR}" \
    -e HOME="${WORKSPACE_DIR}" \
    -e USER="$(whoami)" \
    -e VIRTUAL_ENV="${WORKSPACE_DIR}/hailo_venv" \
    ubuntu:22.04 \
    "${WORKSPACE_DIR}/hailo_venv/bin/hailo" "$@"
