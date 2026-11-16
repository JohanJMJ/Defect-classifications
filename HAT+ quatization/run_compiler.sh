#!/usr/bin/env bash
# ==============================================================================
# Hailo Dataflow Compiler Runner for Raspberry Pi AI HAT+
# Bridges Ubuntu 20.04 host to Ubuntu 22.04 runtime (satisfying GLIBC 2.34+)
# ==============================================================================

set -e

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIFORGE_DIR="/home/acer/miniforge3"

# Ensure output directory exists
mkdir -p "${WORKSPACE_DIR}/hef_output"

echo "---------------------------------------------------------------"
echo "  Hailo Dataflow Compiler Runner (HAT+ / Raspberry Pi 5)"
echo "  Workspace: ${WORKSPACE_DIR}"
echo "---------------------------------------------------------------"

# Run inside Ubuntu 22.04 container with host user permissions
docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "${MINIFORGE_DIR}:${MINIFORGE_DIR}:ro" \
    -v "${WORKSPACE_DIR}:${WORKSPACE_DIR}" \
    -w "${WORKSPACE_DIR}" \
    -e HOME="${WORKSPACE_DIR}" \
    -e USER="$(whoami)" \
    -e VIRTUAL_ENV="${WORKSPACE_DIR}/hailo_venv" \
    ubuntu:22.04 \
    "${WORKSPACE_DIR}/hailo_venv/bin/python" compile_hef.py "$@"
