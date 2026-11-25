#!/usr/bin/env bash
# ==============================================================================
# Automated Setup Script for Raspberry Pi 5 + AI HAT+ (26 TOPS)
# Pure Native HailoRT (NO PyTorch / NO Ultralytics)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "================================================================"
echo "  Setting up Raspberry Pi 5 for Hailo-8 AI HAT+ (26 TOPS)"
echo "  Runtime: Pure Native HailoRT (No PyTorch / No Ultralytics)"
echo "================================================================"

# 1. Check OS architecture
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ]; then
    echo "Warning: Detected architecture '$ARCH'. Raspberry Pi OS 64-bit (aarch64) is strongly recommended."
fi

# 2. Check and install lightweight system packages
echo "Installing system packages (HailoRT drivers, python3-opencv, numpy)..."
sudo apt update
sudo apt install -y hailo-all python3-pip python3-venv python3-opencv python3-numpy libgl1

# 3. Check /boot/firmware/config.txt for PCIe enablement
CONFIG_FILE="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="/boot/config.txt"
fi

if [ -f "$CONFIG_FILE" ]; then
    if ! grep -q "^dtparam=pciex1" "$CONFIG_FILE"; then
        echo "Enabling PCIe in $CONFIG_FILE..."
        echo "dtparam=pciex1" | sudo tee -a "$CONFIG_FILE"
    fi
    if ! grep -q "^dtparam=pciex1_gen" "$CONFIG_FILE"; then
        echo "dtparam=pciex1_gen=3" | sudo tee -a "$CONFIG_FILE"
    fi
fi

# 4. Check /boot/firmware/cmdline.txt to prevent PCIe ASPM idle timeouts
CMDLINE_FILE="/boot/firmware/cmdline.txt"
if [ ! -f "$CMDLINE_FILE" ]; then
    CMDLINE_FILE="/boot/cmdline.txt"
fi

if [ -f "$CMDLINE_FILE" ]; then
    if ! grep -q "pcie_aspm=off" "$CMDLINE_FILE"; then
        echo "Adding pcie_aspm=off to $CMDLINE_FILE to prevent PCIe low-power sleep timeouts..."
        sudo sed -i.bak 's/$/ pcie_aspm=off/' "$CMDLINE_FILE"
    fi
fi

# 5. Create lightweight virtual environment (inheriting system HailoRT & OpenCV)
echo "Creating lightweight Python virtual environment..."
python3 -m venv --system-site-packages "${SCRIPT_DIR}/venv"

# 6. Ensure numpy, opencv, and web dashboard dependencies are ready
echo "Verifying lightweight Python dependencies..."
"${SCRIPT_DIR}/venv/bin/pip" install --upgrade pip
"${SCRIPT_DIR}/venv/bin/pip" install numpy opencv-python flask flask-cors pillow psutil pyyaml

# 7. Test Hailo hardware recognition
echo "Testing Hailo-8 hardware connection..."
if command -v hailortcli &> /dev/null; then
    if hailortcli fw-control identify; then
        echo ""
        echo "SUCCESS: Hailo-8 AI HAT+ is detected and operational!"
    else
        echo ""
        echo "Note: Hailo device did not respond. If this is the first install, please reboot your Pi:"
        echo "sudo reboot"
    fi
fi

echo ""
echo "================================================================"
echo "  Setup Complete! Zero PyTorch / Zero Ultralytics installed."
echo "  To run inference:"
echo "  ./run_inference.sh"
echo "================================================================"
