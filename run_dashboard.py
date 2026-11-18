#!/usr/bin/env python3
"""
Launcher script for the Industrial Quality Control Assistant Dashboard.
Works on PC (Windows/Linux/macOS) in Simulation Mode and Raspberry Pi 5 in Native Mode.
"""

import sys
import os
from pathlib import Path

# Add pi_deployment to path
CURRENT_DIR = Path(__file__).resolve().parent
PI_DEPLOYMENT_DIR = CURRENT_DIR / "pi_deployment"

sys.path.insert(0, str(PI_DEPLOYMENT_DIR))
sys.path.insert(0, str(PI_DEPLOYMENT_DIR / "dashboard"))

if __name__ == "__main__":
    os.chdir(str(PI_DEPLOYMENT_DIR / "dashboard"))
    from dashboard.app import app, config, vision_engine

    import socket
    local_ip = "127.0.0.1"
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass

    print("==================================================================")
    print("  ON-DEVICE AI INDUSTRIAL QUALITY CONTROL ASSISTANT")
    print(f"  PC Browser URL        : http://localhost:{config.PORT}")
    print(f"  Mobile / Network URL  : http://{local_ip}:{config.PORT}")
    print(f"  Vision Engine         : {'HAILO-8 NPU (26 TOPS)' if not vision_engine.is_simulation_mode() else 'SIMULATION / PC MODE'}")
    print("==================================================================")
    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)
