#!/usr/bin/env python3
"""
Convenience entrypoint for the Industrial Edge Power Profiler.
Directs execution to pi_deployment/measure_power.py.
"""
import sys
from pathlib import Path

script = Path(__file__).resolve().parent / "pi_deployment" / "measure_power.py"
if not script.exists():
    print(f"Error: {script} not found.")
    sys.exit(1)

with open(script, "rb") as f:
    code = compile(f.read(), str(script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(script)})
