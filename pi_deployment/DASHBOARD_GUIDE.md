# Edge Industrial Quality Control Assistant - Dashboard Guide

This guide explains how to test the dashboard locally on your PC (in Simulation Mode) and how to deploy and host it on the Raspberry Pi 5 with Hailo-8 AI HAT+.

---

## 1. Architecture & File Layout

All deployable web, backend, and model files live inside `pi_deployment/`:

```text
pi_deployment/
├── backend/
│   ├── hailo_engine.py       # Dual-mode Hailo-8 NPU / PC Simulation vision engine
│   ├── llama_engine.py       # Multi-backend Llama advisory (ExecuTorch / Server / Simulation)
│   ├── rule_engine.py        # Deterministic fast-path QC rules (< 0.1 ms triage)
│   ├── camera_streamer.py    # USB / CSI Camera streamer & MJPEG preview feed
│   ├── alert_manager.py      # Physical GPIO LEDs & Buzzer alerts (with PC safe mode)
│   └── database.py           # SQLite database for audit trails & CSV export
├── dashboard/
│   ├── app.py                # Main Flask application & REST endpoints
│   ├── config.py             # Server config, hardware telemetry, and model registry
│   ├── static/
│   │   ├── css/dashboard.css # Industrial high-contrast dark theme
│   │   └── js/dashboard.js   # Interactive UI, sample picker, and dual-AI displays
│   ├── templates/
│   │   ├── base.html         # Shell layout with top hardware HUD
│   │   ├── index.html        # Main visual inspection studio
│   │   └── history.html      # Audit log with search, filters, and CSV download
│   └── data/
│       ├── inspections.db    # SQLite database
│       └── uploads/          # Saved raw captures and AI-annotated overlays
├── models/                   # 9 Pre-compiled Hailo-8 HEF executables (26 TOPS)
├── sample_images/            # Categorized test images for each model
├── service/
│   └── qc-dashboard.service  # Systemd service for auto-start on Pi boot
├── setup_pi.sh               # 1-click setup script for Raspberry Pi OS
└── run_dashboard.sh          # 1-click launcher for Raspberry Pi
```

---

## 2. Testing on Your PC (Simulation Mode)

On your PC, you do not need the Hailo-8 hardware or ExecuTorch installed. The application detects when it is on a PC and automatically runs in **Simulation Mode**:
- **Vision Engine**: Analyzes sample images or uploaded photos, generates bounding boxes, polygon segmentation masks, and simulates Hailo-8 latency (~11.8 ms).
- **Llama Advisory**: Produces intelligent, structured 3-point QC advisories based on category and defect metrics.
- **Alerts**: Safely mocks GPIO pins and buzzer beeps in software without errors.

### How to Run on PC:
In your project directory terminal:
```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Run dashboard launcher
python run_dashboard.py
```

Open your browser to:
**`http://localhost:5000`**

### What to test in the UI:
1. **Model Selector**: Switch between `Tile`, `Bottle`, `Cable`, `Capsule`, `Carpet`, `Grid`, `Hazelnut`, `Pill`, `Zipper`.
2. **Sample Test Library tab**: Click any sample image (`defective_crack.png`, `good.png`, etc.) to run an instant inspection.
3. **Upload tab**: Drag and drop any component image from your computer.
4. **Camera tab**: View live webcam stream and click **Capture & Inspect Frame**.
5. **Viewfinder Controls**: Toggle the **Overlay Masks** switch to compare raw image vs. AI-annotated mask and bounding box.
6. **Audit Log**: Click **Audit Log** in the header to view all logged inspections, filter by decision (PASS / MONITOR / REJECT), view details modal, and click **Export CSV**.

---

## 3. Deploying and Hosting on Raspberry Pi 5

When you are ready to move to the Pi:

### Step 1: Copy `pi_deployment` to the Pi
From your PC terminal:
```bash
scp -r pi_deployment pi@<raspberry-pi-ip>:~/
```

### Step 2: Run Setup on the Pi
SSH into the Raspberry Pi:
```bash
ssh pi@<raspberry-pi-ip>
cd ~/pi_deployment
chmod +x setup_pi.sh run_dashboard.sh
./setup_pi.sh
```
*(If prompted to enable PCIe Gen 3 for the AI HAT+, reboot once with `sudo reboot`)*.

### Step 3: Launch Dashboard
```bash
cd ~/pi_deployment
./run_dashboard.sh
```

The server binds to `0.0.0.0:5000`. You can now access the dashboard from:
- **On the Pi itself**: `http://localhost:5000`
- **From your laptop or tablet on the same Wi-Fi / LAN**: `http://<raspberry-pi-ip>:5000` or `http://raspberrypi.local:5000`

---

## 4. Setting up Auto-Start on Boot (Factory Operation)

To make the dashboard launch automatically when the Raspberry Pi powers up:

```bash
# Copy systemd unit file
sudo cp ~/pi_deployment/service/qc-dashboard.service /etc/systemd/system/

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable qc-dashboard.service
sudo systemctl start qc-dashboard.service
```

To check service status:
```bash
sudo systemctl status qc-dashboard.service
```

---

## 5. In-Memory Persistent Daemon (~2.4s Llama Advisory)

By default, running Llama per-frame via ExecuTorch incurs ~4.0s of process initialization and memory allocation overhead on every defect, resulting in ~6.5s–8.2s total latency. 

The **In-Memory Persistent Daemon** keeps the model weights, KV cache, and thread pool permanently resident in RAM:
- **Cold-load latency**: 0.0s (Permanently eliminated)
- **Prompt evaluation**: < 50ms with prompt caching (`--prompt-cache`)
- **Total advisory time**: **~2.4 seconds** (Down from ~8.2s)
- **Idle CPU**: 0.0% (Sleeps on local socket)

### 1-Click Setup on Raspberry Pi 5:
```bash
cd ~/pi_deployment
chmod +x setup_llama_daemon.sh start_llama_daemon.sh stop_llama_daemon.sh
./setup_llama_daemon.sh
```
This compiles native `llama-server` tuned for ARM64 Cortex-A76, downloads the 740 MB quantized `Llama-3.2-1B-Instruct-Q4_K_M.gguf` model, tests the HTTP API, and verifies latency.

### Starting and Stopping the Daemon:
```bash
# Start in background (allocates ~700MB in RAM, binds to port 8080)
./start_llama_daemon.sh

# Stop daemon and free RAM
./stop_llama_daemon.sh
```

### Auto-Integration with Dashboard & CLI:
- `./run_dashboard.sh` automatically detects if the daemon is active on port 8080 and routes queries to it. If not running, it gracefully falls back to ExecuTorch.
- `python3 inference_llama_optimized.py` defaults to `--llama-backend auto`, automatically connecting to the persistent daemon if running.

### Optional: Run Daemon on Boot via Systemd:
```bash
sudo cp ~/pi_deployment/service/llama-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llama-daemon.service
```

