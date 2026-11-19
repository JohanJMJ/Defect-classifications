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
1. **Model Selector**: Switch between verified production models (`Tile`, `Bottle`, `Grid`).
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

## 5. On-Device Arm ExecuTorch Llama Advisory Engine

For the **ARM Bharat AI-SoC Challenge**, the system uses **Meta ExecuTorch** tuned specifically for 64-bit Arm Cortex-A architecture:

- **Model**: `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte`
- **Runner**: Native Arm64 `llama_main` binary (compiled with Arm XNNPACK / KleidiAI backend)
- **Token Speed**: **13.5 tokens/sec** on Raspberry Pi 5 quad-core Cortex-A76 CPU
- **Decode Time**: **~2.5 seconds** for standard 34-token Six-Sigma CQE root-cause assessment
- **Zero-Disk Cold Load**: Model weights (`.pte`) and tokenizer are mirrored to `/dev/shm` (POSIX RAM disk) at startup for pure DDR4/DDR5 execution.
- **CPU Resource Balance**: Configured for 3 execution threads (`--cpu_threads 3`), leaving 1 core dedicated to camera I/O, Hailo NPU DMA, and web serving.
- **Asynchronous Advisory**: Advisories execute in a background thread pool (`_async_llama_worker`), ensuring the Hailo-8 vision pipeline runs at full **35+ FPS** with zero conveyor blocking.

### ExecuTorch File Directory Layout:
Ensure your deployment files are placed at `~/executorch-deployment/`:
```text
~/executorch-deployment/
├── llama_main                                     # Native ARM64 ExecuTorch executable
├── Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte   # INT4 Quantized model weights
├── tokenizer.model                                # Tokenizer vocabulary
└── libllama_runner.so                             # Runtime dynamic shared library
```

*(The system automatically auto-discovers this directory, configures `LD_LIBRARY_PATH`, and mirrors files to `/dev/shm`).*


