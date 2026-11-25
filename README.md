# Industrial Edge AI Quality Control Assistant
### Real-Time Hailo-8 Vision Defect Segmentation & On-Device ExecuTorch Llama Advisory

An industrial-grade, edge-deployed machine vision and AI quality control system operating **100% on-device with zero cloud dependencies**. Designed for high-speed manufacturing conveyor lines running on **Raspberry Pi 5 with Raspberry Pi AI HAT+ (26 TOPS Hailo-8 NPU)**, backed by an intelligent on-device Small Language Model (**Llama 3.2 1B INT4 on ExecuTorch**) for root-cause failure mode assessment and disposition action items.

It also includes PC Simulation Mode for local development, demo, and testing on Windows, macOS, or Linux.

---

## System Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │             Conveyor Camera / Optical Sensor           │
       └───────────────────────────┬────────────────────────────┘
                                   │ Raw Frame (BGR)
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │         Hailo-8 AI HAT+ (26 TOPS PCIe M.2 NPU)         │
       │   YOLO Models: YOLO11-seg / YOLOv8-P2 (Native HailoRT) │
       │   ~22 ms Latency | Bounding Boxes | Polygon Masks      │
       └───────────────────────────┬────────────────────────────┘
                                   │ Anomaly BBox, Mask %, Certainty
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │      Neuro-Fuzzy Triage Engine (< 0.05 ms)             │
       │   Continuous Risk Score (0-100) | Fast-Path Bypass     │
       └───────────────────────────┬────────────────────────────┘
                                   │ Structured Vision Telemetry
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │     Raspberry Pi 5 ARM CPU (ExecuTorch Runtime)        │
       │       Llama 3.2 1B Instruct SpinQuant INT4 SLM         │
       │   • Severity: [Minor / Moderate / Major / Critical]    │
       │   • Assessment: Technical failure mode & material risk │
       │   • Action: 1. Disposition, 2. Quarantine, 3. CAPA     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │    Modern Industrial Web Dashboard (Port 5000)         │
       │  Decoupled 2-Stage UI | Live Stream | Audit Database   │
       └────────────────────────────────────────────────────────┘
```

---

## Key Highlights

- **Ultra-Fast Edge Vision AI**: Native HailoRT (`hailo_platform`) PCIe inference executes in **~22 ms** (~45+ FPS) with zero PyTorch or Ultralytics runtime overhead.
- **High-Accuracy HEF Industrial Models**: Pre-quantized and compiled for Hailo-8 (`steel_defects`, `bottle`, `grid`, `tile`).
- **Decoupled Asynchronous LLM Architecture**: Hailo-8 vision segmentation masks and pass/reject verdicts display **immediately (< 25 ms)**, while Llama 3.2 1B INT4 resolves root-cause advice asynchronously (**~2.5s**) with an animated UI progress indicator.
- **Continuous Conveyor Mode**: Automated cycle inspection (1s to 10s intervals) with sub-millisecond deterministic triage bypass for non-stop manufacturing lines.
- **Self-Healing Hardware Pipeline**: Thread-safe mutex locking (`RLock`) and automatic PCIe reset (`hailortcli fw-control reset`) with instant 1-shot retry to eliminate PCIe timeouts.
- **100% Offline Standalone Operation**: Zero cloud connections, zero external font CDNs, local SQLite database, and automated CSV audit logging.
- **Neuro-Fuzzy Edge Triage**: Continuous 0–100 risk scoring evaluating defect area percentage, model confidence, and hazard severity with sub-millisecond fast-path bypass for minor blemishes.
- **Heterogeneous Dual-Compute Pipeline**: Hardware acceleration split between Hailo-8 NPU (vision inference in ~22 ms) and Raspberry Pi 5 ARM Cortex-A76 CPU (LLaMA 3.2 1B INT4 on ExecuTorch via `/dev/shm` RAM disk). For complete mathematical formulations, severity tiers, and hardware benchmarks, see [pi_deployment/README.md](pi_deployment/README.md).

---

## Repository Structure

```text
Defect-classifications/
├── run_dashboard.py        # 1-Click launcher for PC Development (Simulation Mode)
├── test_dashboard.py       # Automated test suite & health check
├── requirements.txt        # Core Python dependencies
├── full_training/          # Production PyTorch weights & benchmark specifications
├── HAT+ quatization/       # Hailo-8 quantization & HEF compilation pipeline
├── llama/                  # LLaMA 3.2 1B dataset generator & INT4 export scripts
└── pi_deployment/          # Complete edge runtime for Raspberry Pi 5 & Hailo-8
```

Each primary directory contains its own comprehensive, dedicated documentation:
- **[`full_training/README.md`](full_training/README.md)**: Production model training configs, dataset curation, and 90%+ mAP benchmarks.
- **[`HAT+ quatization/README.md`](HAT+%20quatization/README.md)**: Hailo Dataflow Compiler (DFC) pipeline and Docker build environment.
- **[`llama/README.md`](llama/README.md)**: LLaMA 3.2 1B dataset generation, LoRA fine-tuning, and INT4 ExecuTorch export.
- **[`pi_deployment/README.md`](pi_deployment/README.md)**: Complete Raspberry Pi 5 deployment guide, Web Dashboard, and evaluation scripts.

---

## 1. Running on Local Machine (PC / Laptop)

You do **NOT** need a Raspberry Pi or Hailo-8 NPU to develop, test, or demonstrate this system. The software automatically detects the operating system and runs in **PC Simulation Mode**:
- **Vision Engine**: Detects anomalies using computer vision heuristics or authentic sample data, generates bounding boxes, polygon masks, and simulates Hailo-8 latency (~11.8 ms).
- **Llama Advisory**: Uses Six-Sigma failure mode knowledge bases to generate structured engineering recommendations.
- **Alerts**: Safely mocks GPIO pins and buzzer beeps in software without errors.

### Prerequisites (PC)
- **OS:** Windows 10/11, macOS, or Ubuntu / Linux
- **Python:** Version `3.9`, `3.10`, `3.11`, or `3.12`
- **Webcam:** Optional (for Live Camera feed testing)

### Step-by-Step Installation & Launch:

#### Step 1: Clone the repository
```bash
git clone https://github.com/JohanJMJ/Defect-classifications.git
cd Defect-classifications
```

#### Step 2: Create a virtual environment & install dependencies
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# Linux / macOS Terminal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Step 3: Launch the Web Dashboard
```bash
python run_dashboard.py
```

The terminal will print:
```text
==================================================================
  ON-DEVICE AI INDUSTRIAL QUALITY CONTROL ASSISTANT
  PC Browser URL        : http://localhost:5000
  Mobile / Network URL  : http://192.168.x.x:5000
  Vision Engine         : SIMULATION / PC MODE
==================================================================
```

#### Step 4: Open your browser
Navigate to **`http://localhost:5000`** (or access from your smartphone using the network IP).
- Switch models using the **Defect Detection Model** dropdown (`Steel Defects`, `Tile`, `Bottle`, `Grid`).
- Try the **Sample Test Library** tab to inspect pre-loaded defect images.
- Drag & drop images onto the **Upload Image** tab.
- Click **Live Camera Feed** to test webcam streaming and continuous inspection.
- Check the **Audit Log** tab to search inspection history and export CSV records.

---

## 2. Running on Raspberry Pi 5 with Hailo-8 AI HAT+

On the Raspberry Pi 5, the system unlocks full hardware acceleration:
- Vision inference executes on the **Hailo-8 NPU (26 TOPS)** over PCIe Gen 3 in **~22 ms**.
- Root-cause advisories execute on the Pi's ARM CPU via **ExecuTorch** running **Llama 3.2 1B Instruct SpinQuant INT4** in **~2.5 seconds**.

### Hardware Requirements
- **Raspberry Pi 5** (8 GB RAM recommended, 4 GB minimum)
- **Raspberry Pi AI HAT+** (26 TOPS Hailo-8 or 13 TOPS Hailo-8L)
- **Power Supply**: Official Raspberry Pi 27W USB-C Power Supply (5V / 5A)
- **Camera**: Raspberry Pi Camera Module 3 (CSI) or standard USB Industrial WebCam
- **MicroSD / NVMe**: 32 GB+ with **Raspberry Pi OS 64-bit (Debian Bookworm)**

---

### Step-by-Step Raspberry Pi Setup:

#### Step 1: Copy Deployment Package to Raspberry Pi
From your PC PowerShell / Terminal:
```bash
scp -r pi_deployment raspberrypi5@<PI_IP>:~/Desktop/
```
*(Alternatively, transfer `update_code.zip` and extract inside `~/Desktop/pi_deployment/`)*.

#### Step 2: Run the 1-Click Automated Setup
SSH into your Raspberry Pi:
```bash
ssh raspberrypi5@<PI_IP>
cd ~/Desktop/pi_deployment
chmod +x setup_pi.sh run_dashboard.sh run_inference.sh evaluate_hef.py run_eval.sh
./setup_pi.sh
```

**What `setup_pi.sh` does automatically in ~30 seconds:**
1. Installs official `hailo-all`, `python3-opencv`, `python3-numpy`, and `libgl1`.
2. Configures `/boot/firmware/config.txt` to enable PCIe Gen 3 (`dtparam=pciex1_gen=3`).
3. Appends `pcie_aspm=off` to `/boot/firmware/cmdline.txt` to prevent PCIe power-management idle timeouts.
4. Creates a lightweight Python virtual environment inheriting system HailoRT bindings.
5. Verifies NPU hardware detection via `hailortcli fw-control identify`.

*(If this is your first time enabling the AI HAT+, reboot once with `sudo reboot` and return to `~/Desktop/pi_deployment`)*.

---

#### Step 3: Setup ExecuTorch Llama 3.2 1B (Optional but Recommended)
To enable on-device Small Language Model advisories:
Ensure your ExecuTorch deployment directory is placed at `/home/raspberrypi5/executorch-new/`:
```text
/home/raspberrypi5/executorch-new/
├── llama_main                                     # ExecuTorch runner binary
├── libllama_runner.so                             # Dynamic runner library
├── model_int4.pte                                 # 1B Quantized model (~650 MB)
└── tokenizer.json                                 # Llama 3 tokenizer
```

The advisory engine automatically discovers this path and loads the model on startup:
```text
[LlamaAdvisoryEngine] Arm ExecuTorch Runtime Active: /home/raspberrypi5/executorch-new
[LlamaAdvisoryEngine] Model (.pte): model_int4.pte
```

*(If ExecuTorch is not present, the system automatically falls back to sub-millisecond fast-path Six-Sigma rule advisories without crashing).*

---

#### Step 4: Launch the Edge Dashboard
```bash
cd ~/Desktop/pi_deployment
./run_dashboard.sh
```

Now open a browser on any device connected to the same local network:
- **On the Pi**: `http://localhost:5000`
- **From another PC / Phone**: `http://<PI_IP>:5000`

---

#### Step 5: (Optional) Enable Headless Auto-Start on Factory Boot
To run the dashboard as a background systemd service that starts automatically when the Pi powers on:
```bash
cd ~/Desktop/pi_deployment
sudo cp service/qc-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qc-dashboard
sudo systemctl start qc-dashboard
```

Check service status anytime:
```bash
sudo systemctl status qc-dashboard
```

---

## 3. Web Dashboard Features

| Feature | Description |
| :--- | :--- |
| **Model Switcher** | Instant switching between 9 pre-compiled industrial HEFs without restarting the server. |
| **Decoupled 2-Stage Pipeline** | Hailo-8 Vision returns masks in **22 ms**. Llama 3.2 1B finishes root-cause advice in **~2.5s** with an animated timer. |
| **On-Demand Camera** | Physical camera sensor stays completely powered off in standby mode; activates instantly upon opening the Live tab. |
| **Continuous Capture** | Automated conveyor loop capturing frames at 1s to 10s intervals with sub-millisecond triage. |
| **Inspection Viewfinder** | High-contrast visual overlay showing bounding boxes, translucent defect segmentation masks, and certainty scores. |
| **Audit Log & Analytics** | Historical inspection database, filter by verdict (PASS / MONITOR / REJECT), modal inspection details, and 1-click CSV export. |
| **100% Offline Font Stack** | Zero external Google Fonts or CDN dependencies; renders natively on isolated factory air-gapped networks. |

---

## 4. Hardware Evaluation & CLI Tools

In addition to the Web Dashboard, `pi_deployment` includes standalone CLI tools for benchmarking and mAP validation:

### 1. Benchmark Hailo-8 NPU Latency & Throughput
```bash
cd ~/Desktop/pi_deployment
./run_inference.sh --model tile --benchmark
```
*Outputs true hardware metrics over 100 iterations (average ~11.6 ms latency, ~85+ FPS).*

### 2. On-Device Box & Mask mAP Evaluation
```bash
cd ~/Desktop/pi_deployment
./run_eval.sh
```
*Evaluates post-quantization Box mAP50, Box mAP50-95, Mask mAP50, Mask mAP50-95, precision, recall, and pass/fail classification accuracy directly against ground truth labels.*

### 3. Direct Camera Feed CLI
```bash
python3 inference.py --model bottle --source 0 --show
```

---

## 5. Troubleshooting & FAQ

### Q1: `HAILO_TIMEOUT(4)` (10000ms) on D2H layers
**Root Cause**: Raspberry Pi 5 kernel automatically enables PCIe ASPM (Active State Power Management). When the system sits idle, the PCIe bus drops into low-power states (L0s/L1) and fails to wake up quickly enough for DMA transfers.  
**Fix**:
1. Run a hardware reset: `hailortcli fw-control reset && hailortcli fw-control identify`
2. Permanently disable PCIe ASPM: Add `pcie_aspm=off` to the end of the single line in `/boot/firmware/cmdline.txt` and reboot:
   ```text
   console=serial0 ... rootwait fsck.repair=yes pcie_aspm=off
   ```
*(Note: `HailoVisionEngine` now includes automatic self-healing that detects this timeout and recovers the NPU automatically).*

### Q2: Detecting an image shown on a phone screen
**Root Cause**: Holding a phone in front of a camera introduces active backlight glare, screen moiré pixel grids, trapezoidal angle distortion, and shrinks the defect size relative to the full frame.  
**Best Practices**:
1. **Zoom in / Move closer**: Make sure the phone screen fills **85%–90% of the camera viewfinder** (no fingers or bezels visible).
2. **Lower brightness**: Dim the phone screen to **25%–35%** to stop exposure blowout.
3. **Upload directly**: Open `http://<PI_IP>:5000` on your phone browser and upload the photo directly via the **Upload Image** tab.

### Q3: `error while loading shared libraries: libllama_runner.so`
**Fix**: Ensure `LD_LIBRARY_PATH` points to your ExecuTorch directory:
```bash
export LD_LIBRARY_PATH=/home/raspberrypi5/executorch-deployment:$LD_LIBRARY_PATH
```
*(The backend `llama_engine.py` injects this path automatically upon subprocess execution).*

---

## License

This project is released under the [Apache 2.0 License](LICENSE). Models are trained on the MVTec Anomaly Detection (AD) dataset for research and industrial quality control.
