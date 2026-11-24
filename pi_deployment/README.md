# Raspberry Pi AI HAT+ (26 TOPS) — Multi-Model Defect Segmentation & Advisory
### Pure Native HailoRT Runtime (Zero PyTorch / Zero Ultralytics)

This package contains a complete, production-ready edge deployment suite for running real-time YOLO vision models (**YOLO11s-seg** for multi-class defect segmentation and **YOLOv8s-P2** for small steel defect detection), deterministic physical triage, and on-device Small Language Model (LLaMA 3.2 1B) root-cause advisory on a **Raspberry Pi 5** equipped with the **Raspberry Pi AI HAT+ (26 TOPS / Hailo-8 NPU)**.

> This deployment package uses **pure Native HailoRT (`hailo_platform`)**, **NumPy**, and **OpenCV**.  
> **No PyTorch or Ultralytics is required or installed on the Raspberry Pi.**

---

## Package Structure

```text
pi_deployment/
├── backend/                     # Dual-AI & hardware abstraction engines
│   ├── hailo_engine.py          # Native HailoRT NPU runner with self-healing PCIe reset & PC simulation
│   ├── llama_engine.py          # ExecuTorch Llama 3.2 1B runner with Six-Sigma CQE prompt engine
│   ├── rule_engine.py           # Sub-millisecond deterministic physical triage (< 0.1 ms)
│   ├── camera_streamer.py       # Zero-contention USB/CSI camera streamer with MJPEG feed
│   ├── alert_manager.py         # Industrial GPIO buzzer & LED indicator management
│   └── database.py              # SQLite audit logging & historical analytics
├── dashboard/                   # High-contrast Industrial Web Dashboard
│   ├── app.py                   # Decoupled 2-stage Flask API backend (< 25 ms vision response)
│   ├── config.py                # System telemetry, paths, and hardware monitor
│   ├── templates/               # Offline HTML templates (base, index, history)
│   └── static/                  # Modular CSS architecture, icons, and async JavaScript
├── models/                      # High-Accuracy Hailo-8 HEF executables (26 TOPS)
│   ├── steel_defects_hailo8.hef # Steel defect detection (YOLOv8s-P2, 6 classes, 21.1 MB)
│   ├── bottle_hailo8.hef        # Bottle defect segmentation (YOLO11s-seg, 3 classes, 23.0 MB)
│   ├── grid_hailo8.hef          # Grid defect segmentation (YOLO11s-seg, 5 classes, 23.8 MB)
│   ├── tile_hailo8.hef          # Tile defect segmentation (YOLO11s-seg, 5 classes, 22.5 MB)
│   └── models.yaml              # Models registry & specifications
├── sample_images/               # Categorized test images for each model
├── service/                     # Systemd service for auto-start on factory boot
├── results/                     # Output folder for annotated images & JSON reports
├── inference.py                 # Pure HailoRT inference engine (HUD, masks, JSON logs)
├── inference_llama_optimized.py # End-to-end CLI inspection (Hailo-8 YOLO + ExecuTorch LLaMA 3.2 1B)
├── evaluate_hef.py              # On-device Box & Mask mAP evaluation script
├── measure_power.py             # Raspberry Pi 5 PMIC live power & energy profiler (mJ/frame)
├── run_dashboard.sh             # 1-Click Web Dashboard launcher for Raspberry Pi
├── run_pipeline.sh              # 1-Click end-to-end test pipeline (Hailo + LLaMA)
├── run_eval.sh                  # 1-Click mAP and accuracy evaluation runner
├── run_inference.sh             # 1-Click terminal runner for images/camera
├── setup_pi.sh                  # Automated Raspberry Pi OS setup script (~30 sec)
└── README.md                    # Documentation & usage guide
```

---

## Model Architecture & Domain Specifications

Each industrial inspection domain utilizes a specialized, mathematically tuned YOLO architecture:

| Domain | Category | Task | Architecture | Target Defect Classes | Model Size | Hailo-8 Binary |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **Ceramic Tile** | `tile` | Instance Segmentation | **YOLO11s-seg** | 5 classes (`crack`, `glue_strip`, `gray_stroke`, `oil`, `rough`) | 22.5 MB | `tile_hailo8.hef` |
| **Glass Bottle** | `bottle` | Instance Segmentation | **YOLO11s-seg** | 3 classes (`broken_large`, `broken_small`, `contamination`) | 23.0 MB | `bottle_hailo8.hef` |
| **Wire Mesh Grid** | `grid` | Instance Segmentation | **YOLO11s-seg** | 5 classes (`bent`, `broken`, `glue`, `metal_contamination`, `thread`) | 23.8 MB | `grid_hailo8.hef` |
| **Steel Strip** | `steel_defects` | Object Detection | **YOLOv8s-P2** | 6 classes (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`) | 21.1 MB | `steel_defects_hailo8.hef` |

- **Why different architectures?**
  - **Tile, Bottle, and Grid** require pixel-precise polygon boundary masks to measure physical defect surface area ($\%$) and contour deformation, making **YOLO11s-seg** the optimal choice.
  - **Hot-Rolled Steel (NEU-DET)** contains microscopic surface hairline cracks and scale where bounding-box localization is critical; it integrates a high-resolution **P2 small-feature detection head into YOLOv8s-P2** to capture ultra-fine defects without mask overhead.

---

## Heterogeneous Edge Compute Pipeline

To maximize throughput and eliminate pipeline contention, the system decouples vision inference from language reasoning across two distinct compute units on the Raspberry Pi 5:

| Pipeline Stage | Compute Unit | Model & Runtime | Latency | Responsibility |
| :--- | :--- | :--- | :---: | :--- |
| **Vision Inference** | Hailo-8 NPU (26 TOPS, PCIe Gen 3) | YOLO11s-seg / YOLOv8s-P2 (`.hef` via Native HailoRT) | ~22 ms | Spatial bounding box localization, polygon mask segmentation, and defect classification. |
| **Physical Triage** | Broadcom BCM2712 Host CPU | Neuro-Fuzzy Rule Engine (`rule_engine.py`) | < 0.05 ms | Calculates continuous risk score (0–100), determines severity, and handles fast-path clearance. |
| **Root-Cause Advisory** | Quad-Core ARM Cortex-A76 @ 2.4 GHz | LLaMA 3.2 1B Instruct SpinQuant INT4 (`model_int4.pte` via Meta ExecuTorch + XNNPACK) | ~2.5 s | Async Six-Sigma CQE failure mode analysis, material risk evaluation, and disposition action items. |

- **RAM Disk Execution (`/dev/shm`)**: The 789 MB quantized LLaMA binary (`model_int4.pte`) is pre-staged directly into the Raspberry Pi 5's LPDDR4X shared memory (`/dev/shm`), avoiding random read bottlenecks and flash memory wear on MicroSD storage.
- **Asynchronous Decoupling**: High-speed conveyor inspection continues uninterrupted because vision segmentation masks and initial pass/reject decisions are returned in under 25 ms, while the LLaMA advisory generates asynchronously in the background.

---

## Neuro-Fuzzy Triage & Industrial Severity Matrix

Conveyor lines operating at industrial speeds cannot afford to invoke a multi-second language model for every minor dust speck or visual artifact. The system integrates a sub-millisecond **Neuro-Fuzzy Triage Engine** (`rule_engine.py`) that computes a continuous **Fuzzy Risk Score ($0 \text{ to } 100$)** across three weighted physical membership functions:

$$\text{Fuzzy Risk Score} = \text{Area Score (40 pts)} + \text{Confidence Factor (25 pts)} + \text{Hazard Weight (35 pts)}$$

1. **Defect Surface Area Risk ($0 \text{ to } 40$ points)**: Non-linear piecewise scaling based on total defect polygon mask pixel percentage:
   - $\le 0.2\%$ surface area $\rightarrow 0 \text{ to } 8$ pts
   - $0.2\% - 1.0\%$ surface area $\rightarrow 8 \text{ to } 22$ pts
   - $1.0\% - 5.0\%$ surface area $\rightarrow 22 \text{ to } 36$ pts
   - $> 5.0\%$ surface area $\rightarrow 36 \text{ to } 40$ pts
2. **Model Confidence Factor ($0 \text{ to } 25$ points)**: Scaled detection probability certainty ($\text{conf} \ge 0.2 \rightarrow 0 \text{ to } 25$ pts).
3. **Hazard Severity Weight ($0 \text{ to } 35$ points)**: Domain-specific physical risk matrix representing structural vs cosmetic severity:
   - *High Hazard ($0.75 - 0.95$)*: `broken_large` (0.95), `crack` (0.90), `broken` (0.90), `crazing` (0.85), `rolled-in_scale` (0.75)
   - *Moderate Hazard ($0.50 - 0.70$)*: `inclusion` (0.70), `metal_contamination` (0.65), `pitted_surface` (0.60), `oil` (0.60), `glue` (0.50)
   - *Low Hazard ($0.35 - 0.45$)*: `bent` (0.45), `rough` (0.45), `scratches` (0.40), `gray_stroke` (0.40), `thread` (0.35)

### Industrial Severity & Disposition Action Matrix

| Severity Tier | Fuzzy Score Range | Conveyor Verdict | Factory Disposition Action | LLaMA 3.2 1B Invocation |
| :--- | :---: | :---: | :--- | :---: |
| **None** | $0.0$ | `PASS` | Surface uniformity verified. Clear component to downstream station. | Bypassed ($0$ ms) |
| **Minor** | $< 25.0$ | `PASS` | Minor cosmetic blemish within visual tolerance. Pass without line delay. | **Fast-Path Bypass** ($< 0.05$ ms) |
| **Moderate** | $25.0 - 54.9$ | `DIVERT` | Divert part from main conveyor line to buffer, rework lane, or wash bay. | **Triggered** (Async ~2.5s) |
| **Major** | $55.0 - 79.9$ | `INSPECT` | Route component to quality gate for manual inspector verification and sign-off. | **Triggered** (Async ~2.5s) |
| **Critical** | $\ge 80.0$ | `REJECT` | Auto-eject part to scrap bin immediately. Trigger visual alarm strobe and siren. | **Triggered** (Async ~2.5s) |

---

## Quick Start on Raspberry Pi 5

### Step 1: Copy this folder to your Raspberry Pi 5
From your PC terminal:
```bash
scp -r pi_deployment raspberrypi5@raspberrypi.local:~/Desktop/
```

### Step 2: Run the One-Time Setup on the Pi
SSH into your Raspberry Pi and run:
```bash
cd ~/Desktop/pi_deployment
chmod +x setup_pi.sh run_dashboard.sh run_pipeline.sh run_inference.sh run_eval.sh inference.py evaluate_hef.py
./setup_pi.sh
```
*Setup takes under 30 seconds. It enables the PCIe Gen 3 bus, disables PCIe ASPM sleep timeouts, installs the official `hailo-all` drivers, and creates a lightweight virtual environment.*

*(If this is the first time enabling the AI HAT+ PCIe bus, reboot once with `sudo reboot` and return to `~/Desktop/pi_deployment`).*

---

## Operational Execution Modes

### 1. Launch the Industrial Web Dashboard
This is the primary user interface for factory operators and inspection stations:
```bash
./run_dashboard.sh
```
- Open Chromium on the Pi or your computer browser at: **`http://localhost:5000`** (or `http://<pi_ip>:5000`).
- **Features:**
  - Automated RAM Disk pre-staging (`/dev/shm`) for zero cold-load latency.
  - Live camera view with instant single-shot or continuous conveyor cycle inspection.
  - Decoupled two-stage execution: Hailo-8 vision mask rendered in **< 25 ms**, followed by asynchronous LLaMA 3.2 1B root-cause advisory with progress indicator.
  - High-contrast industrial theme (dark/light toggle) with emergency audio-visual alert banner.
  - Historical audit log table with search filters and one-click CSV export.

---

### 2. Run the End-to-End Pipeline Test (`run_pipeline.sh`)
To verify the complete dual-AI pipeline (Hailo-8 YOLO + ExecuTorch LLaMA 3.2 1B) directly from the terminal across multiple defect categories:
```bash
./run_pipeline.sh
```
This tests:
1. **Tile Model:** Evaluates oil contamination defect.
2. **Bottle Model:** Evaluates foreign particulate contamination.
3. **Steel Defects Model:** Evaluates surface scratches.
Annotated inspection frames and JSON diagnostic reports are saved to `results/`.

---

### 3. Run Standalone Vision Inference (`run_inference.sh`)
To test Hailo-8 vision segmentation without launching the web server:

```bash
# List all registered HEF models
./run_inference.sh --list-models

# Run on default model (Tile) using sample images
./run_inference.sh tile

# Run on other model categories
./run_inference.sh steel_defects
./run_inference.sh bottle
./run_inference.sh grid

# Live camera stream with real-time HUD (press 'q' to exit):
./run_inference.sh --camera -m tile

# Inspect a custom image file:
./run_inference.sh -m tile -s path/to/sample.jpg
```

---

### 4. Hardware Accuracy & mAP Benchmarking (`run_eval.sh`)
To measure the exact on-device post-quantization accuracy and mAP of the `.hef` models:

```bash
./run_eval.sh tile
```
Or customize confidence and IoU thresholds:
```bash
python3 evaluate_hef.py --model tile --conf 0.25 --iou 0.50
```

#### Sample Evaluation Output:
```text
======================================================================
Hailo-8 Quantized Model mAP & Accuracy Evaluation
Model       : tile_hailo8.hef
Images Dir  : dataset/images
Labels Dir  : dataset/labels
Confidence  : 0.25
IoU NMS     : 0.50
======================================================================
Found 64 images to evaluate.
Running hardware inference and collecting ground truth...

======================================================================
Class        Images   Instances  Box mAP50   Box mAP50-95  Mask mAP50   Mask mAP50-95
----------------------------------------------------------------------
anomaly      64       25         0.958       0.784         0.945        0.751        
======================================================================

Detailed Metric Breakdown:
Box Precision (P)    : 0.942
Box Recall (R)       : 0.920
Box F1-Score         : 0.931
Mask Precision (P)   : 0.931
Mask Recall (R)      : 0.912
Mask F1-Score        : 0.921
Hardware Latency     : 11.65 ms (85.8 FPS)

Industrial Pass / Fail Classification Accuracy:
Overall Accuracy     : 96.9% (62/64 images)
True Negatives (Pass): 43
True Positives (Def) : 19
False Alarms (FP)    : 1
Missed Defects (FN)  : 1

Full evaluation report saved to: results/evaluation_report.json
```

---

### 5. Hardware Power & Energy Profiling (`measure_power.py`)
To measure instantaneous board power (Watts), voltage, current, and energy consumption per frame using the Raspberry Pi 5 onboard PMIC (`vcgencmd pmic_read_adc`):

```bash
# 1. Live real-time terminal power HUD (updates at 2 Hz):
python3 measure_power.py --live

# 2. Benchmark energy consumption on a specific model (50 iterations):
python3 measure_power.py --benchmark --model tile --iterations 50

# 3. Log power time-series to CSV for graphing and academic reports:
python3 measure_power.py --live --duration 60 --log-csv power_log.csv
```

---

## System Requirements & Specifications

- **Single Board Computer:** Raspberry Pi 5 (4GB or 8GB recommended).
- **AI Accelerator:** Raspberry Pi AI HAT+ with Hailo-8 NPU (26 TOPS, M.2 PCIe Gen 3).
- **Operating System:** Raspberry Pi OS 64-bit (`Bookworm`, `aarch64`).
- **Dependencies:** `hailo-all`, `python3-opencv`, `numpy`, `flask`, `flask-cors`, `pillow`, `psutil`, `pyyaml`.
- **LLaMA Engine:** Meta ExecuTorch runtime with LLaMA 3.2 1B SpinQuant INT4 (`/home/raspberrypi5/executorch-new`).
