# Raspberry Pi AI HAT+ (26 TOPS) — Multi-Model Defect Segmentation
### Pure Native HailoRT Runtime (Zero PyTorch / Zero Ultralytics)

This folder contains a **complete, lightweight deployment package** for running real-time YOLO11 segmentation and **post-quantization accuracy/mAP evaluation** on a Raspberry Pi 5 equipped with the **Raspberry Pi AI HAT+ (26 TOPS / Hailo-8)**.

> [!NOTE]
> This deployment package uses **pure Native HailoRT (`hailo_platform`)**, **NumPy**, and **OpenCV**.  
> **No PyTorch or Ultralytics is required or installed on the Raspberry Pi.**

---

## Package Contents

```text
pi_deployment/
├── backend/                     # Dual-AI & hardware engines
│   ├── hailo_engine.py          # Native HailoRT NPU runner with self-healing PCIe reset & PC simulation
│   ├── llama_engine.py          # ExecuTorch Llama 3.2 1B runner with Six-Sigma CQE prompt engine
│   ├── rule_engine.py           # Sub-millisecond deterministic physical triage (< 0.1 ms)
│   ├── camera_streamer.py       # On-demand USB/CSI camera streamer with MJPEG feed
│   ├── alert_manager.py         # Industrial GPIO buzzer & LED indicator management
│   └── database.py              # SQLite audit logging & historical analytics
├── dashboard/                   # High-contrast Industrial Web Dashboard
│   ├── app.py                   # Decoupled 2-stage Flask API backend (< 25 ms vision response)
│   ├── config.py                # System telemetry, paths, and hardware monitor
│   ├── templates/               # Offline HTML templates (base, index, history)
│   └── static/                  # Responsive CSS, icons, and async JavaScript polling
├── models/                      # High-Accuracy Hailo-8 HEF executables (26 TOPS)
│   ├── bottle_hailo8.hef        # Bottle defect segmentation (21.7 MB)
│   ├── grid_hailo8.hef          # Grid defect segmentation (21.4 MB)
│   ├── tile_hailo8.hef          # Tile defect segmentation (21.6 MB)
│   ├── models.yaml              # Models registry & specifications
│   └── metadata.yaml            # Base model metadata & segmentation classes
├── sample_images/               # Categorized test images for each model
├── service/                     # Systemd service for auto-start on factory boot
├── results/                     # Output folder for annotated images & JSON reports
├── inference.py                 # Pure HailoRT inference engine (HUD, masks, JSON logs)
├── evaluate_hef.py              # Evaluates Box mAP & Mask mAP across all images
├── run_dashboard.sh             # 1-click Web Dashboard launcher for Raspberry Pi
├── run_eval.sh                  # 1-click mAP and accuracy evaluation runner
├── run_inference.sh             # One-click terminal runner for images/camera
├── setup_pi.sh                  # 1-click Raspberry Pi OS setup script (fast, ~30 sec)
└── README.md                    # Documentation & usage guide
```

---

## Quick Start (in 3 Steps)

### Step 1: Copy this folder to your Raspberry Pi 5
From your PC terminal:
```bash
scp -r pi_deployment <pi_user>@<pi_ip>:~/
```
*(Example: `scp -r pi_deployment pi@192.168.1.100:~/`)*

### Step 2: Run the 1-Click Setup on the Pi
SSH into your Raspberry Pi and run:
```bash
cd ~/pi_deployment
chmod +x setup_pi.sh run_inference.sh inference.py evaluate_hef.py run_eval.sh
./setup_pi.sh
```
*Setup takes under 30 seconds! It installs the official `hailo-all` drivers, enables the PCIe bus, and installs NumPy and OpenCV.*

*(If this is your first time enabling the AI HAT+ PCIe bus, reboot once with `sudo reboot` and return to `~/pi_deployment`).*

### Step 3: Run Defect Inspection!

```bash
# List all 9 available models
./run_inference.sh --list-models

# Run defect inspection on default model (tile):
./run_inference.sh

# Select any model by category name (auto-loads matching sample images):
./run_inference.sh bottle
./run_inference.sh grid
./run_inference.sh tile

# Or use standard --model / -m flags:
./run_inference.sh --model bottle
python3 inference.py --model grid
```

---

## Checking Accuracy & mAP on the Board

To measure the exact post-quantization accuracy and mAP of the `.hef` model while running on the Raspberry Pi AI HAT+:

```bash
./run_eval.sh
```
*Or customize confidence threshold:*
```bash
python3 evaluate_hef.py --conf 0.25 --iou 0.50
```

### What it Output in Terminal:
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
   [10/64] processed (11.8 ms/frame)...
   [20/64] processed (11.7 ms/frame)...
   ...
   [64/64] processed (11.6 ms/frame)...

======================================================================
Class        Images   Instances  Box mAP50   Box mAP50-95  Mask mAP50   Mask mAP50-95
----------------------------------------------------------------------
anomaly      64       25         0.958       0.784         0.945        0.751        
======================================================================

Detailed Metric Breakdown:
   • Box Precision (P)    : 0.942
   • Box Recall (R)       : 0.920
   • Box F1-Score         : 0.931
   • Mask Precision (P)   : 0.931
   • Mask Recall (R)      : 0.912
   • Mask F1-Score        : 0.921
   • Hardware Latency     : 11.65 ms (85.8 FPS)

Industrial Pass / Fail Classification Accuracy:
   • Overall Accuracy     : 96.9% (62/64 images)
   • True Negatives (Pass): 43
   • True Positives (Def) : 19
   • False Alarms (FP)    : 1
   • Missed Defects (FN)  : 1

Full evaluation report saved to: results/evaluation_report.json
```

---

## Other Inference Options

### 1. Live Camera Stream (USB / CSI Camera)
```bash
# Stream from camera index 0 and display GUI window with HUD (FPS & Latency)
./run_inference.sh --camera

# Or specify a custom camera device:
python3 inference.py --source /dev/video0 --show
```
*Press `q` in the display window to exit.*

---

### 2. Inspect Specific Models & Custom Images
```bash
# Run defect inspection on a specific category model
./run_inference.sh bottle
./run_inference.sh --model tile
./run_inference.sh --model grid

# Inspect single custom image with specific model:
./run_inference.sh --model bottle --source path/to/my_bottle.jpg

# Inspect entire custom folder of images:
./run_inference.sh --model tile --source path/to/tile_batch/
```

---

### 3. Run Hardware NPU Benchmark
Measure the true hardware frames per second (FPS) and millisecond latency:
```bash
./run_inference.sh --benchmark
```
Runs 100 timed iterations directly on the Hailo-8 chip and outputs:
* Average Throughput (FPS)
* Average & Median Latency (ms)
* Min / Max Latency (ms)
