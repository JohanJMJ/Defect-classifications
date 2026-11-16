# Hailo Dataflow Compiler Setup & Compilation Guide for Raspberry Pi AI HAT+

This workspace is fully configured to compile YOLO models (such as YOLO11-seg defect detection models) from `.pt` format to Hailo Executable Format (`.hef`) for execution on the **Raspberry Pi AI HAT+**.

---

## 1. Overview of the Setup

### Key Challenges Solved:
1. **Python Version Compatibility**: Hailo Dataflow Compiler v3.34.0 requires **Python 3.10**. The virtual environment (`hailo_venv`) has been constructed with Python 3.10 and populated with all necessary compiler dependencies, PyTorch CPU, Ultralytics, OpenCV, and Jax/TensorFlow.
2. **GLIBC 2.34 Requirement**: Hailo DFC's emulator (`HSim.so`) requires **GLIBC >= 2.34**. Because the host operating system is Ubuntu 20.04 (which provides GLIBC 2.31), execution is seamlessly bridged through a lightweight Ubuntu 22.04 container (`run_compiler.sh` / `hailo_cli.sh`) which provides GLIBC 2.35 without modifying host system libraries.
3. **Calibration Data Paths**: All 15 `data.yaml` files inside `hailo_calibration_data/` have been updated from Kaggle cloud paths (`/kaggle/working/...`) to local workspace paths for INT8 calibration.

---

## 2. Target Hardware: Choosing Your HAT+ Architecture

The Raspberry Pi AI HAT+ comes in two distinct hardware variants:
- **13 TOPS AI HAT+**: Uses the **Hailo-8L** chip. Use target `--arch hailo8l`.
- **26 TOPS AI HAT+**: Uses the full **Hailo-8** chip. Use target `--arch hailo8`.

> **Important**: A model compiled for `hailo8` will not load on a 13 TOPS HAT+ (`hailo8l`). Always compile targeting your specific hardware.

---

## 3. How to Compile Models to .hef

### Option A: Using the Runner Script (Recommended)
The runner script handles Docker and environment variables automatically:

```bash
# Compile a single category (default: tile, target: hailo8l)
./run_compiler.sh --category tile --arch hailo8l

# Compile for 26 TOPS AI HAT+ (Hailo-8)
./run_compiler.sh --category tile --arch hailo8

# Compile a different category (e.g., bottle, screw, wood)
./run_compiler.sh --category bottle --arch hailo8l

# Compile all 15 categories sequentially
./run_compiler.sh --all --arch hailo8l
```

### Option B: Compiling with Custom Weights or Datasets
```bash
./run_compiler.sh \
  --model /path/to/custom_weights.pt \
  --data /path/to/custom_data.yaml \
  --arch hailo8l \
  --output-dir hef_output
```

### Option C: Using Native Hailo CLI Directly
To run the Hailo Dataflow Compiler tools directly:
```bash
./hailo_cli.sh --version
./hailo_cli.sh --help
./hailo_cli.sh parser onnx <model>.onnx --hw-arch hailo8l
```

---

## 4. Output HEF Artifacts

Compiled `.hef` files are saved into:
```
/home/acer/hailo_workspace/hef_output/
```
Format: `<category>_<arch>.hef` (e.g., `tile_hailo8l.hef`).

---

## 5. Deploying and Running on Raspberry Pi 5 with AI HAT+

Once compiled, copy your `.hef` file to your Raspberry Pi 5:

```bash
scp hef_output/tile_hailo8l.hef pi@raspberrypi.local:~/
```

On the Raspberry Pi 5:
1. Ensure the Hailo software is installed:
   ```bash
   sudo apt update
   sudo apt install hailo-all
   ```
2. Check the device is recognized:
   ```bash
   hailortcli scan
   hailortcli fw-control identify
   ```
3. Run inference using HailoRT Python API or `rpicam-apps` pipeline.
