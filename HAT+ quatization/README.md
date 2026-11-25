# Raspberry Pi AI HAT+ (Hailo-8) Quantization & HEF Compiler Guide

This directory contains the quantization pipeline and toolchain for compiling PyTorch floating-point defect classification/segmentation models (`full_training/*.pt`) into optimized **Hailo Executable Format (`.hef`)** binaries for deployment on the **Raspberry Pi AI HAT+**.

> [!NOTE]
> **Pre-compiled HEF binaries are already included:** The fully optimized, production-grade `.hef` binaries for all 4 industrial categories are maintained in [`pi_deployment/models/`](file:///d:/Defect-classifications/pi_deployment/models/). To conserve repository space and prevent duplicate binary weight (~90 MB), redundant copies in a local `hef_output/` folder have been removed. Calibration image datasets (`hailo_calibration_data/`) are kept locally / uncommitted (via `.gitignore`) because the pre-quantized binaries are already checked into the deployment folder.

---

## 1. Toolchain & Execution Architecture

Compiling neural networks for Hailo hardware requires the **Hailo Dataflow Compiler (DFC)**. To guarantee cross-platform compatibility and satisfy runtime dependencies, compilation is structured through a containerized bridge:

```
[ PyTorch .pt Model ] ──(Ultralytics)──> [ ONNX Graph ]
                                              │
                                     (Hailo DFC Parser)
                                              ▼
                                     [ Hailo Archive (HAR) ]
                                              │
                                 ┌────────────┴────────────┐
                                 │ .alls Model Script      │
                                 │ - Normalization [0-255] │
                                 │ - Output Node Routing   │
                                 │ - NMS Configuration     │
                                 └────────────┬────────────┘
                                              ▼
[ Calibration Dataset (INT8) ] ──> [ Quantization Optimizer ]
                                   (PTQ / Level 2 QAT Tuning)
                                              ▼
                                    [ Quantized HAR ]
                                              │
                                    (Hailo DFC Compiler)
                                              ▼
                                  [ Target .hef Binary ]
```

### Key Engineering Constraints Solved
1. **GLIBC 2.34+ Compatibility:** Hailo DFC's hardware emulator (`HSim.so`) requires `GLIBC >= 2.34`. When running on older Linux hosts (e.g. Ubuntu 20.04 with GLIBC 2.31), [`run_compiler.sh`](file:///d:/Defect-classifications/HAT+%20quatization/run_compiler.sh) and [`hailo_cli.sh`](file:///d:/Defect-classifications/HAT+%20quatization/hailo_cli.sh) transparently execute the toolchain within a lightweight Ubuntu 22.04 container.
2. **Python 3.10 Virtual Environment:** DFC v3.34.0 is pinned to Python 3.10 with specialized wheels for `hailo_sdk_client`, `hailo_model_zoo`, and PyTorch CPU.
3. **Hardware Target Differentiation:**
   - **Hailo-8 (26 TOPS):** Target `--arch hailo8` for the 26 TOPS AI HAT+.
   - **Hailo-8L (13 TOPS):** Target `--arch hailo8l` for the 13 TOPS AI HAT+.

---

## 2. Quantization Workflow (Technical Deep Dive)

### Step 1: ONNX Export & Graph Simplification
Trained models are loaded from `full_training/*.pt` using Ultralytics. The network graph is exported to ONNX with:
- Fixed spatial input resolution: `640 x 640 x 3`
- Dynamic batch size set to static batch 1 (optimized for real-time camera streaming)
- Constant folding enabled via `onnx-simplifier`

### Step 2: Hailo Parsing (ONNX to HAR)
The Hailo Parser ingests the ONNX graph and converts PyTorch operators into Hailo hardware primitives:
```bash
hailo parser onnx <model>.onnx --hw-arch hailo8 --start-node-names "images" --end-node-names <output_heads>
```
The resulting unquantized graph is packaged into a Hailo Archive (`.har`).

---

### Step 3: Model Configuration & Hailo Script (`.alls`)
Each model is configured using Hailo script commands (`.alls`) that define hardware-level behavior:

#### A. Hardware Normalization
Hailo-8 NPUs feature dedicated input preprocessing hardware. Rather than wasting host CPU cycles dividing pixel values by 255.0, normalization is mapped directly to the NPU input stream:
```alls
normalization1 = norm([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
```
This allows the Raspberry Pi camera to pass raw 8-bit `uint8` image buffers straight to the NPU.

#### B. Architectural Output Configurations

The four production models are divided into two distinct hardware output topologies:

| Model Category | Task Type | Output Configuration on Hailo-8 | Tensor Breakdown |
|---|---|---|---|
| **`steel_defects`** | Bounding Box Detection | **On-Chip Hailo NMS** | 1 output tensor (`[1, 6, 80, 5]`): Bounding boxes, scores, and class IDs decoded directly on the Hailo silicon. |
| **`tile`** | Multi-Class Segmentation | **Multi-Tensor Raw Heads** | 10 output tensors (Box + Cls across strides 8, 16, 32 + Proto mask `[1, 32, 160, 160]`). Decoded on Pi 5 via dynamic matrix multiplication. |
| **`bottle`** | Multi-Class Segmentation | **Multi-Tensor Raw Heads** | 10 output tensors (Strides 8, 16, 32 + Proto mask `[1, 32, 160, 160]`). |
| **`grid`** | Multi-Class Segmentation | **Multi-Tensor Raw Heads** | 10 output tensors (Strides 8, 16, 32 + Proto mask `[1, 32, 160, 160]`). |

##### Detection (`steel_defects_hailo8.hef`)
Configured with Hailo on-chip NMS post-processing to offload bounding box filtering from the Raspberry Pi 5 CPU:
```alls
nms_post_process(
    meta_arch=yolov8,
    engine=cpu,
    classes=6,
    score_threshold=0.25,
    iou_threshold=0.50
)
```

##### Segmentation (`tile`, `bottle`, `grid`)
Preserves the high-resolution prototype mask channel (`proto`) to enable precise polygon boundary extraction for micro-defects (e.g. hairline tile cracks, small glass chips, thin threads). The host decoder (`pi_deployment/backend/hailo_engine.py`) performs sigmoid activation and spatial matrix multiplication ($160 \times 160 \rightarrow 640 \times 640$).

---

### Step 4: Calibration & INT8 Quantization
To transform 32-bit floating-point weights into 8-bit fixed-point integers (`INT8`) without sacrificing defect recall:

1. **Calibration Set:** A representative subset (~100 to 300 images per category) containing both defect-free samples and diverse defect classes is drawn from a local calibration dataset referenced by a `data.yaml`.
   > [!TIP]
   > **Storage Conservation:** Calibration image datasets are kept external/local and are ignored in git (`.gitignore`) to avoid bloating the repository with hundreds of megabytes of redundant raw images. Pre-quantized `.hef` binaries for all categories are already compiled and stored in [`pi_deployment/models/`](file:///d:/Defect-classifications/pi_deployment/models/).
2. **Dynamic Range Profiling:** The activations of each layer are evaluated to compute optimal scaling factors and zero-points ($S$ and $Z$):
   $$q = \text{round}\left(\frac{x}{S}\right) + Z$$
3. **Optimization Flavors:**
   - **Fast PTQ (Level 1, `--fast`):** Computes per-channel weight scaling and KL-divergence activation clipping (~1.5 minutes).
   - **Full QAT Fine-Tuning (Level 2, default):** Employs equalization and bias correction across 4 optimization epochs (~20 minutes) to prevent clipping of small, low-contrast defect signatures.

---

### Step 5: Compilation to HEF
The quantized HAR is compiled into hardware binary microcode:
```bash
hailo compiler <quantized_model>.har --hw-arch hailo8
```
The compiler maps layers to the Hailo-8's core clusters, orchestrates on-chip SRAM memory reuse, and writes the final `.hef` file directly into [`pi_deployment/models/`](file:///d:/Defect-classifications/pi_deployment/models/).

---

## 3. Usage & CLI Commands

### Compiling Directly via Python
If working inside a Hailo Dataflow Compiler virtual environment (`hailo_venv` on an Ubuntu 22.04 / GLIBC >= 2.34 environment):

```bash
# 1. Tile Defect Segmentation (output directly to pi_deployment/models)
python compile_hef.py \
    --category tile \
    --arch hailo8 \
    --model ../full_training/tile_production_best.pt \
    --data /path/to/calibration_data/tile/data.yaml

# 2. Steel Defect Detection (using steel_production_best.pt)
python compile_hef.py \
    --category steel_defects \
    --arch hailo8 \
    --model ../full_training/steel_production_best.pt \
    --data /path/to/calibration_data/steel/data.yaml

# 3. Fast Post-Training Quantization (~1.5 min)
python compile_hef.py \
    --category bottle \
    --arch hailo8 \
    --fast \
    --data /path/to/calibration_data/bottle/data.yaml
```

### CLI Arguments Reference

| Flag | Type | Default | Description |
|---|---|---|---|
| `-c`, `--category` | String | `tile` | Defect category to compile (`steel_defects`, `bottle`, `grid`, `tile`). |
| `-a`, `--arch` | Choice | `hailo8` | Target hardware: `hailo8` (26 TOPS) or `hailo8l` (13 TOPS). |
| `-m`, `--model` | Path | `None` | Path to `.pt` weights (defaults to `full_training/<category>_production_best.pt`, or `steel_production_best.pt` for steel). |
| `-d`, `--data` | Path | `None` | Path to local calibration `data.yaml` (kept external to save repo storage). |
| `--imgsz` | Integer | `640` | Square input resolution. |
| `--fast` | Flag | `False` | Enables Fast PTQ (Level 1) instead of full 4-epoch QAT fine-tuning. |
| `--all` | Flag | `False` | Compiles all 4 active production models sequentially. |
| `-o`, `--output-dir` | Path | `../pi_deployment/models` | Output folder for compiled `.hef` binaries (defaults directly to deployment models directory). |

---

## 4. Deployed Binary Mapping

All compiled binaries reside in [`pi_deployment/models/`](file:///d:/Defect-classifications/pi_deployment/models/) ready for production execution on the Raspberry Pi:

| Model | Compiled HEF File | Input Format | NMS Location | Target Hardware |
|---|---|---|---|---|
| Tile Defect Segmenter | `tile_hailo8.hef` | $640 \times 640 \times 3$ uint8 | Host HailoRT | Hailo-8 (26 TOPS) |
| Bottle Defect Segmenter | `bottle_hailo8.hef` | $640 \times 640 \times 3$ uint8 | Host HailoRT | Hailo-8 (26 TOPS) |
| Grid Mesh Segmenter | `grid_hailo8.hef` | $640 \times 640 \times 3$ uint8 | Host HailoRT | Hailo-8 (26 TOPS) |
| Steel Defect Detector | `steel_defects_hailo8.hef` | $640 \times 640 \times 3$ uint8 | On-Chip Hailo NMS | Hailo-8 (26 TOPS) |
