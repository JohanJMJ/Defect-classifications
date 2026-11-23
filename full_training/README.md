# Production Model Weights and Benchmark Specifications

This directory contains the production-grade PyTorch (`.pt`) model weights and intermediate ONNX representations for the four industrial quality inspection domains: Ceramic Tile, Hot-Rolled Steel Strip, Glass Bottle Packaging, and Wire Mesh Grid.

These models serve as the high-precision floating-point (FP32/FP16) source of truth prior to post-training quantization (INT8) and compilation into Hailo Executable Format (HEF) binaries for the Hailo-8 AI HAT+ (26 TOPS).

---

## 1. Executive Performance Summary

| Model Weights File | Task | Architecture | Parameters | Target Classes | Precision | Recall | F1-Score | mAP@50 | mAP@50:95 | Model Size |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `tile_production_best.pt` | Segmentation | YOLO11s-seg | 10.08M | 5 classes | **89.29%** | **94.87%** | **91.99%** | **98.50%** | **68.16%** | 20.5 MB |
| `bottle_production_best.pt` | Segmentation | YOLO11s-seg | 10.08M | 3 classes | **76.91%** | **82.22%** | **79.48%** | **80.79%** | **56.28%** | 20.5 MB |
| `steel_production_best.pt` | Detection | YOLOv8s | 10.64M | 6 classes | **76.23%** | **80.55%** | **78.33%** | **82.83%** | **45.28%** | 21.7 MB |
| `grid_production_best.pt` | Segmentation | YOLO11s-seg | 10.08M | 5 classes | **63.56%** | **75.30%** | **68.94%** | **69.75%** | **36.76%** | 20.5 MB |

*Note: For segmentation models (`tile`, `bottle`, `grid`), reported values reflect pixel-level mask metrics evaluated at IoU thresholds of 0.50 and 0.50:0.95.*

---

## 2. Dataset Curation, Sanitization, and Augmentation

To resolve historical annotation artifacts and severe class imbalance in public benchmarks (NEU-DET and MVTec AD), all datasets underwent a systematic preparation and sanitization pipeline prior to model training.

### 2.1 NEU-DET Steel Strip Defect Sanitization
- **Deduplication:** Perceptual and cryptographic MD5 hash filtering removed identical and rotated duplicate frames.
- **Cross-Contamination Correction:** Rectified 130 bounding boxes mislabeled across conflicting defect classes (e.g., isolated inclusion specks erroneously labeled as scratches).
- **Bounding Box Harmonization ($\text{IoU} \ge 0.15$):** Replaced fragmented, overlapping micro-boxes with convex bounding envelopes, resolving contradictory training loss penalties on continuous defects (`crazing` and `rolled-in_scale`).
- **Contrast Sharpening (CLAHE):** Applied Contrast Limited Adaptive Histogram Equalization ($2.5$ clip limit, $8\times 8$ tile grid) on the luminance channel to expose low-contrast surface crazing.

### 2.2 MVTec AD Multi-Class Boundary Alignment
- **Physical Deformation Alignment (Grid):** Corrected human annotator truncation error in the `bent` wire class, expanding the arbitrary 70 px apex annotation to match the continuous 135 px structural wire deformation.
- **Morphological Polygon Extraction:** Converted raw binary anomaly masks into clean multi-class instance segmentation polygons using morphological closing with elliptical structuring elements ($k=31\times 31$).

### 2.3 Training Augmentations & Terminal Mosaic Suppression
- **Stratified Splitting:** 80% Train, 10% Validation, 10% Test across all defect classes.
- **Geometric Transformations:** RandomAffine rotation ($\pm 45^\circ$, scale $\pm 0.3$) and bidirectional flips ($p=0.5$).
- **Defect Blending:** CopyPaste ($p=0.3$) and Random Erasing ($p=0.4$) to simulate partial visual occlusions.
- **Terminal Mosaic Suppression:** Disabled mosaic augmentation during the final 10 epochs of training. This allowed the CNN prototype mask layers to converge cleanly on natural defect borders without boundary artifacts.

---

## 3. Detailed Model Specifications

### 3.1 Tile Surface Defect Segmenter (`tile_production_best.pt`)
- **Domain:** Industrial ceramic and glazed surface quality control.
- **Architecture:** YOLO11s-seg (`nc=5`, `ch=3`, `imgsz=640`, parameters: 10,084,223).
- **Target Classes:**
  - `0: crack` (Severe structural fracture)
  - `1: glue_strip` (Foreign adhesive deposit)
  - `2: gray_stroke` (Abrasive surface discoloration)
  - `3: oil` (Hydrocarbon contamination puddle)
  - `4: rough` (Texture deformation / glaze unevenness)
- **Training Setup:** 100 epochs, AdamW optimizer (`lr0=0.001`, `lrf=0.01`, weight decay 0.005), Cosine Annealing learning rate schedule.
- **Validation Metrics:**
  - Mask Precision: **89.29%** | Mask Recall: **94.87%** | **F1: 91.99%**
  - Mask mAP@50: **98.50%** | Mask mAP@50:95: **68.16%**
  - Box mAP@50: **98.50%** | Box mAP@50:95: **63.80%**

### 3.2 Bottle Packaging Defect Segmenter (`bottle_production_best.pt`)
- **Domain:** Glass and plastic container bottling line inspection (neck, rim, body, base).
- **Architecture:** YOLO11s-seg (`nc=3`, `ch=3`, `imgsz=640`, parameters: 10,083,449).
- **Target Classes:**
  - `0: broken_large` (Major glass fracture / missing wall section)
  - `1: broken_small` (Chipped lip / rim fissure)
  - `2: contamination` (Foreign internal particulate / fluid residue)
- **Training Setup:** 60 epochs, AdamW optimizer (`lr0=0.001`, `lrf=0.01`, weight decay 0.0005), Cosine Annealing with 10-epoch mosaic suppression.
- **Validation Metrics:**
  - Mask Precision: **76.91%** | Mask Recall: **82.22%** | **F1: 79.48%**
  - Mask mAP@50: **80.79%** | Mask mAP@50:95: **56.28%**
  - Box mAP@50: **80.79%** | Box mAP@50:95: **59.66%**

### 3.3 Steel Strip Surface Defect Detector (`steel_production_best.pt`)
- **Domain:** High-speed hot-rolled steel strip surface inspection (NEU-DET).
- **Architecture:** YOLOv8s Object Detector (`nc=6`, `ch=3`, `imgsz=640`, parameters: 10,638,536).
- **Target Classes:**
  - `0: crazing` (Interconnected thermal micro-fissure network)
  - `1: inclusion` (Non-metallic refractory / slag particle entrapment)
  - `2: patches` (Localized oxidation plaques)
  - `3: pitted_surface` (Cavity depressions)
  - `4: rolled-in_scale` (Pressed mill oxide scale)
  - `5: scratches` (Longitudinal mechanical abrasions)
- **Training Setup:** 40 epochs, AdamW optimizer (`lr0=0.001`, `lrf=0.01`), on-chip Hailo NMS offload target.
- **Validation Metrics:**
  - Precision: **76.23%** | Recall: **80.55%** | **F1: 78.33%**
  - Box mAP@50: **82.83%** (Literature baseline Faster R-CNN: 75.80%)
  - Box mAP@50:95: **45.28%**

### 3.4 Grid Structural Mesh Segmenter (`grid_production_best.pt`)
- **Domain:** Wire mesh, woven netting, and periodic patterned surfaces.
- **Architecture:** YOLO11s-seg (`nc=5`, `ch=3`, `imgsz=640`, parameters: 10,084,223).
- **Target Classes:**
  - `0: bent` (Deformed wire strut)
  - `1: broken` (Severed wire intersection / mesh aperture)
  - `2: glue` (Excess binding agent blob)
  - `3: metal_contamination` (Conductive metallic flakes)
  - `4: thread` (Stray fibrous textile inclusion)
- **Training Setup:** 30 epochs, AdamW optimizer (`lr0=0.001`, `lrf=0.01`), physical deformation mask alignment.
- **Validation Metrics:**
  - Mask Precision: **63.56%** | Mask Recall: **75.30%** | **F1: 68.94%**
  - Mask mAP@50: **69.75%** | Mask mAP@50:95: **36.76%**
  - Per-Class Highlights: `glue` (98.75%), `metal_contamination` (95.41%), `broken` (90.94%), `thread` (86.48%).

---

## 4. Export and Edge Hardware Deployment

The trained `.pt` checkpoints are compiled for hardware acceleration through the following stages:

```text
PyTorch Checkpoint (.pt)
  │
  ▼
ONNX Export (opset=17, 640x640 static input)
  │
  ▼
Hailo Dataflow Compiler (DFC) Calibration & INT8 Quantization
  │
  ▼
Compiled Hailo Executable Format (.hef)
  │
  ▼
Raspberry Pi 5 + Hailo-8 M.2 AI HAT+ (26 TOPS, 23.5 - 25.9 ms Latency)
```

The resulting compiled binaries are deployed in `pi_deployment/models/`:
- `tile_hailo8.hef` (19.2 MB)
- `bottle_hailo8.hef` (19.2 MB)
- `steel_defects_hailo8.hef` (20.1 MB)
- `grid_hailo8.hef` (19.2 MB)
