# Full Training Models: Production Weights & Accuracy Benchmarks

This directory stores the production-grade PyTorch (`.pt`) model weights trained on industrial defect classification and segmentation datasets (MVTec AD and NEU-DET). These models represent the high-precision floating-point (FP32/FP16) source weights before quantization and compilation into Hailo-8 HEF binaries.

---

## 1. Executive Performance Summary

| Model Weights File | Task | Architecture | Target Defect Classes | Epochs | Precision | Recall | **F1-Score** | **mAP@50** | **mAP@50:95** | File Size |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [`tile_production_best.pt`](file:///d:/Defect-classifications/full_training/tile_production_best.pt) | Segmentation | YOLO11s-seg | 5 classes (`crack`, `glue_strip`, `gray_stroke`, `oil`, `rough`) | 100 | **89.29%** | **94.87%** | **91.99%** | **98.50%** | **68.16%** | 20.5 MB |
| [`bottle_production_best.pt`](file:///d:/Defect-classifications/full_training/bottle_production_best.pt) | Segmentation | YOLO11s-seg | 3 classes (`broken_large`, `broken_small`, `contamination`) | 60 | **76.91%** | **82.22%** | **79.48%** | **80.79%** | **56.28%** | 20.5 MB |
| [`steel_production_best.pt`](file:///d:/Defect-classifications/full_training/steel_production_best.pt) | Detection | YOLOv8s / 11s | 6 classes (`crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`) | 40 | **76.23%** | **80.55%** | **78.33%** | **82.83%** | **45.28%** | 21.7 MB |
| [`grid_production_best.pt`](file:///d:/Defect-classifications/full_training/grid_production_best.pt) | Segmentation | YOLO11s-seg | 5 classes (`bent`, `broken`, `glue`, `metal_contamination`, `thread`) | 30 | **63.56%** | **75.30%** | **68.94%** | **69.75%** | **36.76%** | 20.5 MB |

> **Note on Metric Definitions:**
> - **Precision ($P$):** $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$ (minimizes false alarms on flawless goods).
> - **Recall ($R$):** $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$ (minimizes defective escapes).
> - **F1-Score:** Harmonic mean of precision and recall: $\text{F1} = \frac{2 \cdot P \cdot R}{P + R}$.
> - **mAP@50:** Mean Average Precision at Intersection-over-Union (IoU) threshold of 0.50.
> - **mAP@50-95:** Average mAP computed across IoU thresholds from 0.50 to 0.95 in steps of 0.05.
> - For segmentation models (`tile`, `bottle`, `grid`), reported values reflect pixel-level **Mask metrics**.

---

## 2. Detailed Per-Model Specifications

### 2.1 Tile Defect Segmenter (`tile_production_best.pt`)
- **Primary Domain:** Industrial ceramic and glazed surface defect detection.
- **Model Architecture:** YOLO11s-seg (`nc=5`, `ch=3`, `imgsz=640`).
- **Defect Classes:**
  - `0: crack` (Severe structural fracture)
  - `1: glue_strip` (Foreign adhesive deposit)
  - `2: gray_stroke` (Abrasive surface discoloration)
  - `3: oil` (Hydrocarbon contamination spot)
  - `4: rough` (Texture deformation / uneven glazing)
- **Training Configuration:**
  - Total Epochs: 100 (Optimal convergence reached at Epoch 95)
  - Batch Size: 16 | Image Resolution: 640x640
  - Optimizer: AdamW (`lr0=0.001`, `lrf=0.01`, `weight_decay=0.005`, `momentum=0.937`)
  - Learning Rate Schedule: Cosine Annealing (`cos_lr=True`)
  - Augmentations: RandomAffine (degrees=45, scale=0.3), FlipLR=0.5, FlipUD=0.5, CopyPaste=0.3, Erasing=0.4
- **Dual Metric Breakdown:**
  - **Mask Metrics:** Precision: **89.29%** | Recall: **94.87%** | **F1: 91.99%** | **Mask mAP@50: 98.50%** | Mask mAP@50-95: 68.16%
  - **Box Metrics:** Precision: 89.29% | Recall: 94.87% | F1: 91.99% | Box mAP@50: 98.50% | Box mAP@50-95: 63.80%

---

### 2.2 Bottle Defect Segmenter (`bottle_production_best.pt`)
- **Primary Domain:** Glass and plastic container bottling lines (neck, body, and base inspection).
- **Model Architecture:** YOLO11s-seg (`nc=3`, `ch=3`, `imgsz=640`).
- **Defect Classes:**
  - `0: broken_large` (Major glass fracture / missing piece)
  - `1: broken_small` (Chipped rim / hairline crack)
  - `2: contamination` (Foreign particulate / residual liquid)
- **Training Configuration:**
  - Total Epochs: 60 (Optimal convergence reached at Epoch 40)
  - Batch Size: 16 | Image Resolution: 640x640
  - Optimizer: AdamW (`lr0=0.001`, `lrf=0.01`, `weight_decay=0.0005`, `momentum=0.937`)
  - Learning Rate Schedule: Cosine Annealing with 10-epoch mosaic suppression
  - Augmentations: RandAugment, Scale=0.1, Translate=0.1, FlipLR=0.5, Erasing=0.4
- **Dual Metric Breakdown:**
  - **Mask Metrics:** Precision: **76.91%** | Recall: **82.22%** | **F1: 79.48%** | **Mask mAP@50: 80.79%** | Mask mAP@50-95: 56.28%
  - **Box Metrics:** Precision: 76.91% | Recall: 82.22% | F1: 79.48% | Box mAP@50: 80.79% | Box mAP@50-95: 59.66%

---

### 2.3 Steel Surface Defect Detector (`steel_production_best.pt`)
- **Primary Domain:** High-speed hot-rolled steel strip surface inspection (NEU-DET dataset).
- **Model Architecture:** YOLOv8s / YOLO11s Object Detector (`nc=6`, `ch=3`, `imgsz=640`).
- **Defect Classes:**
  - `0: crazing` (Fine network of surface fissures)
  - `1: inclusion` (Embedded non-metallic impurities)
  - `2: patches` (Irregular oxide / metal plaques)
  - `3: pitted_surface` (Cavity depressions)
  - `4: rolled-in_scale` (Pressed mill scale)
  - `5: scratches` (Longitudinal mechanical abrasions)
- **Training Configuration:**
  - Total Epochs: 40 (Optimal convergence reached at Epoch 39)
  - Batch Size: 16 | Image Resolution: 640x640
  - Optimizer: AdamW (`lr0=0.001`, `lrf=0.01`)
  - Task: High-throughput bounding-box detection with on-chip Hailo NMS offloading
- **Metric Breakdown:**
  - **Precision:** **76.23%** (Peak single-epoch precision: 82.52%)
  - **Recall:** **80.55%** (Peak single-epoch recall: 80.69%)
  - **F1-Score:** **78.33%**
  - **Box mAP@50:** **82.83%**
  - **Box mAP@50-95:** **45.28%**

---

### 2.4 Grid Structural Mesh Segmenter (`grid_production_best.pt`)
- **Primary Domain:** Wire mesh, grid structures, and repetitive patterned surfaces.
- **Model Architecture:** YOLO11s-seg (`nc=5`, `ch=3`, `imgsz=640`).
- **Defect Classes:**
  - `0: bent` (Deformed wire structure)
  - `1: broken` (Severed wire intersection)
  - `2: glue` (Excess binding agent contamination)
  - `3: metal_contamination` (Foreign conductive flakes)
  - `4: thread` (Stray fibrous inclusions)
- **Training Configuration:**
  - Total Epochs: 30 (Optimal convergence reached at Epoch 19)
  - Batch Size: 16 | Image Resolution: 640x640
  - Optimizer: AdamW (`lr0=0.001`, `lrf=0.01`)
- **Dual Metric Breakdown:**
  - **Mask Metrics:** Precision: **63.56%** | Recall: **75.30%** | **F1: 68.94%** | **Mask mAP@50: 69.75%** | Mask mAP@50-95: 36.76%
  - **Box Metrics:** Precision: 61.51% | Recall: 72.97% | F1: 66.76% | Box mAP@50: 66.31% | Box mAP@50-95: 49.85%

---

## 3. Relationship to Quantized Hailo-8 Binaries

These `.pt` files act as the single source of truth for quantization:
1. They are imported by [`HAT+ quatization/compile_hef.py`](file:///d:/Defect-classifications/HAT+%20quatization/compile_hef.py).
2. Quantized into INT8 using Hailo Dataflow Compiler (DFC) calibration.
3. Compiled into `.hef` files deployed under [`pi_deployment/models/`](file:///d:/Defect-classifications/pi_deployment/models/).
