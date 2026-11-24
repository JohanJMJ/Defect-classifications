# Results

## 2.1 Detection & Segmentation Accuracy (Production Models)

All four production models achieve over 90% mAP@0.5 on physically aligned industrial test splits:

| Domain | Model (Task) | Params | Precision | Recall | F1 | mAP@0.5 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Tile** | YOLO11s-seg (mask) | 10.08 M | 89.29% | 94.87% | 91.99% | **98.50%** |
| **Bottle** | YOLO11s-seg (mask) | 10.08 M | 88.54% | 92.80% | 90.62% | **92.83%** |
| **Grid** | YOLO11s-seg (mask) | 10.08 M | 89.45% | 91.28% | 90.35% | **91.13%** |
| **Steel** | YOLOv8s-P2 (box) | 10.64 M | 86.40% | 91.02% | 88.65% | **90.23%** |

*All metrics evaluated on curated test sets. Grid additionally achieves 90.85% Box mAP@0.5.*

---

## 2.2 Hardware Latency & Throughput Benchmarks

Inference benchmarks executed natively on the Raspberry Pi 5 + Hailo-8 AI HAT+ demonstrate real-time deterministic performance across all pipeline stages:

| Pipeline Stage | Executing Hardware | Latency | Throughput |
| :--- | :--- | :---: | :---: |
| **Total Vision Perception** | Hailo-8 AI HAT+ (26 TOPS) | 23.5–25.9 ms | 38.6–42.5 FPS |
| **Fuzzy Triage** | ARM Cortex-A76 (CPU) | 0.05–0.08 ms | >12,000 FPS |
| **LLaMA Prompt Prefill (TTFT)** | ARM Cortex-A76 (4 Threads) | 320–380 ms | 110–130 tok/s |
| **LLM Token Decode Generation** | ARM Cortex-A76 (4 Threads) | 3.0–3.5 s | 7.88–7.95 tok/s |
| **Total Advisory Wall-Clock Time** | ARM Cortex-A76 (ExecuTorch) | 6.2–7.8 s | Async Advisory |

The physical sorting line is gated exclusively by vision perception and triage (< 26 ms total), maintaining $\approx 40\text{ FPS}$ line rate while the ExecuTorch LLaMA 3.2 1B engine generates root-cause and CAPA advisories asynchronously in a background thread.

---

## 2.3 Resource Utilization, Power & Thermal Stability

Real-time hardware telemetry captured across all four operational states on the Raspberry Pi 5 + Hailo-8 AI HAT+ confirms sustained thermal and power stability for 24/7 industrial deployment:

| Metric | Idle | Vision Only (Hailo-8) | LLM Only (ExecuTorch) | Combined Peak Load |
| :--- | :---: | :---: | :---: | :---: |
| **CPU Utilization** | 1.2% | 3.8% | 96.4% | **98.1%** |
| **System RAM (RSS)** | 740 MB | 860 MB | 2,780 MB | **2,870 MB** |
| **System Power Draw** | 4.2 W | 6.7 W | 9.8 W | **11.8 W** |
| **SoC Core Temp** | 42.6 °C | 45.1 °C | 56.8 °C | **58.6 °C** |
| **Hailo-8 Die Temp** | 38.2 °C | 47.4 °C | 39.0 °C | **47.8 °C** |
