#!/usr/bin/env python3
"""
Lightweight Native HailoRT Inference for Raspberry Pi AI HAT+ (26 TOPS)
=======================================================================
ZERO PyTorch / ZERO Ultralytics dependencies.
Requires ONLY:
  - hailo_platform (installed via: sudo apt install hailo-all)
  - numpy
  - opencv-python (or python3-opencv)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Try importing HailoRT Python bindings
try:
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        ConfigureParams,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType
    )
    HAILORT_AVAILABLE = True
except ImportError:
    HAILORT_AVAILABLE = False


# Class labels
MODEL_CLASSES = {
    "steel_defects": {
        0: "crazing",
        1: "inclusion",
        2: "patches",
        3: "pitted_surface",
        4: "rolled-in_scale",
        5: "scratches",
    },
    "steel": {
        0: "crazing",
        1: "inclusion",
        2: "patches",
        3: "pitted_surface",
        4: "rolled-in_scale",
        5: "scratches",
    },
    "bottle": {
        0: "broken_large",
        1: "broken_small",
        2: "contamination",
    },
    "grid": {
        0: "bent",
        1: "broken",
        2: "glue",
        3: "metal_contamination",
        4: "thread",
    },
    "tile": {
        0: "crack",
        1: "glue_strip",
        2: "gray_stroke",
        3: "oil",
        4: "rough",
    },
}
CLASS_NAMES = {0: "anomaly"}
STEEL_CLASSES = MODEL_CLASSES["steel_defects"]


def get_model_classes(model_name, models_dir=None):
    """Retrieve defect class dictionary for a model, checking models.yaml then MODEL_CLASSES."""
    try:
        import yaml
        yaml_path = (Path(models_dir) if models_dir else Path(__file__).resolve().parent / "models") / "models.yaml"
        if yaml_path.is_file():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                models_cfg = data.get("models", {})
                for k, v in models_cfg.items():
                    if k.lower() == str(model_name).lower() or v.get("category", "").lower() == str(model_name).lower():
                        if "classes" in v and isinstance(v["classes"], dict):
                            return {int(idx): str(name) for idx, name in v["classes"].items()}
    except Exception:
        pass

    norm = str(model_name).lower()
    if norm.startswith("steel"):
        norm = "steel_defects"
    elif "bottle" in norm:
        norm = "bottle"
    elif "grid" in norm:
        norm = "grid"
    elif "tile" in norm:
        norm = "tile"

    return MODEL_CLASSES.get(norm, {0: "anomaly"})


# Colors for visualization (BGR)
MASK_COLOR = (0, 0, 255)      # Red for defect overlay
BOX_COLOR = (0, 165, 255)     # Orange for bounding box
TEXT_COLOR = (255, 255, 255)  # White text


def letterbox(img, target_size=(640, 640), fill_color=(114, 114, 114)):
    """Resize and pad image while preserving aspect ratio."""
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = tw - nw
    pad_h = th - nh
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    canvas = np.full((th, tw, 3), fill_color, dtype=np.uint8)
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = resized

    pad_info = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "orig_shape": (h, w),
        "target_shape": (th, tw)
    }
    return canvas, pad_info


def dfl_decode(reg_tensor):
    """
    Decode Distribution Focal Loss (DFL) coordinates.
    reg_tensor: (N, 64) -> 4 box offsets (left, top, right, bottom)
    """
    n = reg_tensor.shape[0]
    reshaped = reg_tensor.reshape(n, 4, 16)
    # Softmax over the 16 bin values
    exp_x = np.exp(reshaped - np.max(reshaped, axis=-1, keepdims=True))
    prob = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    weights = np.arange(16, dtype=np.float32)
    return np.sum(prob * weights, axis=-1)  # (N, 4)


def make_anchors(imgsz=640, strides=(8, 16, 32)):
    """Generate anchor coordinates and stride mappings for 640x640 resolution."""
    anchors = []
    stride_tensor = []
    for s in strides:
        grid_size = imgsz // s
        y, x = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
        grid = np.stack([x, y], axis=-1).reshape(-1, 2) + 0.5
        anchors.append(grid)
        stride_tensor.append(np.full((grid.shape[0], 1), s, dtype=np.float32))

    return np.concatenate(anchors, axis=0), np.concatenate(stride_tensor, axis=0)


class HailoYOLOSegmentor:
    """Lightweight HailoRT YOLO Segmentation and Detection Engine."""

    def __init__(self, hef_path, category=None):
        if not HAILORT_AVAILABLE:
            raise RuntimeError(
                "HailoRT ('hailo_platform') is not installed.\n"
                "On Raspberry Pi OS, install it using: sudo apt install hailo-all"
            )

        self.hef_path = str(hef_path)
        self.model_name = category or Path(hef_path).stem.replace("_hailo8", "").replace("_hailo8l", "")
        self.class_map = get_model_classes(self.model_name, Path(hef_path).parent)
        self.hef = HEF(self.hef_path)
        self.target = VDevice()

        # Configure network group on PCIe
        configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.target.configure(self.hef, configure_params)[0]
        self.network_group_params = self.network_group.create_params()

        # Input & Output Stream Parameters (UINT8 input for zero CPU conversion, FLOAT32 output)
        self.input_params = InputVStreamParams.make(self.network_group, format_type=FormatType.UINT8)
        self.output_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.output_infos = self.hef.get_output_vstream_infos()
        self.output_infos_names = [x.name for x in self.output_infos]

        # Persistent inference pipeline
        self._activated = self.network_group.activate(self.network_group_params)
        self._activated.__enter__()
        self.pipeline = InferVStreams(self.network_group, self.input_params, self.output_params)
        self.pipeline.__enter__()

        # Precompute anchors for 640x640
        self.anchors, self.strides = make_anchors(imgsz=640)

    def __del__(self):
        try:
            if getattr(self, "pipeline", None) is not None:
                self.pipeline.__exit__(None, None, None)
            if getattr(self, "_activated", None) is not None:
                self._activated.__exit__(None, None, None)
        except Exception:
            pass

    def run_raw_inference(self, image_rgb):
        """Run input frame through Hailo-8 hardware pipeline."""
        input_batch = np.expand_dims(image_rgb, axis=0)
        results = self.pipeline.infer({self.input_info.name: input_batch})
        return [results[x.name] for x in self.output_infos]

    def decode_outputs(self, outputs, pad_info, conf_thresh=0.25, iou_thresh=0.50):
        """
        Pure NumPy post-processing for YOLO11 multi-class segmentation and detection.
        Zero PyTorch required.
        """
        if isinstance(outputs, dict):
            outputs = list(outputs.values())

        out_names = getattr(self, "output_infos_names", [])
        is_nms = (len(outputs) == 1) or any(
            isinstance(x, str) and ("postprocess" in x.lower() or "nms" in x.lower())
            for x in out_names
        )

        if is_nms:
            return self._decode_hailo_nms(outputs[0], pad_info, conf_thresh)

        # Multi-Class Segmentation with prototype mask (shape 160x160x32)
        proto_tensor = None
        remaining = []
        for o in outputs:
            if (o.ndim == 4 and o.shape[1] == 160 and o.shape[2] == 160 and o.shape[3] == 32) or \
               (o.ndim == 3 and o.shape[0] == 160 and o.shape[1] == 160 and o.shape[2] == 32):
                proto_tensor = o
            else:
                remaining.append(o)

        if proto_tensor is not None:
            proto = proto_tensor[0] if proto_tensor.ndim == 4 else proto_tensor
            return self._decode_segmentation(remaining, proto, pad_info, conf_thresh, iou_thresh)

        return self._decode_raw_detection(outputs, pad_info, conf_thresh, iou_thresh)

    _decode = decode_outputs

    def _decode_hailo_nms(self, output, pad_info, conf_thresh):
        scale = pad_info["scale"]
        pad_left = pad_info["pad_left"]
        pad_top = pad_info["pad_top"]
        orig_h, orig_w = pad_info["orig_shape"]

        detections = []
        class_map = self.class_map

        # Remove batch dimension if present
        if isinstance(output, np.ndarray) and output.ndim >= 3 and output.shape[0] == 1:
            output = output[0]
        elif isinstance(output, (list, tuple)) and len(output) == 1 and isinstance(output[0], (list, tuple, np.ndarray)):
            output = output[0]

        # Reshape flat array if needed
        if isinstance(output, np.ndarray) and output.ndim == 1:
            num_classes = len(class_map)
            if num_classes > 0 and output.size % (num_classes * 5) == 0:
                output = output.reshape(num_classes, -1, 5)
            elif output.size % 6 == 0:
                output = output.reshape(-1, 6)
            elif output.size % 5 == 0:
                output = output.reshape(1, -1, 5)

        # Case A: Grouped by class: [num_classes, max_boxes, 5] -> [ymin, xmin, ymax, xmax, score]
        if isinstance(output, (list, tuple)) or (isinstance(output, np.ndarray) and output.ndim >= 2 and output.shape[-1] == 5):
            for class_id, class_dets in enumerate(output):
                if class_dets is None or len(class_dets) == 0:
                    continue
                class_dets = np.asarray(class_dets)
                if class_dets.ndim == 1 and class_dets.size == 5:
                    class_dets = class_dets.reshape(1, 5)
                elif class_dets.ndim != 2 or class_dets.shape[-1] != 5:
                    continue

                for det in class_dets:
                    ymin, xmin, ymax, xmax, score = [float(v) for v in det]
                    if score < conf_thresh:
                        continue
                    if max(abs(ymin), abs(xmin), abs(ymax), abs(xmax)) <= 1.05:
                        bx1, by1 = xmin * 640.0, ymin * 640.0
                        bx2, by2 = xmax * 640.0, ymax * 640.0
                    else:
                        bx1, by1 = xmin, ymin
                        bx2, by2 = xmax, ymax

                    orig_x1 = max(0.0, min(orig_w, (bx1 - pad_left) / scale))
                    orig_y1 = max(0.0, min(orig_h, (by1 - pad_top) / scale))
                    orig_x2 = max(0.0, min(orig_w, (bx2 - pad_left) / scale))
                    orig_y2 = max(0.0, min(orig_h, (by2 - pad_top) / scale))

                    ix1, iy1 = int(round(orig_x1)), int(round(orig_y1))
                    ix2, iy2 = int(round(orig_x2)), int(round(orig_y2))

                    final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    if ix2 > ix1 and iy2 > iy1:
                        final_mask[iy1:iy2, ix1:ix2] = 1

                    area_px = float((ix2 - ix1) * (iy2 - iy1))
                    polygon = [[ix1, iy1], [ix2, iy1], [ix2, iy2], [ix1, iy2]]
                    label = class_map.get(class_id, f"defect_{class_id}")

                    detections.append({
                        "label": label,
                        "confidence": round(score, 4),
                        "bbox": [round(orig_x1, 1), round(orig_y1, 1), round(orig_x2, 1), round(orig_y2, 1)],
                        "mask_polygon_points": 4,
                        "polygon": polygon,
                        "mask_area_px": round(area_px, 1),
                        "mask_array": final_mask,
                    })

        # Case B: Detections array [N, 6] -> [ymin, xmin, ymax, xmax, score, class_id]
        elif isinstance(output, np.ndarray) and output.ndim == 2 and output.shape[-1] >= 6:
            for det in output:
                score = float(det[4])
                if score < conf_thresh:
                    continue
                cls_id = int(round(float(det[5])))
                ymin, xmin, ymax, xmax = [float(v) for v in det[:4]]
                if max(abs(ymin), abs(xmin), abs(ymax), abs(xmax)) <= 1.05:
                    bx1, by1 = xmin * 640.0, ymin * 640.0
                    bx2, by2 = xmax * 640.0, ymax * 640.0
                else:
                    bx1, by1 = xmin, ymin
                    bx2, by2 = xmax, ymax

                orig_x1 = max(0.0, min(orig_w, (bx1 - pad_left) / scale))
                orig_y1 = max(0.0, min(orig_h, (by1 - pad_top) / scale))
                orig_x2 = max(0.0, min(orig_w, (bx2 - pad_left) / scale))
                orig_y2 = max(0.0, min(orig_h, (by2 - pad_top) / scale))

                ix1, iy1 = int(round(orig_x1)), int(round(orig_y1))
                ix2, iy2 = int(round(orig_x2)), int(round(orig_y2))

                final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                if ix2 > ix1 and iy2 > iy1:
                    final_mask[iy1:iy2, ix1:ix2] = 1

                area_px = float((ix2 - ix1) * (iy2 - iy1))
                polygon = [[ix1, iy1], [ix2, iy1], [ix2, iy2], [ix1, iy2]]
                label = class_map.get(cls_id, f"defect_{cls_id}")

                detections.append({
                    "label": label,
                    "confidence": round(score, 4),
                    "bbox": [round(orig_x1, 1), round(orig_y1, 1), round(orig_x2, 1), round(orig_y2, 1)],
                    "mask_polygon_points": 4,
                    "polygon": polygon,
                    "mask_area_px": round(area_px, 1),
                    "mask_array": final_mask,
                })

        return detections

    def _decode_segmentation(self, tensors, proto, pad_info, conf_thresh, iou_thresh):
        scale = pad_info["scale"]
        pad_left = pad_info["pad_left"]
        pad_top = pad_info["pad_top"]
        orig_h, orig_w = pad_info["orig_shape"]

        reg_tensors = [o for o in tensors if o.shape[-1] == 64]
        cof_tensors = [o for o in tensors if o.shape[-1] == 32]
        cls_tensors = [o for o in tensors if o.shape[-1] not in (64, 32)]

        if len(cls_tensors) == 0 and len(cof_tensors) > 3:
            cls_tensors = cof_tensors[3:]
            cof_tensors = cof_tensors[:3]

        reg_tensors = sorted(reg_tensors, key=lambda x: x.shape[1] * x.shape[2], reverse=True)
        cls_tensors = sorted(cls_tensors, key=lambda x: x.shape[1] * x.shape[2], reverse=True)
        cof_tensors = sorted(cof_tensors, key=lambda x: x.shape[1] * x.shape[2], reverse=True)

        if not reg_tensors or not cls_tensors or not cof_tensors:
            return []

        all_reg = np.concatenate([r[0].reshape(-1, 64) for r in reg_tensors], axis=0)
        all_cof = np.concatenate([c[0].reshape(-1, 32) for c in cof_tensors], axis=0)
        all_cls = np.concatenate([c[0].reshape(-1, c.shape[-1]) for c in cls_tensors], axis=0)

        if np.max(all_cls) > 1.0 or np.min(all_cls) < 0.0:
            all_cls = 1.0 / (1.0 + np.exp(-np.clip(all_cls, -30.0, 30.0)))

        best_cls_ids = np.argmax(all_cls, axis=1)
        best_scores = np.max(all_cls, axis=1)

        candidate_indices = np.where(best_scores >= conf_thresh)[0]
        if len(candidate_indices) == 0:
            return []

        cand_scores = best_scores[candidate_indices]
        cand_classes = best_cls_ids[candidate_indices]
        cand_reg = all_reg[candidate_indices]
        cand_cof = all_cof[candidate_indices]
        cand_anchors = self.anchors[candidate_indices]
        cand_strides = self.strides[candidate_indices]

        dists = dfl_decode(cand_reg)
        x1 = (cand_anchors[:, 0] - dists[:, 0]) * cand_strides[:, 0]
        y1 = (cand_anchors[:, 1] - dists[:, 1]) * cand_strides[:, 0]
        x2 = (cand_anchors[:, 0] + dists[:, 2]) * cand_strides[:, 0]
        y2 = (cand_anchors[:, 1] + dists[:, 3]) * cand_strides[:, 0]

        boxes_for_nms = []
        for i in range(len(candidate_indices)):
            boxes_for_nms.append([float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])])

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, cand_scores.tolist(), float(conf_thresh), float(iou_thresh))
        if len(indices) == 0:
            return []

        keep_indices = indices.flatten() if isinstance(indices, np.ndarray) else [
            i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices
        ]

        detections = []
        class_map = self.class_map

        for ki in keep_indices:
            conf_val = float(cand_scores[ki])
            cls_id = int(cand_classes[ki])
            label = class_map.get(cls_id, f"defect_{cls_id}")

            bx1, by1 = float(x1[ki]), float(y1[ki])
            bx2, by2 = float(x2[ki]), float(y2[ki])

            coeff = cand_cof[ki]
            raw_mask = np.matmul(proto, coeff)
            sigmoid_mask = 1.0 / (1.0 + np.exp(-np.clip(raw_mask, -30.0, 30.0)))
            mask_640 = cv2.resize(sigmoid_mask, (640, 640), interpolation=cv2.INTER_LINEAR)

            crop_x1 = max(0, min(640, int(round(bx1))))
            crop_y1 = max(0, min(640, int(round(by1))))
            crop_x2 = max(0, min(640, int(round(bx2))))
            crop_y2 = max(0, min(640, int(round(by2))))

            binary_mask_640 = np.zeros((640, 640), dtype=np.uint8)
            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                binary_mask_640[crop_y1:crop_y2, crop_x1:crop_x2] = (
                    mask_640[crop_y1:crop_y2, crop_x1:crop_x2] > 0.5
                ).astype(np.uint8)

            unpad_y2 = 640 - pad_top
            unpad_x2 = 640 - pad_left
            unpad_mask = binary_mask_640[pad_top:unpad_y2, pad_left:unpad_x2]

            final_mask = (
                cv2.resize(unpad_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                if unpad_mask.size > 0
                else np.zeros((orig_h, orig_w), dtype=np.uint8)
            )

            orig_x1 = float(max(0.0, min(orig_w, (bx1 - pad_left) / scale)))
            orig_y1 = float(max(0.0, min(orig_h, (by1 - pad_top) / scale)))
            orig_x2 = float(max(0.0, min(orig_w, (bx2 - pad_left) / scale)))
            orig_y2 = float(max(0.0, min(orig_h, (by2 - pad_top) / scale)))

            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly_points = len(contours[0]) if len(contours) > 0 else 0
            area_px = float(np.sum(final_mask))

            polygon = []
            if len(contours) > 0:
                approx = cv2.approxPolyDP(contours[0], epsilon=0.01 * cv2.arcLength(contours[0], True), closed=True)
                polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

            detections.append({
                "label": label,
                "confidence": round(conf_val, 4),
                "bbox": [round(orig_x1, 1), round(orig_y1, 1), round(orig_x2, 1), round(orig_y2, 1)],
                "mask_polygon_points": int(poly_points),
                "polygon": polygon,
                "mask_area_px": round(area_px, 1),
                "mask_array": final_mask,
            })

        return detections

    def _decode_raw_detection(self, outputs, pad_info, conf_thresh, iou_thresh):
        scale = pad_info["scale"]
        pad_left = pad_info["pad_left"]
        pad_top = pad_info["pad_top"]
        orig_h, orig_w = pad_info["orig_shape"]

        reg_tensors = sorted([o for o in outputs if o.shape[-1] == 64], key=lambda x: x.shape[1] * x.shape[2], reverse=True)
        cls_tensors = sorted([o for o in outputs if o.shape[-1] not in (64, 32)], key=lambda x: x.shape[1] * x.shape[2], reverse=True)

        if not reg_tensors or not cls_tensors:
            return []

        all_reg = np.concatenate([r[0].reshape(-1, 64) for r in reg_tensors], axis=0)
        all_cls = np.concatenate([c[0].reshape(-1, c.shape[-1]) for c in cls_tensors], axis=0)

        if np.max(all_cls) > 1.0 or np.min(all_cls) < 0.0:
            all_cls = 1.0 / (1.0 + np.exp(-np.clip(all_cls, -30.0, 30.0)))

        best_cls_ids = np.argmax(all_cls, axis=1)
        best_scores = np.max(all_cls, axis=1)

        candidate_indices = np.where(best_scores >= conf_thresh)[0]
        if len(candidate_indices) == 0:
            return []

        cand_scores = best_scores[candidate_indices]
        cand_classes = best_cls_ids[candidate_indices]
        cand_reg = all_reg[candidate_indices]
        cand_anchors = self.anchors[candidate_indices]
        cand_strides = self.strides[candidate_indices]

        dists = dfl_decode(cand_reg)
        x1 = (cand_anchors[:, 0] - dists[:, 0]) * cand_strides[:, 0]
        y1 = (cand_anchors[:, 1] - dists[:, 1]) * cand_strides[:, 0]
        x2 = (cand_anchors[:, 0] + dists[:, 2]) * cand_strides[:, 0]
        y2 = (cand_anchors[:, 1] + dists[:, 3]) * cand_strides[:, 0]

        boxes_for_nms = []
        for i in range(len(candidate_indices)):
            boxes_for_nms.append([float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])])

        indices = cv2.dnn.NMSBoxes(boxes_for_nms, cand_scores.tolist(), float(conf_thresh), float(iou_thresh))
        if len(indices) == 0:
            return []

        keep_indices = indices.flatten() if isinstance(indices, np.ndarray) else [
            i[0] if isinstance(i, (list, tuple, np.ndarray)) else i for i in indices
        ]

        detections = []
        class_map = self.class_map

        for ki in keep_indices:
            conf_val = float(cand_scores[ki])
            cls_id = int(cand_classes[ki])
            label = class_map.get(cls_id, f"defect_{cls_id}")

            bx1, by1 = float(x1[ki]), float(y1[ki])
            bx2, by2 = float(x2[ki]), float(y2[ki])

            orig_x1 = float(max(0.0, min(orig_w, (bx1 - pad_left) / scale)))
            orig_y1 = float(max(0.0, min(orig_h, (by1 - pad_top) / scale)))
            orig_x2 = float(max(0.0, min(orig_w, (bx2 - pad_left) / scale)))
            orig_y2 = float(max(0.0, min(orig_h, (by2 - pad_top) / scale)))

            ix1, iy1 = int(round(orig_x1)), int(round(orig_y1))
            ix2, iy2 = int(round(orig_x2)), int(round(orig_y2))

            final_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            if ix2 > ix1 and iy2 > iy1:
                final_mask[iy1:iy2, ix1:ix2] = 1

            area_px = float((ix2 - ix1) * (iy2 - iy1))
            polygon = [[ix1, iy1], [ix2, iy1], [ix2, iy2], [ix1, iy2]]

            detections.append({
                "label": label,
                "confidence": round(conf_val, 4),
                "bbox": [round(orig_x1, 1), round(orig_y1, 1), round(orig_x2, 1), round(orig_y2, 1)],
                "mask_polygon_points": 4,
                "polygon": polygon,
                "mask_area_px": round(area_px, 1),
                "mask_array": final_mask,
            })

        return detections


def draw_visualizations(image_bgr, detections, latency_ms=None, fps=None):
    """Draw defect bounding boxes, translucent segmentation masks, and HUD overlay."""
    annotated = image_bgr.copy()
    overlay = image_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        mask = det["mask_array"]

        # Apply translucent color overlay where mask == 1
        overlay[mask == 1] = MASK_COLOR

        # Draw contour line around mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Draw text label
        label = f"{det['label']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, max(th + 8, y1)), BOX_COLOR, -1)
        cv2.putText(annotated, label, (x1 + 2, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

    # Blend translucent mask (alpha = 0.4)
    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

    # Draw HUD overlay
    if latency_ms is not None:
        fps_text = f" | {fps:.1f} FPS" if fps is not None else ""
        hud = f"Hailo-8 NPU (Pure HailoRT) | Latency: {latency_ms:.1f}ms{fps_text}"
        cv2.putText(annotated, hud, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return annotated


def run_benchmark(engine, sample_path, iterations=100):
    """Evaluate pure hardware HailoRT latency and throughput."""
    img = cv2.imread(str(sample_path))
    canvas, _ = letterbox(img, (640, 640))
    rgb_img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    print("\n" + "=" * 60)
    print("⚡ Running Pure HailoRT Hardware Benchmark (No PyTorch / No Ultralytics)")
    print(f"   Model       : {Path(engine.hef_path).name}")
    print(f"   Iterations  : {iterations}")
    print("=" * 60)

    # Warmup
    print("Warming up Hailo-8 NPU...")
    for _ in range(10):
        _ = engine.run_raw_inference(rgb_img)

    latencies = []
    print("Benchmarking NPU hardware inference...")
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.run_raw_inference(rgb_img)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    avg_lat = np.mean(latencies)
    med_lat = np.median(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    fps = 1000.0 / avg_lat

    print("\n📊 Pure Hailo-8 Hardware Performance:")
    print(f"   Throughput     : {fps:.2f} FPS")
    print(f"   Average Latency: {avg_lat:.2f} ms")
    print(f"   Median Latency : {med_lat:.2f} ms")
    print(f"   Min / Max      : {min_lat:.2f} ms / {max_lat:.2f} ms")
    print("=" * 60 + "\n")


def get_available_models(models_dir):
    """Scan models_dir and return dict mapping category/model aliases to Path."""
    models = {}
    if not models_dir.exists():
        return models
    for p in sorted(models_dir.glob("*.hef")):
        stem = p.stem  # e.g. bottle_hailo8
        clean = stem.replace("_hailo8l", "").replace("_hailo8", "").lower()
        models[clean] = p
        models[stem.lower()] = p
        models[p.name.lower()] = p
        if clean.startswith("steel"):
            models["steel"] = p
            models["steel_defects"] = p
            models["steel_defect"] = p
        else:
            models[stem.split("_")[0].lower()] = p
    return models


def list_available_models(models_dir, sample_images_dir=None):
    """Print a clean table of all installed .hef models and sample image status."""
    if not models_dir.exists():
        print(f"❌ Error: Models directory '{models_dir}' not found.")
        return
    hef_files = sorted(models_dir.glob("*.hef"))
    if not hef_files:
        print("❌ No .hef models found in models directory.")
        return

    print("=" * 72)
    print(f"📦 Installed Hailo-8 HEF Models ({len(hef_files)} available)")
    print("=" * 72)
    print(f"{'Category':<14} {'Model Filename':<26} {'Size (MB)':<12} {'Sample Images':<16}")
    print("-" * 72)
    for p in hef_files:
        cat = "steel_defects" if p.stem.startswith("steel") else p.stem.split("_")[0]
        size_mb = p.stat().st_size / (1024 * 1024)
        has_samples = sample_images_dir and (sample_images_dir / cat).is_dir()
        sample_status = f"sample_images/{cat}" if has_samples else "Root folder"
        print(f"{cat:<14} {p.name:<26} {size_mb:<12.2f} {sample_status:<16}")
    print("=" * 72)
    print("Quick Run Examples:")
    print("  python3 inference.py --model bottle")
    print("  ./run_inference.sh --model tile")
    print("  ./run_inference.sh grid")
    print("  ./run_inference.sh tile --camera\n")


def resolve_model_path(model_arg, models_dir):
    """
    Resolve model path from:
      1. Direct path or file in cwd: 'models/bottle_hailo8.hef'
      2. File directly in models_dir: 'bottle_hailo8.hef'
      3. Category name: 'bottle' -> models/bottle_hailo8.hef
      4. Stem: 'bottle_hailo8' -> models/bottle_hailo8.hef
    """
    if not model_arg:
        # Default fallback
        steel_path = models_dir / "steel_defects_hailo8.hef"
        if steel_path.exists():
            return steel_path
        tile_path = models_dir / "tile_hailo8.hef"
        if tile_path.exists():
            return tile_path
        hefs = sorted(models_dir.glob("*.hef"))
        return hefs[0] if hefs else None

    # 1. Direct path check
    direct_p = Path(model_arg)
    if direct_p.is_file():
        return direct_p.resolve()

    # 2. Check inside models_dir directly or via basename
    base_name = direct_p.name
    cand = models_dir / base_name
    if cand.is_file():
        return cand.resolve()

    cand_hef = models_dir / f"{base_name}.hef"
    if cand_hef.is_file():
        return cand_hef.resolve()

    cand_h8 = models_dir / f"{base_name}_hailo8.hef"
    if cand_h8.is_file():
        return cand_h8.resolve()

    cand_h8l = models_dir / f"{base_name}_hailo8l.hef"
    if cand_h8l.is_file():
        return cand_h8l.resolve()

    # 3. Lookup in available models alias dict
    clean_arg = base_name.replace(".hef", "").strip().lower()
    if clean_arg in ["steel", "steel_defect", "steel_defects"]:
        steel_p = models_dir / "steel_defects_hailo8.hef"
        if steel_p.is_file():
            return steel_p.resolve()

    avail = get_available_models(models_dir)
    if clean_arg in avail:
        return avail[clean_arg].resolve()

    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Native HailoRT YOLO segmentation and detection inference (No PyTorch, No Ultralytics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Model name, category (e.g. 'steel_defects', 'tile', 'bottle', 'grid'), or path to .hef model"
    )
    parser.add_argument(
        "-c", "--category",
        type=str,
        default=None,
        help="Alias for --model (e.g. steel_defects, tile, bottle, grid)"
    )
    parser.add_argument(
        "-l", "--list-models",
        action="store_true",
        help="List all installed .hef models and their sample images, then exit"
    )
    parser.add_argument(
        "-s", "--source",
        type=str,
        default=None,
        help="Image file, directory, or camera index (default: sample_images/<category>)"
    )
    parser.add_argument("-o", "--output-dir", type=str, default="results", help="Directory for output files")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument("--show", action="store_true", help="Display GUI window with masks")
    parser.add_argument("--no-save", action="store_true", help="Do not save output images")
    parser.add_argument("--benchmark", action="store_true", help="Run hardware speed benchmark")
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    sample_images_dir = script_dir / "sample_images"

    # Handle --list-models flag
    if args.list_models:
        list_available_models(models_dir, sample_images_dir)
        return

    # Select model
    selected_model_arg = args.category or args.model or "tile"
    model_path = resolve_model_path(selected_model_arg, models_dir)

    if model_path is None or not model_path.exists():
        print(f"❌ Error: Model '{selected_model_arg}' not found.")
        list_available_models(models_dir, sample_images_dir)
        sys.exit(1)

    category = "steel_defects" if model_path.stem.startswith("steel") else model_path.stem.split("_")[0].lower()

    # Determine default source if not provided
    if args.source is None:
        cat_sample_dir = sample_images_dir / category
        if cat_sample_dir.is_dir() and any(cat_sample_dir.iterdir()):
            source_path = cat_sample_dir
        elif sample_images_dir.is_dir():
            source_path = sample_images_dir
        else:
            source_path = Path("sample_images")
    else:
        source_path = Path(args.source)

    out_dir = Path(args.output_dir).resolve()
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🏭 Hailo-8 Native Inspection Engine (Lightweight Pure HailoRT)")
    print(f"   Category    : {category.upper()}")
    print(f"   Model       : {model_path.name}")
    print(f"   Model Path  : {model_path}")
    print(f"   Source      : {source_path}")
    print(f"   Dependencies: hailo_platform, numpy, cv2 (NO Ultralytics / NO PyTorch)")
    print("=" * 60)

    # Initialize Hailo engine
    try:
        engine = HailoYOLOSegmentor(model_path, category=category)
    except Exception as e:
        print(f"❌ Failed to initialize Hailo-8 engine: {e}")
        sys.exit(1)

    # Benchmark mode
    if args.benchmark:
        src = source_path
        if src.is_file():
            sample_img = src
        elif src.is_dir():
            sample_img = next((f for f in src.glob("*.*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"}), None)
        else:
            sample_img = None
        if sample_img:
            run_benchmark(engine, sample_img)
        else:
            print("❌ Error: No valid image found for benchmark.")
        return

    # Camera mode
    source_str = str(args.source) if args.source is not None else ""
    is_camera = source_str.isdigit() or source_str.startswith("/dev/video")
    if is_camera:
        cam_idx = int(source_str) if source_str.isdigit() else source_str
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open camera at {source_str}")
            sys.exit(1)

        print(f"📷 Streaming live camera from {source_str} (Press 'q' in window to exit)...")
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            canvas, pad_info = letterbox(frame, (640, 640))
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            outputs = engine.run_raw_inference(rgb)
            detections = engine.decode_outputs(outputs, pad_info, conf_thresh=args.conf, iou_thresh=args.iou)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
            prev_time = curr_time

            annotated = draw_visualizations(frame, detections, latency_ms=latency_ms, fps=fps)

            if args.show:
                cv2.imshow("Hailo-8 Live Defect Inspection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        return

    # Image / Directory mode
    src_path = source_path.resolve()
    if src_path.is_file():
        image_files = [src_path]
    elif src_path.is_dir():
        # If source is sample_images and category subfolder exists, use that subfolder
        cat_sub = src_path / category
        if src_path == sample_images_dir.resolve() and cat_sub.is_dir() and any(cat_sub.iterdir()):
            image_files = sorted(
                [f for f in cat_sub.iterdir() if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            )
        else:
            image_files = sorted(
                [f for f in src_path.iterdir() if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            )
    else:
        print(f"❌ Error: Source '{source_path}' does not exist.")
        sys.exit(1)

    print(f"📂 Processing {len(image_files)} image(s)...")
    summary_report = []

    for idx, img_path in enumerate(image_files, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        canvas, pad_info = letterbox(img_bgr, (640, 640))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        outputs = engine.run_raw_inference(rgb)
        detections = engine.decode_outputs(outputs, pad_info, conf_thresh=args.conf, iou_thresh=args.iou)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        status = f"🚨 DEFECT DETECTED ({len(detections)})" if len(detections) > 0 else "✅ PASS (Normal)"
        print(f"[{idx}/{len(image_files)}] {img_path.name:<25} -> {status:<25} ({latency_ms:.1f} ms)")

        det_summary = []
        for det in detections:
            print(f"     └─ {det['label']} (conf: {det['confidence']:.2f}, area: {det['mask_area_px']} px, bbox: {det['bbox']})")
            det_summary.append({
                "label": det["label"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "mask_area_px": det["mask_area_px"],
                "mask_polygon_points": det["mask_polygon_points"]
            })

        if not args.no_save:
            annotated = draw_visualizations(img_bgr, detections, latency_ms=latency_ms)
            out_file = out_dir / f"annotated_{img_path.name}"
            cv2.imwrite(str(out_file), annotated)

        summary_report.append({
            "filename": img_path.name,
            "status": "DEFECTIVE" if len(detections) > 0 else "PASS",
            "latency_ms": round(latency_ms, 2),
            "detections": det_summary
        })

        if args.show:
            annotated = draw_visualizations(img_bgr, detections, latency_ms=latency_ms)
            cv2.imshow("Hailo-8 Inspection", annotated)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    if args.show:
        cv2.destroyAllWindows()

    if not args.no_save:
        summary_path = out_dir / "detection_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_report, f, indent=2)

        print("\n" + "=" * 60)
        print("✅ Inspection Completed!")
        print(f"   Annotated images saved to : {out_dir}")
        print(f"   Summary JSON report saved : {summary_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
