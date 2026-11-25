import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from inference import HailoYOLOSegmentor, letterbox, CLASS_NAMES, resolve_model_path, list_available_models


def parse_yolo_labels(label_path, img_w, img_h):
    boxes = []
    masks = []

    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes, masks

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            coords = [float(p) for p in parts[1:]]
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]

            bx1 = max(0.0, min(xs))
            by1 = max(0.0, min(ys))
            bx2 = min(float(img_w), max(xs))
            by2 = min(float(img_h), max(ys))

            pts = np.array([[int(round(x)), int(round(y))] for x, y in zip(xs, ys)], dtype=np.int32)
            gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(gt_mask, [pts], 1)

            boxes.append([bx1, by1, bx2, by2])
            masks.append(gt_mask)

    return boxes, masks


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return float(ap)


def evaluate_matches(all_preds, all_gts, iou_thresholds, is_mask=False):
    total_gts = len(all_gts)
    if total_gts == 0:
        return 0.0, 0.0, 0.0, 0.0

    if len(all_preds) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Sort all predictions by confidence descending
    sorted_preds = sorted(all_preds, key=lambda x: x["conf"], reverse=True)

    gt_by_img = {}
    for gt in all_gts:
        gt_by_img.setdefault(gt["img_id"], []).append(gt)

    aps_per_threshold = []
    p50, r50 = 0.0, 0.0

    for idx, thresh in enumerate(iou_thresholds):
        tp = np.zeros(len(sorted_preds))
        fp = np.zeros(len(sorted_preds))

        # Track matched GTs per image: {img_id: set of matched gt_indices}
        matched_gts = {img_id: set() for img_id in gt_by_img}

        for p_idx, pred in enumerate(sorted_preds):
            img_id = pred["img_id"]
            img_gts = gt_by_img.get(img_id, [])

            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, gt in enumerate(img_gts):
                if is_mask:
                    curr_iou = mask_iou(pred["mask"], gt["mask"])
                else:
                    curr_iou = box_iou(pred["bbox"], gt["bbox"])

                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_gt_idx = g_idx

            if best_iou >= thresh and best_gt_idx not in matched_gts[img_id]:
                tp[p_idx] = 1
                matched_gts[img_id].add(best_gt_idx)
            else:
                fp[p_idx] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / (cum_tp + cum_fp)
        recall = cum_tp / float(total_gts)

        ap = compute_ap(recall, precision)
        aps_per_threshold.append(ap)

        if abs(thresh - 0.50) < 1e-4:
            p50 = float(precision[-1]) if len(precision) > 0 else 0.0
            r50 = float(recall[-1]) if len(recall) > 0 else 0.0

    map50 = aps_per_threshold[0]
    map50_95 = float(np.mean(aps_per_threshold))

    return p50, r50, map50, map50_95


def main():
    parser = argparse.ArgumentParser(
        description="Check mAP of quantized YOLO HEF model on Raspberry Pi AI HAT+",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=None,
        help="Path, name, or category of compiled .hef file (e.g. 'bottle', 'tile', 'grid')"
    )
    parser.add_argument(
        "-c", "--category",
        type=str,
        default=None,
        help="Alias for --model (specify defect category)"
    )
    parser.add_argument(
        "-l", "--list-models",
        action="store_true",
        help="List all installed .hef models and exit"
    )
    parser.add_argument("-i", "--images", type=str, default="dataset/images", help="Path to validation images directory")
    parser.add_argument("-l_dir", "--labels", type=str, default="dataset/labels", help="Path to validation labels directory")
    parser.add_argument("-o", "--output", type=str, default="results/evaluation_report.json", help="Path for JSON results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    sample_images_dir = script_dir / "sample_images"

    if args.list_models:
        list_available_models(models_dir, sample_images_dir)
        return

    selected_model_arg = args.category or args.model or "tile"
    model_path = resolve_model_path(selected_model_arg, models_dir)

    if model_path is None or not model_path.exists():
        print(f"Error: Model '{selected_model_arg}' could not be resolved.")
        list_available_models(models_dir, sample_images_dir)
        sys.exit(1)

    images_dir = Path(args.images).resolve()
    labels_dir = Path(args.labels).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"Error: Images directory not found at {images_dir}")
        sys.exit(1)

    category = model_path.stem.split("_")[0].upper()
    print("=" * 70)
    print("Hailo-8 Quantized Model mAP & Accuracy Evaluation")
    print(f"Category    : {category}")
    print(f"Model       : {model_path.name}")
    print(f"Model Path  : {model_path}")
    print(f"Images Dir  : {images_dir}")
    print(f"Labels Dir  : {labels_dir}")
    print(f"Confidence  : {args.conf}")
    print(f"IoU NMS     : {args.iou}")
    print("=" * 70)

    # Load Hailo engine
    print("Loading compiled HEF onto Hailo-8 NPU...")
    try:
        engine = HailoYOLOSegmentor(model_path)
    except Exception as e:
        print(f"Failed to initialize Hailo-8: {e}")
        sys.exit(1)

    # Collect images
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )

    if len(image_paths) == 0:
        print("Error: No images found in dataset directory.")
        sys.exit(1)

    print(f"Found {len(image_paths)} images to evaluate.")
    print("Running hardware inference and collecting ground truth...")

    all_preds_box = []
    all_preds_mask = []
    all_gts_box = []
    all_gts_mask = []

    latencies = []
    per_image_results = []

    cm = {"true_negative": 0, "true_positive": 0, "false_positive": 0, "false_negative": 0}

    for idx, img_p in enumerate(image_paths, 1):
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        # 1. Parse Ground Truth
        lbl_p = labels_dir / f"{img_p.stem}.txt"
        gt_boxes, gt_masks = parse_yolo_labels(lbl_p, w, h)
        is_actually_defective = len(gt_boxes) > 0

        for b, m in zip(gt_boxes, gt_masks):
            all_gts_box.append({"img_id": idx, "bbox": b})
            all_gts_mask.append({"img_id": idx, "mask": m})

        # 2. Run Hailo Inference
        canvas, pad_info = letterbox(img_bgr, (640, 640))
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        outputs = engine.run_raw_inference(rgb)
        detections = engine.decode_outputs(outputs, pad_info, conf_thresh=args.conf, iou_thresh=args.iou)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        is_predicted_defective = len(detections) > 0
        if not is_actually_defective and not is_predicted_defective:
            cm["true_negative"] += 1  # Correct Pass
        elif is_actually_defective and is_predicted_defective:
            cm["true_positive"] += 1  # Correct Defect Detected
        elif not is_actually_defective and is_predicted_defective:
            cm["false_positive"] += 1  # False Alarm
        elif is_actually_defective and not is_predicted_defective:
            cm["false_negative"] += 1  # Missed Defect

        for det in detections:
            all_preds_box.append({
                "img_id": idx,
                "bbox": det["bbox"],
                "conf": det["confidence"]
            })
            all_preds_mask.append({
                "img_id": idx,
                "mask": det["mask_array"],
                "conf": det["confidence"]
            })

        if idx % 10 == 0 or idx == len(image_paths):
            print(f"   [{idx}/{len(image_paths)}] processed ({lat:.1f} ms/frame)...")

    # 3. Calculate mAP across IoU thresholds: 0.50 to 0.95 (step 0.05)
    iou_thresholds = np.linspace(0.50, 0.95, 10)

    # Box mAP
    box_p, box_r, box_map50, box_map50_95 = evaluate_matches(
        all_preds_box, all_gts_box, iou_thresholds, is_mask=False
    )
    box_f1 = (2 * box_p * box_r / (box_p + box_r)) if (box_p + box_r) > 0 else 0.0

    # Mask mAP
    mask_p, mask_r, mask_map50, mask_map50_95 = evaluate_matches(
        all_preds_mask, all_gts_mask, iou_thresholds, is_mask=True
    )
    mask_f1 = (2 * mask_p * mask_r / (mask_p + mask_r)) if (mask_p + mask_r) > 0 else 0.0

    # Performance
    avg_latency = float(np.mean(latencies))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # Overall Image-level Classification Accuracy
    total_imgs = len(image_paths)
    correct_imgs = cm["true_positive"] + cm["true_negative"]
    image_acc = (correct_imgs / total_imgs) * 100.0 if total_imgs > 0 else 0.0

    # Print Official Summary Table
    print("\n" + "=" * 70)
    print(f"{'Class':<12} {'Images':<8} {'Instances':<10} {'Box mAP50':<11} {'Box mAP50-95':<13} {'Mask mAP50':<12} {'Mask mAP50-95':<13}")
    print("-" * 70)
    print(f"{'anomaly':<12} {total_imgs:<8} {len(all_gts_box):<10} {box_map50:<11.3f} {box_map50_95:<13.3f} {mask_map50:<12.3f} {mask_map50_95:<13.3f}")
    print("=" * 70)

    print("\nDetailed Metric Breakdown:")
    print(f"Box Precision (P)    : {box_p:.3f}")
    print(f"Box Recall (R)       : {box_r:.3f}")
    print(f"Box F1-Score         : {box_f1:.3f}")
    print(f"Mask Precision (P)   : {mask_p:.3f}")
    print(f"Mask Recall (R)      : {mask_r:.3f}")
    print(f"Mask F1-Score        : {mask_f1:.3f}")
    print(f"Hardware Latency     : {avg_latency:.2f} ms ({fps:.1f} FPS)")

    print("\nIndustrial Pass / Fail Classification Accuracy:")
    print(f"Overall Accuracy     : {image_acc:.1f}% ({correct_imgs}/{total_imgs} images)")
    print(f"True Negatives (Pass): {cm['true_negative']}")
    print(f"True Positives (Def) : {cm['true_positive']}")
    print(f"False Alarms (FP)    : {cm['false_positive']}")
    print(f"Missed Defects (FN)  : {cm['false_negative']}")

    report = {
        "model": model_path.name,
        "dataset_images": total_imgs,
        "ground_truth_instances": len(all_gts_box),
        "box_metrics": {
            "mAP50": round(box_map50, 4),
            "mAP50_95": round(box_map50_95, 4),
            "precision": round(box_p, 4),
            "recall": round(box_r, 4),
            "f1_score": round(box_f1, 4)
        },
        "mask_metrics": {
            "mAP50": round(mask_map50, 4),
            "mAP50_95": round(mask_map50_95, 4),
            "precision": round(mask_p, 4),
            "recall": round(mask_r, 4),
            "f1_score": round(mask_f1, 4)
        },
        "image_classification": {
            "accuracy_percent": round(image_acc, 2),
            "confusion_matrix": cm
        },
        "hardware_performance": {
            "avg_latency_ms": round(avg_latency, 2),
            "fps": round(fps, 2)
        }
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull evaluation report saved to: {output_path}")


if __name__ == "__main__":
    main()
