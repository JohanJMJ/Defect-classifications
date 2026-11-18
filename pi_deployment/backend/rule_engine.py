"""
Fast-Path Industrial Quality Rule Engine (< 0.1 ms latency)
Performs instant physical triage of defect detections without LLM delay.
"""

# Thresholds
HIGH_SEVERITY_MASK_PCT = 10.0   # Defect covering >10% of image is critical
MED_SEVERITY_MASK_PCT = 3.0     # Defect covering >3% of image is medium
HIGH_CONF_THRESHOLD = 0.70      # High detection certainty

def evaluate_rules(detections, image_shape, category="general"):
    """
    Evaluates rule-based verdict for detected anomalies.

    Args:
        detections: list of dicts with keys 'confidence', 'mask_area_px', 'label'
        image_shape: tuple of (height, width, ...)
        category: industrial category (e.g. tile, bottle, zipper, etc.)

    Returns:
        dict: verdict, severity, max_confidence, total_mask_pct, rule_action
    """
    if not detections:
        return {
            "verdict": "PASS",
            "severity": "NONE",
            "max_confidence": 0.0,
            "total_mask_pct": 0.0,
            "rule_action": "Quality standards met. Route to next production station.",
        }

    h, w = image_shape[:2]
    total_px = float(h * w) if (h * w) > 0 else 1.0

    total_mask_px = sum(d.get("mask_area_px", 0) for d in detections)
    total_mask_pct = (total_mask_px / total_px) * 100.0
    max_conf = max(d.get("confidence", 0.0) for d in detections)

    # Category-specific terminology
    cat_lower = category.lower()

    if total_mask_pct >= HIGH_SEVERITY_MASK_PCT:
        severity = "CRITICAL"
        verdict = "REJECT"
        if "cable" in cat_lower or "wire" in cat_lower:
            action = f"Critical insulation/wire fault ({total_mask_pct:.1f}% area). Halt cable harness line immediately."
        elif "bottle" in cat_lower or "capsule" in cat_lower or "pill" in cat_lower:
            action = f"Severe containment rupture or contamination ({total_mask_pct:.1f}%). Reject batch and isolate filler unit."
        elif "zipper" in cat_lower:
            action = f"Complete teeth displacement ({total_mask_pct:.1f}%). Reject zipper track and inspect alignment pins."
        else:
            action = f"Severe structural defect ({total_mask_pct:.1f}% surface area). Auto-reject part to scrap bin."

    elif total_mask_pct >= MED_SEVERITY_MASK_PCT or max_conf >= HIGH_CONF_THRESHOLD:
        severity = "HIGH"
        verdict = "REJECT"
        action = f"Noticeable defect detected ({total_mask_pct:.1f}% area, {max_conf*100:.1f}% conf). Reject part for manual rework."

    elif total_mask_pct >= 0.8:
        severity = "MEDIUM"
        verdict = "FLAGGED_MONITOR"
        action = f"Moderate surface anomaly ({total_mask_pct:.1f}% area). Route to secondary optical inspection."

    else:
        severity = "LOW"
        verdict = "FLAGGED_MONITOR"
        action = f"Minor microscopic imperfection ({total_mask_pct:.2f}% area). Log anomaly and monitor station variance."

    return {
        "verdict": verdict,
        "severity": severity,
        "max_confidence": round(float(max_conf), 4),
        "total_mask_pct": round(float(total_mask_pct), 2),
        "rule_action": action,
    }
