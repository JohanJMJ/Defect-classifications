DEFECT_SEVERITY_WEIGHTS = {
    # Steel defects
    "crazing": 0.85,
    "inclusion": 0.70,
    "patches": 0.50,
    "pitted_surface": 0.60,
    "rolled-in_scale": 0.75,
    "scratches": 0.40,
    # Bottle defects
    "broken_large": 0.95,
    "broken_small": 0.75,
    "contamination": 0.55,
    # Grid defects
    "bent": 0.45,
    "broken": 0.90,
    "glue": 0.50,
    "metal_contamination": 0.65,
    "thread": 0.35,
    # Tile defects
    "crack": 0.90,
    "glue_strip": 0.50,
    "gray_stroke": 0.40,
    "oil": 0.60,
    "rough": 0.45,
}

def calculate_fuzzy_risk(total_mask_pct: float, max_conf: float, detections: list) -> float:
    # Computes the Fuzzy risk score (0 to 100).
    if not detections or total_mask_pct <= 0.0:
        return 0.0

    # 1. Defect Area Risk (0 to 40 pts)
    # Area scaling: 0.1% -> 5 pts, 1.0% -> 20 pts, >5.0% -> 40 pts
    if total_mask_pct <= 0.2:
        area_score = (total_mask_pct / 0.2) * 8.0
    elif total_mask_pct <= 1.0:
        area_score = 8.0 + ((total_mask_pct - 0.2) / 0.8) * 14.0
    elif total_mask_pct <= 5.0:
        area_score = 22.0 + ((total_mask_pct - 1.0) / 4.0) * 14.0
    else:
        area_score = 36.0 + min(4.0, (total_mask_pct - 5.0) * 0.4)

    # 2. Confidence Factor (0 to 25 pts)
    conf_score = max(0.0, min(25.0, (max_conf - 0.2) / 0.7 * 25.0))

    # 3. Defect Hazard Weight (0 to 35 pts)
    max_hazard = 0.40
    for d in detections:
        lbl = d.get("label", "").lower()
        hazard = DEFECT_SEVERITY_WEIGHTS.get(lbl, 0.50)
        if hazard > max_hazard:
            max_hazard = hazard
    hazard_score = max_hazard * 35.0

    total_fuzzy = min(100.0, max(0.0, area_score + conf_score + hazard_score))
    return round(total_fuzzy, 1)


def evaluate_rules(detections=None, image_shape=(640, 640), category="tile", **kwargs) -> dict:
    if isinstance(detections, str):
        cat_arg = detections
        det_arg = image_shape if isinstance(image_shape, list) else []
        shape_arg = category if isinstance(category, tuple) else (640, 640)
        category = cat_arg
        detections = det_arg
        image_shape = shape_arg

    if "category" in kwargs:
        category = kwargs["category"]

    if detections is None:
        detections = []

    if not detections:
        return {
            "verdict": "PASS",
            "severity": "None",
            "fuzzy_score": 0.0,
            "run_llama": False,
            "trigger_banner": False,
            "max_confidence": 0.0,
            "total_mask_pct": 0.0,
            "rule_action": "Surface uniformity verified within nominal tolerance. Component cleared for production flow.",
        }

    h, w = image_shape[:2]
    total_px = float(h * w) if (h * w) > 0 else 1.0

    total_mask_px = sum(d.get("mask_area_px", 0) for d in detections)
    total_mask_pct = (total_mask_px / total_px) * 100.0
    max_conf = max(d.get("confidence", 0.0) for d in detections)

    fuzzy_score = calculate_fuzzy_risk(total_mask_pct, max_conf, detections)

    cat_lower = str(category).lower()
    primary_defect = detections[0].get("label", "").lower() if detections else ""

    # Minor Defect (Score < 25.0) - PASS with fast-path rule (NO LLaMA)
    if fuzzy_score < 25.0:
        severity = "Minor"
        verdict = "PASS"
        run_llama = False
        trigger_banner = False
        action = (
            f"Minor surface blemish ({total_mask_pct:.2f}% area, fuzzy risk {fuzzy_score:.0f}/100). "
            f"Within allowable visual tolerance; pass to next station without line delay."
        )

    # Moderate Defect (Score 25.0 - 54.9) - DIVERT (Runs LLaMA)
    elif fuzzy_score < 55.0:
        severity = "Moderate"
        verdict = "DIVERT"
        run_llama = True
        trigger_banner = False
        if "steel" in cat_lower:
            action = f"Moderate surface irregularity ({primary_defect}, {total_mask_pct:.1f}% area). Divert coil/sheet to secondary finishing track."
        elif "bottle" in cat_lower:
            action = f"Moderate surface scuff or cosmetic variance ({total_mask_pct:.1f}%). Divert bottle to wash/re-inspection lane."
        elif "grid" in cat_lower:
            action = f"Moderate mesh strut deviation ({total_mask_pct:.1f}%). Divert grid to alignment fixture."
        elif "tile" in cat_lower:
            if "oil" in primary_defect:
                action = f"Moderate oil residue ({total_mask_pct:.1f}%). Divert tile to degreasing wipe station."
            elif "glue" in primary_defect:
                action = f"Moderate glue smudge ({total_mask_pct:.1f}%). Divert tile to surface cleanup buffer."
            else:
                action = f"Moderate glaze or surface variation ({total_mask_pct:.1f}%). Divert tile to secondary grading sorter."
        else:
            action = f"Moderate anomaly detected ({total_mask_pct:.1f}% area). Divert part from main conveyor to buffer/rework lane."

    # Major Defect (Score 55.0 - 79.9) - INSPECT (Runs LLaMA)
    elif fuzzy_score < 80.0:
        severity = "Major"
        verdict = "INSPECT"
        run_llama = True
        trigger_banner = False
        if "steel" in cat_lower:
            action = f"Major surface flaw ({primary_defect}, {total_mask_pct:.1f}% area). Route coil to quality gate for manual inspector sign-off."
        elif "bottle" in cat_lower:
            if "contam" in primary_defect:
                action = f"Major particulate contamination ({total_mask_pct:.1f}%). Hold bottle at optical inspection gate for wash re-cycle."
            else:
                action = f"Major containment wall anomaly ({total_mask_pct:.1f}%). Hold bottle at optical inspection gate."
        elif "grid" in cat_lower:
            action = f"Major strut fracture/bend ({total_mask_pct:.1f}%). Route grid to manual inspection station."
        elif "tile" in cat_lower:
            if "oil" in primary_defect:
                action = f"Major oil splatter ({total_mask_pct:.1f}%). Hold tile for solvent wash; check conveyor chain lubrication."
            elif "glue" in primary_defect:
                action = f"Major glue streak ({total_mask_pct:.1f}%). Hold tile for manual cleanup; service adhesive nozzle."
            elif "crack" in primary_defect:
                action = f"Major structural crack ({total_mask_pct:.1f}%). Hold tile at inspection station; check press die calibration."
            else:
                action = f"Major tile anomaly ({total_mask_pct:.1f}%). Hold tile at inspection station."
        else:
            action = f"Major defect detected ({total_mask_pct:.1f}% area). Hold component for secondary manual inspection."

    # Critical Defect (Score >= 80.0) -> REJECT (Runs LLaMA, Shows Banner & Siren)
    else:
        severity = "Critical"
        verdict = "REJECT"
        run_llama = True
        trigger_banner = True
        if "steel" in cat_lower:
            action = f"Critical steel defect ({primary_defect}, {total_mask_pct:.1f}% area, fuzzy risk {fuzzy_score:.0f}/100). Auto-reject sheet and quarantine coil."
        elif "bottle" in cat_lower:
            if "contam" in primary_defect:
                action = f"Critical internal contamination ({total_mask_pct:.1f}%). Auto-reject bottle; inspect wash station rinse nozzles."
            else:
                action = f"Critical structural fracture/rupture ({total_mask_pct:.1f}%). Auto-reject bottle to scrap bin immediately."
        elif "grid" in cat_lower:
            if "glue" in primary_defect or "metal" in primary_defect:
                action = f"Critical foreign material contamination ({total_mask_pct:.1f}%). Auto-reject grid; inspect feeder and cleaning station."
            else:
                action = f"Critical mesh structural fracture ({total_mask_pct:.1f}%). Auto-reject grid; inspect stamping die."
        elif "tile" in cat_lower:
            if "oil" in primary_defect:
                action = f"Critical oil contamination ({total_mask_pct:.1f}%). Eject tile to wash bay and inspect hydraulic line seals."
            elif "glue" in primary_defect:
                action = f"Critical adhesive deposit ({total_mask_pct:.1f}%). Eject tile to scrap bin and check glue applicator shutoff."
            elif "crack" in primary_defect:
                action = f"Critical tile fracture ({total_mask_pct:.1f}%). Eject tile to scrap bin and check press balance."
            else:
                action = f"Critical tile defect ({total_mask_pct:.1f}%). Eject tile to scrap bin and inspect kiln conveyor."
        else:
            action = f"Critical defect detected ({total_mask_pct:.1f}% area). Auto-reject component immediately to scrap bin."

    return {
        "verdict": verdict,
        "severity": severity,
        "fuzzy_score": fuzzy_score,
        "run_llama": run_llama,
        "trigger_banner": trigger_banner,
        "max_confidence": round(float(max_conf), 4),
        "total_mask_pct": round(float(total_mask_pct), 2),
        "rule_action": action,
    }
