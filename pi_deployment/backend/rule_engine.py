"""
Neuro-Fuzzy Industrial Quality Triage Engine (< 0.05 ms latency)
Performs continuous microsecond fuzzy risk assessment and gating for LLaMA 3.2 1B.
Tiers: None, Minor, Moderate, Major, Critical
Verdicts: PASS, DIVERT, INSPECT, REJECT
"""

# Defect class inherent hazard weights (0.0 to 1.0)
HAZARD_WEIGHTS = {
    # Structural / Fatal Flaws (0.85 - 1.0)
    "crack": 1.0,
    "tile_crack": 1.0,
    "fracture": 1.0,
    "broken": 1.0,
    "broken_large": 1.0,
    "broken_small": 1.0,
    "bottle_broken_large": 1.0,
    "bottle_broken_small": 1.0,
    "grid_broken": 1.0,
    "hole": 0.95,
    "wood_hole": 0.95,
    "cut": 0.90,
    "leather_cut": 0.90,
    "crazing": 0.85,
    "inclusion": 0.80,
    "pitted_surface": 0.80,
    "damaged_case": 0.75,
    # Process / Surface Flaws (0.50 - 0.70)
    "contamination": 0.70,
    "bottle_contamination": 0.70,
    "bent": 0.65,
    "grid_bent": 0.65,
    "metal_nut_bent": 0.65,
    "transistor_bent_lead": 0.65,
    "oil": 0.60,
    "tile_oil": 0.60,
    "patches": 0.55,
    "rolled-in_scale": 0.55,
    "glue": 0.55,
    "tile_glue_strip": 0.55,
    "grid_glue": 0.55,
    "leather_glue": 0.55,
    # Cosmetic / Surface Marks (0.20 - 0.35)
    "scratches": 0.35,
    "wood_scratch": 0.35,
    "metal_nut_scratch": 0.35,
    "screw_scratch_neck": 0.35,
    "thread": 0.35,
    "screw_thread_side": 0.35,
    "color": 0.30,
    "metal_nut_color": 0.30,
    "poke": 0.30,
    "leather_poke": 0.30,
}


def get_hazard_weight(label: str) -> float:
    """Returns inherent risk weight for a defect label."""
    if not label:
        return 0.50
    lbl = str(label).lower().strip()
    if lbl in HAZARD_WEIGHTS:
        return HAZARD_WEIGHTS[lbl]
    # Substring matching for composite labels
    for key, weight in HAZARD_WEIGHTS.items():
        if key in lbl:
            return weight
    return 0.50


def calculate_fuzzy_risk(total_mask_pct: float, max_conf: float, detections: list) -> float:
    """
    Computes continuous Fuzzy Risk Score (0.0 to 100.0) in < 0.02ms.
    Formula balances:
      - 55% Defect Area Percentage (non-linear smooth growth)
      - 25% Defect Inherent Hazard (damped for microscopic areas)
      - 20% Model Confidence (damped for microscopic areas)
    """
    if total_mask_pct <= 0.0 or not detections:
        return 0.0

    # 1. Non-linear area membership (0.0 to 1.0)
    mu_area = min(1.0, max(0.0, total_mask_pct / 8.0)) ** 1.2

    # 2. Confidence membership (0.0 to 1.0)
    mu_conf = min(1.0, max(0.0, (max_conf - 0.25) / 0.75))

    # 3. Maximum hazard weight among detected labels
    hazard = max((get_hazard_weight(d.get("label", "")) for d in detections), default=0.50)

    # 4. Area dampening factor (prevents tiny < 1.0% flaws from shooting into Major/Critical)
    area_factor = min(1.0, max(0.15, total_mask_pct / 2.0))

    # 5. Weighted Fuzzy Risk Score (0 to 100)
    raw_risk = (
        (mu_area * 55.0) +
        (hazard * 25.0 * area_factor) +
        (mu_conf * 20.0 * area_factor) +
        5.0  # Base detection presence offset
    )

    return round(min(100.0, max(0.0, raw_risk)), 1)


def evaluate_rules(detections, image_shape, category="general"):
    """
    Evaluates rule-based and fuzzy verdict for detected anomalies.

    Tiers:
      - None     (Score = 0.0)   -> PASS    | No LLaMA     | No Banner
      - Minor    (Score < 25.0)  -> PASS    | Fast Rule    | No Banner
      - Moderate (Score < 55.0)  -> DIVERT  | Runs LLaMA   | No Banner
      - Major    (Score < 80.0)  -> INSPECT | Runs LLaMA   | No Banner
      - Critical (Score >= 80.0) -> REJECT  | Runs LLaMA   | SHOWS BANNER & SIREN

    Returns:
        dict: verdict, severity, fuzzy_score, run_llama, trigger_banner,
              max_confidence, total_mask_pct, rule_action
    """
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

    # Compute Fuzzy Risk Score
    fuzzy_score = calculate_fuzzy_risk(total_mask_pct, max_conf, detections)

    cat_lower = category.lower()

    # Tier 1: Minor Defect (Score < 25.0) -> PASS with fast-path rule (NO LLaMA!)
    if fuzzy_score < 25.0:
        severity = "Minor"
        verdict = "PASS"
        run_llama = False
        trigger_banner = False
        action = (
            f"Minor surface blemish ({total_mask_pct:.2f}% area, fuzzy risk {fuzzy_score:.0f}/100). "
            f"Within allowable visual tolerance; pass to next station without line delay."
        )

    # Tier 2: Moderate Defect (Score 25.0 - 54.9) -> DIVERT (Runs LLaMA)
    elif fuzzy_score < 55.0:
        severity = "Moderate"
        verdict = "DIVERT"
        run_llama = True
        trigger_banner = False
        if "steel" in cat_lower:
            action = f"Moderate surface irregularity ({total_mask_pct:.1f}% area). Divert coil/sheet to secondary finishing track."
        elif "bottle" in cat_lower:
            action = f"Moderate surface scuff or cosmetic variance ({total_mask_pct:.1f}%). Divert bottle to wash/re-inspection lane."
        elif "grid" in cat_lower:
            action = f"Moderate mesh strut deviation ({total_mask_pct:.1f}%). Divert grid to alignment fixture."
        elif "tile" in cat_lower:
            action = f"Moderate glaze or surface variation ({total_mask_pct:.1f}%). Divert tile to secondary grading sorter."
        else:
            action = f"Moderate anomaly detected ({total_mask_pct:.1f}% area). Divert part from main conveyor to buffer/rework lane."

    # Tier 3: Major Defect (Score 55.0 - 79.9) -> INSPECT (Runs LLaMA, No Alarm Banner)
    elif fuzzy_score < 80.0:
        severity = "Major"
        verdict = "INSPECT"
        run_llama = True
        trigger_banner = False
        if "steel" in cat_lower:
            action = f"Major surface flaw ({total_mask_pct:.1f}% area). Route coil to quality gate for manual inspector sign-off."
        elif "bottle" in cat_lower:
            action = f"Major containment wall anomaly ({total_mask_pct:.1f}%). Hold bottle at optical inspection gate."
        elif "grid" in cat_lower:
            action = f"Major strut fracture/bend ({total_mask_pct:.1f}%). Route grid to manual inspection station."
        elif "tile" in cat_lower:
            action = f"Major structural crack ({total_mask_pct:.1f}%). Hold tile at inspection station; check press die calibration."
        else:
            action = f"Major defect detected ({total_mask_pct:.1f}% area). Hold component for secondary manual inspection."

    # Tier 4: Critical Defect (Score >= 80.0) -> REJECT (Runs LLaMA, Shows Banner & Siren)
    else:
        severity = "Critical"
        verdict = "REJECT"
        run_llama = True
        trigger_banner = True
        if "steel" in cat_lower:
            action = f"Critical steel defect ({total_mask_pct:.1f}% area, fuzzy risk {fuzzy_score:.0f}/100). Auto-reject sheet and quarantine coil."
        elif "bottle" in cat_lower:
            action = f"Critical structural fracture/rupture ({total_mask_pct:.1f}%). Auto-reject bottle to scrap bin immediately."
        elif "grid" in cat_lower:
            action = f"Critical mesh structural fracture ({total_mask_pct:.1f}%). Auto-reject grid; inspect stamping die."
        elif "tile" in cat_lower:
            action = f"Critical tile fracture ({total_mask_pct:.1f}%). Eject tile to scrap bin and check press balance."
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
