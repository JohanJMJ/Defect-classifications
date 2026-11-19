"""
Industrial Quality Control Web Application & API
Flask Server for Raspberry Pi 5 + Hailo-8 AI HAT+ & PC Development
"""

import os
from pathlib import Path
import threading
import time
import uuid
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from flask_cors import CORS

import sys
# Ensure dashboard and backend modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.hailo_engine import HailoVisionEngine
from backend.llama_engine import LlamaAdvisoryEngine
from backend.rule_engine import evaluate_rules
from backend.alert_manager import AlertManager
from backend.database import QCInspectionDB
from backend.camera_streamer import CameraStreamer

try:
    from dashboard import config
except ImportError:
    import config

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)
CORS(app)

# Initialize engines
db = QCInspectionDB(config.DB_PATH)
vision_engine = HailoVisionEngine(config.MODELS_DIR)
llama_engine = LlamaAdvisoryEngine(backend="auto")
alert_mgr = AlertManager(enable_gpio=True)
camera = CameraStreamer(camera_index=0)

# Asynchronous Background Llama Job Registry
advisory_jobs = {}
advisory_lock = threading.Lock()

def _async_llama_worker(inspection_id, category, detections, image_shape, rule_eval):
    t0 = time.perf_counter()
    try:
        print(f"[Async Llama] Running Llama advisory ({llama_engine.backend}) for Inspection #{inspection_id}...")
        llama_text, llama_lat = llama_engine.generate_advisory(
            category, detections, image_shape, rule_eval
        )
    except Exception as e:
        print(f"[Async Llama Error] {e}")
        llama_text = rule_eval.get("rule_action", f"Error: {e}")
        llama_lat = (time.perf_counter() - t0) * 1000.0

    with advisory_lock:
        advisory_jobs[inspection_id] = {
            "status": "completed",
            "llama_advisory": llama_text,
            "latency_llama_ms": round(llama_lat, 2),
            "severity": rule_eval.get("severity", "NONE"),
        }
    try:
        db.update_advisory(inspection_id, llama_text, llama_lat, rule_eval.get("severity"))
        print(f"[Async Llama] Inspection #{inspection_id} completed advisory in {llama_lat:.1f}ms")
    except Exception as e:
        print(f"[Async Llama DB Error] {e}")


@app.route("/")
def index():
    models = config.get_available_models()
    return render_template(
        "index.html",
        models=models,
        is_simulation=vision_engine.is_simulation_mode(),
    )


@app.route("/history")
def history_page():
    models = config.get_available_models()
    return render_template(
        "history.html",
        models=models,
    )


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(str(config.UPLOADS_DIR), filename)


@app.route("/sample_images/<category>/<path:filename>")
def serve_sample(category, filename):
    cat_dir = config.SAMPLE_IMAGES_DIR / category
    return send_from_directory(str(cat_dir), filename)


@app.route("/video_feed")
def video_feed():
    """MJPEG Live Camera Stream Route (activates camera on connection, releases on disconnect)."""
    return Response(
        camera.generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    success = camera.open_camera()
    return jsonify({"success": success, "active": camera.is_active()})


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    camera.release()
    return jsonify({"success": True, "active": False})


@app.route("/api/camera/status", methods=["GET"])
def api_camera_status():
    return jsonify({"success": True, "active": camera.is_active()})


# ---------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------

@app.route("/api/models", methods=["GET"])
def api_models():
    models = config.get_available_models()
    return jsonify({
        "success": True,
        "models": models,
        "active_model": vision_engine.current_model_name or "steel_defects",
        "simulation_mode": vision_engine.is_simulation_mode(),
    })


@app.route("/api/telemetry", methods=["GET"])
def api_telemetry():
    telemetry = config.get_system_telemetry()
    telemetry["npu_active"] = not vision_engine.is_simulation_mode()
    telemetry["npu_name"] = "Hailo-8 (26 TOPS)" if not vision_engine.is_simulation_mode() else "PC Simulation Mode"
    telemetry["llama_backend"] = llama_engine.backend
    telemetry["llama_ram_disk"] = getattr(llama_engine, "is_ram_disk", False)
    telemetry["llama_model_name"] = llama_engine.model_path.name if getattr(llama_engine, "model_path", None) else "None"
    if getattr(llama_engine, "is_ram_disk", False):
        telemetry["llama_location"] = "ExecuTorch (RAM /dev/shm)"
    elif llama_engine.backend == "executorch":
        telemetry["llama_location"] = "ExecuTorch (Disk)"
    else:
        telemetry["llama_location"] = "Simulation Mode"
    return jsonify({"success": True, "telemetry": telemetry})


@app.route("/api/samples/<category>", methods=["GET"])
def api_samples(category):
    cat_dir = config.SAMPLE_IMAGES_DIR / category
    samples = []
    if cat_dir.is_dir():
        for f in cat_dir.glob("*.*"):
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                samples.append({
                    "name": f.name,
                    "url": f"/sample_images/{category}/{f.name}",
                    "is_defective": not any(kw in f.name.lower() for kw in ["good", "normal", "pass", "clean"]),
                })
    return jsonify({"success": True, "category": category, "samples": samples})


@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    """
    Core inspection execution endpoint.
    Accepts:
      - File upload ('file') OR
      - Camera capture ('source_type'='camera') OR
      - Sample library reference ('sample_path' and 'source_type'='sample')
    """
    model_category = request.form.get("model", "steel_defects").lower()
    component_id = request.form.get("component_id") or f"COMP-{uuid.uuid4().hex[:6].upper()}"
    source_type = request.form.get("source_type", "upload")

    image_bgr = None
    orig_filename = ""

    # 1. Acquire Image
    print(f"[Inspect] Received request: source_type='{source_type}', model='{model_category}'")

    if source_type == "camera":
        image_bgr = camera.capture_snapshot()
        orig_filename = f"camera_{int(time.time())}.jpg"

    elif source_type == "sample":
        rel_sample = request.form.get("sample_path", "")
        # Sanitize path across Linux and Windows
        clean_rel = rel_sample.replace("/sample_images/", "").lstrip("/\\")
        sample_path = config.SAMPLE_IMAGES_DIR / clean_rel
        if sample_path.is_file():
            image_bgr = cv2.imread(str(sample_path))
            orig_filename = clean_rel  # Preserve relative path e.g. "tile/defective_crack.png"
        else:
            print(f"[Inspect] Sample file not found: {sample_path}")
            return jsonify({"success": False, "error": f"Sample not found: {clean_rel}"}), 404

    else:
        # File upload
        if "file" not in request.files or request.files["file"].filename == "":
            print("[Inspect] No image file provided in request.files!")
            return jsonify({"success": False, "error": "No image file was received. Please select an image first."}), 400

        file = request.files["file"]
        orig_filename = file.filename
        file_bytes = file.read()
        print(f"[Inspect] Uploaded file: {orig_filename} ({len(file_bytes)} bytes)")

        import numpy as np
        nparr = np.frombuffer(file_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Fallback to PIL Image if OpenCV fails to decode (e.g. WebP, mobile orientations)
        if image_bgr is None:
            try:
                from PIL import Image
                import io
                pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                print(f"[Inspect] Successfully decoded with PIL fallback: {image_bgr.shape}")
            except Exception as e:
                print(f"[Inspect] PIL fallback decode failed: {e}")

    if image_bgr is None:
        print(f"[Inspect] Failed to decode image: {orig_filename}")
        return jsonify({"success": False, "error": "Unable to decode image file. Please use JPG or PNG."}), 400

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:6]
    base_save_name = f"{timestamp_str}_{model_category}_{unique_id}.jpg"
    raw_path = config.UPLOADS_DIR / f"raw_{base_save_name}"
    annotated_path = config.UPLOADS_DIR / f"annotated_{base_save_name}"

    # Save original
    cv2.imwrite(str(raw_path), image_bgr)

    # 2. Vision AI Inference (Hailo-8 / Mock)
    try:
        detections, vision_lat = vision_engine.infer(image_bgr, model_name=model_category, image_name=orig_filename)
    except Exception as e:
        return jsonify({"success": False, "error": f"Vision inference failed: {e}"}), 500

    # 3. Rule Engine Triage (< 0.1ms)
    rule_eval = evaluate_rules(detections, image_bgr.shape, category=model_category)

    # 4. Render Visualization & Save Annotated Image
    annotated_bgr = vision_engine.draw_visualizations(
        image_bgr, detections, rule_eval, latency_ms=vision_lat
    )
    cv2.imwrite(str(annotated_path), annotated_bgr)

    # 5. Physical & Dashboard Alerts (< 1ms)
    alert_mgr.trigger_inspection_alert(rule_eval["verdict"], rule_eval["severity"])

    # 6. Store to SQLite DB
    defect_detected = len(detections) > 0
    defect_label = detections[0]["label"] if defect_detected else "None"

    is_continuous_mode = request.form.get("continuous") == "1"

    if is_continuous_mode:
        # In fast continuous mode, use instant deterministic fast-path triage (< 1ms)
        llama_text = (
            f"• Severity: {rule_eval.get('severity', 'NONE')}\n"
            f"• Assessment: Continuous inspection triage verified ({rule_eval.get('total_mask_pct', 0)}% defect area).\n"
            f"• Action: {rule_eval.get('rule_action', 'Continue continuous conveyor line.')}"
        )
        llama_lat = 0.4
        llama_status = "completed"
    else:
        # Asynchronous decoupled Llama: Return Vision results immediately, generate Llama in background!
        llama_text = "PENDING"
        llama_lat = 0.0
        llama_status = "pending"

    # Prepare serializable detections (strip numpy mask array)
    json_detections = []
    for d in detections:
        item = {
            "label": d.get("label", "anomaly"),
            "confidence": float(d.get("confidence", 0)),
            "bbox": d.get("bbox", []),
            "mask_polygon_points": d.get("mask_polygon_points", 0),
            "polygon": d.get("polygon", []),
            "mask_area_px": float(d.get("mask_area_px", 0)),
        }
        json_detections.append(item)

    record_data = {
        "component_id": component_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_category": model_category,
        "defect_detected": defect_detected,
        "defect_class": defect_label,
        "confidence": rule_eval["max_confidence"],
        "mask_area_pct": rule_eval["total_mask_pct"],
        "verdict": rule_eval["verdict"],
        "severity": rule_eval["severity"],
        "rule_action": rule_eval["rule_action"],
        "llama_advisory": llama_text,
        "latency_vision_ms": vision_lat,
        "latency_llama_ms": llama_lat,
        "image_path": f"/uploads/raw_{base_save_name}",
        "annotated_path": f"/uploads/annotated_{base_save_name}",
    }

    inspection_id = db.add_inspection(record_data)
    record_data["id"] = inspection_id
    record_data["detections"] = json_detections
    record_data["llama_status"] = llama_status

    # Launch Llama 3B in background worker thread so Vision results return immediately (< 25 ms)!
    if not is_continuous_mode:
        with advisory_lock:
            advisory_jobs[inspection_id] = {
                "status": "pending",
                "start_time": time.time(),
            }
        t = threading.Thread(
            target=_async_llama_worker,
            args=(inspection_id, model_category, detections, image_bgr.shape, rule_eval),
            daemon=True,
        )
        t.start()

    return jsonify({
        "success": True,
        "inspection": record_data
    })


@app.route("/api/advisory/<int:inspection_id>", methods=["GET"])
def api_get_advisory(inspection_id):
    """Polls status and result of asynchronous Llama 3B generation."""
    with advisory_lock:
        job = advisory_jobs.get(inspection_id)
    if job:
        return jsonify({"success": True, **job})

    # Check database in case completed or restarted
    try:
        row = db.get_inspection_by_id(inspection_id)
        if row and row.get("llama_advisory") and row["llama_advisory"] != "PENDING":
            return jsonify({
                "success": True,
                "status": "completed",
                "llama_advisory": row["llama_advisory"],
                "latency_llama_ms": row.get("latency_llama_ms", 0),
                "severity": row.get("severity", "NONE"),
            })
    except Exception:
        pass

    return jsonify({"success": True, "status": "pending"})


@app.route("/api/history", methods=["GET"])
def api_history():
    limit = int(request.args.get("limit", 25))
    offset = int(request.args.get("offset", 0))
    verdict = request.args.get("verdict", None)
    model = request.args.get("model", None)
    search = request.args.get("search", None)

    rows, total = db.get_inspections(limit, offset, verdict, model, search)
    stats = db.get_summary_stats()

    return jsonify({
        "success": True,
        "total": total,
        "inspections": rows,
        "stats": stats,
    })


@app.route("/api/history/<int:insp_id>", methods=["GET"])
def api_history_detail(insp_id):
    row = db.get_inspection_by_id(insp_id)
    if not row:
        return jsonify({"success": False, "error": "Not found"}), 404
    return jsonify({"success": True, "inspection": row})


@app.route("/api/export/csv", methods=["GET"])
def api_export_csv():
    csv_data = db.export_csv_string()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=qc_inspections_{int(time.time())}.csv"}
    )


@app.route("/api/alerts/test", methods=["POST"])
def api_test_alert():
    verdict = request.json.get("verdict", "PASS")
    alert_mgr.trigger_inspection_alert(verdict, severity="TEST")
    return jsonify({"success": True, "verdict": verdict, "status": alert_mgr.get_status()})


if __name__ == "__main__":
    print(f"============================================================")
    print(f"  Industrial Quality Control Assistant - Edge Web Dashboard")
    print(f"  Host: {config.HOST}:{config.PORT}")
    print(f"  Simulation Mode: {vision_engine.is_simulation_mode()}")
    print(f"============================================================")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG, threaded=True)
