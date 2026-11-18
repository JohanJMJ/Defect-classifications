"""
Llama AI Advisory Engine for Industrial QC
Provides intelligent defect severity assessment and actionable recommendations.
Supports ExecuTorch (ARM64 CPU), llama-server / Ollama HTTP, and PC Simulation Mode.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import urllib.error


def mirror_to_ram_disk(src_path: Path, subfolder="executorch") -> tuple:
    """
    Mirrors a model or tokenizer file to Linux /dev/shm (POSIX RAM disk).
    Returns (effective_path, is_in_ram).
    If /dev/shm is unavailable, not writable, or has insufficient free memory, returns (src_path, False).
    """
    if not src_path:
        return src_path, False
    src_path = Path(src_path)
    if not src_path.is_file():
        return src_path, False

    shm_root = Path("/dev/shm")
    if not shm_root.is_dir() or not os.access(str(shm_root), os.W_OK):
        return src_path, False

    try:
        dest_dir = shm_root / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / src_path.name

        src_size = src_path.stat().st_size

        # Check if already cached in RAM with matching size
        if dest_path.is_file() and dest_path.stat().st_size == src_size:
            print(f"[RAM Disk] Model file already warm in RAM: {dest_path} ({src_size / (1024*1024):.1f} MB)")
            return dest_path, True

        # Check available RAM space (require file size + 100MB margin)
        usage = shutil.disk_usage(str(shm_root))
        if usage.free < (src_size + 100 * 1024 * 1024):
            print(
                f"[RAM Disk] Insufficient free space in /dev/shm ({usage.free / (1024*1024):.1f} MB free, "
                f"need {src_size / (1024*1024):.1f} MB). Keeping on disk."
            )
            return src_path, False

        print(f"[RAM Disk] ⚡ Loading {src_path.name} ({src_size / (1024*1024):.1f} MB) into RAM disk (/dev/shm)...")
        t0 = time.perf_counter()
        shutil.copy2(str(src_path), str(dest_path))
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[RAM Disk] ✅ Successfully staged in RAM in {dt:.1f}ms! Direct DDR4/DDR5 execution enabled.")
        return dest_path, True

    except Exception as e:
        print(f"[RAM Disk] Staging to /dev/shm skipped: {e}")
        return src_path, False


def check_persistent_server(url="http://127.0.0.1:8080"):
    """Checks if a persistent llama-server is listening for requests."""
    try:
        health_url = url.rstrip("/") + "/health"
        req = urllib.request.Request(health_url, headers={"User-Agent": "QC-Advisory-Probe"})
        with urllib.request.urlopen(req, timeout=0.4) as resp:
            return resp.status == 200
    except Exception:
        return False


def auto_discover_llama():
    """Searches common paths for ExecuTorch runner, .pte model, and tokenizer, prioritizing 1B models for edge speed."""
    possible_dirs = [
        Path.home() / "executorch-deployment",
        Path("/home/raspberrypi5/executorch-deployment"),
        Path.home() / "Desktop" / "executorch-deployment",
        Path.home() / "Desktop" / "executorch-1b",
        Path.home() / "executorch-1b",
        Path.home() / "Desktop" / "executorch",
        Path.home() / "executorch",
        Path.home() / "Desktop" / "executorch-3b",
        Path.home() / "executorch-3b",
        Path("/home/raspberrypi5/Desktop/executorch-3b"),
    ]

    for d in possible_dirs:
        try:
            if d.is_dir():
                runner = d / "llama_main"
                pte_files = list(d.glob("*.pte"))
                tok_files = list(d.glob("*tokenizer*"))

                if runner.is_file() and pte_files and tok_files:
                    # Prioritize 1B models first for edge speed, otherwise first available .pte
                    models_1b = [p for p in pte_files if "1b" in p.name.lower()]
                    chosen_model = models_1b[0] if models_1b else pte_files[0]

                    # Prefer tokenizer.bin or tokenizer.model for Llama
                    model_toks = [t for t in tok_files if t.name.endswith(".model") or t.name.endswith(".bin")]
                    chosen_tok = model_toks[0] if model_toks else tok_files[0]

                    return runner, chosen_model, chosen_tok
                elif runner.is_file():
                    model = pte_files[0] if pte_files else (d / "llama-1b.pte")
                    tok = tok_files[0] if tok_files else (d / "tokenizer.bin")
                    return runner, model, tok
        except Exception:
            continue

    return None, None, None


# Domain-Specific Industrial Failure Modes & CAPA Knowledge Base
INDUSTRIAL_KB = {
    "tile": {
        "material": "Ceramic / Porcelain Flooring",
        "failure_modes": "Mechanical press stress cracks, thermal firing fractures, conveyor oil leaks, glue overflow",
        "risk_threshold": "Cracks > 3% cause catastrophic failure under 1500N floor load (ISO 10545-4).",
        "tooling_capa": "Inspect hydraulic press die balance, verify Kiln Zone 3 cooling ramp (<15°C/min), check pneumatic chain lubricator."
    },
    "bottle": {
        "material": "PET / Glass Beverage Container",
        "failure_modes": "Neck finish cracks, sidewall parison bubbles, foreign particulate contamination",
        "risk_threshold": "Neck defects destroy hermetic seal, causing pressure loss and microbial spoilage.",
        "tooling_capa": "Inspect blow-mold clamp alignment, check preform heater lamp calibration, inspect capper torque."
    },
    "cable": {
        "material": "Copper Conductor / Extruded PVC Sheath",
        "failure_modes": "Conductor strand breakage, cut/pinched insulation, crimp deformation",
        "risk_threshold": "Exposed copper causes dielectric breakdown, short circuits, and fire hazard.",
        "tooling_capa": "Check wire-stripping blade clearance, inspect feed guide tension rollers, calibrate crimp applicator."
    },
    "capsule": {
        "material": "Gelatin / HPMC Pharmaceutical Capsule",
        "failure_modes": "Cap/body split, pinhole puncture, dosage leakage, surface dent",
        "risk_threshold": "Wall rupture compromises active ingredient dosage integrity and shelf-life.",
        "tooling_capa": "Inspect capsule rectifier track, check closing station pin depth, verify vacuum transfer cups."
    },
    "carpet": {
        "material": "Woven Textile / Polypropylene Fiber",
        "failure_modes": "Tuft pull-out, loop cut, metal particulate, uneven pile density",
        "risk_threshold": "Thread tears propagate under foot traffic; metal fragments violate safety standards.",
        "tooling_capa": "Examine tufting needle sharpness, check yarn tension brakes, sweep inline magnetic metal detector."
    },
    "grid": {
        "material": "Stamped Sheet Metal / Aluminum Honeycomb",
        "failure_modes": "Bent mesh strut, broken weld node, adhesive drip, burr deformation",
        "risk_threshold": "Strut fractures reduce structural shear resistance and block airflow volume.",
        "tooling_capa": "Inspect stamping die clearance, examine spot-weld electrode tips, check trim blade burrs."
    },
    "hazelnut": {
        "material": "Organic Food / Agricultural Nut",
        "failure_modes": "Shell crack, insect bore hole, fungal aflatoxin mold, internal rot",
        "risk_threshold": "Internal fungal rot or aflatoxins fail food safety standards (FDA/EU).",
        "tooling_capa": "Calibrate optical color sorter sensitivity, check pneumatic reject nozzle pressure and timing."
    },
    "leather": {
        "material": "Natural Hide / Automotive Upholstery Leather",
        "failure_modes": "Surface cut, scar deformation, glue smear, needle puncture",
        "risk_threshold": "Grain punctures rupture under seat tension load; aesthetic grade reduction.",
        "tooling_capa": "Inspect CNC oscillating cutting blade sharpness, check hide tension vacuum bed, clean feed rollers."
    },
    "metal_nut": {
        "material": "Hardened Grade 8 Carbon Steel Hex Nut",
        "failure_modes": "Stripped internal thread, bent chamfer corner, surface oxidation scratch",
        "risk_threshold": "Thread deformation causes thread galling or total joint torque loss under dynamic load.",
        "tooling_capa": "Inspect thread tapping tool wear, check cold-heading punch concentricity, verify rust-inhibitor bath."
    },
    "pill": {
        "material": "Compressed Solid Dose Tablet",
        "failure_modes": "Tablet capping/lamination, edge chipping, surface contamination, deboss defect",
        "risk_threshold": "Chipping causes variable drug dosage; capping exposes formulation to rapid degradation.",
        "tooling_capa": "Check tablet press punch tooling wear, verify pre-compression force, inspect ejection cam."
    },
    "screw": {
        "material": "Grade 10.9 Zinc-Plated Fastener",
        "failure_modes": "Stripped Torx/hex head drive, thread pitch damage, head-to-shank neck scratch",
        "risk_threshold": "Damaged drive prevents automated robotic assembly torque; neck defects trigger fatigue fracture.",
        "tooling_capa": "Inspect thread-rolling dies, calibrate heading punch strike alignment, check zinc plating rack contacts."
    },
    "toothbrush": {
        "material": "Molded Polypropylene & Nylon Bristles",
        "failure_modes": "Splayed bristle tuft, missing anchor plate, bristle shedding, handle flash",
        "risk_threshold": "Loose bristles present choking risk and gum abrasion during hygiene use.",
        "tooling_capa": "Inspect bristle tufting stapling machine, verify anchor wire feed tension, examine bristle end-rounding buffer."
    },
    "transistor": {
        "material": "Semiconductor TO-220 / DPAK Epoxy Package",
        "failure_modes": "Bent lead terminal, cracked epoxy casing, heatsink tab scratch, solder bridging",
        "risk_threshold": "Bent leads block PCB through-hole insertion; cracked packaging permits moisture ingress.",
        "tooling_capa": "Inspect lead trim-and-form tooling die, calibrate SMD tape-and-reel feeder, check vacuum pickup nozzle."
    },
    "wood": {
        "material": "Hardwood / Architectural Veneer Board",
        "failure_modes": "Drilled hole, resin/liquid stain, surface gouge scratch, knot cracking",
        "risk_threshold": "Deep gouges ruin decorative finish grade; knot fractures reduce structural bending strength.",
        "tooling_capa": "Check planer blade sharpness, inspect automated sanding belt grit, clean conveyor drive belts."
    },
    "zipper": {
        "material": "Molded Plastic / Brass Interlocking Teeth",
        "failure_modes": "Missing tooth, bent pitch spacing, misaligned slider stop, slider catch",
        "risk_threshold": "Tooth displacement prevents tooth interlock, causing complete separation under lateral pull.",
        "tooling_capa": "Inspect tooth injection mold cavity, check ultrasonic top-stop welding horn, examine chain guide."
    }
}


class LlamaAdvisoryEngine:
    def __init__(
        self,
        backend="auto",
        runner_path=None,
        model_path=None,
        tokenizer_path=None,
        server_url="http://127.0.0.1:8080/completion",
        max_tokens=65,
        temperature=0.0,
    ):
        disc_runner, disc_model, disc_tok = auto_discover_llama()

        self.runner_path = Path(runner_path) if runner_path else disc_runner
        self.model_path = Path(model_path) if model_path else disc_model
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else disc_tok
        self.server_url = server_url
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

        # Determine backend
        if backend == "auto":
            # 1. Check if persistent in-memory server is listening (0s cold load, ~2s generation)
            server_host = self.server_url.rsplit("/", 1)[0]
            if check_persistent_server(server_host):
                self.backend = "server"
                print(f"[LlamaAdvisoryEngine] Connected to Persistent In-Memory Server at {self.server_url}")
            # 2. Check if ExecuTorch binary & model are available
            elif (
                self.runner_path
                and self.runner_path.is_file()
                and os.access(self.runner_path, os.X_OK)
                and self.model_path
                and self.model_path.is_file()
            ):
                self.backend = "executorch"
                print(f"[LlamaAdvisoryEngine] Found ExecuTorch at: {self.runner_path.parent}")
                print(f"[LlamaAdvisoryEngine] Model: {self.model_path.name}")
            else:
                self.backend = "simulation"
                print("[LlamaAdvisoryEngine] ExecuTorch model files not fully present. Using Fast-Path Rule Advisory mode.")
        else:
            self.backend = backend

        # Fast-Path RAM Disk Staging: Mirror ExecuTorch model to Linux /dev/shm for zero-disk cold load
        self.is_ram_disk = False
        self.original_model_path = self.model_path
        if self.backend == "executorch" and self.model_path and self.model_path.is_file():
            ram_model, model_in_ram = mirror_to_ram_disk(self.model_path)
            self.model_path = ram_model
            self.is_ram_disk = model_in_ram

            if self.tokenizer_path and self.tokenizer_path.is_file():
                ram_tok, _ = mirror_to_ram_disk(self.tokenizer_path)
                self.tokenizer_path = ram_tok

            if self.is_ram_disk:
                print(f"[LlamaAdvisoryEngine] 🚀 ExecuTorch active from RAM disk (/dev/shm): {self.model_path.name}")
            else:
                print(f"[LlamaAdvisoryEngine] ExecuTorch active from Disk: {self.model_path}")

        print(f"[LlamaAdvisoryEngine] Active advisory backend: {self.backend} (RAM Resident: {self.is_ram_disk})")

    def build_compact_prompt(self, category, detections, image_shape, rule_eval):
        h, w = image_shape[:2]
        cat_key = category.lower()
        kb = INDUSTRIAL_KB.get(cat_key, INDUSTRIAL_KB["tile"])

        total_pct = rule_eval.get("total_mask_pct", 0)
        verdict = rule_eval.get("verdict", "REJECT")
        severity = rule_eval.get("severity", "CRITICAL")

        det_summary = []
        for i, d in enumerate(detections, 1):
            pct = (d.get("mask_area_px", 0) / (h * w)) * 100.0 if (h * w) > 0 else 0
            lbl = d.get('label', 'defect')
            det_summary.append(
                f"{lbl} #{i} ({pct:.1f}% area, {d.get('confidence', 0)*100:.0f}% certainty)"
            )
        det_str = "; ".join(det_summary) if det_summary else "No anomalies"

        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"You are a Six-Sigma Quality Engineer. An optical conveyor camera has detected a physical surface anomaly on an industrial part.\n"
            f"Respond strictly in 3 concise engineering bullet points:\n"
            f"- Severity: {severity}\n"
            f"- Assessment: [Technical failure mode and functional risk to {kb['material']} based on {total_pct}% defect area]\n"
            f"- Action: [1. Part disposition, 2. Batch quarantine, 3. Tooling CAPA: check {kb['tooling_capa']}]\n"
            f"\n"
            f"Example:\n"
            f"- Severity: CRITICAL\n"
            f"- Assessment: Thermal firing crack covering 16% surface. Loss of structural flexural strength; fails ISO standard.\n"
            f"- Action: 1. Auto-reject to scrap hopper. 2. Quarantine preceding 20 units. 3. Check kiln cooling ramp and press die balance.\n"
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Component: {category.upper()} ({kb['material']}).\n"
            f"Optical Camera Finding: {det_str} covering {total_pct}% surface area.\n"
            f"Quality Gate Verdict: {verdict} (Severity: {severity}).\n"
            f"Known factory failure modes: {kb['failure_modes']}.\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        return prompt

    def generate_advisory(self, category, detections, image_shape, rule_eval):
        t0 = time.perf_counter()

        if not detections:
            return (
                "• Severity: None\n"
                "• Assessment: Visual inspection verified surface uniformity within tolerance.\n"
                "• Action: Accept component for continuous manufacturing flow.",
                12.5
            )

        if self.backend == "simulation":
            advisory, latency = self._generate_simulation(category, detections, image_shape, rule_eval)
            return advisory, latency

        if self.backend == "server":
            advisory = self._generate_server(category, detections, image_shape, rule_eval)
            latency = (time.perf_counter() - t0) * 1000.0
            return advisory, round(latency, 2)

        if self.backend == "executorch":
            advisory = self._generate_executorch(category, detections, image_shape, rule_eval)
            latency = (time.perf_counter() - t0) * 1000.0
            return advisory, round(latency, 2)

        return rule_eval.get("rule_action", "No action specified."), 5.0

    def _generate_simulation(self, category, detections, image_shape, rule_eval):
        """Intelligent contextual simulation for PC testing."""
        severity = rule_eval.get("severity", "MEDIUM")
        verdict = rule_eval.get("verdict", "FLAGGED_MONITOR")
        pct = rule_eval.get("total_mask_pct", 1.5)
        conf = rule_eval.get("max_confidence", 0.90) * 100.0

        if severity == "CRITICAL":
            assessment = f"Severe anomaly detected across {pct}% of {category} surface with {conf:.1f}% confidence, indicating structural breach or contamination."
            action = f"Immediate rejection required. Quarantine lot, stop feeder line, and inspect tooling calibration."
        elif severity == "HIGH":
            assessment = f"Pronounced surface defect covering {pct}% detected ({conf:.1f}% certitude), exceeding nominal tolerance limit."
            action = f"Divert part to rework bin. Schedule secondary optical probe and check batch alignment."
        elif severity == "MEDIUM":
            assessment = f"Moderate surface variation covering {pct}% observed; does not impair core structural integrity but fails aesthetic QA."
            action = f"Flag for secondary manual inspection. Clean component surface and evaluate next inspection station."
        else:
            assessment = f"Minor cosmetic anomaly ({pct}%) within acceptable marginal operating thresholds."
            action = f"Pass with monitoring flag. Track batch defect density over next 50 cycles."

        advisory_text = (
            f"• Severity: {severity}\n"
            f"• Assessment: {assessment}\n"
            f"• Action: {action}"
        )
        sim_latency = round(45.0 + (len(category) * 3.5), 1)
        return advisory_text, sim_latency

    def _generate_server(self, category, detections, image_shape, rule_eval):
        prompt = self.build_compact_prompt(category, detections, image_shape, rule_eval)
        payload = json.dumps({
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "stop": ["<|eot_id|>", "<|end_of_text|>"],
        }).encode("utf-8")

        req = urllib.request.Request(
            self.server_url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("content", "").strip()
        except Exception as e:
            return f"Error contacting Llama server: {e}"

    def _generate_executorch(self, category, detections, image_shape, rule_eval):
        prompt = self.build_compact_prompt(category, detections, image_shape, rule_eval)
        cmd = [
            str(self.runner_path),
            "--model_path", str(self.model_path),
            "--tokenizer_path", str(self.tokenizer_path),
            "--prompt", prompt,
            "--max_new_tokens", str(self.max_tokens),
            "--temperature", str(self.temperature),
            "--cpu_threads", "4",
        ]
        try:
            runner_dir = self.runner_path.parent
            # Ensure libllama_runner.so in the runner directory is discoverable by the dynamic linker
            env = os.environ.copy()
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{runner_dir}:{existing_ld}".rstrip(":")

            loc_desc = "RAM (/dev/shm)" if getattr(self, "is_ram_disk", False) else "Disk (MicroSD)"
            print(f"[ExecuTorch] Executing from {loc_desc}: {self.model_path.name}...")
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                cwd=str(runner_dir),
                env=env,
            )
            if res.returncode != 0:
                err_msg = res.stderr.strip() or f"exit code {res.returncode}"
                print(f"[ExecuTorch Failed] Exit {res.returncode}: {err_msg}")
                return (
                    f"• Severity: {rule_eval.get('severity', 'UNKNOWN')}\n"
                    f"• Assessment: Fast-Path Rule fallback (ExecuTorch failed: {err_msg[:60]})\n"
                    f"• Action: {rule_eval.get('rule_action', 'Verify ExecuTorch configuration.')}"
                )

            out = res.stdout

            # Extract PyTorchObserver telemetry and strip from user-facing text
            if "PyTorchObserver" in out:
                parts = out.split("PyTorchObserver", 1)
                out = parts[0]
                try:
                    observer_json = parts[1].strip()
                    obs_data = json.loads(observer_json)
                    decode_speed = obs_data.get("decode_token_per_sec", 0)
                    gen_tokens = obs_data.get("generated_tokens", 0)
                    print(f"[ExecuTorch Observer] Speed: {decode_speed:.2f} tok/s, Tokens: {gen_tokens}")
                except Exception:
                    pass

            marker_llama = "<|start_header_id|>assistant<|end_header_id|>"
            if marker_llama in out:
                out = out.rsplit(marker_llama, 1)[-1]
            elif prompt in out:
                out = out.replace(prompt, "")

            out = out.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()
            return out if out else (
                f"• Severity: {rule_eval.get('severity', 'UNKNOWN')}\n"
                f"• Assessment: ExecuTorch generated empty output.\n"
                f"• Action: {rule_eval.get('rule_action', 'Proceed with rule engine action.')}"
            )
        except subprocess.TimeoutExpired:
            print("[ExecuTorch] Timed out after 60s. Falling back to fast rule engine action.")
            return (
                f"• Severity: {rule_eval.get('severity', 'UNKNOWN')}\n"
                f"• Assessment: Fast-Path Rule fallback (ExecuTorch timed out after 60s).\n"
                f"• Action: {rule_eval.get('rule_action', 'Proceed with rule engine action.')}"
            )
        except Exception as e:
            print(f"[ExecuTorch Exception]: {e}")
            return (
                f"• Severity: {rule_eval.get('severity', 'UNKNOWN')}\n"
                f"• Assessment: ExecuTorch exception: {e}\n"
                f"• Action: {rule_eval.get('rule_action', 'Proceed with rule engine action.')}"
            )
