"""
Llama AI Advisory Engine for Industrial QC
Provides intelligent defect severity assessment and actionable recommendations.
Supports ExecuTorch (ARM64 CPU), llama-server / Ollama HTTP, and PC Simulation Mode.
"""

import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import urllib.error

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
                    # Prioritize 1B model if present
                    models_1b = [p for p in pte_files if "1b" in p.name.lower()]
                    chosen_model = models_1b[0] if models_1b else pte_files[0]
                    return runner, chosen_model, tok_files[0]
                elif runner.is_file():
                    model = pte_files[0] if pte_files else (d / "llama-1b.pte")
                    tok = tok_files[0] if tok_files else (d / "tokenizer.json")
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
    "pill": {
        "material": "Compressed Solid Dose Tablet",
        "failure_modes": "Tablet capping/lamination, edge chipping, surface contamination, deboss defect",
        "risk_threshold": "Chipping causes variable drug dosage; capping exposes formulation to rapid degradation.",
        "tooling_capa": "Check tablet press punch tooling wear, verify pre-compression force, inspect ejection cam."
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
        max_tokens=125,
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
            if (
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

        print(f"[LlamaAdvisoryEngine] Active advisory backend: {self.backend}")

    def build_compact_prompt(self, category, detections, image_shape, rule_eval):
        h, w = image_shape[:2]
        cat_key = category.lower()
        kb = INDUSTRIAL_KB.get(cat_key, INDUSTRIAL_KB["tile"])

        total_pct = rule_eval.get("total_mask_pct", 0)
        verdict = rule_eval.get("verdict", "REJECT")
        severity = rule_eval.get("severity", "CRITICAL")
        max_conf = rule_eval.get("max_confidence", 0.85)

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

            print(f"[ExecuTorch] Executing: {self.runner_path.name} in {runner_dir}...")
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
                # Return explicit Fast-Path Rule Fallback so the user knows Llama didn't execute
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

            marker = "<|start_header_id|>assistant<|end_header_id|>"
            if marker in out:
                out = out.rsplit(marker, 1)[-1]
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
