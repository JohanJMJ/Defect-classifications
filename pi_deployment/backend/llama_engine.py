import json
import os
from pathlib import Path
import shutil
import subprocess
import time

def mirror_to_ram_disk(src_path: Path, subfolder="executorch") -> tuple:
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

        print(f"[RAM Disk] Loading {src_path.name} ({src_size / (1024*1024):.1f} MB) into RAM disk (/dev/shm)...")
        t0 = time.perf_counter()
        shutil.copy2(str(src_path), str(dest_path))
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[RAM Disk] Successfully staged in RAM in {dt:.1f}ms Direct DDR4/DDR5 execution enabled.")
        return dest_path, True

    except Exception as e:
        print(f"[RAM Disk] Staging to /dev/shm skipped: {e}")
        return src_path, False


def auto_discover_llama():
    possible_dirs = [
        Path("/home/raspberrypi5/executorch-new"),
        Path.home() / "executorch-new",
        Path(__file__).resolve().parent.parent,
        Path.cwd(),
    ]
    for d in possible_dirs:
        try:
            if d.is_dir():
                runner = d / "llama_main"
                pte_files = list(d.glob("*.pte"))
                tok_files = list(d.glob("*tokenizer*"))
                if runner.is_file() and pte_files and tok_files:
                    # Pick the .pte model
                    int4_pte = [p for p in pte_files if p.name == "model_int4.pte"]
                    chosen_model = int4_pte[0] if int4_pte else pte_files[0]
                    # Pick the tokenizer 
                    json_toks = [t for t in tok_files if t.name.endswith(".json")]
                    chosen_tok = json_toks[0] if json_toks else tok_files[0]
                    return runner, chosen_model, chosen_tok
        except Exception:
            continue
    return None, None, None


# Knowledge base
INDUSTRIAL_KB = {
    "steel_defects": {
        "material": "Hot/Cold-Rolled Carbon Steel Sheet & Strip",
        "failure_modes": "Crazing micro-cracks, non-metallic inclusions, surface patches, pitting corrosion, rolled-in oxide scale, longitudinal scratches",
        "risk_threshold": "Surface cracks and rolled-in scale cause catastrophic tensile fracture under stamping loads (ASTM A1008/A568).",
        "tooling_capa": "Inspect continuous caster mold level, check roughing stand roll wear, verify descaler spray pressure (>15 MPa), and clean coil guide rollers."
    },
    "steel": {
        "material": "Hot/Cold-Rolled Carbon Steel Sheet & Strip",
        "failure_modes": "Crazing micro-cracks, non-metallic inclusions, surface patches, pitting corrosion, rolled-in oxide scale, longitudinal scratches",
        "risk_threshold": "Surface cracks and rolled-in scale cause catastrophic tensile fracture under stamping loads (ASTM A1008/A568).",
        "tooling_capa": "Inspect continuous caster mold level, check roughing stand roll wear, verify descaler spray pressure (>15 MPa), and clean coil guide rollers."
    },
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
    "grid": {
        "material": "Stamped Sheet Metal / Aluminum Honeycomb",
        "failure_modes": "Bent mesh strut, broken weld node, adhesive drip, burr deformation",
        "risk_threshold": "Strut fractures reduce structural shear resistance and block airflow volume.",
        "tooling_capa": "Inspect stamping die clearance, examine spot-weld electrode tips, check trim blade burrs."
    }
}


class LlamaAdvisoryEngine:
    def __init__(
        self,
        backend="auto",
        runner_path=None,
        model_path=None,
        tokenizer_path=None,
        server_url=None,
        max_tokens=48,
        temperature=0.0,
    ):
        disc_runner, disc_model, disc_tok = auto_discover_llama()

        self.runner_path = Path(runner_path).resolve() if runner_path else (disc_runner.resolve() if disc_runner else None)
        self.model_path = Path(model_path).resolve() if model_path else (disc_model.resolve() if disc_model else None)
        self.tokenizer_path = Path(tokenizer_path).resolve() if tokenizer_path else (disc_tok.resolve() if disc_tok else None)
        self.server_url = server_url
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

        # Determine backend: Primary is Meta ExecuTorch on Arm64 CPU
        if backend == "auto":
            if (
                self.runner_path
                and self.runner_path.is_file()
                and os.access(self.runner_path, os.X_OK)
                and self.model_path
                and self.model_path.is_file()
            ):
                self.backend = "executorch"
                print(f"[LlamaAdvisoryEngine] Arm ExecuTorch Runtime Active: {self.runner_path.parent}")
                print(f"[LlamaAdvisoryEngine] Model (.pte): {self.model_path.name}")
            else:
                self.backend = "simulation"
                print("[LlamaAdvisoryEngine] ExecuTorch model files not present. Using Fast-Path Rule Advisory mode (Simulation).")
        else:
            self.backend = backend

        # Fast-Path RAM Disk Staging: Mirror ExecuTorch model to Linux /dev/shm for zero-disk cold load
        self.is_ram_disk = False
        self.original_model_path = self.model_path
        if self.model_path and self.model_path.is_file():
            ram_model, model_in_ram = mirror_to_ram_disk(self.model_path)
            self.model_path = ram_model
            self.is_ram_disk = model_in_ram

            if self.tokenizer_path and self.tokenizer_path.is_file():
                ram_tok, _ = mirror_to_ram_disk(self.tokenizer_path)
                self.tokenizer_path = ram_tok

            if self.is_ram_disk and self.backend == "executorch":
                print(f"[LlamaAdvisoryEngine] ExecuTorch active from RAM disk (/dev/shm): {self.model_path.name}")
            elif self.backend == "executorch":
                print(f"[LlamaAdvisoryEngine] ExecuTorch active from Disk: {self.model_path}")

        print(f"[LlamaAdvisoryEngine] Active advisory backend: {self.backend} (RAM Resident: {self.is_ram_disk})")

    # Each anchors the model's vocabulary to the right industrial domain.
    _CATEGORY_EXAMPLES = {
        "steel_defects": ("Rolled-in scale induces fracture propagation during deep drawing stamping.",
                          "Quarantine hot coil, check high-pressure descaling water headers."),
        "steel":         ("Rolled-in scale induces fracture propagation during deep drawing stamping.",
                          "Quarantine hot coil, check high-pressure descaling water headers."),
        "tile":       ("Stress crack propagates under 1500N floor load per ISO 10545.",
                       "Stop press, inspect Kiln Zone 3 cooling ramp."),
        "bottle":     ("Neck finish crack breaks hermetic seal, risks spoilage.",
                       "Reject lot, check blow-mold clamp alignment."),
        "grid":       ("Bent strut reduces structural shear resistance per spec.",
                       "Divert part, inspect stamping die clearance."),
    }

    def build_compact_prompt(self, category, detections, image_shape, rule_eval):
        cat_key = category.lower()
        kb = INDUSTRIAL_KB.get(cat_key, INDUSTRIAL_KB["steel_defects"])

        total_pct = rule_eval.get("total_mask_pct", 0.0)
        verdict = rule_eval.get("verdict", "REJECT")
        severity_tier = str(rule_eval.get("severity", "CRITICAL")).upper()
        conf_pct = rule_eval.get("max_confidence", 0.85) * 100.0

        # Clean defect labels
        labels = sorted(list(set(d.get('label', 'defect') for d in detections)))
        det_names = "/".join(labels) if labels else "defect"

        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are an expert Industrial Quality Control (QC) Engineer.\n"
            "- Severity level (CRITICAL, MAJOR, MODERATE, MINOR)\n"
            "- Technical root cause engineering assessment\n"
            "- Immediate manufacturing corrective action<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Part: {category.upper()} ({kb['material']}). "
            f"Defect: {det_names} (area: {total_pct:.1f}%, confidence: {conf_pct:.0f}%). "
            f"Preliminary Rule Verdict: {verdict}.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"- Severity: {severity_tier}\n"
            f"- Assessment:"
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

        if self.backend == "executorch":
            advisory = self._generate_executorch(category, detections, image_shape, rule_eval)
            latency = (time.perf_counter() - t0) * 1000.0
            return advisory, round(latency, 2)

        # For the simulation Mode
        advisory, latency = self._generate_simulation(category, detections, image_shape, rule_eval)
        return advisory, latency

    def _generate_simulation(self, category, detections, image_shape, rule_eval):
        severity = rule_eval.get("severity", "MEDIUM")
        verdict = rule_eval.get("verdict", "FLAGGED_MONITOR")
        pct = rule_eval.get("total_mask_pct", 1.5)
        conf = rule_eval.get("max_confidence", 0.90) * 100.0

        sev_upper = str(severity).upper()
        if sev_upper == "CRITICAL":
            assessment = f"Critical defect detected across {pct}% of {category} surface ({conf:.1f}% confidence), indicating severe structural breach or material defect."
            action = f"Immediate rejection required. Auto-reject to scrap bin, stop feeder line, and inspect tooling calibration."
        elif sev_upper == "MAJOR":
            assessment = f"Major surface defect covering {pct}% detected ({conf:.1f}% certitude), exceeding nominal operating tolerance."
            action = f"Hold component at inspection station. Require manual quality inspector sign-off before proceeding."
        elif sev_upper == "MODERATE":
            assessment = f"Moderate surface anomaly covering {pct}% observed. Functional integrity intact, but requires diversion."
            action = f"Divert component from main conveyor to buffer/rework lane. Monitor station variance."
        else:
            assessment = f"Minor cosmetic anomaly ({pct}%) within allowable marginal operating thresholds."
            action = f"Pass with cosmetic notice. Cleared for standard continuous manufacturing flow."

        advisory_text = (
            f"• Severity: {severity}\n"
            f"• Assessment: {assessment}\n"
            f"• Action: {action}"
        )
        sim_latency = round(45.0 + (len(category) * 3.5), 1)
        return advisory_text, sim_latency

    def _generate_executorch(self, category, detections, image_shape, rule_eval):
        prompt = self.build_compact_prompt(category, detections, image_shape, rule_eval)
        cmd = [
            str(self.runner_path),
            "--model_path", str(self.model_path),
            "--tokenizer_path", str(self.tokenizer_path),
            "--prompt", prompt,
            "--max_new_tokens", str(self.max_tokens),
            "--temperature", str(self.temperature),
            "--cpu_threads", "3",
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

            # Strip anything after end of text
            for end_tok in ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]:
                if end_tok in out:
                    out = out.split(end_tok)[0]

            out = out.strip()

            severity_upper = str(rule_eval.get("severity", "CRITICAL")).upper()
            cat_key = category.lower()
            kb = INDUSTRIAL_KB.get(cat_key, INDUSTRIAL_KB["steel_defects"])

            if "- Assessment:" in out:
                assessment_part = out.split("- Assessment:", 1)[1].strip()
                out = f"- Severity: {severity_upper}\n- Assessment: {assessment_part}"
            elif not out.startswith("- Severity:") and not out.startswith("• Severity:"):
                out = f"- Severity: {severity_upper}\n- Assessment: {out.lstrip('- ')}"
            else:
                lines = out.split("\n", 1)
                rest = lines[1] if len(lines) > 1 else ""
                out = f"- Severity: {severity_upper}\n{rest}"

            # If the llama gets cutoff or forgets the action in between, it appends a pre-verified correcive action from the rule engine or knowledge base
            fallback_act = rule_eval.get("rule_action", kb["tooling_capa"])
            has_action = any(tag in out for tag in ["- Action:", "• Action:", "Action:", "- Immediate Action:"])
            if not has_action:
                out = out.rstrip().rstrip(".") + f".\n- Action: {fallback_act}"
            else:
                for tag in ["- Action:", "• Action:", "- Immediate Action:", "Action:"]:
                    if tag in out:
                        parts = out.split(tag, 1)
                        act_text = parts[1].strip()
                        if len(act_text) < 15 or not act_text.endswith((".", "!", "?")):
                            out = parts[0].strip() + f"\n- Action: {fallback_act}"
                        break

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
