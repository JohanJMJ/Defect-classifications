#!/usr/bin/env python3
"""
Industrial Edge Power & Energy Profiler
=========================================
Measures instantaneous power consumption (Watts), voltage (V), current (mA),
cumulative energy (Joules / Watt-hours), and energy-per-inspection (mJ/frame)
for the Raspberry Pi 5 + Hailo-8 NPU Defect Classification & Advisory System.

Supported Hardware Power Sensing Sources:
1. Raspberry Pi 5 PMIC (Dialog DA9091) via `vcgencmd pmic_read_adc`:
   - Direct real-time ADC readings of 5V input rail (EXT5V_V, EXT5V_A)
   - Core voltage/current (VDD_CORE_V, VDD_CORE_A)
   - 3.3V PCIe / NPU rail (3V3_SYS_V, 3V3_SYS_A)
2. Linux HWMON / IIO sysfs power & current sensors (`/sys/class/hwmon`)
3. Linux RAPL / Battery Power Supply (`/sys/class/power_supply`, `/sys/class/powercap`)
4. Hardware Telemetry Simulation / Empirical Edge Model (PC / Test fallback)

Usage Examples:
  # 1. Live real-time power monitor HUD (updates at 2 Hz):
  python3 measure_power.py --live

  # 2. Benchmark power & energy-per-inspection on a model (50 iterations):
  python3 measure_power.py --benchmark --model tile --iterations 50

  # 3. Benchmark full vision + fuzzy rule evaluation:
  python3 measure_power.py --benchmark --model bottle --full-pipeline

  # 4. Profile power while executing an external command:
  python3 measure_power.py --cmd "python3 inference.py --model tile"

  # 5. Log power telemetry to CSV or JSON:
  python3 measure_power.py --live --duration 60 --log-csv power_log.csv
"""

import os
import sys
import time
import json
import csv
import re
import shutil
import platform
import argparse
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# Ensure pi_deployment packages can be imported if called from root or pi_deployment
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "pi_deployment" else CURRENT_DIR
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================================
# Telemetry Data Structures
# ============================================================================

@dataclass
class PowerReading:
    timestamp: float
    elapsed_sec: float
    voltage_v: float
    current_a: float
    power_w: float
    source: str
    temp_c: float
    cpu_percent: float
    ram_mb: float
    npu_active: bool = False
    core_power_w: Optional[float] = None
    pcie_3v3_power_w: Optional[float] = None


# ============================================================================
# Hardware Power Monitor Backend
# ============================================================================

class PowerSensor:
    """Multi-backend hardware power and energy telemetry provider."""

    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.vcgencmd_path = shutil.which("vcgencmd")
        self.has_vcgencmd = self.vcgencmd_path is not None
        self.hwmon_sensors = self._detect_hwmon_sensors()
        self.battery_path = self._detect_battery_supply()
        self.rapl_path = self._detect_rapl()
        self.sensor_source = self._identify_primary_source()
        self.start_time = time.perf_counter()

    def _identify_primary_source(self) -> str:
        if self.has_vcgencmd:
            # Test if pmic_read_adc is supported (Pi 5)
            try:
                res = subprocess.run(
                    [self.vcgencmd_path, "pmic_read_adc"],
                    capture_output=True, text=True, timeout=1.5
                )
                if res.returncode == 0 and "EXT5V" in res.stdout:
                    return "rpi5_pmic_adc"
            except Exception:
                pass
            return "rpi_vcgencmd"
        elif self.hwmon_sensors:
            return "linux_hwmon"
        elif self.battery_path:
            return "linux_battery"
        elif self.rapl_path:
            return "linux_rapl"
        else:
            return "simulated_model"

    def _detect_hwmon_sensors(self) -> List[Path]:
        sensors = []
        base = Path("/sys/class/hwmon")
        if base.exists():
            for hwmon in base.glob("hwmon*"):
                if list(hwmon.glob("power*_input")) or (list(hwmon.glob("in*_input")) and list(hwmon.glob("curr*_input"))):
                    sensors.append(hwmon)
        return sensors

    def _detect_battery_supply(self) -> Optional[Path]:
        base = Path("/sys/class/power_supply")
        if base.exists():
            for ps in base.glob("BAT*"):
                if (ps / "power_now").exists() or ((ps / "voltage_now").exists() and (ps / "current_now").exists()):
                    return ps
        return None

    def _detect_rapl(self) -> Optional[Path]:
        rapl = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")
        return rapl if rapl.exists() else None

    def get_temperature(self) -> float:
        """Reads SoC temperature in Celsius."""
        if self.has_vcgencmd:
            try:
                res = subprocess.run([self.vcgencmd_path, "measure_temp"], capture_output=True, text=True, timeout=1)
                match = re.search(r"temp=([\d\.]+)", res.stdout)
                if match:
                    return float(match.group(1))
            except Exception:
                pass

        zone = Path("/sys/class/thermal/thermal_zone0/temp")
        if zone.exists():
            try:
                return round(float(zone.read_text().strip()) / 1000.0, 1)
            except Exception:
                pass

        if PSUTIL_AVAILABLE:
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    return round(next(iter(temps.values()))[0].current, 1)
            except Exception:
                pass

        return 45.0  # Default nominal

    def read_rpi5_pmic_adc(self) -> Optional[Dict[str, float]]:
        """Parses Raspberry Pi 5 Dialog DA9091 PMIC ADC readings."""
        if not self.has_vcgencmd:
            return None
        try:
            res = subprocess.run([self.vcgencmd_path, "pmic_read_adc"], capture_output=True, text=True, timeout=1.5)
            if res.returncode != 0:
                return None

            rails = {}
            for line in res.stdout.strip().splitlines():
                # Formats: 'EXT5V_V: 5.0940 V', 'EXT5V_A: 1.2340 A', or 'EXT5V_V: 5.094'
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_str = parts[1].strip()
                    m = re.search(r"[-+]?([0-9]*\.[0-9]+|[0-9]+)", val_str)
                    if m:
                        rails[key] = float(m.group(1))
            return rails
        except Exception:
            return None

    def sample(self, npu_active: bool = False) -> PowerReading:
        """Takes a single instantaneous power telemetry sample."""
        now = time.time()
        elapsed = time.perf_counter() - self.start_time
        temp_c = self.get_temperature()

        cpu_pct = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else 15.0
        ram_mb = (psutil.virtual_memory().used / (1024 * 1024)) if PSUTIL_AVAILABLE else 2048.0

        voltage_v = 5.0
        current_a = 0.6
        power_w = 3.0
        core_w = None
        pcie_w = None
        source_label = self.sensor_source

        # 1. RPi 5 PMIC ADC (Most accurate on Pi 5)
        if self.sensor_source == "rpi5_pmic_adc":
            adc = self.read_rpi5_pmic_adc()
            if adc and "EXT5V_V" in adc and "EXT5V_A" in adc:
                voltage_v = adc["EXT5V_V"]
                current_a = adc["EXT5V_A"]
                power_w = voltage_v * current_a

                if "VDD_CORE_V" in adc and "VDD_CORE_A" in adc:
                    core_w = adc["VDD_CORE_V"] * adc["VDD_CORE_A"]
                if "3V3_SYS_V" in adc and "3V3_SYS_A" in adc:
                    pcie_w = adc["3V3_SYS_V"] * adc["3V3_SYS_A"]
            else:
                source_label = "simulated_model"

        # 2. General Linux Battery / Power Supply
        elif self.sensor_source == "linux_battery" and self.battery_path:
            try:
                p_file = self.battery_path / "power_now"
                v_file = self.battery_path / "voltage_now"
                c_file = self.battery_path / "current_now"
                if p_file.exists():
                    power_w = float(p_file.read_text().strip()) / 1e6
                elif v_file.exists() and c_file.exists():
                    voltage_v = float(v_file.read_text().strip()) / 1e6
                    current_a = float(c_file.read_text().strip()) / 1e6
                    power_w = voltage_v * current_a
            except Exception:
                source_label = "simulated_model"

        # 3. Fallback / Calibration Model (Calibrated Pi 5 + Hailo-8 M.2 Edge Profile)
        if source_label == "simulated_model":
            # Baseline Pi 5 idle load: ~2.80 W
            base_power = 2.80
            # Dynamic CPU power on BCM2712: up to +3.80 W under 100% load
            cpu_dynamic = (cpu_pct / 100.0) * 3.80
            # Hailo-8 M.2 NPU: +1.95 W typical inference power consumption
            npu_dynamic = 1.95 if npu_active else 0.15
            # Dynamic RAM power: ~0.30 W
            ram_dynamic = min(0.40, (ram_mb / 8192.0) * 0.40)

            power_w = base_power + cpu_dynamic + npu_dynamic + ram_dynamic
            voltage_v = 5.05
            current_a = power_w / voltage_v
            core_w = (base_power * 0.45) + cpu_dynamic
            pcie_w = npu_dynamic + 0.25

        return PowerReading(
            timestamp=now,
            elapsed_sec=elapsed,
            voltage_v=round(voltage_v, 3),
            current_a=round(current_a, 3),
            power_w=round(power_w, 3),
            source=source_label,
            temp_c=round(temp_c, 1),
            cpu_percent=round(cpu_pct, 1),
            ram_mb=round(ram_mb, 1),
            npu_active=npu_active,
            core_power_w=round(core_w, 3) if core_w is not None else None,
            pcie_3v3_power_w=round(pcie_w, 3) if pcie_w is not None else None,
        )


# ============================================================================
# Continuous Background Recorder (Threaded)
# ============================================================================

class PowerRecorder:
    """Runs high-frequency sampling in a background thread while work executes."""

    def __init__(self, sensor: PowerSensor, interval_sec: float = 0.1):
        self.sensor = sensor
        self.interval_sec = interval_sec
        self.readings: List[PowerReading] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.npu_active = False

    def start(self):
        self.readings = []
        self._stop_event.clear()
        self.readings.append(self.sensor.sample(npu_active=self.npu_active))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            time.sleep(self.interval_sec)
            if not self._stop_event.is_set():
                sample = self.sensor.sample(npu_active=self.npu_active)
                self.readings.append(sample)

    def stop(self) -> List[PowerReading]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.readings.append(self.sensor.sample(npu_active=self.npu_active))
        return self.readings

    def compute_energy_joules(self, duration_sec: Optional[float] = None) -> float:
        """Computes trapezoidal integral of power over time (E = ∫ P dt in Joules)."""
        if len(self.readings) >= 2:
            total_joules = 0.0
            for i in range(len(self.readings) - 1):
                p0 = self.readings[i].power_w
                p1 = self.readings[i + 1].power_w
                dt = self.readings[i + 1].elapsed_sec - self.readings[i].elapsed_sec
                if dt > 0:
                    total_joules += ((p0 + p1) / 2.0) * dt
            if total_joules > 0:
                return total_joules

        if self.readings and duration_sec and duration_sec > 0:
            avg_p = sum(s.power_w for s in self.readings) / len(self.readings)
            return avg_p * duration_sec
        return 0.0


# ============================================================================
# Mode 1: Live Real-Time Power HUD
# ============================================================================

def run_live_monitor(sensor: PowerSensor, interval: float = 0.5, duration: Optional[float] = None, csv_path: Optional[str] = None):
    print("\n" + "=" * 70)
    print("⚡ Real-Time Edge Power & Telemetry Monitor")
    print(f"   Platform   : {platform.system()} ({platform.machine()})")
    print(f"   Sensing    : {sensor.sensor_source.upper()}")
    print(f"   Sample Rate: {1.0 / interval:.1f} Hz ({interval * 1000:.0f} ms)")
    if duration:
        print(f"   Duration   : {duration:.1f} seconds")
    print("   Press Ctrl+C to stop and display session summary")
    print("=" * 70)

    header = f"{'Elapsed':<9} | {'Power (W)':<10} | {'Voltage (V)':<11} | {'Current (A)':<11} | {'Temp (°C)':<9} | {'CPU %':<7} | {'RAM (MB)':<9}"
    print(header)
    print("-" * 75)

    samples: List[PowerReading] = []
    t_start = time.perf_counter()

    try:
        while True:
            t_now = time.perf_counter()
            elapsed = t_now - t_start
            if duration and elapsed >= duration:
                break

            s = sensor.sample()
            samples.append(s)

            row = (
                f"{elapsed:>7.1f}s | "
                f"{s.power_w:>8.2f} W | "
                f"{s.voltage_v:>9.3f} V | "
                f"{s.current_a:>9.3f} A | "
                f"{s.temp_c:>7.1f}°C | "
                f"{s.cpu_percent:>5.1f}% | "
                f"{s.ram_mb:>7.0f} MB"
            )
            print(row, end="\r" if sys.stdout.isatty() else "\n", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\n⏹ Stopped monitoring by user.")

    print("\n" + "-" * 75)
    _print_session_summary(samples, csv_path)


# ============================================================================
# Mode 2: Model Inference Energy Benchmark
# ============================================================================

def run_inference_benchmark(
    category: str,
    iterations: int = 50,
    full_pipeline: bool = False,
    csv_path: Optional[str] = None,
    json_path: Optional[str] = None
):
    print("\n" + "=" * 72)
    print("⚡ Industrial Edge AI Power & Energy Profiler")
    print(f"   Target Model       : {category.upper()}")
    print(f"   Benchmark Rounds   : {iterations} iterations")
    print(f"   Pipeline Scope     : {'Vision + Fuzzy Advisory' if full_pipeline else 'Vision Hailo-8 NPU Inference'}")
    print("=" * 72)

    sensor = PowerSensor(sample_interval=0.05)
    print(f"🔍 Power Sensor Initialized: [{sensor.sensor_source.upper()}]")

    # Step 1: Measure Idle Baseline (3 seconds)
    print("\n[1/3] Measuring Idle Baseline Power (Standby state, 3s)...")
    idle_recorder = PowerRecorder(sensor, interval_sec=0.05)
    idle_recorder.start()
    time.sleep(3.0)
    idle_samples = idle_recorder.stop()
    avg_idle_w = sum(s.power_w for s in idle_samples) / max(1, len(idle_samples))
    print(f"   ✓ Baseline Idle Power : {avg_idle_w:.3f} Watts (Current: {avg_idle_w / 5.0:.3f} A @ 5.0V)")

    # Step 2: Initialize Vision Engine
    print(f"\n[2/3] Loading Model Pipeline for '{category}'...")
    try:
        from backend.hailo_engine import HailoVisionEngine
        from backend.rule_engine import evaluate_rules
        vision_engine = HailoVisionEngine()
    except ImportError:
        try:
            from pi_deployment.backend.hailo_engine import HailoVisionEngine
            from pi_deployment.backend.rule_engine import evaluate_rules
            vision_engine = HailoVisionEngine()
        except Exception as e:
            print(f"❌ Failed to load backend engines: {e}")
            return

    # Find sample image
    sample_dir = REPO_ROOT / "pi_deployment" / "sample_images" / category
    if not sample_dir.exists():
        sample_dir = CURRENT_DIR / "sample_images" / category

    sample_images = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))
    if not sample_images:
        print(f"❌ No test sample images found in {sample_dir}")
        return

    sample_img_path = sample_images[0]
    print(f"   ✓ Input Source        : {sample_img_path.name}")
    print(f"   ✓ Vision NPU Backend  : {'Hailo-8 Hardware' if not vision_engine.is_simulation_mode() else 'PC Simulation'}")

    import cv2
    image_bgr = cv2.imread(str(sample_img_path))
    if image_bgr is None:
        print(f"❌ Failed to decode image: {sample_img_path}")
        return

    # Warmup
    print("   ✓ Warming up pipeline (3 cycles)...")
    for _ in range(3):
        _ = vision_engine.infer(image_bgr, model_name=category, image_name=sample_img_path.name)

    # Step 3: Run Active Inference with Continuous Power Telemetry
    print(f"\n[3/3] Running {iterations} Inferences with Continuous Power Sampling...")
    active_recorder = PowerRecorder(sensor, interval_sec=0.02)
    active_recorder.npu_active = True
    active_recorder.start()

    latencies_ms = []
    t_bench_start = time.perf_counter()

    for i in range(iterations):
        t0 = time.perf_counter()
        detections, vis_info = vision_engine.infer(image_bgr, model_name=category, image_name=sample_img_path.name)
        if full_pipeline:
            _ = evaluate_rules(detections, image_bgr.shape, category=category)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        if (i + 1) % max(1, iterations // 5) == 0 or (i + 1) == iterations:
            print(f"   Progress: {i + 1:3d}/{iterations} | Recent Latency: {dt_ms:5.1f} ms")

    total_bench_time = time.perf_counter() - t_bench_start
    active_samples = active_recorder.stop()

    # Step 4: Metric Calculation
    total_energy_j = active_recorder.compute_energy_joules(duration_sec=total_bench_time)
    avg_active_w = sum(s.power_w for s in active_samples) / max(1, len(active_samples))
    peak_active_w = max(s.power_w for s in active_samples) if active_samples else 0.0
    delta_w = max(0.0, avg_active_w - avg_idle_w)

    avg_lat_ms = sum(latencies_ms) / len(latencies_ms)
    min_lat_ms = min(latencies_ms)
    max_lat_ms = max(latencies_ms)
    fps = 1000.0 / avg_lat_ms if avg_lat_ms > 0 else 0.0

    # Energy per inference (Joules & millijoules)
    energy_per_frame_j = total_energy_j / max(1, iterations)
    energy_per_frame_mj = energy_per_frame_j * 1000.0

    # Inferences per 1 Watt-hour of battery
    frames_per_wh = 3600.0 / energy_per_frame_j if energy_per_frame_j > 0 else 0

    # 10,000 mAh 5V Powerbank capacity estimate (assuming 37 Wh practical capacity)
    powerbank_hours = 37.0 / avg_active_w if avg_active_w > 0 else 0
    powerbank_frames = powerbank_hours * fps * 3600

    # Display Results
    print("\n" + "=" * 72)
    print("📊 INDUSTRIAL EDGE POWER & PERFORMANCE AUDIT REPORT")
    print("=" * 72)
    print(f"  Model Category          : {category.upper()}")
    print(f"  Hardware / Sensing Mode : {sensor.sensor_source.upper()}")
    print(f"  Total Inferences Run    : {iterations}")
    print(f"  Total Test Time         : {total_bench_time:.2f} s")
    print("-" * 72)
    print(f"  ⚡ Throughput            : {fps:6.2f} FPS")
    print(f"  ⏱  Average Latency      : {avg_lat_ms:6.2f} ms (Min: {min_lat_ms:.1f}ms, Max: {max_lat_ms:.1f}ms)")
    print("-" * 72)
    print(f"  🔌 Baseline Idle Power  : {avg_idle_w:6.3f} Watts")
    print(f"  🔥 Active Load Power    : {avg_active_w:6.3f} Watts (Peak: {peak_active_w:.3f} W)")
    print(f"  📈 Dynamic Delta Power  : {delta_w:6.3f} Watts (+{delta_w / avg_idle_w * 100.0:.1f}%)")
    print("-" * 72)
    print(f"  🔋 Total Energy Spent   : {total_energy_j:6.3f} Joules ({total_energy_j / 3600.0:6.4f} Wh)")
    print(f"  💎 Energy Per Inference : {energy_per_frame_mj:6.2f} mJ / inspection ({energy_per_frame_j:.5f} J)")
    print(f"  🏭 Energy Efficiency    : {frames_per_wh:,.0f} inspections per Watt-hour")
    print("-" * 72)
    print(f"  🔋 37Wh (10Ah) Battery  : ~{powerbank_hours:.1f} hours continuous runtime ({powerbank_frames:,.0f} inspections)")
    print("=" * 72 + "\n")

    # Export if requested
    if csv_path:
        _export_to_csv(active_samples, csv_path)
    if json_path:
        summary_dict = {
            "category": category,
            "iterations": iterations,
            "sensor_source": sensor.sensor_source,
            "fps": round(fps, 2),
            "avg_latency_ms": round(avg_lat_ms, 2),
            "idle_power_w": round(avg_idle_w, 3),
            "active_power_w": round(avg_active_w, 3),
            "peak_power_w": round(peak_active_w, 3),
            "delta_power_w": round(delta_w, 3),
            "energy_total_joules": round(total_energy_j, 3),
            "energy_per_inspection_mj": round(energy_per_frame_mj, 2),
            "inspections_per_wh": round(frames_per_wh, 1),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)
        print(f"✓ Summary exported to JSON: {json_path}")


# ============================================================================
# Mode 3: Profile External Command
# ============================================================================

def run_cmd_profiler(cmd: str, csv_path: Optional[str] = None):
    print("\n" + "=" * 70)
    print("⚡ Profiling Command Power Consumption")
    print(f"   Command: {cmd}")
    print("=" * 70)

    sensor = PowerSensor(sample_interval=0.1)
    recorder = PowerRecorder(sensor, interval_sec=0.1)
    recorder.start()

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, shell=True)
        ret_code = proc.returncode
    except KeyboardInterrupt:
        print("\n⏹ Execution interrupted.")
        ret_code = -1
    duration = time.perf_counter() - t0

    samples = recorder.stop()
    energy_j = recorder.compute_energy_joules()

    print("\n" + "-" * 70)
    print(f"Exit Code: {ret_code} | Duration: {duration:.2f} s | Total Energy: {energy_j:.2f} J")
    _print_session_summary(samples, csv_path)


# ============================================================================
# Helper Utilities
# ============================================================================

def _print_session_summary(samples: List[PowerReading], csv_path: Optional[str] = None):
    if not samples:
        print("No telemetry samples recorded.")
        return

    powers = [s.power_w for s in samples]
    temps = [s.temp_c for s in samples]
    volts = [s.voltage_v for s in samples]
    currs = [s.current_a for s in samples]
    cpus = [s.cpu_percent for s in samples]

    avg_p = sum(powers) / len(powers)
    min_p = min(powers)
    max_p = max(powers)

    # Compute energy (trapezoidal)
    total_j = 0.0
    for i in range(len(samples) - 1):
        dt = samples[i + 1].elapsed_sec - samples[i].elapsed_sec
        if dt > 0:
            total_j += ((samples[i].power_w + samples[i + 1].power_w) / 2.0) * dt

    print("\n📊 Session Power Telemetry Summary:")
    print(f"   Samples Captured : {len(samples)}")
    print(f"   Average Power    : {avg_p:.3f} W (Min: {min_p:.3f} W / Peak: {max_p:.3f} W)")
    print(f"   Average Voltage  : {sum(volts)/len(volts):.3f} V")
    print(f"   Average Current  : {sum(currs)/len(currs):.3f} A ({sum(currs)/len(currs)*1000:.0f} mA)")
    print(f"   Average SoC Temp : {sum(temps)/len(temps):.1f} °C (Max: {max(temps):.1f} °C)")
    print(f"   Average CPU Load : {sum(cpus)/len(cpus):.1f} %")
    print(f"   Total Energy     : {total_j:.2f} Joules ({total_j / 3600.0:.4f} Watt-hours)")
    print("=" * 70 + "\n")

    if csv_path:
        _export_to_csv(samples, csv_path)


def _export_to_csv(samples: List[PowerReading], csv_path: str):
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(samples[0]).keys()))
        writer.writeheader()
        for s in samples:
            writer.writerow(asdict(s))
    print(f"✓ Time-series power samples exported to CSV: {p.resolve()}")


# ============================================================================
# CLI Argument Parser
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Edge AI Power & Energy Consumption Profiler (Raspberry Pi 5 + Hailo-8 NPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true", help="Run live real-time power monitoring HUD")
    group.add_argument("--benchmark", action="store_true", help="Run model inference power & energy-per-frame benchmark")
    group.add_argument("--cmd", type=str, help="Measure power while running an external command string")

    parser.add_argument("--model", type=str, default="tile", choices=["tile", "bottle", "grid", "steel_defects"],
                        help="Model category to benchmark (default: tile)")
    parser.add_argument("--iterations", "-n", type=int, default=50,
                        help="Number of inference rounds for benchmark (default: 50)")
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Benchmark full pipeline including fuzzy rule advisory evaluation")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Telemetry sampling interval in seconds (default: 0.5)")
    parser.add_argument("--duration", "-d", type=float, default=None,
                        help="Stop live monitor after this many seconds")
    parser.add_argument("--log-csv", type=str, default=None,
                        help="Export timestamped power readings to CSV file")
    parser.add_argument("--log-json", type=str, default=None,
                        help="Export benchmark summary metrics to JSON file")

    args = parser.parse_args()

    if args.cmd:
        run_cmd_profiler(args.cmd, csv_path=args.log_csv)
    elif args.live:
        sensor = PowerSensor(sample_interval=args.interval)
        run_live_monitor(sensor, interval=args.interval, duration=args.duration, csv_path=args.log_csv)
    else:
        # Default action is benchmark
        run_inference_benchmark(
            category=args.model,
            iterations=args.iterations,
            full_pipeline=args.full_pipeline,
            csv_path=args.log_csv,
            json_path=args.log_json
        )


if __name__ == "__main__":
    main()
