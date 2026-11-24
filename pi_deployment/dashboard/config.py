from pathlib import Path
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import os
import platform

BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_YAML = MODELS_DIR / "models.yaml"
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
DATA_DIR = DASHBOARD_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "inspections.db"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

def get_available_models():
    if YAML_AVAILABLE and MODELS_YAML.is_file():
        try:
            with open(MODELS_YAML, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("models", {})
        except Exception as e:
            print(f"[Config] Error reading {MODELS_YAML}: {e}")

    # Fallback default models list
    return {
        "steel_defects": {"category": "steel_defects", "description": "Steel Defects"},
        "tile": {"category": "tile", "description": "Tile"},
        "bottle": {"category": "bottle", "description": "Bottle"},
        "grid": {"category": "grid", "description": "Grid"},
    }

def get_system_telemetry():
    if PSUTIL_AVAILABLE:
        cpu_pct = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory()
        ram_used_mb = round((ram.total - ram.available) / (1024 * 1024), 0)
        ram_total_mb = round(ram.total / (1024 * 1024), 0)
        ram_pct = ram.percent
    else:
        cpu_pct = 14.5
        ram_used_mb = 3120
        ram_total_mb = 8192
        ram_pct = 38.1

    temp_c = None
    if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_c = round(float(f.read().strip()) / 1000.0, 1)
        except Exception:
            pass

    if temp_c is None and PSUTIL_AVAILABLE:
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                first_key = next(iter(temps))
                temp_c = round(temps[first_key][0].current, 1)
        except Exception:
            pass

    if temp_c is None:
        temp_c = 44.5  # Nominal fallback for display

    return {
        "os": f"{platform.system()} ({platform.machine()})",
        "cpu_percent": cpu_pct,
        "ram_used_mb": ram_used_mb,
        "ram_total_mb": ram_total_mb,
        "ram_percent": ram_pct,
        "temperature_c": temp_c,
    }
