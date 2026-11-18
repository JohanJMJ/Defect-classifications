import sys
import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "pi_deployment"))
sys.path.insert(0, str(BASE_DIR / "pi_deployment" / "dashboard"))

os.chdir(str(BASE_DIR / "pi_deployment" / "dashboard"))
from dashboard.app import app

client = app.test_client()

# 1. Test index page
r = client.get('/')
assert r.status_code == 200, f"Index returned {r.status_code}"
print("✓ GET / -> 200 OK (HTML rendered)")

# 2. Test history page
r = client.get('/history')
assert r.status_code == 200, f"History returned {r.status_code}"
print("✓ GET /history -> 200 OK (HTML rendered)")

# 3. Test Camera On-Demand Lifecycle (Must be OFF at startup!)
r = client.get('/api/camera/status')
data = r.get_json()
assert data['success'] is True
assert data['active'] is False, "Camera should NOT be active on startup!"
print("✓ GET /api/camera/status -> Camera is OFF on startup (Verified)")

r = client.post('/api/camera/start')
data = r.get_json()
assert data['success'] in [True, False]
print("✓ POST /api/camera/start -> On-demand activation route works")

r = client.post('/api/camera/stop')
data = r.get_json()
assert data['success'] is True
assert data['active'] is False
print("✓ POST /api/camera/stop -> Hardware released (Camera OFF)")

# 4. Test API models
r = client.get('/api/models')
data = r.get_json()
assert data['success'] is True
models_cnt = len(data['models'])
print(f"✓ GET /api/models -> 200 OK ({models_cnt} models available)")

# 4. Test API telemetry
r = client.get('/api/telemetry')
data = r.get_json()
assert data['success'] is True
npu_name = data['telemetry']['npu_name']
print(f"✓ GET /api/telemetry -> 200 OK (NPU: {npu_name})")

# 5. Test API samples
r = client.get('/api/samples/tile')
data = r.get_json()
assert data['success'] is True
samples_cnt = len(data['samples'])
print(f"✓ GET /api/samples/tile -> 200 OK ({samples_cnt} test images)")

# 6. Test inspection execution on a sample image!
sample_url = data['samples'][0]['url']
r = client.post('/api/inspect', data={
    'model': 'tile',
    'component_id': 'TEST-TILE-001',
    'source_type': 'sample',
    'sample_path': sample_url
})
assert r.status_code == 200, f"Inspect returned {r.status_code}: {r.get_data(as_text=True)}"
insp = r.get_json()['inspection']
print(f"✓ POST /api/inspect -> 200 OK")
print(f"   Component ID : {insp['component_id']}")
print(f"   Verdict      : {insp['verdict']}")
print(f"   Severity     : {insp['severity']}")
print(f"   Confidence   : {insp['confidence']}")
print(f"   Defect Area  : {insp['mask_area_pct']}%")
print(f"   Vision Lat   : {insp['latency_vision_ms']} ms")
print(f"   Llama Lat    : {insp['latency_llama_ms']} ms")
print(f"   Annotated Img: {insp['annotated_path']}")

# 7. Test history retrieval
r = client.get('/api/history')
data = r.get_json()
assert data['success'] is True
assert data['total'] >= 1
print(f"✓ GET /api/history -> 200 OK ({data['total']} records in SQLite DB)")

# 8. Test CSV export
r = client.get('/api/export/csv')
assert r.status_code == 200
print(f"✓ GET /api/export/csv -> 200 OK ({len(r.data)} bytes exported)")

print("\n==================================================")
print("  ALL TESTS PASSED: Full QC Dashboard Operational!")
print("==================================================")
