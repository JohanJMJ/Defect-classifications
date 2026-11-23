@echo off
REM Automated Test Suite for Windows Laptop
cd /d "%~dp0"

echo ==========================================================
echo Edge QC Inspection Test Suite (Local PC Simulation Mode)
echo ==========================================================

echo.
echo [1/3] Testing TILE (Oil Defect)...
python inference_llama_optimized.py -m models\tile_hailo8.hef -s sample_images\tile\defective_oil.png --llama-backend simulation

echo.
echo [2/3] Testing BOTTLE (Contamination Defect)...
python inference_llama_optimized.py -m models\bottle_hailo8.hef -s sample_images\bottle\defective_contamination.png --llama-backend simulation

echo.
echo [3/3] Testing STEEL DEFECTS (Scratches)...
python inference_llama_optimized.py -m models\steel_defects_hailo8.hef -s sample_images\steel_defects\scratches_3.jpg --llama-backend simulation

echo.
echo ==========================================================
echo All tests completed! Annotated images saved to .\results\
echo ==========================================================
pause
