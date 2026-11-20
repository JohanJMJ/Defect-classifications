# Automated Test Suite for Windows PowerShell
Set-Location -Path $PSScriptRoot

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Edge QC Inspection Test Suite (Local PC Simulation Mode)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Testing TILE (Oil Defect)..." -ForegroundColor Yellow
python inference_llama_optimized.py -m models\tile_hailo8.hef -s sample_images\tile\defective_oil.png --llama-backend simulation

Write-Host "`n[2/3] Testing BOTTLE (Contamination Defect)..." -ForegroundColor Yellow
python inference_llama_optimized.py -m models\bottle_hailo8.hef -s sample_images\bottle\defective_contamination.png --llama-backend simulation

Write-Host "`n[3/3] Testing STEEL DEFECTS (Scratches)..." -ForegroundColor Yellow
python inference_llama_optimized.py -m models\steel_defects_hailo8.hef -s sample_images\steel_defects\scratches_3.jpg --llama-backend simulation

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "All tests completed! Annotated images saved to .\results\" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
