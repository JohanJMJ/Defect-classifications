/**
 * Industrial Quality Control Assistant - Dashboard Frontend Logic
 * Includes On-Demand Camera Control & Automatic Continuous Capture Engine
 */

let currentSelectedFile = null;
let currentSamplePath = null;
let currentInspectionData = null;
let recentInspections = [];

// Camera State & Continuous Capture Engine
let isCameraActive = false;
let isContinuousRunning = false;
let continuousTimer = null;
let continuousCountdownTimer = null;
let continuousCycleCount = 0;
let continuousIntervalSeconds = 2;
let remainingCountdownMs = 2000;
let isInspectionInProgress = false;

// DOM Elements
const modelSelect = document.getElementById('modelSelect');
const modelDesc = document.getElementById('modelDesc');
const componentIdInput = document.getElementById('componentId');
const btnGenId = document.getElementById('btnGenId');

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const samplesGallery = document.getElementById('samplesGallery');

const btnRunInspect = document.getElementById('btnRunInspect');
const inspectionImg = document.getElementById('inspectionImage');
const emptyPrompt = document.getElementById('emptyPrompt');
const loadingOverlay = document.getElementById('loadingOverlay');
const toggleOverlay = document.getElementById('toggleOverlay');

// Camera Elements
const cameraStreamImg = document.getElementById('cameraStreamImg');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const btnToggleCamera = document.getElementById('btnToggleCamera');
const btnToggleCameraText = document.getElementById('btnToggleCameraText');
const cameraStateDot = document.getElementById('cameraStateDot');
const cameraStateText = document.getElementById('cameraStateText');
const btnCaptureInspect = document.getElementById('btnCaptureInspect');

// Continuous Controls
const btnToggleContinuous = document.getElementById('btnToggleContinuous');
const btnContinuousText = document.getElementById('btnContinuousText');
const continuousPulse = document.getElementById('continuousPulse');
const continuousIntervalSelect = document.getElementById('continuousInterval');
const continuousStatusBar = document.getElementById('continuousStatusBar');
const continuousCountdown = document.getElementById('continuousCountdown');
const continuousCycleCountEl = document.getElementById('continuousCycleCount');

// Verdict & AI Metric Elements
const verdictCard = document.getElementById('verdictCard');
const verdictTitle = document.getElementById('verdictTitle');
const verdictSummary = document.getElementById('verdictSummary');

const valDefectClass = document.getElementById('valDefectClass');
const valConfidence = document.getElementById('valConfidence');
const valDefectArea = document.getElementById('valDefectArea');
const valVisionLatency = document.getElementById('valVisionLatency');
const fillCoverage = document.getElementById('fillCoverage');
const labelCoverage = document.getElementById('labelCoverage');

const valSeverity = document.getElementById('valSeverity');
const valLlamaLatency = document.getElementById('valLlamaLatency');
const valAssessment = document.getElementById('valAssessment');
const valAction = document.getElementById('valAction');

const llamaLoadingBox = document.getElementById('llamaLoadingBox');
const llamaTimer = document.getElementById('llamaTimer');
const advisoryContentBox = document.getElementById('advisoryContentBox');

let llamaPollInterval = null;
let llamaTimerInterval = null;
let activeLlamaInspId = null;

const recentTableBody = document.getElementById('recentTableBody');

// Emergency Audio / Visual Alarm Banner Elements
const emergencyAlarmBanner = document.getElementById('emergencyAlarmBanner');
const alarmCompId = document.getElementById('alarmCompId');
const alarmDefectDesc = document.getElementById('alarmDefectDesc');
const btnAcknowledgeAlarm = document.getElementById('btnAcknowledgeAlarm');
const btnToggleAlarmSound = document.getElementById('btnToggleAlarmSound');
const alarmSoundIcon = document.getElementById('alarmSoundIcon');

// ----------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    generateSerialId();
    setupTabs();
    setupDropzone();
    setupModelSelector();
    setupCameraControls();
    setupAlarmBannerListeners();
    loadSamplesForCategory(modelSelect.value);

    // Run inspect button
    btnRunInspect.addEventListener('click', () => {
        executeInspection();
    });

    // Toggle Raw vs Annotated image
    toggleOverlay.addEventListener('change', () => {
        if (!currentInspectionData) return;
        if (toggleOverlay.checked) {
            inspectionImg.src = currentInspectionData.annotated_path || currentInspectionData.image_path;
        } else {
            inspectionImg.src = currentInspectionData.image_path;
        }
    });

    // Auto ID button
    btnGenId.addEventListener('click', generateSerialId);
});


// ----------------------------------------------------------------------
// Serial ID Generator
// ----------------------------------------------------------------------
function generateSerialId() {
    const cat = modelSelect.value.toUpperCase();
    const rand = Math.floor(100000 + Math.random() * 900000);
    componentIdInput.value = `${cat}-SN-${rand}`;
}


// ----------------------------------------------------------------------
// Tab Management (Auto-starts/stops camera on tab switch)
// ----------------------------------------------------------------------
function setupTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const target = btn.getAttribute('data-tab');
            const targetPane = document.getElementById(target);
            if (targetPane) targetPane.classList.add('active');

            // Handle Camera Lifecycle on Tab Navigation
            if (target === 'tab-camera') {
                // Switching to camera tab: auto-start camera
                startCameraStream();
            } else {
                // Leaving camera tab: stop continuous capture and release camera hardware!
                if (isContinuousRunning) {
                    stopContinuousCapture();
                }
                stopCameraStream();
            }

            // Reset opposing source when switching tabs to avoid confusion
            if (target === 'tab-upload') {
                currentSamplePath = null;
            } else if (target === 'tab-samples') {
                currentSelectedFile = null;
            }
        });
    });
}


// ----------------------------------------------------------------------
// On-Demand Camera Lifecycle Controls
// ----------------------------------------------------------------------
function setupCameraControls() {
    // Manual Toggle Camera button
    btnToggleCamera.addEventListener('click', () => {
        if (isCameraActive) {
            if (isContinuousRunning) stopContinuousCapture();
            stopCameraStream();
        } else {
            startCameraStream();
        }
    });

    // Capture & Inspect button
    btnCaptureInspect.addEventListener('click', () => {
        executeInspection(true, false);
    });

    // Interval selector change
    continuousIntervalSelect.addEventListener('change', () => {
        continuousIntervalSeconds = parseInt(continuousIntervalSelect.value, 10) || 2;
        if (isContinuousRunning) {
            // Restart cycle with new interval
            resetContinuousCountdown();
        }
    });

    // Toggle Continuous Inspection button
    btnToggleContinuous.addEventListener('click', () => {
        if (isContinuousRunning) {
            stopContinuousCapture();
        } else {
            startContinuousCapture();
        }
    });
}

function startCameraStream() {
    if (isCameraActive) return;

    fetch('/api/camera/start', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            isCameraActive = true;
            cameraStreamImg.src = '/video_feed?t=' + Date.now();
            cameraStreamImg.classList.remove('hidden');
            cameraPlaceholder.classList.add('hidden');

            cameraStateDot.style.background = '#10B981';
            cameraStateDot.className = 'status-indicator-dot pulse-green';
            cameraStateText.innerText = 'Camera Hardware: LIVE';
            btnToggleCameraText.innerText = 'Turn Camera OFF';

            btnCaptureInspect.disabled = false;
            btnToggleContinuous.disabled = false;
        })
        .catch(err => {
            console.error('Error starting camera:', err);
        });
}

function stopCameraStream() {
    if (!isCameraActive && cameraPlaceholder.classList.contains('hidden') === false) return;

    isCameraActive = false;
    cameraStreamImg.src = '';
    cameraStreamImg.classList.add('hidden');
    cameraPlaceholder.classList.remove('hidden');

    cameraStateDot.style.background = '#6B7280';
    cameraStateDot.className = 'status-indicator-dot';
    cameraStateText.innerText = 'Camera Hardware: OFF';
    btnToggleCameraText.innerText = 'Turn Camera ON';

    btnCaptureInspect.disabled = true;
    btnToggleContinuous.disabled = true;

    fetch('/api/camera/stop', { method: 'POST' }).catch(() => {});
}


// ----------------------------------------------------------------------
// Continuous Automatic Inspection Engine
// ----------------------------------------------------------------------
function startContinuousCapture() {
    if (!isCameraActive) {
        startCameraStream();
    }

    isContinuousRunning = true;
    continuousCycleCount = 0;
    continuousIntervalSeconds = parseInt(continuousIntervalSelect.value, 10) || 2;

    // UI Updates
    btnContinuousText.innerText = 'Stop Continuous Capture';
    btnToggleContinuous.classList.add('btn-continuous-active');
    continuousPulse.classList.remove('hidden');
    continuousStatusBar.classList.remove('hidden');
    continuousCycleCountEl.innerText = '0';

    // Disable single run buttons to prevent contention
    btnCaptureInspect.disabled = true;

    resetContinuousCountdown();
    runContinuousLoop();
}

function stopContinuousCapture() {
    isContinuousRunning = false;
    if (continuousTimer) clearTimeout(continuousTimer);
    if (continuousCountdownTimer) clearInterval(continuousCountdownTimer);

    btnContinuousText.innerText = 'Start Continuous Capture';
    btnToggleContinuous.classList.remove('btn-continuous-active');
    continuousPulse.classList.add('hidden');
    continuousStatusBar.classList.add('hidden');

    btnCaptureInspect.disabled = !isCameraActive;
}

function resetContinuousCountdown() {
    remainingCountdownMs = continuousIntervalSeconds * 1000;
    updateCountdownUI();

    if (continuousCountdownTimer) clearInterval(continuousCountdownTimer);
    continuousCountdownTimer = setInterval(() => {
        if (!isContinuousRunning) {
            clearInterval(continuousCountdownTimer);
            return;
        }
        remainingCountdownMs = Math.max(0, remainingCountdownMs - 100);
        updateCountdownUI();
    }, 100);
}

function updateCountdownUI() {
    const sec = (remainingCountdownMs / 1000).toFixed(1);
    continuousCountdown.innerText = sec;
}

function runContinuousLoop() {
    if (!isContinuousRunning) return;

    resetContinuousCountdown();

    continuousTimer = setTimeout(() => {
        if (!isContinuousRunning) return;

        if (isInspectionInProgress) {
            // Previous inspection still finishing, re-schedule quickly
            continuousTimer = setTimeout(runContinuousLoop, 200);
            return;
        }

        continuousCycleCount++;
        continuousCycleCountEl.innerText = continuousCycleCount;

        executeInspection(true, true, () => {
            if (isContinuousRunning) {
                runContinuousLoop();
            }
        });
    }, continuousIntervalSeconds * 1000);
}


// ----------------------------------------------------------------------
// Model Selector & Sample Loader
// ----------------------------------------------------------------------
function setupModelSelector() {
    modelSelect.addEventListener('change', () => {
        const cat = modelSelect.value;
        generateSerialId();
        loadSamplesForCategory(cat);
        updateModelBanner(cat);
    });
}

function updateModelBanner(cat) {
    const selectedText = modelSelect.options[modelSelect.selectedIndex].text;
    modelDesc.innerHTML = `Active Model: <strong>${selectedText}</strong>`;
}

function loadSamplesForCategory(category) {
    samplesGallery.innerHTML = '<div class="loading-spinner">Loading sample images...</div>';
    fetch(`/api/samples/${category}`)
        .then(r => r.json())
        .then(data => {
            if (!data.success || !data.samples || data.samples.length === 0) {
                samplesGallery.innerHTML = `<div class="text-muted" style="padding: 10px;">No sample images found for category '${category}'. Use Upload or Camera.</div>`;
                return;
            }

            let html = '';
            data.samples.forEach(s => {
                const tagColor = s.is_defective ? '#EF4444' : '#10B981';
                const tagText = s.is_defective ? 'DEFECT' : 'NORMAL';
                html += `
                    <div class="sample-card" onclick="selectSample('${s.url}', '${category}')">
                        <img src="${s.url}" class="sample-img" alt="${s.name}" loading="lazy">
                        <div class="sample-caption" title="${s.name}">
                            <span style="color: ${tagColor}; font-weight: bold;">[${tagText}]</span> ${s.name}
                        </div>
                    </div>
                `;
            });
            samplesGallery.innerHTML = html;
        })
        .catch(() => {
            samplesGallery.innerHTML = '<div class="text-muted">Error loading samples.</div>';
        });
}

function selectSample(sampleUrl, category) {
    currentSamplePath = sampleUrl;
    currentSelectedFile = null;

    displayPreview(sampleUrl);
    btnRunInspect.disabled = false;

    // Auto-execute inspection on sample click
    executeInspection(false, false);
}


// ----------------------------------------------------------------------
// File Upload & Drag and Drop
// ----------------------------------------------------------------------
function setupDropzone() {
    // dropZone is a <label> wrapping the hidden <input type="file">.
    // Clicking the label naturally opens the file picker — no extra .click() needed.
    // We clear fileInput.value on mousedown (BEFORE the dialog opens) so that
    // selecting the same file a second time still fires the 'change' event.
    dropZone.addEventListener('mousedown', () => { fileInput.value = ''; });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Drag-and-drop onto the box
    ['dragenter', 'dragover'].forEach(ev => {
        dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(ev => {
        dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files && files.length > 0) handleSelectedFile(files[0]);
    });
}

function handleSelectedFile(file) {
    console.log('[Upload] Handling selected file:', file.name, file.size, file.type);
    const isImage = file.type ? file.type.startsWith('image/') : /\.(jpe?g|png|webp|bmp|tiff?)$/i.test(file.name);
    if (!isImage) {
        alert('Please select a valid image file (PNG, JPG, JPEG, BMP).');
        return;
    }
    currentSelectedFile = file;
    currentSamplePath = null;

    // Visual feedback in dropzone
    const mainText = document.getElementById('dropzoneMainText');
    const subText = document.getElementById('dropzoneSubText');
    if (mainText) {
        const sz = Math.round(file.size / 1024);
        mainText.innerHTML = `<span style="color: #10B981; font-weight: 700;">✓ Selected:</span> ${file.name} (${sz} KB)`;
    }
    if (subText) {
        subText.innerText = 'Analyzing with Hailo-8 NPU...';
    }

    // Instant preview using URL.createObjectURL
    try {
        const previewUrl = URL.createObjectURL(file);
        displayPreview(previewUrl);
    } catch (e) {
        const reader = new FileReader();
        reader.onload = (ev) => displayPreview(ev.target.result);
        reader.readAsDataURL(file);
    }

    btnRunInspect.disabled = false;

    // Auto-run inspection immediately on file upload
    executeInspection(false, false);
}

function displayPreview(src) {
    emptyPrompt.classList.add('hidden');
    inspectionImg.src = src;
    inspectionImg.classList.remove('hidden');
}


// ----------------------------------------------------------------------
// Inspection Execution Pipeline
// ----------------------------------------------------------------------
function executeInspection(isCamera = false, isContinuous = false, onComplete = null) {
    if (isInspectionInProgress && !onComplete) return; // Prevent double-fire

    const model = modelSelect.value;
    const componentId = componentIdInput.value.trim() || `COMP-${Date.now()}`;

    const formData = new FormData();
    formData.append('model', model);
    formData.append('component_id', componentId);

    const activeTab = document.querySelector('.tab-btn.active')?.getAttribute('data-tab');

    if (isCamera || activeTab === 'tab-camera') {
        formData.append('source_type', 'camera');
    } else if (currentSelectedFile) {
        // Always use the file if we have one (upload path)
        formData.append('source_type', 'upload');
        formData.append('file', currentSelectedFile, currentSelectedFile.name);
        console.log('[Inspect] Uploading file:', currentSelectedFile.name, currentSelectedFile.size, 'bytes');
    } else if (currentSamplePath) {
        formData.append('source_type', 'sample');
        formData.append('sample_path', currentSamplePath);
    } else {
        alert('Please upload an image, select a sample, or capture a frame.');
        if (onComplete) onComplete();
        return;
    }

    if (isContinuous) {
        formData.append('continuous', '1');
    }

    isInspectionInProgress = true;

    // In continuous mode, do NOT pop up the blocking overlay so the live camera isn't obstructed
    if (!isContinuous) {
        loadingOverlay.classList.remove('hidden');
        btnRunInspect.disabled = true;
    }

    fetch('/api/inspect', {
        method: 'POST',
        body: formData,
    })
    .then(r => r.json())
    .then(data => {
        isInspectionInProgress = false;
        if (!isContinuous) {
            loadingOverlay.classList.add('hidden');
            btnRunInspect.disabled = false;
        }

        if (!data.success) {
            if (!isContinuous) alert('Inspection failed: ' + (data.error || 'Unknown error'));
            if (onComplete) onComplete();
            return;
        }

        currentInspectionData = data.inspection;
        renderInspectionResults(data.inspection);
        generateSerialId(); // Prepare next component serial

        if (onComplete) onComplete();
    })
    .catch(err => {
        isInspectionInProgress = false;
        if (!isContinuous) {
            loadingOverlay.classList.add('hidden');
            btnRunInspect.disabled = false;
            alert('Inspection network error: ' + err.message);
        }
        if (onComplete) onComplete();
    });
}


// ----------------------------------------------------------------------
// Results Rendering
// ----------------------------------------------------------------------
function renderInspectionResults(insp) {
    // 1. Update Viewfinder Image (Annotated)
    emptyPrompt.classList.add('hidden');
    inspectionImg.classList.remove('hidden');

    if (toggleOverlay.checked) {
        inspectionImg.src = insp.annotated_path || insp.image_path;
    } else {
        inspectionImg.src = insp.image_path;
    }

    // 2. Verdict Banner Card
    verdictCard.className = 'verdict-card';
    const verd = (insp.verdict || 'PASS').toUpperCase();
    if (verd === 'PASS') {
        verdictCard.classList.add('verdict-pass');
        verdictTitle.innerText = 'PASS';
        if (insp.severity === 'Minor') {
            verdictSummary.innerText = 'Minor surface variance detected within allowable visual tolerance. Cleared to proceed.';
        } else {
            verdictSummary.innerText = 'Component satisfies quality threshold. No significant defect detected.';
        }
    } else if (verd === 'DIVERT') {
        verdictCard.classList.add('verdict-divert');
        verdictTitle.innerText = 'DIVERT';
        verdictSummary.innerText = 'Moderate anomaly detected. Divert component from main conveyor to buffer/rework lane.';
    } else if (verd === 'INSPECT' || verd === 'FLAGGED_MONITOR' || verd === 'MONITOR') {
        verdictCard.classList.add('verdict-inspect');
        verdictTitle.innerText = 'INSPECT';
        verdictSummary.innerText = 'Major surface defect detected. Hold component for secondary manual inspector sign-off.';
    } else {
        verdictCard.classList.add('verdict-reject');
        verdictTitle.innerText = 'REJECT';
        verdictSummary.innerText = 'Critical structural defect exceeds tolerance. Auto-reject to scrap bin immediately.';
    }

    // 3. Vision Subsystem Metrics
    valDefectClass.innerText = insp.defect_class || 'None';
    valConfidence.innerText = insp.confidence ? (insp.confidence * 100).toFixed(1) + '%' : '0.0%';
    valDefectArea.innerText = insp.mask_area_pct ? insp.mask_area_pct.toFixed(2) + '%' : '0.0%';
    valVisionLatency.innerText = (insp.latency_vision_ms || 11.8) + ' ms';

    const coverage = Math.min(100, (insp.mask_area_pct || 0) * 5); // visual scaling
    fillCoverage.style.width = coverage + '%';
    labelCoverage.innerText = (insp.mask_area_pct || 0).toFixed(2) + '%';

    // 4. Llama Advisory (Decoupled Asynchronous Execution)
    if (insp.llama_status === 'pending' && insp.id) {
        startLlamaPolling(insp.id, insp.rule_action);
    } else {
        if (llamaLoadingBox) llamaLoadingBox.classList.add('hidden');
        valSeverity.className = 'severity-badge';
        const sevName = insp.severity || 'None';
        valSeverity.classList.add(`badge-${sevName.toLowerCase()}`);
        valSeverity.innerText = sevName;
        valLlamaLatency.innerText = `Llama: ${insp.latency_llama_ms || 25.0} ms`;

        // Parse structured Llama text
        const text = insp.llama_advisory || insp.rule_action || '';
        parseAndDisplayLlama(text, insp.rule_action);
    }

    // 5. Flash Hardware GPIO simulation indicator
    flashGpioIndicator(insp.verdict);

    // 6. Emergency Audio-Visual Alarm Banner (Only for Critical and Major!)
    const sName = (insp.severity || '').toLowerCase();
    const isMajorOrCritical = (sName === 'critical' || sName === 'major' || verd === 'REJECT' || insp.trigger_banner);
    if (isMajorOrCritical) {
        triggerSevereDefectAlarm(insp);
    } else {
        silenceSevereDefectAlarm();
    }

    // 7. Prepend to Recent Table
    prependRecentInspection(insp);
}

function startLlamaPolling(inspId, fallbackAction) {
    // Clear any previous running polling
    if (llamaPollInterval) clearInterval(llamaPollInterval);
    if (llamaTimerInterval) clearInterval(llamaTimerInterval);

    activeLlamaInspId = inspId;
    if (llamaLoadingBox) llamaLoadingBox.classList.remove('hidden');
    valLlamaLatency.innerText = 'Llama: Analyzing...';
    valAssessment.innerText = 'Evaluating surface morphology and defect density on Pi 5 ARM CPU...';
    valAction.innerText = 'Synthesizing actionable engineering disposition...';

    const startTime = Date.now();
    llamaTimerInterval = setInterval(() => {
        if (llamaTimer) {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            llamaTimer.innerText = `${elapsed}s`;
        }
    }, 100);

    llamaPollInterval = setInterval(() => {
        fetch(`/api/advisory/${inspId}`)
            .then(r => r.json())
            .then(data => {
                if (data.success && data.status === 'completed') {
                    clearInterval(llamaPollInterval);
                    clearInterval(llamaTimerInterval);
                    llamaPollInterval = null;
                    llamaTimerInterval = null;

                    if (llamaLoadingBox) llamaLoadingBox.classList.add('hidden');
                    valLlamaLatency.innerText = `Llama: ${data.latency_llama_ms || 18000} ms`;

                    if (data.severity) {
                        valSeverity.className = 'severity-badge';
                        valSeverity.classList.add(`badge-${data.severity.toLowerCase()}`);
                        valSeverity.innerText = data.severity;
                    }

                    parseAndDisplayLlama(data.llama_advisory, fallbackAction);

                    // Update in current inspection object
                    if (currentInspectionData && currentInspectionData.id === inspId) {
                        currentInspectionData.llama_advisory = data.llama_advisory;
                        currentInspectionData.latency_llama_ms = data.latency_llama_ms;
                        currentInspectionData.severity = data.severity || currentInspectionData.severity;
                    }
                }
            })
            .catch(() => {});
    }, 1200);
}

function parseAndDisplayLlama(llamaText, fallbackAction) {
    let assessment = "Surface analyzed against tolerance parameters.";
    let action = fallbackAction || "Proceed with standard flow.";

    const lines = llamaText.split('\n');
    lines.forEach(l => {
        const clean = l.replace(/^[•\-\*]\s*/, '').trim();
        if (clean.toLowerCase().startsWith('assessment:')) {
            assessment = clean.substring(11).trim();
        } else if (clean.toLowerCase().startsWith('action:')) {
            action = clean.substring(7).trim();
        } else if (clean.toLowerCase().startsWith('severity:')) {
            const sev = clean.substring(9).trim();
            const sevUpper = sev.toUpperCase();
            if (['NONE', 'MINOR', 'MODERATE', 'MAJOR', 'CRITICAL', 'LOW', 'MEDIUM', 'HIGH'].includes(sevUpper)) {
                valSeverity.className = 'severity-badge badge-' + sev.toLowerCase();
                valSeverity.innerText = sev;
            }
        }
    });

    valAssessment.innerText = assessment;
    valAction.innerText = action;
}

function flashGpioIndicator(verdict) {
    const g = document.getElementById('gpioGreen');
    const y = document.getElementById('gpioYellow');
    const r = document.getElementById('gpioRed');
    const b = document.getElementById('gpioBuzzer');

    [g, y, r, b].forEach(el => el.classList.remove('active'));

    const v = (verdict || 'PASS').toUpperCase();
    if (v === 'PASS') {
        g.classList.add('active');
        setTimeout(() => g.classList.remove('active'), 1500);
    } else if (v === 'DIVERT') {
        y.classList.add('active');
        setTimeout(() => y.classList.remove('active'), 1800);
    } else if (v === 'INSPECT' || v === 'FLAGGED_MONITOR') {
        y.classList.add('active');
        b.classList.add('active');
        setTimeout(() => {
            y.classList.remove('active');
            b.classList.remove('active');
        }, 1200);
    } else if (v === 'REJECT') {
        r.classList.add('active');
        b.classList.add('active');
        setTimeout(() => {
            r.classList.remove('active');
            b.classList.remove('active');
        }, 2000);
    }
}

function prependRecentInspection(insp) {
    recentInspections.unshift(insp);
    if (recentInspections.length > 8) recentInspections.pop();

    let html = '';
    recentInspections.forEach(r => {
        const v = (r.verdict || 'PASS').toUpperCase();
        let badgeClass = 'badge-pass';
        if (v === 'REJECT') badgeClass = 'badge-reject';
        else if (v === 'DIVERT') badgeClass = 'badge-divert';
        else if (v === 'INSPECT' || v === 'FLAGGED_MONITOR') badgeClass = 'badge-inspect';
        else badgeClass = 'badge-pass';

        const conf = r.confidence ? (r.confidence * 100).toFixed(1) + '%' : '--';
        const area = r.mask_area_pct ? r.mask_area_pct.toFixed(2) + '%' : '0.0%';
        const sevName = r.severity || 'None';
        const sevClass = 'badge-' + sevName.toLowerCase();

        html += `
            <tr>
                <td><strong>#${r.id}</strong></td>
                <td class="text-muted" style="font-size: 0.8rem;">${r.timestamp.split(' ')[1] || r.timestamp}</td>
                <td><code>${r.component_id}</code></td>
                <td><span class="category-tag">${r.model_category}</span></td>
                <td>${r.defect_class || 'None'}</td>
                <td>${area}</td>
                <td>${conf}</td>
                <td><span class="status-badge ${badgeClass}">${v}</span></td>
                <td><span class="severity-pill ${sevClass}">${sevName}</span></td>
                <td style="font-size: 0.8rem;">${r.latency_vision_ms}ms</td>
            </tr>
        `;
    });
    recentTableBody.innerHTML = html;
}

// ----------------------------------------------------------------------
// Audio Alarm & Emergency Banner Subsystem (Web Audio API)
// ----------------------------------------------------------------------
let audioCtx = null;
let alarmOscillatorInterval = null;
let isAlarmActive = false;
let isAlarmMuted = localStorage.getItem('dashboard_alarm_muted') === 'true';

function initAudioContext() {
    if (!audioCtx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) audioCtx = new AudioContextClass();
    }
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
}

function playIndustrialAlarmBeep(freq = 880, duration = 0.18) {
    if (isAlarmMuted || !audioCtx) return;
    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sawtooth'; // Industrial warning buzzer tone
        osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
        gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        console.warn('[AudioAlarm] Error playing beep:', e);
    }
}

function startAlarmAudio() {
    initAudioContext();
    if (alarmOscillatorInterval) clearInterval(alarmOscillatorInterval);
    if (isAlarmMuted) return;

    let highTone = true;
    playIndustrialAlarmBeep(880, 0.20);
    alarmOscillatorInterval = setInterval(() => {
        if (!isAlarmActive || isAlarmMuted) {
            clearInterval(alarmOscillatorInterval);
            alarmOscillatorInterval = null;
            return;
        }
        highTone = !highTone;
        playIndustrialAlarmBeep(highTone ? 880 : 660, 0.20);
    }, 380);
}

function stopAlarmAudio() {
    if (alarmOscillatorInterval) {
        clearInterval(alarmOscillatorInterval);
        alarmOscillatorInterval = null;
    }
}

function triggerSevereDefectAlarm(insp) {
    isAlarmActive = true;
    if (alarmCompId) alarmCompId.innerText = insp.component_id || 'UNKNOWN';
    if (alarmDefectDesc) {
        const defName = (insp.defect_class || 'Surface Anomaly').replace('_', ' ').toUpperCase();
        const areaStr = insp.mask_area_pct ? `(${insp.mask_area_pct.toFixed(1)}% area)` : '';
        alarmDefectDesc.innerText = `${defName} ${areaStr} exceeds tolerance limit. Divert component immediately.`;
    }

    if (emergencyAlarmBanner) {
        emergencyAlarmBanner.classList.remove('hidden');
    }
    startAlarmAudio();
}

function silenceSevereDefectAlarm() {
    isAlarmActive = false;
    stopAlarmAudio();
    if (emergencyAlarmBanner) {
        emergencyAlarmBanner.classList.add('hidden');
    }
}

function setupAlarmBannerListeners() {
    if (btnAcknowledgeAlarm) {
        btnAcknowledgeAlarm.addEventListener('click', () => {
            silenceSevereDefectAlarm();
        });
    }

    if (btnToggleAlarmSound) {
        updateAlarmSoundIcon();
        btnToggleAlarmSound.addEventListener('click', () => {
            isAlarmMuted = !isAlarmMuted;
            localStorage.setItem('dashboard_alarm_muted', isAlarmMuted ? 'true' : 'false');
            updateAlarmSoundIcon();
            if (isAlarmMuted) {
                stopAlarmAudio();
            } else if (isAlarmActive) {
                startAlarmAudio();
            }
        });
    }
}

function updateAlarmSoundIcon() {
    if (alarmSoundIcon) {
        alarmSoundIcon.innerText = isAlarmMuted ? '🔇' : '🔊';
    }
    if (btnToggleAlarmSound) {
        btnToggleAlarmSound.title = isAlarmMuted ? 'Unmute Alarm Sound' : 'Mute Alarm Sound';
    }
}

// Global test alert helper
window.testAlert = function(verdict) {
    flashGpioIndicator(verdict);
    if (verdict === 'REJECT') {
        triggerSevereDefectAlarm({
            component_id: 'SIMULATED-CRITICAL',
            defect_class: 'Critical Fracture',
            mask_area_pct: 14.8,
            severity: 'Critical',
            verdict: 'REJECT'
        });
    } else if (verdict === 'INSPECT') {
        triggerSevereDefectAlarm({
            component_id: 'SIMULATED-MAJOR',
            defect_class: 'Major Crack',
            mask_area_pct: 6.2,
            severity: 'Major',
            verdict: 'INSPECT'
        });
    } else if (verdict === 'PASS' || verdict === 'DIVERT') {
        silenceSevereDefectAlarm();
    }
    fetch('/api/alerts/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ verdict: verdict }),
    }).catch(() => {});
};
