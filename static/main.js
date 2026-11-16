document.addEventListener('DOMContentLoaded', () => {
    // Medical compatibility dictionary for blood groups
    const MEDICAL_DATA = {
        'A+': { donate: 'A+, AB+', receive: 'A+, A-, O+, O-', rh: 'Positive (+)', rarity: '~35.7% (Common)' },
        'A-': { donate: 'A+, A-, AB+, AB-', receive: 'A-, O-', rh: 'Negative (-)', rarity: '~6.3% (Uncommon)' },
        'B+': { donate: 'B+, AB+', receive: 'B+, B-, O+, O-', rh: 'Positive (+)', rarity: '~8.5% (Moderate)' },
        'B-': { donate: 'B+, B-, AB+, AB-', receive: 'B-, O-', rh: 'Negative (-)', rarity: '~1.5% (Rare)' },
        'AB+': { donate: 'AB+ Only (Universal Recipient)', receive: 'All Groups (Universal Recipient)', rh: 'Positive (+)', rarity: '~3.4% (Rare)' },
        'AB-': { donate: 'AB+, AB-', receive: 'AB-, A-, B-, O-', rh: 'Negative (-)', rarity: '~0.6% (Very Rare)' },
        'O+': { donate: 'O+, A+, B+, AB+', receive: 'O+, O-', rh: 'Positive (+)', rarity: '~37.4% (Most Common)' },
        'O-': { donate: 'All Groups (Universal Donor)', receive: 'O- Only', rh: 'Negative (-)', rarity: '~6.6% (High Demand)' }
    };

    // State Variables
    let selectedFile = null;
    let selectedSamplePath = null;
    let scannedImageBase64 = null;
    let currentInputMode = 'upload'; // 'upload' or 'scanner'
    let currentFilter = 'normal';
    let selectedModelType = 'vgg16_hybrid';

    // Elements
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const modelBtns = document.querySelectorAll('.model-btn');

    // Input Mode Elements
    const modeUploadBtn = document.getElementById('mode-upload-btn');
    const modeScannerBtn = document.getElementById('mode-scanner-btn');
    const scannerTabDot = document.getElementById('scanner-tab-dot');
    const scannerPanel = document.getElementById('scanner-panel');
    const samplesSection = document.querySelector('.samples-section');

    // Scanner UI Elements
    const scannerSelect = document.getElementById('scanner-select');
    const refreshScannersBtn = document.getElementById('refresh-scanners-btn');
    const telemetryDeviceName = document.getElementById('telemetry-device-name');
    const telemetryStatusBadge = document.getElementById('telemetry-status-badge');
    const telemetrySlotBadge = document.getElementById('telemetry-slot-badge');
    const telVendor = document.getElementById('tel-vendor');
    const telVid = document.getElementById('tel-vid');
    const telPid = document.getElementById('tel-pid');
    const telConnection = document.getElementById('tel-connection');
    const telSdk = document.getElementById('tel-sdk');
    const telClass = document.getElementById('tel-class');
    const scannerNoticeBox = document.getElementById('scanner-notice-box');
    const noticeTitle = document.getElementById('notice-title');
    const noticeDesc = document.getElementById('notice-desc');
    const triggerScanBtn = document.getElementById('trigger-scan-btn');
    const cancelScanBtn = document.getElementById('cancel-scan-btn');
    const scannerLiveStatus = document.getElementById('scanner-live-status');
    const scannerLiveStatusText = document.getElementById('scanner-live-status-text');
    const scannerCaptureReview = document.getElementById('scanner-capture-review');
    const scannerReviewImg = document.getElementById('scanner-review-img');
    const captureQualityBadge = document.getElementById('capture-quality-badge');
    const captureDimBadge = document.getElementById('capture-dim-badge');
    const captureSourceVal = document.getElementById('capture-source-val');
    const captureClarityVal = document.getElementById('capture-clarity-val');
    const captureTimeVal = document.getElementById('capture-time-val');
    const useFingerprintBtn = document.getElementById('use-fingerprint-btn');
    const retakeScanBtn = document.getElementById('retake-scan-btn');
    let lastCapturedScanData = null;

    const usbInspectorToggle = document.getElementById('usb-inspector-toggle');
    const usbInspectorContent = document.getElementById('usb-inspector-content');
    const usbInspectorIcon = document.getElementById('usb-inspector-icon');
    const usbTotalCount = document.getElementById('usb-total-count');
    const usbDeviceList = document.getElementById('usb-device-list');

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const dropzoneInner = document.getElementById('dropzone-inner');
    const previewArea = document.getElementById('preview-area');
    const previewImg = document.getElementById('preview-img');
    const scannerLaser = document.getElementById('scanner-laser');
    const imgNameDisplay = document.getElementById('img-name-display');
    const clearBtn = document.getElementById('clear-btn');

    const predictBtn = document.getElementById('predict-btn');
    const compareBtn = document.getElementById('compare-btn');
    const filterBtns = document.querySelectorAll('.filter-btn');

    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultError = document.getElementById('result-error');
    const resultErrorMsg = document.getElementById('result-error-msg');
    const resultLoading = document.getElementById('result-loading');
    const resultDisplay = document.getElementById('result-display');
    const loadingStatusText = document.getElementById('loading-status-text');

    const comparisonSection = document.getElementById('comparison-section');
    const modelComparisonGrid = document.getElementById('model-comparison-grid');
    const consensusVal = document.getElementById('consensus-val');

    const predictedGroup = document.getElementById('predicted-group');
    const confidenceScore = document.getElementById('confidence-score');
    const confidenceBarFill = document.getElementById('confidence-bar-fill');
    const probGrid = document.getElementById('prob-grid');

    const donateToVal = document.getElementById('donate-to-val');
    const receiveFromVal = document.getElementById('receive-from-val');
    const rhFactorVal = document.getElementById('rh-factor-val');
    const rarityVal = document.getElementById('rarity-val');

    const samplesGrid = document.getElementById('samples-grid');
    const sampleCount = document.getElementById('sample-count');

    const modelsGrid = document.getElementById('models-grid');
    const graphModal = document.getElementById('graph-modal');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    const closeModalBtn = document.getElementById('close-modal-btn');

    // 1. Navigation Tabs
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(t => t.classList.remove('active'));

            btn.classList.add('active');
            const targetTab = btn.getAttribute('data-tab');
            document.getElementById(targetTab).classList.add('active');
        });
    });

    // 2. Drag and Drop File Input
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        selectedFile = file;
        selectedSamplePath = null;
        scannedImageBase64 = null;
        imgNameDisplay.textContent = file.name;

        // Deselect sample items
        document.querySelectorAll('.sample-item').forEach(s => s.classList.remove('selected'));

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            showPreview();
        };
        reader.readAsDataURL(file);
    }

    function showPreview() {
        if (dropzone) dropzone.classList.remove('hidden');
        dropzoneInner.classList.add('hidden');
        previewArea.classList.remove('hidden');
        if (scannerPanel) scannerPanel.classList.add('hidden');
        if (predictBtn) predictBtn.disabled = false;
        if (compareBtn) compareBtn.disabled = false;
        applyFilter(currentFilter);
    }

    clearBtn.addEventListener('click', () => {
        selectedFile = null;
        selectedSamplePath = null;
        scannedImageBase64 = null;
        fileInput.value = '';
        previewImg.src = '';
        dropzoneInner.classList.remove('hidden');
        previewArea.classList.add('hidden');
        if (predictBtn) predictBtn.disabled = true;
        if (compareBtn) compareBtn.disabled = true;
        
        // Reset results UI
        resultPlaceholder.classList.remove('hidden');
        resultLoading.classList.add('hidden');
        resultDisplay.classList.add('hidden');

        // Reset Explainability Lab UI
        resetLabUI();

        // If in scanner mode, show scanner panel and hide dropzone
        if (currentInputMode === 'scanner') {
            if (scannerPanel) scannerPanel.classList.remove('hidden');
            if (dropzone) dropzone.classList.add('hidden');
        }
    });

    // 3. Filters & Model selection
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.classList.contains('model-btn')) return;
            filterBtns.forEach(b => {
                if (!b.classList.contains('model-btn')) b.classList.remove('active');
            });
            btn.classList.add('active');
            currentFilter = btn.getAttribute('data-filter');
            applyFilter(currentFilter);
        });
    });

    const headerStatusText = document.getElementById('header-status-text');
    const modelLabels = {
        'vgg16_hybrid': 'Hybrid VGG16-SVM Active',
        'hybrid_svm': 'Hybrid ResNet-SVM Active',
        'resnet': 'ResNet-34 Active',
        'svm': 'SVM (HOG) Active',
        'compare_all': 'Cross-Model Comparison Active'
    };

    modelBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            modelBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedModelType = btn.getAttribute('data-model');
            if (headerStatusText && modelLabels[selectedModelType]) {
                headerStatusText.textContent = modelLabels[selectedModelType];
            }
        });
    });

    function applyFilter(filter) {
        if (!previewImg) return;
        switch (filter) {
            case 'contrast':
                previewImg.style.filter = 'contrast(200%) brightness(110%)';
                break;
            case 'grayscale':
                previewImg.style.filter = 'grayscale(100%) contrast(150%)';
                break;
            case 'invert':
                previewImg.style.filter = 'invert(100%) grayscale(100%)';
                break;
            default:
                previewImg.style.filter = 'none';
                break;
        }
    }

    // 4. Fetch Sample Images
    fetchSamples();
    function fetchSamples() {
        fetch('/api/samples')
            .then(res => res.json())
            .then(data => {
                const samples = data.samples || [];
                sampleCount.textContent = `${samples.length} items`;
                samplesGrid.innerHTML = '';

                samples.forEach((sample, idx) => {
                    const item = document.createElement('div');
                    item.className = 'sample-item';
                    item.title = sample.name;
                    
                    const imgUrl = `/image-file?path=${encodeURIComponent(sample.path)}`;
                    item.innerHTML = `
                        <img src="${imgUrl}" alt="${sample.name}" loading="lazy">
                        <span class="sample-tag">${sample.label}</span>
                    `;

                    item.addEventListener('click', () => {
                        document.querySelectorAll('.sample-item').forEach(s => s.classList.remove('selected'));
                        item.classList.add('selected');
                        
                        selectedFile = null;
                        selectedSamplePath = sample.path;
                        scannedImageBase64 = null;
                        imgNameDisplay.textContent = sample.name;
                        previewImg.src = imgUrl;
                        showPreview();
                    });

                    samplesGrid.appendChild(item);
                });
            })
            .catch(err => {
                console.error("Error fetching samples:", err);
                sampleCount.textContent = 'Error loading';
            });
    }

    // 5. Prediction Execution
    if (predictBtn) {
        predictBtn.addEventListener('click', () => {
            executeInference(selectedModelType);
        });
    }

    if (compareBtn) {
        compareBtn.addEventListener('click', () => {
            executeInference('compare_all');
        });
    }

    function executeInference(modelTypeToRun) {
        if (!selectedFile && !selectedSamplePath && !scannedImageBase64) return;

        // UI Loading State
        resultPlaceholder.classList.add('hidden');
        if (resultError) resultError.classList.add('hidden');
        resultDisplay.classList.add('hidden');
        resultLoading.classList.remove('hidden');
        scannerLaser.classList.remove('hidden');
        if (predictBtn) predictBtn.disabled = true;
        if (compareBtn) compareBtn.disabled = true;

        // Animate AI Pipeline stages progression
        animatePipelineProgress(true);

        if (loadingStatusText) {
            if (modelTypeToRun === 'compare_all') {
                loadingStatusText.textContent = "Running Cross-Model Evaluation Across All 4 Architectures...";
            } else {
                loadingStatusText.textContent = "Extracting Ridge Biometrics & Deep Features...";
            }
        }

        const formData = new FormData();
        let fetchConfig = {};

        if (scannedImageBase64) {
            fetchConfig = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_base64: scannedImageBase64, model_type: modelTypeToRun })
            };
        } else if (selectedFile) {
            formData.append('file', selectedFile);
            formData.append('model_type', modelTypeToRun);
            fetchConfig = { method: 'POST', body: formData };
        } else if (selectedSamplePath) {
            fetchConfig = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sample_path: selectedSamplePath, model_type: modelTypeToRun })
            };
        }

        setTimeout(() => {
            fetch('/api/predict', fetchConfig)
                .then(res => res.json())
                .then(res => {
                    scannerLaser.classList.add('hidden');
                    resultLoading.classList.add('hidden');
                    if (predictBtn) predictBtn.disabled = false;
                    if (compareBtn) compareBtn.disabled = false;

                    if (res.success) {
                        animatePipelineProgress(false, true);
                        if (resultError) resultError.classList.add('hidden');
                        displayResults(res.data);
                    } else {
                        animatePipelineProgress(false, false);
                        const errMsg = res.error || 'Unable to classify this fingerprint. Please provide a clear fingerprint image.';
                        if (resultError) {
                            if (resultErrorMsg) resultErrorMsg.textContent = errMsg;
                            resultError.classList.remove('hidden');
                            resultPlaceholder.classList.add('hidden');
                        } else {
                            alert('Classification notice: ' + errMsg);
                            resultPlaceholder.classList.remove('hidden');
                        }
                    }
                })
                .catch(err => {
                    scannerLaser.classList.add('hidden');
                    resultLoading.classList.add('hidden');
                    animatePipelineProgress(false, false);
                    if (predictBtn) predictBtn.disabled = false;
                    if (compareBtn) compareBtn.disabled = false;
                    resultPlaceholder.classList.remove('hidden');
                    alert('Server communication error.');
                });
        }, 1100);
    }

    function displayResults(data) {
        resultDisplay.classList.remove('hidden');

        const resultSubtitle = document.getElementById('result-subtitle');
        if (resultSubtitle && data.model_name) {
            resultSubtitle.textContent = `${data.model_name}`;
        }

        // Show/Hide "from user correction" badge
        const userCorrectionTag = document.getElementById('user-correction-tag');
        if (userCorrectionTag) {
            if (data.source === 'from user correction' || data.is_user_correction) {
                userCorrectionTag.classList.remove('hidden');
                if (resultSubtitle) {
                    resultSubtitle.textContent = 'from user correction (Ground Truth Override)';
                }
            } else {
                userCorrectionTag.classList.add('hidden');
            }
        }

        // Reset Feedback Component State for new prediction
        window._currentPredictedBloodGroup = data.prediction;
        const feedbackPromptZone = document.getElementById('feedback-prompt-zone');
        const feedbackCorrectionZone = document.getElementById('feedback-correction-zone');
        const feedbackResponseMsg = document.getElementById('feedback-response-msg');
        const feedbackYesBtn = document.getElementById('feedback-yes-btn');
        const feedbackNoBtn = document.getElementById('feedback-no-btn');
        const correctBloodGroupSelect = document.getElementById('correct-blood-group-select');

        if (feedbackPromptZone) feedbackPromptZone.classList.remove('hidden');
        if (feedbackCorrectionZone) feedbackCorrectionZone.classList.add('hidden');
        if (feedbackResponseMsg) {
            feedbackResponseMsg.classList.add('hidden');
            feedbackResponseMsg.textContent = '';
        }
        if (feedbackYesBtn) feedbackYesBtn.disabled = false;
        if (feedbackNoBtn) feedbackNoBtn.disabled = false;
        if (correctBloodGroupSelect) correctBloodGroupSelect.selectedIndex = 0;

        // Render Cross-Model Comparison Grid if present
        if (data.models_comparison && data.models_comparison.length > 0) {
            comparisonSection.classList.remove('hidden');
            if (consensusVal) {
                consensusVal.innerHTML = `${data.consensus_group} &nbsp;·&nbsp; ${data.agreement_count} (${data.agreement_percentage}%)`;
            }
            if (modelComparisonGrid) {
                modelComparisonGrid.innerHTML = '';
                data.models_comparison.forEach(m => {
                    const card = document.createElement('div');
                    card.className = `compare-card ${m.is_recommended ? 'recommended' : ''}`;
                    card.innerHTML = `
                        ${m.is_recommended ? '<span class="compare-card-badge"><i class="fa-solid fa-crown"></i> Recommended</span>' : ''}
                        <div class="compare-card-title">${m.name}</div>
                        <div class="compare-card-type">${m.type}</div>
                        <div class="compare-prediction-row">
                            <span class="compare-group-badge">${m.prediction}</span>
                            <div class="compare-conf-wrap">
                                <div class="compare-conf-val">${m.confidence}%</div>
                                <div class="compare-rank">${m.accuracy_rank}</div>
                            </div>
                        </div>
                    `;
                    modelComparisonGrid.appendChild(card);
                });
            }
        } else {
            comparisonSection.classList.add('hidden');
        }

        predictedGroup.textContent = data.prediction;
        confidenceScore.textContent = `${data.confidence}%`;
        confidenceBarFill.style.width = `${data.confidence}%`;

        // Render Pattern Rationale
        const patternRationale = document.getElementById('pattern-rationale-text');
        if (patternRationale && data.pattern_rationale) {
            patternRationale.textContent = `"${data.pattern_rationale}"`;
        }

        // Render Biometric Features
        if (data.biometrics) {
            const bioDim = document.getElementById('bio-dim-val');
            const bioSpacing = document.getElementById('bio-spacing-val');
            const bioDensity = document.getElementById('bio-density-val');
            const bioOrientation = document.getElementById('bio-orientation-val');
            const bioCoherence = document.getElementById('bio-coherence-val');
            const bioMinutiae = document.getElementById('bio-minutiae-val');
            
            if (bioDim) bioDim.textContent = data.biometrics.dimensions || '-';
            if (bioSpacing) bioSpacing.textContent = `${data.biometrics.ridge_spacing_px || '-'} px`;
            if (bioDensity) bioDensity.textContent = `${data.biometrics.ridge_density_score} (Gradient)`;
            if (bioOrientation) bioOrientation.textContent = `${data.biometrics.dominant_orientation_deg}°`;
            if (bioCoherence) bioCoherence.textContent = `${data.biometrics.orientation_coherence} (Coherence)`;
            if (bioMinutiae) bioMinutiae.textContent = `${data.biometrics.minutiae_density_score} density`;
        }

        // Render Probabilities
        probGrid.innerHTML = '';
        const sortedClasses = Object.keys(data.probabilities).sort((a, b) => data.probabilities[b] - data.probabilities[a]);

        sortedClasses.forEach(cls => {
            const pct = data.probabilities[cls];
            const isTop = cls === data.prediction;

            const probItem = document.createElement('div');
            probItem.className = `prob-item ${isTop ? 'highlight' : ''}`;
            probItem.innerHTML = `
                <div class="prob-header">
                    <span class="prob-name">${cls}</span>
                    <span class="prob-val">${pct}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: ${pct}%"></div>
                </div>
            `;
            probGrid.appendChild(probItem);
        });

        // Render Medical Info
        const med = MEDICAL_DATA[data.prediction] || { donate: 'N/A', receive: 'N/A', rh: 'N/A', rarity: 'N/A' };
        donateToVal.textContent = med.donate;
        receiveFromVal.textContent = med.receive;
        rhFactorVal.textContent = med.rh;
        rarityVal.textContent = med.rarity;

        // Update & Highlight 3D Blood Group Result Cards
        const majorClasses = ['A+', 'B+', 'O+', 'AB+'];
        majorClasses.forEach(cls => {
            const card = document.getElementById(`bg-card-${cls}`);
            const probEl = document.getElementById(`card-prob-${cls}`);
            const pct = (data.probabilities && data.probabilities[cls] !== undefined) ? data.probabilities[cls] : 0.0;
            if (probEl) probEl.textContent = `${pct}%`;

            if (card) {
                card.classList.remove('predicted-match');
                const existingTag = card.querySelector('.predicted-match-tag');
                if (existingTag) existingTag.remove();

                if (cls === data.prediction) {
                    card.classList.add('predicted-match');
                    const matchTag = document.createElement('span');
                    matchTag.className = 'predicted-match-tag';
                    matchTag.innerHTML = `<i class="fa-solid fa-check"></i> PREDICTED MATCH (${data.confidence}%)`;
                    card.appendChild(matchTag);
                }
            }
        });

        // Update Rh-Negative Matrix Chips
        const negClasses = ['A-', 'B-', 'O-', 'AB-'];
        negClasses.forEach(cls => {
            const chip = document.getElementById(`chip-${cls}`);
            const probEl = document.getElementById(`chip-prob-${cls}`);
            const pct = (data.probabilities && data.probabilities[cls] !== undefined) ? data.probabilities[cls] : 0.0;
            if (probEl) probEl.textContent = `${pct}%`;

            if (chip) {
                if (cls === data.prediction) {
                    chip.classList.add('predicted-match');
                } else {
                    chip.classList.remove('predicted-match');
                }
            }
        });

        // Trigger Explainability & Reliability Lab analysis
        triggerLabAnalysis(data);
    }

    // 6. Fetch Benchmarks
    fetchMetrics();
    function fetchMetrics() {
        fetch('/api/metrics')
            .then(res => res.json())
            .then(data => {
                modelsGrid.innerHTML = '';
                (data.models || []).forEach(m => {
                    const isDeployed = m.status.includes('Deployed');
                    const card = document.createElement('div');
                    card.className = `card glass model-card ${isDeployed ? 'featured' : ''}`;

                    card.innerHTML = `
                        <span class="model-badge ${isDeployed ? 'badge-deployed' : 'badge-eval'}">${m.status}</span>
                        <div class="model-title">${m.name}</div>
                        <div class="model-type">${m.type}</div>

                        <div class="metric-stat">
                            <div class="stat-header">
                                <span>Training Accuracy</span>
                                <strong>${m.accuracy}%</strong>
                            </div>
                            <div class="stat-bar">
                                <div class="stat-fill" style="width: ${m.accuracy}%"></div>
                            </div>
                        </div>

                        <div class="metric-stat">
                            <div class="stat-header">
                                <span>Validation Accuracy</span>
                                <strong>${m.val_accuracy}%</strong>
                            </div>
                            <div class="stat-bar">
                                <div class="stat-fill" style="width: ${m.val_accuracy}%"></div>
                            </div>
                        </div>

                        <button class="btn btn-primary view-graphs-btn" data-model="${m.name}">
                            <i class="fa-solid fa-chart-line"></i> View Graphs
                        </button>
                    `;

                    card.querySelector('.view-graphs-btn').addEventListener('click', () => {
                        openGraphModal(m);
                    });

                    modelsGrid.appendChild(card);
                });
            });
    }

    function openGraphModal(model) {
        modalTitle.textContent = `${model.name} Performance Training Graphs`;
        modalBody.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <h5 style="margin-bottom: 5px; color: #294936; font-weight: 700;">Accuracy History</h5>
                    <img src="${model.graphs.accuracy}" style="width: 100%; border-radius: 8px; border: 1.5px solid #888888;" alt="Accuracy">
                </div>
                <div>
                    <h5 style="margin-bottom: 5px; color: #294936; font-weight: 700;">Validation Accuracy</h5>
                    <img src="${model.graphs.val_accuracy}" style="width: 100%; border-radius: 8px; border: 1.5px solid #888888;" alt="Val Accuracy">
                </div>
                <div>
                    <h5 style="margin-bottom: 5px; color: #294936; font-weight: 700;">Loss Curve</h5>
                    <img src="${model.graphs.loss}" style="width: 100%; border-radius: 8px; border: 1.5px solid #888888;" alt="Loss">
                </div>
                <div>
                    <h5 style="margin-bottom: 5px; color: #294936; font-weight: 700;">Validation Loss</h5>
                    <img src="${model.graphs.val_loss}" style="width: 100%; border-radius: 8px; border: 1.5px solid #888888;" alt="Val Loss">
                </div>
            </div>
        `;
        graphModal.classList.remove('hidden');
    }

    closeModalBtn.addEventListener('click', () => {
        graphModal.classList.add('hidden');
    });

    graphModal.addEventListener('click', (e) => {
        if (e.target === graphModal) {
            graphModal.classList.add('hidden');
        }
    });

    // 7. Feedback Interactive Handlers (Was this correct? Yes/No)
    const feedbackYesBtn = document.getElementById('feedback-yes-btn');
    const feedbackNoBtn = document.getElementById('feedback-no-btn');
    const feedbackSubmitBtn = document.getElementById('feedback-submit-btn');
    const feedbackPromptZone = document.getElementById('feedback-prompt-zone');
    const feedbackCorrectionZone = document.getElementById('feedback-correction-zone');
    const feedbackResponseMsg = document.getElementById('feedback-response-msg');
    const correctBloodGroupSelect = document.getElementById('correct-blood-group-select');

    if (feedbackYesBtn) {
        feedbackYesBtn.addEventListener('click', () => {
            if (feedbackYesBtn) feedbackYesBtn.disabled = true;
            if (feedbackNoBtn) feedbackNoBtn.disabled = true;
            if (feedbackCorrectionZone) feedbackCorrectionZone.classList.add('hidden');
            if (feedbackResponseMsg) {
                feedbackResponseMsg.className = 'glass-inner';
                feedbackResponseMsg.style.color = '#294936';
                feedbackResponseMsg.style.background = '#E8F0E8';
                feedbackResponseMsg.style.border = '1.5px solid #888888';
                feedbackResponseMsg.style.display = 'block';
                feedbackResponseMsg.innerHTML = '<i class="fa-solid fa-circle-check"></i> Thank you for confirming prediction accuracy!';
                feedbackResponseMsg.classList.remove('hidden');
            }
        });
    }

    if (feedbackNoBtn) {
        feedbackNoBtn.addEventListener('click', () => {
            if (feedbackCorrectionZone) {
                feedbackCorrectionZone.classList.remove('hidden');
            }
        });
    }

    if (feedbackSubmitBtn) {
        feedbackSubmitBtn.addEventListener('click', () => {
            const selectedLabel = correctBloodGroupSelect ? correctBloodGroupSelect.value : '';
            if (!selectedLabel) {
                alert('Please select the correct blood group from the dropdown.');
                return;
            }

            feedbackSubmitBtn.disabled = true;
            feedbackSubmitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Submitting...';

            const formData = new FormData();
            if (selectedFile) {
                formData.append('image', selectedFile);
            } else if (selectedSamplePath) {
                formData.append('sample_path', selectedSamplePath);
            } else if (previewImg && previewImg.src) {
                formData.append('image_data', previewImg.src);
            }

            const predToSubmit = window._currentPredictedBloodGroup || (predictedGroup ? predictedGroup.textContent : '');
            formData.append('predicted_label', predToSubmit);
            formData.append('correct_label', selectedLabel);

            fetch('/feedback/submit', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(res => {
                feedbackSubmitBtn.disabled = false;
                feedbackSubmitBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Submit Correction';

                if (res.success) {
                    if (feedbackPromptZone) feedbackPromptZone.classList.add('hidden');
                    if (feedbackCorrectionZone) feedbackCorrectionZone.classList.add('hidden');
                    if (feedbackResponseMsg) {
                        feedbackResponseMsg.className = 'glass-inner';
                        feedbackResponseMsg.style.color = '#294936';
                        feedbackResponseMsg.style.background = '#E8F0E8';
                        feedbackResponseMsg.style.border = '1.5px solid #888888';
                        feedbackResponseMsg.style.display = 'block';
                        feedbackResponseMsg.innerHTML = `<i class="fa-solid fa-circle-check"></i> Correction submitted for administrative review! Ground truth: <strong>${selectedLabel}</strong>`;
                        feedbackResponseMsg.classList.remove('hidden');
                    }
                } else {
                    alert('Error submitting feedback: ' + (res.error || 'Unknown error'));
                }
            })
            .catch(err => {
                console.error("Feedback submit error:", err);
                feedbackSubmitBtn.disabled = false;
                feedbackSubmitBtn.innerHTML = '<i class="fa-solid fa-paper-plane"></i> Submit Correction';
                alert('Communication error submitting correction.');
            });
        });
    }

    // 8. Pipeline Progression Animation Control
    let pipelineInterval = null;
    function animatePipelineProgress(isStarting, isSuccess) {
        if (pipelineInterval) {
            clearInterval(pipelineInterval);
            pipelineInterval = null;
        }

        const stages = [1, 2, 3, 4, 5];
        if (isStarting) {
            // Reset all stages to standby
            stages.forEach(s => {
                const el = document.getElementById(`pipeline-stage-${s}`);
                const stat = document.getElementById(`stage-status-${s}`);
                if (el) el.className = 'pipeline-card';
                if (stat) stat.textContent = 'Standby';
            });

            let currentStep = 1;
            const stepStage = () => {
                stages.forEach(s => {
                    const el = document.getElementById(`pipeline-stage-${s}`);
                    const stat = document.getElementById(`stage-status-${s}`);
                    if (!el || !stat) return;

                    if (s < currentStep) {
                        el.className = 'pipeline-card completed';
                        stat.textContent = 'Processed ✓';
                    } else if (s === currentStep) {
                        el.className = 'pipeline-card active';
                        stat.textContent = 'Analyzing...';
                    } else {
                        el.className = 'pipeline-card';
                        stat.textContent = 'Pending';
                    }
                });
                if (currentStep < 4) {
                    currentStep++;
                }
            };

            stepStage();
            pipelineInterval = setInterval(stepStage, 260);
        } else {
            // Complete or reset
            stages.forEach(s => {
                const el = document.getElementById(`pipeline-stage-${s}`);
                const stat = document.getElementById(`stage-status-${s}`);
                if (!el || !stat) return;

                if (isSuccess) {
                    el.className = 'pipeline-card completed';
                    stat.textContent = s === 5 ? 'Classified ✓' : 'Completed ✓';
                } else {
                    el.className = 'pipeline-card';
                    stat.textContent = 'Standby';
                }
            });
        }
    }

    // 9. 3D Card Tilt Micro-Interactions
    function init3DCardTilt() {
        const tiltTargets = document.querySelectorAll('.pipeline-card, .bg-result-card, .arch-node, .model-card');
        tiltTargets.forEach(card => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const cx = rect.width / 2;
                const cy = rect.height / 2;
                const rotX = ((y - cy) / cy) * -5;
                const rotY = ((x - cx) / cx) * 5;
                card.style.transform = `perspective(800px) rotateX(${rotX}deg) rotateY(${rotY}deg) translateY(-4px)`;
            });

            card.addEventListener('mouseleave', () => {
                card.style.transform = '';
            });
        });
    }

    // 10. Scroll Reveal Animations
    function initScrollReveal() {
        if ('IntersectionObserver' in window) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('is-revealed');
                    }
                });
            }, { threshold: 0.08 });

            document.querySelectorAll('.reveal-on-scroll').forEach(el => observer.observe(el));
        } else {
            document.querySelectorAll('.reveal-on-scroll').forEach(el => el.classList.add('is-revealed'));
        }
    }

    // ==============================================================================
    // 11. Fingerprint Reliability & Explainability Lab Controller
    // ==============================================================================
    let currentLabStages = null;

    // Sub-tab switching
    const labTabBtns = document.querySelectorAll('.lab-tab-btn');
    const labPanels = document.querySelectorAll('.lab-panel');
    labTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            labTabBtns.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            labPanels.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            const targetPanelId = btn.getAttribute('data-lab-target');
            const targetPanel = document.getElementById(targetPanelId);
            if (targetPanel) targetPanel.classList.add('active');
        });
    });

    // Toggle view buttons (Raw vs CLAHE vs Tensor)
    const btnToggleRaw = document.getElementById('btn-toggle-raw');
    const btnToggleClahe = document.getElementById('btn-toggle-clahe');
    const btnToggleTensor = document.getElementById('btn-toggle-tensor');
    const toggleComparisonImg = document.getElementById('toggle-comparison-img');
    const toggleEmptyMsg = document.getElementById('toggle-empty-msg');

    function setToggleImage(type) {
        if (!currentLabStages || !currentLabStages.stages) return;
        [btnToggleRaw, btnToggleClahe, btnToggleTensor].forEach(b => b && b.classList.remove('active'));
        let src = '';
        if (type === 'raw') {
            if (btnToggleRaw) btnToggleRaw.classList.add('active');
            src = currentLabStages.stages[0]?.image_b64 || '';
        } else if (type === 'clahe') {
            if (btnToggleClahe) btnToggleClahe.classList.add('active');
            src = currentLabStages.stages[2]?.image_b64 || '';
        } else if (type === 'tensor') {
            if (btnToggleTensor) btnToggleTensor.classList.add('active');
            src = currentLabStages.stages[3]?.image_b64 || '';
        }
        if (src && toggleComparisonImg) {
            toggleComparisonImg.src = src;
            toggleComparisonImg.classList.remove('hidden');
            if (toggleEmptyMsg) toggleEmptyMsg.classList.add('hidden');
        }
    }

    if (btnToggleRaw) btnToggleRaw.addEventListener('click', () => setToggleImage('raw'));
    if (btnToggleClahe) btnToggleClahe.addEventListener('click', () => setToggleImage('clahe'));
    if (btnToggleTensor) btnToggleTensor.addEventListener('click', () => setToggleImage('tensor'));

    // Accordion in Panel F
    const accordionHeaders = document.querySelectorAll('.accordion-header');
    accordionHeaders.forEach(hdr => {
        hdr.addEventListener('click', () => {
            const item = hdr.parentElement;
            const isCurrentlyActive = item.classList.contains('active');
            document.querySelectorAll('.accordion-item').forEach(i => {
                i.classList.remove('active');
                const btn = i.querySelector('.accordion-header');
                if (btn) btn.setAttribute('aria-expanded', 'false');
            });
            if (!isCurrentlyActive) {
                item.classList.add('active');
                hdr.setAttribute('aria-expanded', 'true');
            }
        });
    });

    // Fetch initial held-out test evaluation benchmarks
    fetchLabEvaluation();
    function fetchLabEvaluation() {
        fetch('/api/lab/evaluation')
            .then(res => res.json())
            .then(data => {
                if (!data.success) return;
                const evalData = data.evaluation;
                renderBenchmarksTable(evalData.model_comparisons);
                renderClassReportTable(evalData.per_class_metrics);
            })
            .catch(err => console.log('Note on lab evaluation fetch:', err));
    }

    function renderBenchmarksTable(comparisons) {
        const tbody = document.getElementById('models-benchmark-tbody');
        if (!tbody || !comparisons || comparisons.length === 0) return;
        tbody.innerHTML = '';
        comparisons.forEach(c => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${c.model_name}</strong></td>
                <td><strong style="color: var(--primary-heading);">${c.test_accuracy}%</strong></td>
                <td><code style="background: var(--bg-alt); padding: 2px 6px; border-radius: 4px;">${c.ci_accuracy}</code></td>
                <td>${c.macro_f1}%</td>
                <td>${c.weighted_f1}%</td>
                <td>${c.correct_ratio}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function renderClassReportTable(metrics) {
        const tbody = document.getElementById('class-report-tbody');
        if (!tbody || !metrics || metrics.length === 0) return;
        tbody.innerHTML = '';
        metrics.forEach(m => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><span class="badge-pill" style="font-weight: 700;">${m.class_name}</span></td>
                <td>${m.precision}%</td>
                <td>${m.recall}%</td>
                <td><strong>${m.f1_score}%</strong></td>
                <td>${m.support}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Trigger full lab analysis when prediction completes
    function triggerLabAnalysis(predictionData) {
        if (!selectedFile && !selectedSamplePath && !scannedImageBase64) return;

        let fetchConfig = {};
        if (scannedImageBase64) {
            fetchConfig = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_base64: scannedImageBase64,
                    predicted_class: predictionData.prediction || '',
                    confidence: predictionData.confidence || 0,
                    probabilities: predictionData.probabilities || {}
                })
            };
        } else if (selectedFile) {
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('predicted_class', predictionData.prediction || '');
            formData.append('confidence', predictionData.confidence || 0);
            fetchConfig = { method: 'POST', body: formData };
        } else if (selectedSamplePath) {
            fetchConfig = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sample_path: selectedSamplePath,
                    predicted_class: predictionData.prediction || '',
                    confidence: predictionData.confidence || 0,
                    probabilities: predictionData.probabilities || {}
                })
            };
        }

        fetch('/api/lab/analyze', fetchConfig)
            .then(res => res.json())
            .then(res => {
                if (!res.success) {
                    console.warn("Lab analysis warning:", res.error);
                    return;
                }
                updateLabUI(res);
            })
            .catch(err => {
                console.error("Lab analysis communication error:", err);
            });
    }

    function updateLabUI(res) {
        // 1. Preprocessing Stages
        if (res.preprocessing_stages && res.preprocessing_stages.stages) {
            currentLabStages = res.preprocessing_stages;
            const stages = res.preprocessing_stages.stages;

            const mapStage = [
                { imgId: 'lab-img-raw', phId: 'raw-placeholder', badgeId: 'raw-dim-badge', data: stages[0] },
                { imgId: 'lab-img-gray', phId: 'gray-placeholder', badgeId: 'gray-dim-badge', data: stages[1] },
                { imgId: 'lab-img-clahe', phId: 'clahe-placeholder', badgeId: 'clahe-dim-badge', data: stages[2] },
                { imgId: 'lab-img-tensor', phId: 'tensor-placeholder', badgeId: 'tensor-dim-badge', data: stages[3] }
            ];

            mapStage.forEach(s => {
                const imgEl = document.getElementById(s.imgId);
                const phEl = document.getElementById(s.phId);
                const badgeEl = document.getElementById(s.badgeId);
                if (imgEl && s.data) {
                    imgEl.src = s.data.image_b64;
                    imgEl.classList.remove('hidden');
                    if (phEl) phEl.classList.add('hidden');
                }
                if (badgeEl && s.data) {
                    badgeEl.textContent = s.data.dimensions;
                }
            });

            // Set default comparison view
            setToggleImage('raw');
        }

        // 2. Quality Analysis
        if (res.quality_assessment) {
            const qa = res.quality_assessment;
            const m = qa.metrics;
            const qDim = document.getElementById('qual-dim-val');
            const qAspect = document.getElementById('qual-aspect-val');
            const qSharp = document.getElementById('qual-sharpness-val');
            const qSharpR = document.getElementById('qual-sharpness-rating');
            const qCont = document.getElementById('qual-contrast-val');
            const qMich = document.getElementById('qual-michelson-val');
            const qCoh = document.getElementById('qual-coherence-val');
            const qCohR = document.getElementById('qual-coherence-rating');

            if (qDim) qDim.textContent = qa.dimensions;
            if (qAspect) qAspect.textContent = `Aspect: ${qa.aspect_ratio}`;
            if (qSharp) qSharp.textContent = m.sharpness_variance;
            if (qSharpR) {
                qSharpR.textContent = m.sharpness_rating;
                qSharpR.style.color = m.sharpness_color || '#294936';
            }
            if (qCont) qCont.textContent = m.contrast_std;
            if (qMich) qMich.textContent = `Michelson: ${m.michelson_contrast}`;
            if (qCoh) qCoh.textContent = m.ridge_coherence;
            if (qCohR) qCohR.textContent = m.coherence_rating;

            // Render Warnings
            const warningsList = document.getElementById('quality-warnings-list');
            if (warningsList && qa.warnings) {
                warningsList.innerHTML = '';
                qa.warnings.forEach(w => {
                    const item = document.createElement('div');
                    item.className = `quality-info-item item-${w.type}`;
                    item.innerHTML = `
                        <i class="fa-solid ${w.icon}"></i>
                        <div>
                            <strong>${w.title}</strong>
                            <p>${w.detail}</p>
                        </div>
                    `;
                    warningsList.appendChild(item);
                });
            }
        }

        // 3. Grad-CAM Explainability
        if (res.explainability_visualization) {
            const cam = res.explainability_visualization;
            const camInImg = document.getElementById('cam-input-img');
            const camInPh = document.getElementById('cam-input-placeholder');
            const camOverImg = document.getElementById('cam-overlay-img');
            const camOverPh = document.getElementById('cam-overlay-placeholder');
            const camFocal = document.getElementById('cam-focal-zone');
            const camCoords = document.getElementById('cam-peak-coords');
            const camNarrative = document.getElementById('cam-narrative-body');
            const camTarget = document.getElementById('cam-target-label');

            if (res.preprocessing_stages && res.preprocessing_stages.stages[3] && camInImg) {
                camInImg.src = res.preprocessing_stages.stages[3].image_b64;
                camInImg.classList.remove('hidden');
                if (camInPh) camInPh.classList.add('hidden');
            }

            if (camOverImg && cam.overlay_b64) {
                camOverImg.src = cam.overlay_b64;
                camOverImg.classList.remove('hidden');
                if (camOverPh) camOverPh.classList.add('hidden');
            }

            if (camFocal) camFocal.textContent = cam.focal_zone || 'Core Ridges';
            if (camCoords) camCoords.textContent = cam.peak_activation_coords || '-';
            if (camNarrative && cam.explanation_narrative) camNarrative.textContent = cam.explanation_narrative;
            if (camTarget && cam.target_layer) camTarget.textContent = `Target: ${cam.target_layer}`;
        }

        // 4. Uncertainty & Entropy Analysis
        if (res.uncertainty_analysis) {
            const unc = res.uncertainty_analysis;
            const tierText = document.getElementById('uncertainty-tier-text');
            const tierBadge = document.getElementById('uncertainty-tier-badge');
            const marginVal = document.getElementById('unc-margin-val');
            const marginBar = document.getElementById('unc-margin-bar');
            const entropyVal = document.getElementById('unc-entropy-val');
            const entropyBar = document.getElementById('unc-entropy-bar');
            const alertBox = document.getElementById('unc-alert-box');
            const alertText = document.getElementById('unc-alert-text');
            const rankedList = document.getElementById('ranked-classes-list');

            if (tierText) tierText.textContent = unc.uncertainty_tier;
            if (tierBadge) {
                if (unc.tier_status === 'confident') {
                    tierBadge.style.borderColor = '#294936';
                    tierBadge.style.color = '#294936';
                    tierBadge.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span>${unc.uncertainty_tier}</span>`;
                } else if (unc.tier_status === 'moderate') {
                    tierBadge.style.borderColor = '#D97706';
                    tierBadge.style.color = '#D97706';
                    tierBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <span>${unc.uncertainty_tier}</span>`;
                } else {
                    tierBadge.style.borderColor = '#DC2626';
                    tierBadge.style.color = '#DC2626';
                    tierBadge.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> <span>${unc.uncertainty_tier}</span>`;
                }
            }

            if (marginVal) marginVal.textContent = `+${unc.margin_delta_pct}%`;
            if (marginBar) marginBar.style.width = `${Math.min(unc.margin_delta_pct, 100)}%`;
            if (entropyVal) entropyVal.textContent = `${unc.entropy_normalized} (max 1.0)`;
            if (entropyBar) entropyBar.style.width = `${Math.min(unc.entropy_normalized * 100, 100)}%`;

            if (alertBox) {
                if (unc.is_uncertain && unc.warning_text) {
                    alertBox.classList.remove('hidden');
                    if (alertText) alertText.textContent = unc.warning_text;
                } else {
                    alertBox.classList.add('hidden');
                }
            }

            if (rankedList && unc.ranked_classes) {
                rankedList.innerHTML = '';
                unc.ranked_classes.forEach(rc => {
                    const row = document.createElement('div');
                    row.className = `ranked-class-row ${rc.is_predicted ? 'predicted-highlight' : ''}`;
                    row.innerHTML = `
                        <span class="ranked-class-name">${rc.class_name}</span>
                        <div class="ranked-bar-wrap">
                            <div class="ranked-bar-fill" style="width: ${rc.probability_pct}%;"></div>
                        </div>
                        <span class="ranked-class-val">${rc.probability_pct}%</span>
                    `;
                    rankedList.appendChild(row);
                });
            }
        }
    }

    function resetLabUI() {
        currentLabStages = null;
        ['lab-img-raw', 'lab-img-gray', 'lab-img-clahe', 'lab-img-tensor', 'toggle-comparison-img', 'cam-input-img', 'cam-overlay-img'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.src = '';
                el.classList.add('hidden');
            }
        });
        ['raw-placeholder', 'gray-placeholder', 'clahe-placeholder', 'tensor-placeholder', 'toggle-empty-msg', 'cam-input-placeholder', 'cam-overlay-placeholder'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.remove('hidden');
        });
        const qualDim = document.getElementById('qual-dim-val');
        if (qualDim) qualDim.textContent = '-';
        const qualSharp = document.getElementById('qual-sharpness-val');
        if (qualSharp) qualSharp.textContent = '-';
        const qualCont = document.getElementById('qual-contrast-val');
        if (qualCont) qualCont.textContent = '-';
        const qualCoh = document.getElementById('qual-coherence-val');
        if (qualCoh) qualCoh.textContent = '-';
        const warningsList = document.getElementById('quality-warnings-list');
        if (warningsList) {
            warningsList.innerHTML = `
                <div class="quality-info-item item-neutral">
                    <i class="fa-solid fa-info-circle"></i>
                    <div>
                        <strong>Awaiting Upload</strong>
                        <p>Upload a fingerprint above to calculate spatial edge sharpness, dynamic range, and ridge coherence.</p>
                    </div>
                </div>
            `;
        }
        const tierText = document.getElementById('uncertainty-tier-text');
        if (tierText) tierText.textContent = 'Awaiting Inference';
        const tierBadge = document.getElementById('uncertainty-tier-badge');
        if (tierBadge) {
            tierBadge.style.borderColor = 'var(--border-sage)';
            tierBadge.style.color = 'var(--primary-heading)';
            tierBadge.innerHTML = `<i class="fa-solid fa-circle-check"></i> <span>Awaiting Inference</span>`;
        }
        const marginVal = document.getElementById('unc-margin-val');
        if (marginVal) marginVal.textContent = '-';
        const entropyVal = document.getElementById('unc-entropy-val');
        if (entropyVal) entropyVal.textContent = '-';
        const alertBox = document.getElementById('unc-alert-box');
        if (alertBox) alertBox.classList.add('hidden');
        const rankedList = document.getElementById('ranked-classes-list');
        if (rankedList) rankedList.innerHTML = `<p style="padding: 15px 0; color: #64748b; font-size: 0.88rem;">Run prediction above to compute class-wise uncertainty ranking.</p>`;
    }

    // ==========================================================================
    // USB Biometric Scanner Management & Live Acquisition Controller
    // ==========================================================================

    let cachedScannerSlots = [];

    function switchInputMode(mode) {
        currentInputMode = mode;
        if (mode === 'upload') {
            if (modeUploadBtn) modeUploadBtn.classList.add('active');
            if (modeScannerBtn) modeScannerBtn.classList.remove('active');
            if (samplesSection) samplesSection.classList.remove('hidden');
            if (scannerPanel) scannerPanel.classList.add('hidden');
            if (dropzone) dropzone.classList.remove('hidden');
            if (!previewImg.src || previewArea.classList.contains('hidden')) {
                if (dropzoneInner) dropzoneInner.classList.remove('hidden');
                if (previewArea) previewArea.classList.add('hidden');
            }
        } else {
            if (modeScannerBtn) modeScannerBtn.classList.add('active');
            if (modeUploadBtn) modeUploadBtn.classList.remove('active');
            if (samplesSection) samplesSection.classList.add('hidden');
            if (!previewImg.src || previewArea.classList.contains('hidden')) {
                if (dropzone) dropzone.classList.add('hidden');
                if (scannerPanel) scannerPanel.classList.remove('hidden');
            }
            fetchScannerTelemetry(false);
        }
    }

    if (modeUploadBtn) {
        modeUploadBtn.addEventListener('click', () => switchInputMode('upload'));
    }
    if (modeScannerBtn) {
        modeScannerBtn.addEventListener('click', () => switchInputMode('scanner'));
    }

    function fetchScannerTelemetry(forceRefresh = false) {
        if (refreshScannersBtn) {
            refreshScannersBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate fa-spin"></i> Checking...';
            refreshScannersBtn.disabled = true;
        }

        const url = forceRefresh ? '/api/scanners/list?refresh=true' : '/api/scanners/list';
        fetch(url)
            .then(res => res.json())
            .then(res => {
                if (refreshScannersBtn) {
                    refreshScannersBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Refresh';
                    refreshScannersBtn.disabled = false;
                }

                if (!res.success) {
                    console.error("Scanner fetch failed:", res.error);
                    return;
                }

                const data = res.data;
                cachedScannerSlots = data.slots || [];
                const activeId = data.active_scanner_id || 'simulated';

                // Populate Dropdown
                if (scannerSelect) {
                    scannerSelect.innerHTML = '';
                    cachedScannerSlots.forEach(slot => {
                        const opt = document.createElement('option');
                        opt.value = slot.adapter_id;
                        opt.selected = (slot.adapter_id === activeId);

                        let badgeText = '';
                        if (slot.adapter_id === 'simulated' || slot.is_simulated) {
                            badgeText = '[Virtual Test Mode]';
                        } else if (slot.is_connected && slot.sdk_available) {
                            badgeText = '[Online & Ready]';
                        } else if (slot.is_connected && !slot.sdk_available) {
                            badgeText = '[Device detected — capture integration required]';
                        } else {
                            badgeText = '[Disconnected]';
                        }

                        opt.textContent = `${slot.display_name} ${badgeText}`;
                        scannerSelect.appendChild(opt);
                    });
                }

                // Update Active Slot Telemetry UI
                const activeSlot = cachedScannerSlots.find(s => s.adapter_id === activeId) || cachedScannerSlots[0];
                if (activeSlot) {
                    updateActiveSlotDisplay(activeSlot);
                }

                // Update USB Peripherals Inspector
                if (data.hardware_summary) {
                    renderUsbInspector(data.hardware_summary);
                }
            })
            .catch(err => {
                if (refreshScannersBtn) {
                    refreshScannersBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Refresh';
                    refreshScannersBtn.disabled = false;
                }
                console.error("Scanner communication error:", err);
            });
    }

    function updateActiveSlotDisplay(slot) {
        if (telemetryDeviceName) {
            telemetryDeviceName.textContent = slot.model_name || slot.display_name;
        }

        if (telemetrySlotBadge) {
            telemetrySlotBadge.textContent = `Slot: ${slot.adapter_id === 'simulated' ? 'Virtual Sensor' : 'Slot ' + slot.slot}`;
        }

        if (telemetryStatusBadge) {
            telemetryStatusBadge.className = 'telemetry-status-badge';
            if (slot.is_simulated) {
                telemetryStatusBadge.classList.add('badge-ready');
                telemetryStatusBadge.innerHTML = '<i class="fa-solid fa-vial"></i> Demo Mode (Virtual)';
            } else if (slot.status === 'ready') {
                telemetryStatusBadge.classList.add('badge-ready');
                telemetryStatusBadge.innerHTML = '<i class="fa-solid fa-circle-check"></i> Connected & Ready';
            } else if (slot.status === 'driver_missing') {
                telemetryStatusBadge.classList.add('badge-missing');
                telemetryStatusBadge.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Device detected — capture integration required';
            } else if (slot.status === 'disconnected') {
                telemetryStatusBadge.classList.add('badge-disconnected');
                telemetryStatusBadge.innerHTML = '<i class="fa-solid fa-circle-xmark"></i> Disconnected';
            } else {
                telemetryStatusBadge.classList.add('badge-unverified');
                telemetryStatusBadge.innerHTML = '<i class="fa-solid fa-circle-question"></i> Unsupported Device';
            }
        }

        if (telVendor) telVendor.textContent = slot.vendor_name || 'N/A';
        if (telVid) telVid.textContent = slot.vid !== 'N/A' ? `0x${slot.vid}` : 'N/A';
        if (telPid) telPid.textContent = slot.pid !== 'N/A' ? `0x${slot.pid}` : 'N/A';
        if (telConnection) telConnection.textContent = slot.is_connected ? 'Connected' : 'Disconnected';
        if (telSdk) telSdk.textContent = slot.sdk_available ? 'Ready / Available' : 'Driver / SDK Missing';
        if (telClass) telClass.textContent = slot.is_biometric ? 'Biometric Fingerprint Scanner' : 'Unsupported Device';

        // Update notice box
        if (scannerNoticeBox && noticeTitle && noticeDesc) {
            if (slot.status === 'driver_missing') {
                scannerNoticeBox.classList.remove('hidden');
                noticeTitle.textContent = 'Device detected — capture integration required';
                noticeDesc.textContent = slot.integration_notes || 'Hardware detected. Official vendor SDK or driver DLL is required for live image capture.';
            } else if (slot.status === 'disconnected') {
                scannerNoticeBox.classList.remove('hidden');
                noticeTitle.textContent = 'Hardware Disconnected';
                noticeDesc.textContent = slot.integration_notes || 'Please plug your USB fingerprint scanner into any USB port and click "Refresh".';
            } else {
                scannerNoticeBox.classList.add('hidden');
            }
        }

        // Update Scan Action Button
        if (triggerScanBtn) {
            if (slot.can_capture) {
                triggerScanBtn.disabled = false;
                triggerScanBtn.innerHTML = '<i class="fa-solid fa-play"></i> <span>Start Scan</span>';
            } else {
                triggerScanBtn.disabled = true;
                triggerScanBtn.innerHTML = '<i class="fa-solid fa-ban"></i> <span>Scanner Not Ready</span>';
            }
        }

        // Update header/tab status indicator dot
        if (scannerTabDot) {
            scannerTabDot.className = 'scanner-status-dot';
            if (slot.can_capture) {
                scannerTabDot.classList.add('online');
            } else if (slot.status === 'driver_missing') {
                scannerTabDot.classList.add('warning');
            } else {
                scannerTabDot.classList.add('offline');
            }
        }
    }

    function renderUsbInspector(summary) {
        if (usbTotalCount) {
            usbTotalCount.textContent = summary.total_usb_peripherals || 0;
        }

        if (!usbDeviceList) return;
        usbDeviceList.innerHTML = '';

        const allDevs = (summary.biometric_candidates || []).concat(summary.unsupported_peripherals || []);
        if (allDevs.length === 0) {
            usbDeviceList.innerHTML = '<div style="padding: 10px; color: var(--text-muted); font-size: 0.8rem;">No USB devices enumerated.</div>';
            return;
        }

        allDevs.forEach(dev => {
            const item = document.createElement('div');
            item.className = 'usb-device-item';

            const isBio = dev.is_biometric_candidate;
            const badgeHtml = isBio
                ? `<span class="usb-badge-biometric"><i class="fa-solid fa-fingerprint"></i> Biometric Candidate</span>`
                : `<span class="usb-badge-unsupported"><i class="fa-solid fa-ban"></i> Non-Biometric Device</span>`;

            item.innerHTML = `
                <div class="usb-item-left">
                    <span class="usb-item-name" title="${dev.name}">${dev.name}</span>
                    <span class="usb-item-meta font-mono">VID: 0x${dev.vid} | PID: 0x${dev.pid} | ${dev.manufacturer || 'Standard'}</span>
                </div>
                ${badgeHtml}
            `;
            usbDeviceList.appendChild(item);
        });
    }

    // Scanner Selection Change Event (Hot-Switching without server restart)
    if (scannerSelect) {
        scannerSelect.addEventListener('change', (e) => {
            const targetId = e.target.value;
            if (scannerLiveStatusText) {
                scannerLiveStatusText.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Switching active scanner slot...';
            }

            fetch('/api/scanners/select', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scanner_id: targetId })
            })
            .then(res => res.json())
            .then(res => {
                if (res.success) {
                    if (scannerLiveStatusText) {
                        scannerLiveStatusText.innerHTML = `<i class="fa-solid fa-circle-check"></i> Switched to ${res.data.telemetry.display_name}. Ready.`;
                    }
                    updateActiveSlotDisplay(res.data.telemetry);
                } else {
                    alert('Scanner selection error: ' + (res.error || 'Unknown error'));
                    fetchScannerTelemetry(false);
                }
            })
            .catch(err => {
                console.error("Device switch error:", err);
                alert("Failed to switch scanner slot.");
            });
        });
    }

    // Refresh Scanners Button Event
    if (refreshScannersBtn) {
        refreshScannersBtn.addEventListener('click', () => {
            fetchScannerTelemetry(true);
        });
    }

    // USB Inspector Accordion Toggle
    if (usbInspectorToggle && usbInspectorContent) {
        usbInspectorToggle.addEventListener('click', () => {
            const isHidden = usbInspectorContent.classList.contains('hidden');
            if (isHidden) {
                usbInspectorContent.classList.remove('hidden');
                usbInspectorToggle.classList.add('open');
            } else {
                usbInspectorContent.classList.add('hidden');
                usbInspectorToggle.classList.remove('open');
            }
        });
    }

    // Live Fingerprint Capture Trigger & Cancellation
    if (triggerScanBtn) {
        triggerScanBtn.addEventListener('click', () => {
            const activeId = scannerSelect ? scannerSelect.value : 'simulated';

            triggerScanBtn.disabled = true;
            triggerScanBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> <span>Acquiring Ridge Biometrics...</span>';
            if (cancelScanBtn) {
                cancelScanBtn.classList.remove('hidden');
                cancelScanBtn.disabled = false;
                cancelScanBtn.innerHTML = '<i class="fa-solid fa-xmark"></i> <span>Cancel Scan</span>';
            }
            if (scannerLiveStatusText) {
                scannerLiveStatusText.innerHTML = '<i class="fa-solid fa-hand-holding-hand fa-fade"></i> Acquiring fingerprint impression... Place finger firmly on sensor surface.';
            }

            // Hide review card while acquiring new scan
            if (scannerCaptureReview) {
                scannerCaptureReview.classList.add('hidden');
            }

            fetch('/api/scanners/capture', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scanner_id: activeId, timeout: 10.0 })
            })
            .then(res => res.json())
            .then(res => {
                triggerScanBtn.disabled = false;
                triggerScanBtn.innerHTML = '<i class="fa-solid fa-play"></i> <span>Start Scan</span>';
                if (cancelScanBtn) {
                    cancelScanBtn.classList.add('hidden');
                }

                if (res.success) {
                    const data = res.data;
                    lastCapturedScanData = data;

                    // Display Captured Image in In-Panel Review Box
                    if (scannerReviewImg) {
                        scannerReviewImg.src = data.image_base64;
                    }
                    if (captureQualityBadge) {
                        captureQualityBadge.textContent = `Quality: ${data.quality_score}%`;
                    }
                    if (captureDimBadge) {
                        captureDimBadge.textContent = `${data.normalized_dimensions ? data.normalized_dimensions[0] + ' × ' + data.normalized_dimensions[1] : '512 × 512'} px`;
                    }
                    if (captureSourceVal) {
                        captureSourceVal.textContent = data.scanner_name || 'USB Scanner';
                    }
                    if (captureClarityVal) {
                        const level = data.quality_score >= 80 ? 'High' : (data.quality_score >= 50 ? 'Moderate' : 'Low');
                        captureClarityVal.textContent = `${level} (${data.quality_score}%)`;
                    }
                    if (captureTimeVal) {
                        captureTimeVal.textContent = `${data.capture_time_ms || 320} ms`;
                    }

                    if (scannerCaptureReview) {
                        scannerCaptureReview.classList.remove('hidden');
                    }

                    if (scannerLiveStatusText) {
                        scannerLiveStatusText.innerHTML = `<i class="fa-solid fa-circle-check" style="color: #2E7D32;"></i> Capture Complete! Review impression below and click <strong>"Use This Fingerprint"</strong>.`;
                    }
                } else {
                    const errMsg = res.error || 'Failed to capture from biometric sensor.';
                    const isCancelled = res.is_cancelled || (errMsg.toLowerCase().includes('cancel'));

                    if (isCancelled) {
                        if (scannerLiveStatusText) {
                            scannerLiveStatusText.innerHTML = `<i class="fa-solid fa-circle-info" style="color: #ED6C02;"></i> Scan operation was cancelled by user.`;
                        }
                    } else {
                        if (scannerLiveStatusText) {
                            scannerLiveStatusText.innerHTML = `<i class="fa-solid fa-circle-xmark" style="color: #C62828;"></i> Capture Failed: ${errMsg}`;
                        }
                        alert(`Scanner Capture Notice:\n\n${errMsg}\n\nTip: You can select 'Demo / Simulated Scanner (Virtual Test Mode)' to test the complete analysis workflow without physical hardware.`);
                    }
                }
            })
            .catch(err => {
                triggerScanBtn.disabled = false;
                triggerScanBtn.innerHTML = '<i class="fa-solid fa-play"></i> <span>Start Scan</span>';
                if (cancelScanBtn) {
                    cancelScanBtn.classList.add('hidden');
                }
                if (scannerLiveStatusText) {
                    scannerLiveStatusText.innerHTML = '<i class="fa-solid fa-circle-xmark" style="color: #C62828;"></i> Scanner communication error.';
                }
            });
        });
    }

    // Cancel Scan Button Event
    if (cancelScanBtn) {
        cancelScanBtn.addEventListener('click', () => {
            cancelScanBtn.disabled = true;
            cancelScanBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Cancelling...';
            if (scannerLiveStatusText) {
                scannerLiveStatusText.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Sending cancellation request to scanner...';
            }

            fetch('/api/scanners/cancel', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            })
            .then(res => res.json())
            .then(res => {
                if (scannerLiveStatusText) {
                    scannerLiveStatusText.innerHTML = '<i class="fa-solid fa-circle-info" style="color: #ED6C02;"></i> Cancellation signal sent to device.';
                }
            })
            .catch(err => {
                console.error("Cancel scan error:", err);
            });
        });
    }

    // "Use This Fingerprint" Button -> Sends captured scan to existing prediction pipeline
    if (useFingerprintBtn) {
        useFingerprintBtn.addEventListener('click', () => {
            if (!lastCapturedScanData || !lastCapturedScanData.image_base64) {
                alert("No captured fingerprint to use. Please capture a fingerprint first.");
                return;
            }

            scannedImageBase64 = lastCapturedScanData.image_base64;
            selectedFile = null;
            selectedSamplePath = null;

            // Deselect any sample picker items
            document.querySelectorAll('.sample-item').forEach(s => s.classList.remove('selected'));

            // Display in Main Pipeline Preview Area
            previewImg.src = scannedImageBase64;
            imgNameDisplay.textContent = `${lastCapturedScanData.scanner_name} (512×512 | Quality: ${lastCapturedScanData.quality_score}%)`;

            showPreview();

            if (scannerLiveStatusText) {
                scannerLiveStatusText.innerHTML = `<i class="fa-solid fa-circle-check" style="color: #2E7D32;"></i> Fingerprint impression loaded into prediction pipeline. Click <strong>"Run Analysis"</strong> below!`;
            }

            // Scroll down smoothly to the Run Analysis button
            const actionBar = document.querySelector('.action-bar');
            if (actionBar) {
                actionBar.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        });
    }

    // "Retake / Discard" Button
    if (retakeScanBtn) {
        retakeScanBtn.addEventListener('click', () => {
            lastCapturedScanData = null;
            if (scannerCaptureReview) {
                scannerCaptureReview.classList.add('hidden');
            }
            if (scannerLiveStatusText) {
                scannerLiveStatusText.innerHTML = '<i class="fa-solid fa-circle-info"></i> Impression discarded. Ready to scan again.';
            }
        });
    }

    // Initialize scanner telemetry on startup
    fetchScannerTelemetry(false);

    init3DCardTilt();
    initScrollReveal();
});


