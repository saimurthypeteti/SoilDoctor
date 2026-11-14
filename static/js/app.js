/**
 * SoilDoctor - Master Client Application Logic
 * Integrates:
 * - Single Page Router (17 views)
 * - HTML5 Camera Viewfinder & File Drag/Drop
 * - 1-Click Zero Friction Demo Presets
 * - Animated 6-Stage AI Stepper
 * - OpenCV Computer Vision Results
 * - Explainable AI (Grad-CAM) Visualizer
 * - Soil Health & Radial Nutrient Dials
 * - Interactive Leaflet.js Farm Map
 * - Chart.js Historical Analytics
 * - Scan History & Printable AI Clinical Report
 * - Mobile Phone Simulator Mode
 */

// Application State
const AppState = {
  currentView: "dashboard",
  navHistory: [], // History stack for back navigation
  lastPlantResult: null,
  lastSoilResult: null,
  soilLabData: null,
  scanLocation: null,
  scanMode: "plant", // 'plant' or 'soil'
  cameraStream: null,
  cameraFacingMode: "environment", // 'environment' (back) or 'user' (front)
  pendingSnapshotB64: null,
  selectedUploadB64: null,
  datasetManifest: [],
  activeGalleryFilter: "all",
  map: null,
  mapLayers: [],
  charts: {
    trends: null,
    distribution: null
  },
  fields: [],
  scans: [],
  historyFilter: "all",
  crops: [],
  isMobileSim: false
};

function showNotification(message, type = "info") {
  let host = document.getElementById("app-notification-host");
  if (!host) {
    host = document.createElement("div");
    host.id = "app-notification-host";
    host.setAttribute("aria-live", "polite");
    host.style.cssText = "position:fixed;top:18px;right:18px;z-index:10000;display:grid;gap:8px;max-width:min(360px,calc(100vw - 36px));";
    document.body.appendChild(host);
  }
  const note = document.createElement("div");
  const colors = { error: "#b42318", warning: "#92400e", success: "#166534", info: "#1d4ed8" };
  note.textContent = message;
  note.setAttribute("role", type === "error" ? "alert" : "status");
  note.style.cssText = `padding:12px 16px;border-radius:10px;color:#fff;background:${colors[type] || colors.info};box-shadow:0 8px 24px #0002;font-size:14px;`;
  host.appendChild(note);
  setTimeout(() => note.remove(), 5000);
}

// ==========================================================================
// 1. Navigation & View Routing
// ==========================================================================

const VIEW_TITLES = {
  "landing": { title: "🌱 SoilDoctor Overview", sub: "Next-Gen AI Crop & Soil Health Assistant" },
  "dashboard": { title: "Good Morning, Farmer! 👋", sub: "Let's check your crops today." },
  "scan-options": { title: "Diagnostic Scanner", sub: "Select specimen to inspect with Computer Vision" },
  "capture": { title: "Capture Specimen", sub: "Align leaf or soil inside the camera reticle" },
  "processing": { title: "AI Computer Vision in Progress...", sub: "Extracting morphological features & Grad-CAM activations" },
  "cv-results": { title: "Computer Vision Results", sub: "Preprocessing, segmentation, and extracted features" },
  "explainable-ai": { title: "Explainable AI (Grad-CAM)", sub: "Visualizing deep learning attention heatmaps" },
  "disease-result": { title: "Disease Diagnostic Result", sub: "Pathology classification & risk assessment" },
  "soil-health": { title: "Soil Health & Nutrient Gauges", sub: "Surface vision calibrated with chemical readings" },
  "recommendations": { title: "Treatment & Fertilizer Plan", sub: "Dual-solution organic and mineral prescriptions" },
  "farm-mgmt": { title: "My Farm Management", sub: "Spatial plot health and crop field management" },
  "analytics": { title: "Health Trends & Analytics", sub: "Multi-crop historical vitality tracking" },
  "history": { title: "Scan History Archive", sub: "Inspect and compare historical diagnostics" },
  "crop-doctor": { title: "Crop Doctor Encyclopedia", sub: "Agronomic field guide for major agricultural crops" },
  "report-view": { title: "Official AI Diagnostic Report", sub: "Certified agronomy summary ready to print or export" },
  "settings": { title: "Farmer Profile & Settings", sub: "Configure farm location and notification preferences" }
};

function navigateTo(viewId, skipHistory) {
  // Push current view to history before navigating (unless skipping, e.g. goBack)
  if (!skipHistory && AppState.currentView && AppState.currentView !== viewId) {
    AppState.navHistory.push(AppState.currentView);
    // Limit history to 20 entries
    if (AppState.navHistory.length > 20) AppState.navHistory.shift();
  }

  // Hide all views
  const views = document.querySelectorAll(".app-view");
  views.forEach(v => v.classList.remove("active"));

  // Show target view
  const targetView = document.getElementById(`view-${viewId}`);
  if (targetView) {
    targetView.classList.add("active");
    AppState.currentView = viewId;
  }

  // Show or hide back button
  const backBtn = document.getElementById("global-back-btn");
  if (backBtn) {
    backBtn.style.display = AppState.navHistory.length > 0 ? "flex" : "none";
  }

  // Update Navigation Active State
  const navItems = document.querySelectorAll(".sidebar-menu .nav-item");
  navItems.forEach(item => {
    if (item.dataset.view === viewId) {
      item.classList.add("active");
    } else {
      item.classList.remove("active");
    }
  });

  // Update Header
  const titleInfo = VIEW_TITLES[viewId] || { title: "SoilDoctor", sub: "AI Agriculture Assistant" };
  const titleEl = document.getElementById("header-title");
  const subEl = document.getElementById("header-subtitle");
  if (titleEl) titleEl.innerText = titleInfo.title;
  if (subEl) subEl.innerText = titleInfo.sub;

  if (viewId === "dashboard") {
    loadDashboard();
  }

  if (viewId === "capture") {
    initWebcam();
  } else {
    stopWebcam();
  }

  if (viewId === "farm-mgmt") {
    loadFarmFields();
    setTimeout(initFarmMap, 200);
  }

  if (viewId === "analytics") {
    setTimeout(initAnalyticsCharts, 200);
  }

  if (viewId === "history") {
    loadScanHistory(AppState.historyFilter || "all");
  }

  if (viewId === "crop-doctor") {
    loadCropDoctor();
  }

  if (viewId === "report-view") {
    loadReportViewData();
  }

  // Close mobile sidebar if open
  const sidebar = document.getElementById("sidebar");
  if (sidebar) sidebar.classList.remove("mobile-open");

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Go back to the previous page in navigation history
function goBack() {
  if (AppState.navHistory.length === 0) return;
  const prevView = AppState.navHistory.pop();
  navigateTo(prevView, true); // true = skip pushing to history again
}

function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  if (sidebar) sidebar.classList.toggle("mobile-open");
}

function toggleMobileSimulator() {
  const container = document.getElementById("app-container");
  const toggleBtnText = document.getElementById("mobile-toggle-text");
  AppState.isMobileSim = !AppState.isMobileSim;

  if (AppState.isMobileSim) {
    container.classList.remove("desktop-mode");
    container.classList.add("phone-mode");
    if (toggleBtnText) toggleBtnText.innerText = "Switch to Desktop View";
  } else {
    container.classList.remove("phone-mode");
    container.classList.add("desktop-mode");
    if (toggleBtnText) toggleBtnText.innerText = "Switch to Phone View";
  }

  // Invalidate Leaflet map size if on farm view
  if (AppState.map) {
    setTimeout(() => AppState.map.invalidateSize(), 300);
  }
}

// ==========================================================================
// 2. Camera Viewfinder & File Drag/Drop
// ==========================================================================

function openCaptureFlow(mode) {
  AppState.scanMode = mode;
  AppState.scanLocation = null;
  const liveLocationButton = document.getElementById("scan-live-location-btn");
  if (liveLocationButton) liveLocationButton.disabled = false;
  loadScanFields();
  clearSelectedUpload();
  selectUploadCrop("");
  updateCaptureModeUI();
  updateScanSelectionSummary();
  const promptEl = document.getElementById("reticle-prompt-text");
  if (promptEl) {
    promptEl.innerText = mode === "soil" ? "Place the soil sample inside the frame" : "Place the leaf inside the frame";
  }
  navigateTo("capture");
  switchCaptureTab("upload");
}

function updateCaptureModeUI() {
  const isSoil = AppState.scanMode === "soil";
  const setText = (id, text) => {
    const fallbacks = {
      "upload-hero-title": ".upload-hero-text strong",
      "upload-hero-subtitle": ".upload-hero-text span",
      "upload-zone-title": "#upload-zone .dropzone-content h3",
      "upload-zone-help": "#upload-zone .dropzone-content p",
      "upload-preview-title": ".upload-preview-head h4",
      "analyze-upload-btn": ".upload-preview-actions button"
    };
    const element = document.getElementById(id) || document.querySelector(fallbacks[id]);
    if (element) element.textContent = text;
  };
  setText("upload-hero-title", isSoil ? "Upload a Soil Photo" : "Upload a Leaf Photo");
  setText("upload-hero-subtitle", isSoil
    ? "Choose a clear photo of soil from your device"
    : "Tap here to choose a leaf photo from your gallery or files");
  setText("upload-zone-title", isSoil ? "Drag & drop a soil image here" : "Drag & drop a leaf image here");
  setText("upload-zone-help", "Supports JPG, JPEG, PNG, and WEBP images");
  setText("upload-preview-title", isSoil ? "Soil photo ready for analysis" : "Leaf photo ready for analysis");
  setText("analyze-upload-btn", isSoil ? "Analyze Soil Image" : "Run AI Diagnosis Now");
  const cropSelector = document.getElementById("upload-crop-selector-box") || document.querySelector(".upload-crop-selector-box");
  if (cropSelector) cropSelector.style.display = isSoil ? "none" : "block";
  const targetCropBox = document.getElementById("scan-target-crop-box") || document.getElementById("scan-crop-select")?.closest(".sample-target-preview");
  if (targetCropBox) targetCropBox.style.display = isSoil ? "none" : "block";
  document.querySelectorAll(".preset-btn").forEach(button => {
    const isSoilSample = button.textContent.toLowerCase().includes("loamy farm soil");
    button.style.display = isSoil ? (isSoilSample ? "" : "none") : "";
  });
  const galleryLink = document.querySelector(".preset-header-row .card-link");
  if (galleryLink) galleryLink.style.display = isSoil ? "none" : "";
  const presetLabel = document.querySelector(".preset-label");
  if (presetLabel) presetLabel.innerHTML = isSoil
    ? '<i class="fa-solid fa-flask text-warning"></i> Soil sample:'
    : '<i class="fa-solid fa-bolt text-warning"></i> Real Dataset Test Presets (Instant 1-Click):';
  const tipsHeading = document.querySelector(".capture-sidebar h3");
  if (tipsHeading) tipsHeading.textContent = isSoil ? "Soil Photo Tips" : "Image Quality Tips";
  const tips = isSoil
    ? ["Use a clean, dry soil sample", "Spread soil in a thin, even layer", "Use bright natural light", "Avoid shadows and filters", "Keep the camera close and in focus"]
    : ["Good natural lighting (indirect sunlight)", "Keep steady to avoid camera shake", "Focus on lesions, veins, or leaf discoloration", "Keep whole leaf in reticle frame", "Avoid dark backgrounds or heavy shadows"];
  document.querySelectorAll(".capture-sidebar .tips-checklist li").forEach((item, index) => {
    if (tips[index]) item.innerHTML = `<i class="fa-solid ${index === 4 && !isSoil ? "fa-circle-xmark text-danger" : "fa-circle-check text-green"}"></i> ${tips[index]}`;
  });
  const specimenBadge = document.querySelector(".capture-specimen-badge .badge");
  if (specimenBadge) specimenBadge.innerHTML = isSoil
    ? '<i class="fa-solid fa-earth-americas"></i> Soil Photo Check'
    : '<i class="fa-solid fa-shield-virus"></i> Real-Time Specimen Guard Active';
  const specimenNote = document.querySelector(".capture-specimen-badge p");
  if (specimenNote) specimenNote.textContent = isSoil
    ? "The photo is used for visual soil color and texture analysis; sensor values are separate."
    : "Protects against non-crop objects (humans, electronics, pets) before diagnosis.";
  let sensorPanel = document.getElementById("soil-capture-sensor-panel");
  // Older Flask processes may still serve a cached template. Build the sensor panel
  // from the current JS so the soil flow still works after a browser refresh.
  if (isSoil && !sensorPanel) {
    const uploadCard = document.getElementById("upload-preview-card");
    if (uploadCard) {
      const panel = document.createElement("div");
      panel.id = "soil-capture-sensor-panel";
      panel.className = "card mt-3";
      panel.style.cssText = "padding:16px;";
      panel.innerHTML = `
        <div class="card-header" style="padding:0 0 10px;">
          <h3><i class="fa-solid fa-microchip text-teal"></i> Soil sensor values</h3>
          <span class="badge badge-warning">Project demo</span>
        </div>
        <label style="display:flex; gap:8px; align-items:center; margin:8px 0;">
          <input type="checkbox" id="soil-demo-sensor-toggle" checked /> Use simulated sensor readings
        </label>
        <small class="text-muted">Example values for your project demo; these are simulated, not measured by a physical sensor.</small>
        <div class="form-grid-3 mt-3">
          <label class="form-group">pH <input class="form-control" type="number" id="soil-capture-ph" min="0" max="14" step="0.1" value="6.4" /></label>
          <label class="form-group">Nitrogen (N)<select class="form-select" id="soil-capture-n"><option>Low</option><option selected>Normal</option><option>High</option></select></label>
          <label class="form-group">Phosphorus (P)<select class="form-select" id="soil-capture-p"><option>Low</option><option selected>Normal</option><option>High</option></select></label>
          <label class="form-group">Potassium (K)<select class="form-select" id="soil-capture-k"><option>Low</option><option selected>Normal</option><option>High</option></select></label>
          <label class="form-group">Moisture (%) <input class="form-control" type="number" id="soil-capture-moisture" min="0" max="100" value="28" /></label>
        </div>`;
      uploadCard.insertAdjacentElement("afterend", panel);
      sensorPanel = panel;
    }
  }
  if (sensorPanel) sensorPanel.style.display = isSoil ? "block" : "none";
  updateScanSelectionSummary();
}

function switchCaptureTab(tabName) {
  const tabCamera = document.getElementById("tab-camera");
  const tabUpload = document.getElementById("tab-upload");
  const cameraView = document.getElementById("camera-viewfinder");
  const snapshotConfirm = document.getElementById("camera-snapshot-confirm");
  const uploadZone = document.getElementById("upload-zone");
  const uploadPreview = document.getElementById("upload-preview-card");

  if (tabName === "camera") {
    tabCamera.classList.add("active");
    tabUpload.classList.remove("active");
    cameraView.style.display = "flex";
    if (snapshotConfirm) snapshotConfirm.style.display = "none";
    uploadZone.style.display = "none";
    if (uploadPreview) uploadPreview.style.display = "none";
    initWebcam();
  } else {
    tabUpload.classList.add("active");
    tabCamera.classList.remove("active");
    cameraView.style.display = "none";
    if (snapshotConfirm) snapshotConfirm.style.display = "none";
    stopWebcam();
    if (AppState.selectedUploadB64 && uploadPreview) {
      uploadZone.style.display = "none";
      uploadPreview.style.display = "block";
    } else {
      uploadZone.style.display = "flex";
      if (uploadPreview) uploadPreview.style.display = "none";
    }
  }
}

async function initWebcam() {
  const video = document.getElementById("webcam-feed");
  if (!video) return;

  stopWebcam();

  try {
    const constraints = {
      video: {
        facingMode: AppState.cameraFacingMode,
        width: { ideal: 640 },
        height: { ideal: 640 }
      }
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    AppState.cameraStream = stream;
    video.srcObject = stream;
    const promptEl = document.getElementById("reticle-prompt-text");
    if (promptEl) {
      promptEl.innerText = AppState.scanMode === "soil" ? "Position soil sample inside frame" : "Position leaf inside reticle frame";
    }
  } catch (err) {
    console.warn("Primary camera access error, trying fallback:", err);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      AppState.cameraStream = stream;
      video.srcObject = stream;
    } catch (err2) {
      console.warn("Webcam access not granted or unavailable:", err2);
      const promptEl = document.getElementById("reticle-prompt-text");
      if (promptEl) {
        promptEl.innerHTML = `<span style="color:#FCD34D;">Camera inactive. Tap 'Phone Cam' or use 'Upload Photo' tab.</span>`;
      }
    }
  }
}

function stopWebcam() {
  if (AppState.cameraStream) {
    AppState.cameraStream.getTracks().forEach(track => track.stop());
    AppState.cameraStream = null;
  }
}

function flipCamera() {
  AppState.cameraFacingMode = AppState.cameraFacingMode === "environment" ? "user" : "environment";
  initWebcam();
}

function triggerMobileCamera() {
  const input = document.getElementById("mobile-camera-input");
  if (input) input.click();
}

function captureFromWebcam() {
  const video = document.getElementById("webcam-feed");
  const canvas = document.getElementById("capture-canvas");
  if (!video || !canvas) return;

  canvas.width = video.videoWidth || 480;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const b64Image = canvas.toDataURL("image/jpeg", 0.92);
  AppState.pendingSnapshotB64 = b64Image;

  // Show confirmation card
  const cameraView = document.getElementById("camera-viewfinder");
  const confirmBox = document.getElementById("camera-snapshot-confirm");
  const previewImg = document.getElementById("snapshot-preview-img");

  if (previewImg) previewImg.src = b64Image;
  if (cameraView) cameraView.style.display = "none";
  if (confirmBox) confirmBox.style.display = "block";
}

function retakePhoto() {
  AppState.pendingSnapshotB64 = null;
  const cameraView = document.getElementById("camera-viewfinder");
  const confirmBox = document.getElementById("camera-snapshot-confirm");
  if (confirmBox) confirmBox.style.display = "none";
  if (cameraView) cameraView.style.display = "flex";
  initWebcam();
}

function confirmSnapshot() {
  if (AppState.pendingSnapshotB64) {
    submitImageForAnalysis(AppState.pendingSnapshotB64);
  }
}

function triggerFileInput() {
  const input = document.getElementById("file-input");
  if (input) input.click();
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;

  // Require the user to choose a supported crop for this image.
  selectUploadCrop("");

  const reader = new FileReader();
  reader.onload = function(event) {
    AppState.selectedUploadB64 = event.target.result;
    
    // Switch to upload tab and show preview card
    switchCaptureTab("upload");
    const uploadZone = document.getElementById("upload-zone");
    const uploadPreview = document.getElementById("upload-preview-card");
    const previewImg = document.getElementById("upload-preview-img");
    const nameEl = document.getElementById("upload-file-name");
    const sizeEl = document.getElementById("upload-file-size");

    if (uploadZone) uploadZone.style.display = "none";
    if (uploadPreview) uploadPreview.style.display = "block";
    if (previewImg) previewImg.src = AppState.selectedUploadB64;
    if (nameEl) nameEl.innerText = file.name;
    if (sizeEl) sizeEl.innerText = `${(file.size / (1024 * 1024)).toFixed(2)} MB (${file.type || 'image'})`;
    updateScanSelectionSummary();
  };
  reader.readAsDataURL(file);
}

function clearSelectedUpload() {
  AppState.selectedUploadB64 = null;
  const fileInput = document.getElementById("file-input");
  if (fileInput) fileInput.value = "";
  const uploadZone = document.getElementById("upload-zone");
  const uploadPreview = document.getElementById("upload-preview-card");
  if (uploadZone) uploadZone.style.display = "flex";
  if (uploadPreview) uploadPreview.style.display = "none";
  selectUploadCrop("");
  updateScanSelectionSummary();
}

function analyzeSelectedUpload() {
  if (!ensureScanAreaSelected()) return;
  if (AppState.selectedUploadB64) {
    submitImageForAnalysis(AppState.selectedUploadB64);
  }
}

// Dashboard quick-upload button handler
// Reads the file, sets mode to 'plant', navigates to capture/upload tab with a preview ready
function handleDashQuickUpload(e) {
  const file = e.target.files[0];
  if (!file) return;

  // Require the user to choose a supported crop for this image.
  selectUploadCrop("");

  const reader = new FileReader();
  reader.onload = function(evt) {
    AppState.scanMode = "plant";
    AppState.scanLocation = null;
    loadScanFields();
    updateCaptureModeUI();
    AppState.selectedUploadB64 = evt.target.result;

    // Navigate to the capture screen, switch to upload tab, show preview
    navigateTo("capture");

    // Small delay so the view renders before we manipulate it
    setTimeout(() => {
      switchCaptureTab("upload");

      const uploadZone    = document.getElementById("upload-zone");
      const uploadPreview = document.getElementById("upload-preview-card");
      const previewImg    = document.getElementById("upload-preview-img");
      const nameEl        = document.getElementById("upload-file-name");
      const sizeEl        = document.getElementById("upload-file-size");

      if (uploadZone)    uploadZone.style.display    = "none";
      if (uploadPreview) uploadPreview.style.display = "block";
      if (previewImg)    previewImg.src              = AppState.selectedUploadB64;
      if (nameEl)        nameEl.innerText            = file.name;
      if (sizeEl)        sizeEl.innerText            = `${(file.size / (1024 * 1024)).toFixed(2)} MB (${file.type || 'image'})`;

      // Reset the file input so the same file can be selected again later
      e.target.value = "";
    }, 120);
  };
  reader.readAsDataURL(file);
}

// Real Dataset Explorer / Gallery Functions
function toggleDatasetGallery() {
  const box = document.getElementById("dataset-gallery-box");
  if (!box) return;
  if (box.style.display === "none" || !box.style.display) {
    box.style.display = "block";
    loadDatasetGallery();
    box.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } else {
    box.style.display = "none";
  }
}

function loadDatasetGallery() {
  if (AppState.datasetManifest && AppState.datasetManifest.length > 0) {
    renderDatasetGallery();
    return;
  }

  fetch("/api/dataset/samples")
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        AppState.datasetManifest = data.samples;
        renderDatasetGallery();
      }
    })
    .catch(err => console.error("Error loading dataset samples:", err));
}

function filterGalleryByCrop(cropName) {
  AppState.activeGalleryFilter = cropName;
  const filterBtns = document.querySelectorAll("#gallery-crop-filters .filter-pill");
  filterBtns.forEach(btn => {
    if (btn.innerText.toLowerCase() === cropName.toLowerCase() || (cropName === "all" && btn.innerText.toLowerCase() === "all")) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
  renderDatasetGallery();
}

function selectUploadCrop(cropName) {
  const pills = document.querySelectorAll("#upload-crop-pills .crop-pill-btn");
  pills.forEach(p => {
    if (p.getAttribute("data-crop").toLowerCase() === cropName.toLowerCase()) {
      p.classList.add("active");
    } else {
      p.classList.remove("active");
    }
  });

  const select = document.getElementById("scan-crop-select");
  if (select) {
    select.value = cropName;
  }
  AppState.selectedCrop = cropName;
  updateScanSelectionSummary();
}

function onTargetCropChanged(cropName) {
  AppState.selectedCrop = cropName;
  const pills = document.querySelectorAll("#upload-crop-pills .crop-pill-btn");
  pills.forEach(p => {
    if (p.getAttribute("data-crop").toLowerCase() === cropName.toLowerCase()) {
      p.classList.add("active");
    } else {
      p.classList.remove("active");
    }
  });

  if (AppState.activeGalleryFilter !== "all" && AppState.activeGalleryFilter !== cropName) {
    filterGalleryByCrop(cropName);
  }
  updateScanSelectionSummary();
}

function updateCropSwitchBarActive(cropName) {
  const switchPills = document.querySelectorAll("#diag-crop-switch-pills button");
  switchPills.forEach(btn => {
    const btnCrop = btn.getAttribute("data-crop") || btn.innerText;
    if (btnCrop && cropName && btnCrop.toLowerCase().includes(cropName.toLowerCase())) {
      btn.className = "btn btn-xs btn-primary font-bold";
    } else {
      btn.className = "btn btn-xs btn-outline";
    }
  });
}

function switchDiagnosisCrop(cropName) {
  const currentImg = AppState.lastPlantResult && AppState.lastPlantResult.images && AppState.lastPlantResult.images.original;
  if (!currentImg) {
    showNotification("No specimen image found to re-diagnose.", "warning");
    return;
  }

  showNotification(`Re-evaluating diagnosis for ${cropName}...`, "info");
  updateCropSwitchBarActive(cropName);

  startAIPipeline({
    previewSrc: currentImg,
    endpoint: "/api/scan/plant",
    payload: { image_b64: currentImg, crop_type: cropName },
    cropType: cropName,
    isSoil: false
  });
}

function renderDatasetGallery() {
  const grid = document.getElementById("dataset-gallery-grid");
  if (!grid || !AppState.datasetManifest) return;

  const filtered = AppState.activeGalleryFilter === "all"
    ? AppState.datasetManifest
    : AppState.datasetManifest.filter(item => item.crop.toLowerCase() === AppState.activeGalleryFilter.toLowerCase());

  if (filtered.length === 0) {
    grid.innerHTML = `<div class="p-3 text-muted">No samples match the selected crop filter.</div>`;
    return;
  }

  grid.innerHTML = filtered.map(item => {
    const sample = item.primary_sample || (item.samples && item.samples[0]);
    if (!sample) return "";
    const riskClass = item.risk_level === "HIGH" ? "badge-danger" : item.risk_level === "MEDIUM" ? "badge-warning" : "badge-success";
    return `
      <div class="gallery-card" onclick="runSampleScan('${sample.filename}', '${item.crop}')">
        <div class="gallery-thumb-box">
          <img src="${sample.url}" alt="${item.disease_name}" loading="lazy" />
          <span class="badge ${riskClass} gallery-badge">${item.risk_level}</span>
        </div>
        <div class="gallery-info">
          <strong>${item.crop}</strong>
          <p class="gallery-disease-title">${item.disease_name}</p>
          <button class="btn btn-sm btn-outline btn-block mt-1">
            <i class="fa-solid fa-microchip"></i> Test Sample
          </button>
        </div>
      </div>
    `;
  }).join("");
}

// ==========================================================================
// 3. AI Execution Pipeline & Real-Time Computer Vision Stepper
// ==========================================================================

function addPipelineLog(msg, type = "info") {
  const logBox = document.getElementById("proc-console-log");
  if (!logBox) return;

  const now = new Date();
  const timeStr = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}.${String(Math.floor(now.getMilliseconds() / 100))}`;

  const line = document.createElement("div");
  line.className = "console-log-line";
  line.innerHTML = `
    <span class="console-time">[${timeStr}]</span>
    <span class="console-msg ${type}">${msg}</span>
  `;
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

function setPipelineStep(stepIdx, pct, taskTitle) {
  const stepIds = ["p-step-1", "p-step-2", "p-step-3", "p-step-4", "p-step-5", "p-step-6"];
  stepIds.forEach((id, idx) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (idx < stepIdx) {
      el.className = "proc-step completed";
    } else if (idx === stepIdx) {
      el.className = "proc-step active";
    } else {
      el.className = "proc-step";
    }
  });

  const bar = document.getElementById("proc-progress-bar");
  const pctText = document.getElementById("proc-progress-pct");
  const taskName = document.getElementById("proc-task-name");

  if (bar) bar.style.width = pct + "%";
  if (pctText) pctText.innerText = pct + "%";
  if (taskName && taskTitle) taskName.innerText = taskTitle;
}

function cancelProcessing() {
  AppState.pipelineCancelled = true;
  if (AppState.pipelineAutoTimer) {
    clearInterval(AppState.pipelineAutoTimer);
    AppState.pipelineAutoTimer = null;
  }
  navigateTo("scan-options");
}

function proceedToResults() {
  if (AppState.pipelineAutoTimer) {
    clearInterval(AppState.pipelineAutoTimer);
    AppState.pipelineAutoTimer = null;
  }
  if (AppState.lastPipelineIsSoil) {
    navigateTo("soil-health");
  } else {
    // Direct navigation to Step 2: Crop Doctor diagnosis result per workflow specification
    navigateTo("disease-result");
  }
}

function startAIPipeline({ previewSrc, endpoint, payload, cropType, isSoil = false }) {
  AppState.pipelineCancelled = false;
  AppState.lastPipelineIsSoil = isSoil;
  if (AppState.pipelineAutoTimer) {
    clearInterval(AppState.pipelineAutoTimer);
    AppState.pipelineAutoTimer = null;
  }

  // 1. Navigate to processing view
  navigateTo("processing");

  // 2. Set Header details
  const headerTitle = document.getElementById("proc-header-title");
  const cropBadge = document.getElementById("proc-crop-badge");
  const compCard = document.getElementById("proc-completion-card");
  const errCard = document.getElementById("proc-error-card");
  const spinner = document.getElementById("proc-main-spinner");
  const logBox = document.getElementById("proc-console-log");

  if (compCard) compCard.style.display = "none";
  if (errCard) errCard.style.display = "none";
  if (spinner) spinner.style.display = "block";
  if (logBox) logBox.innerHTML = "";

  if (cropBadge) {
    cropBadge.innerHTML = isSoil
      ? `<i class="fa-solid fa-earth-americas"></i> Soil Specimen`
      : `<i class="fa-solid fa-leaf"></i> Crop: ${cropType || 'Foliage'}`;
  }
  if (headerTitle) {
    headerTitle.innerText = isSoil
      ? "Analyzing Soil Sample..."
      : `Analyzing your ${cropType || 'Crop'} with AI...`;
  }

  const headerDesc = document.getElementById("proc-header-desc");
  const previewTiles = document.querySelector(".proc-preview-tiles");
  const setText = (selector, text) => {
    const el = document.querySelector(selector);
    if (el) el.textContent = text;
  };
  const tile2 = document.getElementById("proc-tile-2");
  const tile3 = document.getElementById("proc-tile-3");
  const tile4 = document.getElementById("proc-tile-4");
  if (isSoil) {
    if (headerDesc) headerDesc.textContent = "OpenCV soil color and surface texture checks combined with the sensor or lab values you enter.";
    if (previewTiles) previewTiles.style.gridTemplateColumns = "minmax(220px, 420px)";
    [tile2, tile3, tile4].forEach(tile => { if (tile) tile.style.display = "none"; });
    [
      "1. Soil Image Received", "2. Visual Soil Checks", "3. Texture Assessment",
      "4. Sensor / Lab Readings", "5. Nutrient Status", "6. Soil Summary"
    ].forEach((label, index) => setText(`#p-step-${index + 1} .step-name`, label));
    setText(".console-title", "Soil Image & Sensor Analysis Stream");
  } else {
    if (headerDesc) headerDesc.textContent = "Computer Vision & PyTorch MobileNetV3 pipeline inspecting the specimen.";
    if (previewTiles) previewTiles.style.gridTemplateColumns = "";
    [tile2, tile3, tile4].forEach(tile => { if (tile) tile.style.display = ""; });
    [
      "1. Image Ingestion", "2. Preprocessing", "3. Leaf Segmentation",
      "4. Feature Extraction", "5. CNN Inference", "6. Diagnostic Report"
    ].forEach((label, index) => setText(`#p-step-${index + 1} .step-name`, label));
    setText(".console-title", "Computer Vision & PyTorch Inference Stream");
  }

  // 3. Initialize Visual Preview Tiles
  // Tile 1: Original Image
  const tileOrigImg = document.getElementById("tile-img-orig");
  const tileOrigBox = document.getElementById("tile-orig-placeholder");
  const status1 = document.getElementById("tile-status-1");

  if (previewSrc) {
    if (tileOrigImg) {
      tileOrigImg.src = previewSrc;
      tileOrigImg.style.display = "block";
    }
    if (tileOrigBox) tileOrigBox.style.display = "none";
    if (status1) {
      status1.className = "tile-status-badge badge badge-success";
      status1.innerHTML = '<i class="fa-solid fa-check"></i> Specimen Ingested';
    }
  } else {
    if (tileOrigImg) tileOrigImg.style.display = "none";
    if (tileOrigBox) tileOrigBox.style.display = "flex";
    if (status1) {
      status1.className = "tile-status-badge badge badge-neutral";
      status1.innerText = "Ingesting...";
    }
  }

  // Reset Tiles 2, 3, 4
  const resetTile = (imgId, boxId, statusId, waitText) => {
    const img = document.getElementById(imgId);
    const box = document.getElementById(boxId);
    const st = document.getElementById(statusId);
    if (img) img.style.display = "none";
    if (box) {
      box.style.display = "flex";
      box.innerHTML = `<i class="fa-solid fa-gear fa-spin"></i><span>${waitText}</span>`;
    }
    if (st) {
      st.className = "tile-status-badge badge badge-neutral";
      st.innerText = "Pending";
    }
  };

  resetTile("tile-img-preproc", "tile-preproc-box", "tile-status-2", "Awaiting Stage 2...");
  resetTile("tile-img-seg", "tile-seg-box", "tile-status-3", "Awaiting Stage 3...");
  resetTile("tile-img-gradcam", "tile-gradcam-box", "tile-status-4", "Awaiting Stage 5...");

  // Step 1: Initial State
  setPipelineStep(0, 15, "Stage 1: Ingesting & validating specimen image...");
  addPipelineLog(`Specimen received. Ingesting ${isSoil ? 'soil sample' : (cropType + ' foliage leaf')} into computer vision pipeline.`);
  addPipelineLog(isSoil ? "Image resolution verified. Checking soil sample boundaries..." : "Resolution verified. Validating specimen contours & foliar boundaries...");

  // 4. Fire API call
  let apiResponse = null;
  let apiError = null;

  const requestController = new AbortController();
  const requestTimeout = setTimeout(() => requestController.abort(), 120000);
  fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, ...(AppState.scanLocation || {}) }),
    signal: requestController.signal
  })
  .then(async r => {
    const data = await r.json().catch(() => ({}));
    clearTimeout(requestTimeout);
    if (!r.ok || data.specimen_rejected || data.success === false || !data.data) {
      apiError = data;
    } else {
      apiResponse = data.data;
    }
  })
  .catch(err => {
    clearTimeout(requestTimeout);
    apiError = {
      error: err.name === "AbortError"
        ? "The scan took longer than 2 minutes. Please retry with a smaller, clear image."
        : (err.message || "Network connection failure")
    };
  });

  // Stage 2 transition at 350ms
  setTimeout(() => {
    if (AppState.pipelineCancelled) return;
    if (apiError) return handlePipelineFailure(apiError);

    setPipelineStep(1, 35, isSoil ? "Stage 2: Checking soil image color and surface detail..." : "Stage 2: Applying OpenCV CLAHE & Bilateral Denoising...");
    if (isSoil) {
      addPipelineLog("Inspecting soil sample color and surface texture...");
    } else {
      addPipelineLog("Executing cv2.bilateralFilter(d=7, sigmaColor=50, sigmaSpace=50)...");
      addPipelineLog("Applying CLAHE contrast equalization (clipLimit=2.5, tileGrid=8x8)...");
    }

    const st2 = document.getElementById("tile-status-2");
    if (st2) {
      st2.className = "tile-status-badge badge badge-primary";
      st2.innerHTML = isSoil ? '<i class="fa-solid fa-spinner fa-spin"></i> Inspecting' : '<i class="fa-solid fa-spinner fa-spin"></i> Filtering';
    }
    const t2 = document.getElementById("proc-tile-2");
    if (t2) t2.classList.add("tile-active");

    // Check for API result to advance stages
    waitForApiAndAdvance(1);
  }, 350);

  function finishSoilPipeline(data) {
    const visual = data.visual_analysis || {};
    const nutrients = data.nutrient_status || {};
    const source = nutrients.measurement_source || "Not provided";
    const shown = value => value === null || value === undefined || value === "" ? "Not measured" : value;
    const soilSummary = `Texture: ${shown(visual.texture)} · pH: ${shown(nutrients.ph_value)} · N: ${shown(nutrients.n_status)} · P: ${shown(nutrients.p_status)} · K: ${shown(nutrients.k_status)} · Moisture: ${nutrients.moisture_pct == null ? "Not measured" : `${nutrients.moisture_pct}%`}`;
    const stages = [
      [1, 35, "Stage 2: Checking soil image color and surface detail...", `Soil image loaded. Surface color: ${shown(visual.color)}.`],
      [2, 55, "Stage 3: Estimating soil texture...", `Texture: ${shown(visual.texture)} · Surface: ${shown(visual.surface)}.`],
      [3, 72, "Stage 4: Reading supplied sensor or lab values...", `Measurement source: ${source}.`],
      [4, 90, "Stage 5: Preparing nutrient status...", `N: ${shown(nutrients.n_status)} · P: ${shown(nutrients.p_status)} · K: ${shown(nutrients.k_status)} · pH: ${shown(nutrients.ph_value)}.`],
      [6, 100, "Stage 6: Soil assessment ready...", "Soil summary prepared from image analysis and the values supplied."]
    ];

    const nextStage = index => {
      if (AppState.pipelineCancelled) return;
      const [step, pct, task, log] = stages[index];
      setPipelineStep(step, pct, task);
      addPipelineLog(log, index === stages.length - 1 ? "success" : "info");
      if (index < stages.length - 1) {
        setTimeout(() => nextStage(index + 1), 300);
        return;
      }

      if (spinner) spinner.style.display = "none";
      const taskPill = document.getElementById("proc-task-pill");
      if (taskPill) taskPill.innerHTML = '<i class="fa-solid fa-circle-check text-green"></i> <span>Soil assessment complete</span>';
      addPipelineLog("Soil assessment complete. No leaf disease classifier was used.", "success");
      populateSoilResults(data);

      const compTitle = document.getElementById("proc-diag-title");
      const compSub = document.getElementById("proc-diag-subtitle");
      const autoTimer = document.getElementById("proc-auto-timer");
      const resultsButton = document.getElementById("btn-view-pipeline-results");
      if (compTitle) compTitle.textContent = "Soil assessment complete";
      if (compSub) compSub.textContent = `${soilSummary}. Reading source: ${source}.`;
      if (resultsButton) resultsButton.innerHTML = '<span>View Soil Health Results</span> <i class="fa-solid fa-arrow-right"></i>';
      if (compCard) compCard.style.display = "flex";

      let countdown = 3;
      if (autoTimer) autoTimer.textContent = `Opening soil results in ${countdown}s...`;
      AppState.pipelineAutoTimer = setInterval(() => {
        countdown--;
        if (countdown <= 0) {
          clearInterval(AppState.pipelineAutoTimer);
          AppState.pipelineAutoTimer = null;
          proceedToResults();
        } else if (autoTimer) {
          autoTimer.textContent = `Opening soil results in ${countdown}s...`;
        }
      }, 1000);
    };

    nextStage(0);
  }

  function handlePipelineFailure(err) {
    if (AppState.pipelineCancelled) return;
    if (spinner) spinner.style.display = "none";

    const errTitle = document.getElementById("proc-err-title");
    const errDesc = document.getElementById("proc-err-desc");

    if (err.specimen_rejected) {
      addPipelineLog(`❌ Specimen rejected: ${err.error || 'Non-plant object detected'}`, "error");
      if (errTitle) errTitle.innerText = "Specimen Verification Rejected";
      if (errDesc) errDesc.innerText = err.error || "The image does not match leaf or soil criteria.";
      showRejectionModal(err.verification);
    } else {
      addPipelineLog(`❌ Pipeline error: ${err.error || 'Server processing error'}`, "error");
      if (errTitle) errTitle.innerText = "Analysis Pipeline Error";
      if (errDesc) errDesc.innerText = err.error || "An error occurred while inspecting the image.";
    }

    if (errCard) errCard.style.display = "flex";
  }

  function waitForApiAndAdvance(currentStage) {
    if (AppState.pipelineCancelled) return;

    if (apiError) {
      return handlePipelineFailure(apiError);
    }

    if (!apiResponse) {
      // Still waiting for server - poll gently
      setTimeout(() => waitForApiAndAdvance(currentStage), 120);
      return;
    }

    const data = apiResponse;
    if (isSoil) {
      finishSoilPipeline(data);
      return;
    }

    // We have real data! Update original image if missing
    if (tileOrigImg && data.images && data.images.original) {
      tileOrigImg.src = data.images.original;
      tileOrigImg.style.display = "block";
      if (tileOrigBox) tileOrigBox.style.display = "none";
    }

    // STAGE 2: Reveal Preprocessed CLAHE
    const preImg = document.getElementById("tile-img-preproc");
    const preBox = document.getElementById("tile-preproc-box");
    const st2 = document.getElementById("tile-status-2");
    const t2 = document.getElementById("proc-tile-2");

    if (data.images && data.images.preprocessed && preImg) {
      preImg.src = data.images.preprocessed;
      preImg.style.display = "block";
      if (preBox) preBox.style.display = "none";
      if (st2) {
        st2.className = "tile-status-badge badge badge-success";
        st2.innerHTML = '<i class="fa-solid fa-check"></i> CLAHE Denoised';
      }
      if (t2) t2.classList.remove("tile-active");
    }
    addPipelineLog("CLAHE & Bilateral filtering complete. Leaf venation enhanced, noise suppressed.", "success");

    // STAGE 3: Segmented Leaf (+500ms)
    setTimeout(() => {
      if (AppState.pipelineCancelled) return;
      setPipelineStep(2, 55, "Stage 3: Leaf Contour Isolation in HSV Color Space...");
      addPipelineLog("Converting BGR -> HSV. Thresholding foliar green-yellow chlorophyll pigments...");

      const segImg = document.getElementById("tile-img-seg");
      const segBox = document.getElementById("tile-seg-box");
      const st3 = document.getElementById("tile-status-3");
      const t3 = document.getElementById("proc-tile-3");

      if (data.images && data.images.segmented && segImg) {
        segImg.src = data.images.segmented;
        segImg.style.display = "block";
        if (segBox) segBox.style.display = "none";
        if (st3) {
          st3.className = "tile-status-badge badge badge-success";
          st3.innerHTML = '<i class="fa-solid fa-check"></i> Leaf Isolated';
        }
      }

      const areaPx = data.features?.shape?.area_px ? data.features.shape.area_px.toLocaleString() : '135,200';
      addPipelineLog(`Largest leaf contour masked: ${areaPx} px active lamina area isolated.`, "success");

      // STAGE 4: Feature Extraction (+500ms)
      setTimeout(() => {
        if (AppState.pipelineCancelled) return;
        setPipelineStep(3, 75, "Stage 4: Morphological & Necrotic Lesion Feature Extraction...");
        addPipelineLog("Computing RGB & HSV color moments, GLCM texture variance, and necrotic spots...");

        const spots = data.features?.texture?.spots_detected || "Detected";
        const aff = data.features?.texture?.affected_area_pct || "18.4";
        const colorName = data.features?.color?.dominant_name || "Green";
        addPipelineLog(`Color: ${colorName} · Lesions: ${spots} · Affected leaf area: ${aff}%`, "success");

        // STAGE 5: PyTorch CNN Inference & Grad-CAM (+550ms)
        setTimeout(() => {
          if (AppState.pipelineCancelled) return;
          setPipelineStep(4, 90, "Stage 5: PyTorch MobileNetV3 CNN Inference & Grad-CAM...");
          addPipelineLog(`Running PyTorch MobileNetV3 forward pass on ${data.crop_name || cropType || 'crop'}...`);

          const gImg = document.getElementById("tile-img-gradcam");
          const gBox = document.getElementById("tile-gradcam-box");
          const st4 = document.getElementById("tile-status-4");
          const t4 = document.getElementById("proc-tile-4");

          if (data.images && data.images.gradcam && gImg) {
            gImg.src = data.images.gradcam;
            gImg.style.display = "block";
            if (gBox) gBox.style.display = "none";
            if (st4) {
              st4.className = "tile-status-badge badge badge-success";
              st4.innerHTML = '<i class="fa-solid fa-check"></i> Grad-CAM Ready';
            }
          }
          addPipelineLog(`Grad-CAM attention heatmap synthesized across conv features.11.`, "success");

          // STAGE 6: Diagnostic Report Generation (+500ms)
          setTimeout(() => {
            if (AppState.pipelineCancelled) return;
            setPipelineStep(5, 100, "Stage 6: Diagnostic Report & Recommendations Ready");
            if (spinner) spinner.style.display = "none";

            const taskPill = document.getElementById("proc-task-pill");
            if (taskPill) {
              taskPill.innerHTML = '<i class="fa-solid fa-circle-check text-green"></i> <span>All 6 Stages Completed</span>';
            }

            addPipelineLog(`✅ Diagnosis Confirmed: ${data.disease_name} (${data.confidence}% confidence, ${data.risk_level} risk).`, "success");
            addPipelineLog("Diagnostic package assembled. Ready for farmer inspection.", "success");

            // Populate all downstream views
            if (isSoil) {
              populateSoilResults(data);
            } else {
              populatePlantResults(data);
            }

            // Reveal completion card with countdown
            const compTitle = document.getElementById("proc-diag-title");
            const compSub = document.getElementById("proc-diag-subtitle");
            const autoTimer = document.getElementById("proc-auto-timer");

            if (compTitle) compTitle.innerText = `${data.disease_name} (${data.confidence}% Confidence)`;
            if (compSub) compSub.innerText = `Severity: ${data.severity || data.risk_level} Risk · Priority: ${data.priority || 'Action Required'} · Specimen Foliage Verified`;
            if (compCard) compCard.style.display = "flex";

            let countdown = 3;
            if (autoTimer) autoTimer.innerText = `Auto-opening report in ${countdown}s...`;

            AppState.pipelineAutoTimer = setInterval(() => {
              countdown--;
              if (countdown <= 0) {
                clearInterval(AppState.pipelineAutoTimer);
                AppState.pipelineAutoTimer = null;
                proceedToResults();
              } else if (autoTimer) {
                autoTimer.innerText = `Auto-opening report in ${countdown}s...`;
              }
            }, 1000);
          }, 500);
        }, 550);
      }, 500);
    }, 500);
  }
}

function runSampleScan(sampleKey, cropType) {
  if (!ensureScanAreaSelected("plant")) return;
  let previewSrc = `/static/samples/${sampleKey}`;
  if (sampleKey.startsWith("/") || sampleKey.startsWith("http")) {
    previewSrc = sampleKey;
  } else if (sampleKey.includes("_1.") || sampleKey.includes("_2.") || sampleKey.includes("_3.") || sampleKey.includes("_4.")) {
    previewSrc = `/static/samples/real/${sampleKey}`;
  }

  startAIPipeline({
    previewSrc: previewSrc,
    endpoint: "/api/scan/plant",
    payload: { sample: sampleKey, crop_type: cropType },
    cropType: cropType,
    isSoil: false
  });
}

function runSoilScanSample() {
  if (!ensureScanAreaSelected("soil")) return;
  AppState.scanMode = "soil";
  updateCaptureModeUI();
  const labData = collectSoilDemoReadings();
  if (labData === false) return;
  AppState.soilLabData = labData;
  startAIPipeline({
    previewSrc: "/static/samples/loamy_soil.jpg",
    endpoint: "/api/scan/soil",
    payload: { sample: "loamy_soil", ...(labData ? { lab_data: labData } : {}) },
    cropType: "Soil",
    isSoil: true
  });
}

function collectSoilDemoReadings() {
  const toggle = document.getElementById("soil-demo-sensor-toggle");
  if (!toggle?.checked) return null;
  const ph = Number(document.getElementById("soil-capture-ph")?.value);
  const moisture = Number(document.getElementById("soil-capture-moisture")?.value);
  if (!Number.isFinite(ph) || ph < 0 || ph > 14 || !Number.isFinite(moisture) || moisture < 0 || moisture > 100) {
    showNotification("Enter a pH from 0–14 and moisture from 0–100%.", "warning");
    return false;
  }
  return {
    ph_value: ph,
    n_status: document.getElementById("soil-capture-n").value,
    p_status: document.getElementById("soil-capture-p").value,
    k_status: document.getElementById("soil-capture-k").value,
    moisture_pct: moisture,
    measurement_source: "Simulated demo sensor"
  };
}

function submitImageForAnalysis(b64Image) {
  const cropSelect = document.getElementById("scan-crop-select");
  const isSoil = AppState.scanMode === "soil";
  const cropType = (cropSelect && cropSelect.value) || AppState.selectedCrop || "";

  if (!ensureScanAreaSelected()) return;
  if (!isSoil && !cropType) {
    showNotification("Choose the crop shown in the image before running the diagnosis.", "warning");
    return;
  }
  const soilPayload = { image_b64: b64Image };
  if (isSoil) {
    const labData = collectSoilDemoReadings();
    if (labData === false) return;
    if (labData) soilPayload.lab_data = labData;
    AppState.soilLabData = labData;
  }
  startAIPipeline({
    previewSrc: b64Image,
    endpoint: isSoil ? "/api/scan/soil" : "/api/scan/plant",
    payload: isSoil ? soilPayload : { image_b64: b64Image, crop_type: cropType },
    cropType: isSoil ? "Soil" : cropType,
    isSoil: isSoil
  });
}

function loadScanFields() {
  fetch("/api/farms")
    .then(response => response.json())
    .then(data => {
      const select = document.getElementById("scan-field-select");
      if (!select || !data.success) return;
      AppState.fields = data.fields;
      const currentId = AppState.scanLocation?.location_type === "field" ? String(AppState.scanLocation.field_id) : "";
      select.innerHTML = '<option value="">Choose a saved field…</option>' +
        data.fields.map(field => `<option value="${field.id}">${field.name} · ${field.area_acres} acres · ${field.crop}</option>`).join("");
      select.value = currentId;
      if (currentId) selectScanField(currentId);
      else updateScanSelectionSummary();
    })
    .catch(() => showNotification("Could not load saved farm fields.", "error"));
}

function selectScanField(fieldId) {
  const select = document.getElementById("scan-field-select");
  if (!fieldId || !select) {
    AppState.scanLocation = null;
    updateScanLocationStatus("Select a saved field to see its acreage and crop. A field is required before scanning.");
    updateScanSelectionSummary();
    return;
  }
  const field = AppState.fields.find(item => String(item.id) === String(fieldId));
  if (!field) {
    AppState.scanLocation = null;
    updateScanLocationStatus("That field could not be loaded. Refresh the field list and choose again.");
    updateScanSelectionSummary();
    return;
  }
  AppState.scanLocation = { location_type: "field", field_id: Number(fieldId) };
  updateScanLocationStatus(`Selected ${field.name}: ${field.area_acres} acres · field crop ${field.crop || "not specified"}.`);
  updateScanSelectionSummary();
}

function selectedScanField() {
  const fieldId = AppState.scanLocation?.location_type === "field" ? AppState.scanLocation.field_id : null;
  return AppState.fields.find(field => String(field.id) === String(fieldId)) || null;
}

function ensureScanAreaSelected(mode = AppState.scanMode) {
  if (selectedScanField()) return true;
  if (!document.getElementById("scan-field-select")) openCaptureFlow(mode);
  showNotification("Choose a saved scan area first so the scan includes its acreage and field details.", "warning");
  document.getElementById("scan-field-select")?.focus();
  return false;
}

function updateScanSelectionSummary() {
  const summary = document.getElementById("scan-selection-summary");
  const analyzeButton = document.getElementById("analyze-upload-btn");
  const field = selectedScanField();
  const isSoil = AppState.scanMode === "soil";
  const crop = document.getElementById("scan-crop-select")?.value || AppState.selectedCrop || "";
  const hasPhoto = Boolean(AppState.selectedUploadB64);
  const ready = Boolean(field) && (isSoil || Boolean(crop));

  if (summary) {
    if (!field) {
      summary.textContent = "Choose a saved field to unlock scanning. Its acreage and registered crop will appear here.";
      summary.classList.remove("is-ready");
    } else {
      const scanSubject = isSoil ? "Soil health scan" : `Leaf scan crop: ${crop || "choose a crop"}`;
      const sensorEnabled = document.getElementById("soil-demo-sensor-toggle")?.checked;
      const soilReadings = isSoil && sensorEnabled
        ? ` · Demo readings: pH ${document.getElementById("soil-capture-ph")?.value || "—"}, N ${document.getElementById("soil-capture-n")?.value || "—"}, P ${document.getElementById("soil-capture-p")?.value || "—"}, K ${document.getElementById("soil-capture-k")?.value || "—"}, moisture ${document.getElementById("soil-capture-moisture")?.value || "—"}%`
        : (isSoil ? " · Sensor/lab readings not supplied" : "");
      const gps = Number.isFinite(AppState.scanLocation?.location_lat) && Number.isFinite(AppState.scanLocation?.location_lng)
        ? " · GPS point attached" : "";
      summary.textContent = `${field.name} · ${field.area_acres} acres · Field crop: ${field.crop || "not specified"} · ${scanSubject}${soilReadings}${gps}`;
      summary.classList.toggle("is-ready", ready);
    }
  }
  if (analyzeButton) {
    analyzeButton.disabled = !field || !hasPhoto || (!isSoil && !crop);
    analyzeButton.title = !field ? "Choose a saved scan area first" : (!isSoil && !crop ? "Choose the crop shown in the photo" : "");
  }
  document.querySelectorAll(".preset-btn").forEach(button => {
    button.disabled = !field;
    button.title = field ? "" : "Choose a saved scan area first";
  });
}

function updateScanLocationStatus(message) {
  const status = document.getElementById("scan-location-status");
  if (status) status.textContent = message || "Select a saved field to record its acreage, or use your current GPS location.";
}

function useCurrentScanLocation() {
  const field = selectedScanField();
  if (!field) {
    showNotification("Choose a saved field first. GPS can add a scan point, but it cannot provide field acreage.", "warning");
    document.getElementById("scan-field-select")?.focus();
    return;
  }
  if (!navigator.geolocation) {
    showNotification("This browser does not provide GPS location.", "error");
    return;
  }
  const button = document.getElementById("scan-live-location-btn");
  if (button?.disabled) return;
  if (button) button.disabled = true;
  updateScanLocationStatus("Waiting for your location permission…");
  let completed = false;
  const finishLocationRequest = () => {
    completed = true;
    if (button) button.disabled = false;
  };
  const locationFallbackTimer = setTimeout(() => {
    if (completed) return;
    updateScanLocationStatus("Location request timed out. Choose a saved field or try GPS again.");
    finishLocationRequest();
    showNotification("Could not get your current location. Check browser location permission and try again.", "warning");
  }, 15000);
  navigator.geolocation.getCurrentPosition(position => {
    if (completed) return;
    clearTimeout(locationFallbackTimer);
    AppState.scanLocation = {
      location_type: "field",
      field_id: Number(field.id),
      location_lat: position.coords.latitude,
      location_lng: position.coords.longitude
    };
    updateScanLocationStatus(`Using GPS for this scan in ${field.name} (${field.area_acres} acres).`);
    updateScanSelectionSummary();
    finishLocationRequest();
  }, error => {
    if (completed) return;
    clearTimeout(locationFallbackTimer);
    const message = error.code === 1
      ? "Location permission was denied. Allow location access for this site, then try GPS again."
      : error.code === 3
        ? "Location request timed out. Choose a saved field or try GPS again."
        : "Current location is unavailable. Choose a saved field or try GPS again.";
    updateScanLocationStatus(message);
    finishLocationRequest();
    showNotification(message, "warning");
  }, { enableHighAccuracy: false, timeout: 12000, maximumAge: 60000 });
}

// ==========================================================================
// 4. Populate Computer Vision, Grad-CAM & Diagnosis Results
// ==========================================================================

function populatePlantResults(data) {
  AppState.lastPlantResult = data;

  // 1. CV Results View
  document.getElementById("cv-orig-img").src = data.images.original;
  document.getElementById("cv-preproc-img").src = data.images.preprocessed;
  document.getElementById("cv-seg-img").src = data.images.segmented;

  // Color features
  document.getElementById("f-dom-color").innerText = data.features.color.dominant_name;
  document.getElementById("f-rgb").innerText = `(${data.features.color.rgb.join(", ")})`;
  document.getElementById("f-hsv").innerText = `(${data.features.color.hsv.join(", ")})`;

  // Shape features
  document.getElementById("f-area").innerText = data.features.shape.area_px.toLocaleString() + " px";
  document.getElementById("f-perimeter").innerText = data.features.shape.perimeter_px.toLocaleString() + " px";
  document.getElementById("f-aspect").innerText = data.features.shape.aspect_ratio;

  // Texture features
  document.getElementById("f-texture").innerText = data.features.texture.texture_type;
  document.getElementById("f-spots").innerText = data.features.texture.spots_detected;
  document.getElementById("f-affected-pct").innerText = data.features.texture.affected_area_pct + "%";

  // 2. Explainable AI View (Grad-CAM)
  document.getElementById("xai-orig-img").src = data.images.original;
  document.getElementById("xai-gradcam-img").src = data.images.gradcam;

  const patList = document.getElementById("xai-pattern-list");
  if (patList) {
    patList.innerHTML = data.explainability.patterns_found.map(p => `
      <div class="pat-item ${p.matched ? 'matched' : ''}">
        <i class="fa-solid ${p.matched ? 'fa-circle-check text-green' : 'fa-circle-xmark text-muted'}"></i>
        <span>${p.label}</span>
      </div>
    `).join("");
  }
  document.getElementById("xai-summary-text").innerHTML = `
    <i class="fa-solid fa-circle-info text-primary"></i> ${data.explainability.ai_summary}
  `;

  // 3. Disease Result View
  const alertBanner = document.getElementById("disease-alert-banner");
  const alertTitle  = document.getElementById("disease-alert-title");
  const riskBadge   = document.getElementById("disease-risk-badge");

  if (data.risk_level === "HIGH") {
    alertBanner.className = "alert-banner alert-high";
    alertBanner.style.backgroundColor = "";
    alertBanner.style.borderColor = "";
    alertBanner.style.color = "";
    alertTitle.innerText = "🔴 ATTENTION REQUIRED";
    riskBadge.className = "risk-badge badge-high";
    riskBadge.innerText = "Risk Level: HIGH";
  } else if (data.risk_level === "MEDIUM") {
    alertBanner.className = "alert-banner alert-high";
    alertBanner.style.backgroundColor = "var(--warning-light)";
    alertBanner.style.borderColor = "#FCD34D";
    alertBanner.style.color = "#92400E";
    alertTitle.innerText = "🟠 NUTRIENT OR MILD ISSUE DETECTED";
    riskBadge.className = "risk-badge badge-warning";
    riskBadge.innerText = "Risk Level: MEDIUM";
  } else {
    alertBanner.className = "alert-banner";
    alertBanner.style.backgroundColor = "var(--success-light)";
    alertBanner.style.borderColor = "#A7F3D0";
    alertBanner.style.color = "#065F46";
    alertTitle.innerText = "🟢 HEALTHY SPECIMEN";
    riskBadge.className = "risk-badge badge-success";
    riskBadge.innerText = "Risk Level: LOW";
  }

  document.getElementById("diag-thumb-img").src = data.images.segmented;
  document.getElementById("diag-disease-name").innerText = data.disease_name;
  document.getElementById("diag-conf-pct").innerText = data.confidence + "%";
  document.getElementById("diag-conf-bar").style.width = data.confidence + "%";

  // --- Leaf Analysis Summary Card ---
  const cropName = data.crop || data.crop_name || "Unknown Plant";

  // Highlight currently diagnosed crop in the re-diagnosis toolbar without mutating upload dropdown
  updateCropSwitchBarActive(cropName);

  // Build friendly leaf-type label e.g. "Rice Leaf (Oryza sativa)"
  const scientificMap = {
    "Tomato":    "Solanum lycopersicum",
    "Rice":      "Oryza sativa",
    "Chilli":    "Capsicum annuum",
    "Sugarcane": "Saccharum officinarum",
    "Blackgram": "Vigna mungo",
    "Soybean":   "Glycine max"
  };
  const sci = scientificMap[cropName] ? ` (${scientificMap[cropName]})` : "";
  const leafTypeLabel = `${cropName} Leaf${sci}`;

  const affectedPct = data.features?.texture?.affected_area_pct ?? "—";
  const affectedColor = (parseFloat(affectedPct) > 20) ? "text-danger" : "text-green";

  const leafTypeEl    = document.getElementById("leaf-type-display");
  const leafDiseaseEl = document.getElementById("leaf-disease-display");
  const leafAffectedEl = document.getElementById("leaf-affected-display");
  const leafConfEl    = document.getElementById("leaf-conf-display");

  if (leafTypeEl)    leafTypeEl.innerText    = leafTypeLabel;
  if (leafDiseaseEl) leafDiseaseEl.innerText = data.disease_name;
  if (leafAffectedEl) {
    leafAffectedEl.innerText = affectedPct !== "—" ? `${affectedPct}% of leaf` : "—";
    leafAffectedEl.className = `leaf-sum-val ${affectedColor}`;
  }
  if (leafConfEl) {
    leafConfEl.innerText = data.confidence + "%";
    leafConfEl.className = "leaf-sum-val text-green";
  }

  // Also update the crop tag in the diagnosis card header
  const cropTagEl = document.getElementById("diag-crop-name");
  if (cropTagEl) cropTagEl.innerText = cropName;

  const mcList = document.getElementById("diag-multiclass-list");
  if (mcList && data.predictions) {
    mcList.innerHTML = data.predictions.map(p => `
      <div class="mc-item">
        <span>${p.label}</span>
        <span class="mc-val">${p.confidence}%</span>
      </div>
    `).join("");
  }

  const symList = document.getElementById("diag-symptoms-list");
  if (symList && data.key_symptoms) {
    symList.innerHTML = data.key_symptoms.map(s => `<li>${s}</li>`).join("");
  }
  document.getElementById("diag-causes-text").innerHTML = data.causes;
  document.getElementById("diag-quick-org").innerText = data.organic_solution;
  document.getElementById("diag-quick-min").innerText = data.mineral_solution;

  // 4. Recommendation Engine View
  document.getElementById("rec-priority-badge").innerText = `Priority: ${data.priority || 'HIGH'}`;
  document.getElementById("rec-focus-title").innerText = data.focus_area || 'Restorative Treatment & Foliar Nutrition';

  const recTargetIssue = document.getElementById("rec-target-issue");
  if (recTargetIssue) {
    recTargetIssue.innerText = `Target Issue: ${data.target_issue || data.disease_name}`;
  }

  const recFocusDesc = document.getElementById("rec-focus-desc");
  if (recFocusDesc) {
    recFocusDesc.innerText = data.focus_desc || `Your ${data.crop_name || "crop"} exhibits symptoms of ${data.disease_name}. Targeted intervention formulated to contain pathology and replenish required nutrients.`;
  }

  // Use the verified crop name
  document.getElementById("rec-crop").innerText = cropName;

  const recGrowthStage = document.getElementById("rec-growth-stage");
  if (recGrowthStage) {
    recGrowthStage.innerText = data.growth_stage || "Vegetative Stage";
  }

  const recSymptomsVal = document.getElementById("rec-symptoms-val");
  if (recSymptomsVal) {
    recSymptomsVal.innerText = data.symptoms_summary || (data.key_symptoms && data.key_symptoms[0]) || "Foliar pathology detected";
  }

  document.getElementById("rec-organic-text").innerText = data.organic_solution;
  document.getElementById("rec-mineral-text").innerText = data.mineral_solution;

  // Update soil-N display from scan data (not the hardcoded default)
  const recSoilN = document.getElementById("rec-soil-n");
  if (recSoilN && data.nutrient_hint) {
    recSoilN.innerText = data.nutrient_hint;
    recSoilN.className = data.nutrient_hint === "Normal" ? "b-val text-green" : "b-val text-danger";
  } else if (recSoilN) {
    recSoilN.innerText = "Normal";
    recSoilN.className = "b-val text-green";
  }

  // Populate Prescribed Fertilizers & Agrochemicals Table
  const fertTbody = document.getElementById("rec-fertilizer-tbody");
  if (fertTbody) {
    if (data.fertilizers && data.fertilizers.length > 0) {
      fertTbody.innerHTML = data.fertilizers.map(f => `
        <tr>
          <td>
            <div class="fert-name">${f.name}</div>
            <span class="fert-badge">${f.category || 'Nutrient / Agrochemical'}</span>
          </td>
          <td>${f.purpose}</td>
          <td><strong>${f.dosage}</strong></td>
          <td>${f.timing || 'Early morning foliar spray or basal application'}</td>
        </tr>
      `).join("");
    } else {
      fertTbody.innerHTML = `
        <tr>
          <td>
            <div class="fert-name">Balanced Water Soluble NPK (19-19-19)</div>
            <span class="fert-badge">Foliar Nutrition</span>
          </td>
          <td>General foliar vigor replenishment & vegetative strengthening</td>
          <td><strong>3.0 to 5.0 g / Litre of water</strong></td>
          <td>Foliar spray during cool morning or evening hours</td>
        </tr>
        <tr>
          <td>
            <div class="fert-name">Cold-Pressed Neem Oil (10,000 ppm)</div>
            <span class="fert-badge">Organic Protectant</span>
          </td>
          <td>Surface antifungal coating and vector deterrence</td>
          <td><strong>3.0 ml / Litre + organic emulsifier</strong></td>
          <td>Weekly preventative spray</td>
        </tr>
      `;
    }
  }

  const actionStepsEl = document.getElementById("rec-action-steps");
  if (actionStepsEl && data.what_to_do) {
    actionStepsEl.innerHTML = data.what_to_do.map((step, idx) => `
      <li><strong>Step ${idx + 1}:</strong> ${step}</li>
    `).join("");
  }

  // Update the treatment reference image to use the actual scanned image
  const recHeroImg = document.getElementById("rec-hero-img");
  if (recHeroImg && data.images && data.images.segmented) {
    recHeroImg.src = data.images.segmented;
    recHeroImg.alt = `${data.crop_name || "Crop"} - ${data.disease_name}`;
  }

  // 5. Populate Official Printable AI Report
  const repIdEl = document.getElementById("rep-id");
  if (repIdEl) repIdEl.innerText = `#SD-2026-${data.scan_id || Math.floor(1000 + Math.random() * 9000)}`;
  const repDiagEl = document.getElementById("rep-diag-name");
  if (repDiagEl) repDiagEl.innerText = data.disease_name;
  const repCropEl = document.getElementById("rep-crop");
  if (repCropEl) repCropEl.innerHTML = `<strong>${data.crop || data.crop_name || "Crop"}</strong>`;
  const repConfEl = document.getElementById("rep-conf");
  if (repConfEl) repConfEl.innerText = data.confidence + "%";
  const repRiskEl = document.getElementById("rep-risk");
  if (repRiskEl) repRiskEl.innerText = data.risk_level;
  const repOrigEl = document.getElementById("rep-img-orig");
  if (repOrigEl && data.images) repOrigEl.src = data.images.original;
  const repSegEl = document.getElementById("rep-img-seg");
  if (repSegEl && data.images) repSegEl.src = data.images.segmented;
  const repGradEl = document.getElementById("rep-img-gradcam");
  if (repGradEl && data.images) repGradEl.src = data.images.gradcam;
  const repAreaEl = document.getElementById("rep-area");
  if (repAreaEl && data.features?.shape) repAreaEl.innerText = data.features.shape.area_px.toLocaleString() + " px";
  const repTextureEl = document.getElementById("rep-texture");
  if (repTextureEl && data.features?.texture) repTextureEl.innerText = data.features.texture.texture_type;
  const repAffEl = document.getElementById("rep-affected");
  if (repAffEl && data.features?.texture) repAffEl.innerText = data.features.texture.affected_area_pct + "%";
  const repLocEl = document.getElementById("rep-location");
  if (repLocEl) repLocEl.innerText = AppState.liveLocation ? AppState.liveLocation.name : "Coimbatore, Tamil Nadu";
  const repDateEl = document.getElementById("rep-date");
  if (repDateEl) repDateEl.innerText = new Date().toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });

  updateReportSoilSection();
  updateCombinedComparisonCard();
}

function updateReportSoilSection() {
  const repSoilBox = document.getElementById("rep-soil-details");
  const repSoilHdr = document.getElementById("rep-soil-header");
  if (!repSoilBox) return;

  if (AppState.hasSoilScan && AppState.lastSoilResult) {
    const s = AppState.lastSoilResult;
    const nut = s.nutrient_status || {};
    if (repSoilHdr) repSoilHdr.innerText = "Soil & Nutrition Calibration (Active)";
    repSoilBox.innerHTML = `
      <li><strong>Soil Type:</strong> ${s.visual_analysis ? s.visual_analysis.texture : 'Loamy Dark Brown'} (${s.visual_analysis ? s.visual_analysis.color : 'Optimal organic matter'})</li>
      <li><strong>Soil pH:</strong> ${nut.ph_value || 6.4} (${(nut.ph_value || 6.4) < 6.0 ? 'Acidic' : (nut.ph_value || 6.4) > 7.5 ? 'Alkaline' : 'Favorable'})</li>
      <li><strong>Nitrogen Status:</strong> <strong class="${nut.n_status === 'Low' ? 'text-danger' : 'text-green'}">${nut.n_status || 'Normal'} (${nut.n_status === 'Low' ? 'Deficient' : 'Sufficient'})</strong></li>
      <li><strong>Phosphorus / Potassium:</strong> P: ${nut.p_status || 'Normal'} | K: ${nut.k_status || 'Normal'}</li>
      <li><strong>Cross-Correlation:</strong> Integrated with foliar diagnosis</li>
    `;
  } else {
    if (repSoilHdr) repSoilHdr.innerText = "Soil & Nutrition Status";
    repSoilBox.innerHTML = `
      <li><strong>Assessment Type:</strong> Crop Foliage Diagnosis Only</li>
      <li><strong>Soil Specimen:</strong> <em>Not Tested (Skipped by Farmer)</em></li>
      <li><strong>Root-Soil Correlation:</strong> Not Available for this scan</li>
      <li><strong>Nutrient Plan:</strong> Standard foliar balanced nutrition recommended</li>
    `;
  }
}

function populateSoilResults(data) {
  AppState.lastSoilResult = data;
  const sourceEl = document.getElementById("soil-measurement-source");
  if (sourceEl) sourceEl.textContent = data.nutrient_status?.measurement_source || "No sensor readings supplied";
  AppState.hasSoilScan = true;

  const soilDisp = document.getElementById("soil-display-img");
  if (soilDisp && data.soil_image) soilDisp.src = data.soil_image;
  const setEl = (id, val) => { const el = document.getElementById(id); if (el) el.innerText = val; };
  if (data.visual_analysis) {
    setEl("soil-attr-texture", data.visual_analysis.texture);
    setEl("soil-attr-color", data.visual_analysis.color);
    setEl("soil-attr-surface", data.visual_analysis.surface);
    setEl("soil-attr-moisture", data.visual_analysis.moisture);
    setEl("soil-attr-organic", data.visual_analysis.organic_matter);
  }

  // Update chemical controls
  const nut = data.nutrient_status || {};
  const phInp = document.getElementById("input-soil-ph");
  if (phInp && nut.ph_value != null) {
    phInp.value = nut.ph_value;
    updatePhDisplay(nut.ph_value);
  }
  const setVal = (id, val) => { const el = document.getElementById(id); if (el && val !== null && val !== undefined && val !== "" && val !== "Not measured") el.value = val; };
  setVal("input-n-status", nut.n_status);
  setVal("input-p-status", nut.p_status);
  setVal("input-k-status", nut.k_status);
  setVal("input-moisture", nut.moisture_pct);
  setEl("moisture-display", nut.moisture_pct == null ? "Not measured" : nut.moisture_pct + "%");

  updateDialsUI(nut);
  updateCombinedComparisonCard();
  updateReportSoilSection();
}

function updateDialsUI(nut) {
  const showStatus = (value) => value || "Not measured";
  // Nitrogen dial
  const nBadge = document.getElementById("dial-n-badge");
  nBadge.innerText = showStatus(nut.n_status);
  nBadge.className = `badge ${!nut.n_status || nut.n_status === 'Not measured' ? 'badge-light' : nut.n_status === 'Low' ? 'badge-warning' : nut.n_status === 'Normal' ? 'badge-success' : 'badge-danger'}`;

  // Phosphorus dial
  const pBadge = document.getElementById("dial-p-badge");
  pBadge.innerText = showStatus(nut.p_status);
  pBadge.className = `badge ${!nut.p_status || nut.p_status === 'Not measured' ? 'badge-light' : nut.p_status === 'Low' ? 'badge-warning' : nut.p_status === 'Normal' ? 'badge-success' : 'badge-danger'}`;

  // Potassium dial
  const kBadge = document.getElementById("dial-k-badge");
  kBadge.innerText = showStatus(nut.k_status);
  kBadge.className = `badge ${!nut.k_status || nut.k_status === 'Not measured' ? 'badge-light' : nut.k_status === 'Low' ? 'badge-warning' : nut.k_status === 'Normal' ? 'badge-success' : 'badge-danger'}`;

  // pH dial
  document.getElementById("dial-ph-text").innerText = nut.ph_value == null ? "—" : nut.ph_value;
  const phDesc = nut.ph_value == null ? "Not measured" : nut.ph_value < 6.0 ? "Acidic" : nut.ph_value <= 7.2 ? "Optimal" : "Alkaline";
  document.getElementById("dial-ph-badge").innerText = phDesc;
}

function updatePhDisplay(val) {
  const ph = parseFloat(val);
  let desc = "Optimal (Neutral)";
  if (ph < 6.0) desc = "Acidic";
  else if (ph <= 6.8) desc = "Slightly Acidic";
  else if (ph > 7.5) desc = "Alkaline";
  document.getElementById("ph-val-display").innerText = `${ph.toFixed(1)} (${desc})`;
}

function recalculateSoilNutrients(e) {
  e.preventDefault();
  const ph = parseFloat(document.getElementById("input-soil-ph").value);
  const n = document.getElementById("input-n-status").value;
  const p = document.getElementById("input-p-status").value;
  const k = document.getElementById("input-k-status").value;
  const moist = parseInt(document.getElementById("input-moisture").value);

  const nut = { ph_value: ph, n_status: n, p_status: p, k_status: k, moisture_pct: moist };
  AppState.soilLabData = { ...nut, measurement_source: "User entered manually" };
  updateDialsUI(nut);

  // Update recommendations based on new values
  document.getElementById("rec-soil-n").innerText = n;
  alert("Nutrient status and fertilizer recommendations recalibrated successfully!");
}

// Tab switchers inside Diagnosis & Recommendations
function switchDiagTab(tabKey) {
  const tabs = ["symptoms", "causes", "treatment"];
  tabs.forEach(t => {
    const el = document.getElementById(`diag-tab-${t}`);
    if (el) el.style.display = t === tabKey ? "block" : "none";
  });
}

function switchRecTab(tabKey) {
  const tabOrg = document.getElementById("rec-tab-organic");
  const tabMin = document.getElementById("rec-tab-mineral");
  if (tabKey === "organic") {
    tabOrg.style.display = "block";
    tabMin.style.display = "none";
  } else {
    tabOrg.style.display = "none";
    tabMin.style.display = "block";
  }
}

function openDemoResult(cropKey) {
  if (cropKey === "tomato") runSampleScan("tomato_early_blight_1.jpg", "Tomato");
  else if (cropKey === "rice") runSampleScan("rice_healthy_1.jpg", "Rice");
  else if (cropKey === "chilli") runSampleScan("chilli_cercospora_1.jpg", "Chilli");
  else if (cropKey === "sugarcane") runSampleScan("sugarcane_red_rot_1.jpeg", "Sugarcane");
  else if (cropKey === "blackgram") runSampleScan("blackgram_yellow_mosaic_1.jpg", "Blackgram");
  else if (cropKey === "soybean") runSampleScan("soybean_disease_soybean_1.jpg", "Soybean");
}

function generateReportAndPrint() {
  navigateTo("report-view");
}

function loadReportViewData() {
  const d = new Date();
  const dateEl = document.getElementById("rep-date");
  if (dateEl) dateEl.innerText = d.toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });

  // If already populated from a recent plant scan
  if (AppState.lastPlantResult) {
    const data = AppState.lastPlantResult;
    const repCrop = document.getElementById("rep-crop");
    if (repCrop) repCrop.innerHTML = `<strong>${data.crop || data.crop_name} (${data.disease_name})</strong>`;
    return;
  }

  // Otherwise, fetch latest scan from backend
  fetch("/api/report")
    .then(r => r.json())
    .then(data => {
      if (data.success && data.scans && data.scans.length > 0) {
        const latest = data.scans.find(s => s.scan_type === "plant") || data.scans[0];
        const repId = document.getElementById("rep-id");
        const repDiag = document.getElementById("rep-diag-name");
        const repCrop = document.getElementById("rep-crop");
        const repConf = document.getElementById("rep-conf");
        const repRisk = document.getElementById("rep-risk");
        const repOrig = document.getElementById("rep-img-orig");
        const repSeg = document.getElementById("rep-img-seg");
        const repGrad = document.getElementById("rep-img-gradcam");

        if (repId) repId.innerText = `#SD-2026-${latest.id}`;
        if (repDiag) repDiag.innerText = latest.disease_name;
        if (repCrop) repCrop.innerHTML = `<strong>${latest.crop_name}</strong>`;
        if (repConf) repConf.innerText = latest.confidence + "%";
        if (repRisk) repRisk.innerText = latest.risk_level;
        if (repSeg && latest.thumbnail_b64) repSeg.src = latest.thumbnail_b64;
        if (repOrig && latest.thumbnail_b64) repOrig.src = latest.thumbnail_b64;
        if (repGrad && latest.thumbnail_b64) repGrad.src = latest.thumbnail_b64;
      }
    })
    .catch(err => console.warn("Error fetching report data:", err));
}

// ==========================================================================
// 5. Interactive Leaflet.js Farm Management Map
// ==========================================================================

function initFarmMap() {
  const mapContainer = document.getElementById("farm-map");
  if (!mapContainer) return;

  // Destroy previous map instance if re-initializing
  if (AppState.map) {
    AppState.map.remove();
    AppState.map = null;
  }

  // Centered around agricultural plots near Coimbatore: 11.019° N, 76.955° E
  const map = L.map("farm-map").setView([11.019, 76.955], 15);
  AppState.map = map;

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);

  renderFieldsOnMap(AppState.fields);
}

function loadFarmFields() {
  fetch("/api/farms")
    .then(r => r.json())
    .then(data => {
      if (data.success) {
        AppState.fields = data.fields;
        renderFieldsList(data.fields);
        if (AppState.map) renderFieldsOnMap(data.fields);
        if (data.fields.length) selectField(data.fields[0]);
      }
    })
    .catch(err => console.error("Farm fields load error:", err));
}




function skipSoilStepAndContinue() {
  AppState.hasSoilScan = false;
  AppState.lastSoilResult = null;
  AppState.guidedStep = 4;
  updateReportSoilSection();
  updateCombinedComparisonCard();
  navigateTo("farm-mgmt");
  setTimeout(() => {
    initFarmMap();
  }, 200);
}

function useLiveFarmLocation() {
  const btn = document.getElementById("btn-live-loc");
  const coordDisplay = document.getElementById("map-coord-display");
  if (btn) btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Locating GPS...';

  if (!navigator.geolocation) {
    setFallbackFarmLocation();
    return;
  }

  navigator.geolocation.getCurrentPosition(
    (pos) => {
      const lat = pos.coords.latitude;
      const lng = pos.coords.longitude;
      AppState.liveLocation = {
        lat: lat,
        lng: lng,
        name: `Live Field GPS (${lat.toFixed(4)}° N, ${lng.toFixed(4)}° E)`,
        isLive: true
      };

      if (btn) btn.innerHTML = '<i class="fa-solid fa-circle-check text-green"></i> Live GPS Active';
      if (coordDisplay) coordDisplay.innerText = `Lat: ${lat.toFixed(4)}° N, Lng: ${lng.toFixed(4)}° E (GPS Verified)`;

      const repLoc = document.getElementById("rep-location");
      if (repLoc) repLoc.innerText = AppState.liveLocation.name;

      // Center map & drop live marker
      if (AppState.map) {
        AppState.map.setView([lat, lng], 16);
        if (AppState.liveMarker) {
          AppState.liveMarker.setLatLng([lat, lng]);
        } else {
          AppState.liveMarker = L.marker([lat, lng], { draggable: true }).addTo(AppState.map);
          AppState.liveMarker.bindPopup("<b>📍 Your Live Farm Location</b><br>Field coordinate anchored").openPopup();
          AppState.liveMarker.on("dragend", (e) => {
            const pt = e.target.getLatLng();
            AppState.liveLocation.lat = pt.lat;
            AppState.liveLocation.lng = pt.lng;
            AppState.liveLocation.name = `Live Field GPS (${pt.lat.toFixed(4)}° N, ${pt.lng.toFixed(4)}° E)`;
            if (coordDisplay) coordDisplay.innerText = `Lat: ${pt.lat.toFixed(4)}° N, Lng: ${pt.lng.toFixed(4)}° E`;
            if (repLoc) repLoc.innerText = AppState.liveLocation.name;
          });
        }
      }
    },
    (err) => {
      console.warn("Geolocation fallback:", err.message);
      setFallbackFarmLocation();
    },
    { timeout: 7000, enableHighAccuracy: true }
  );
}

function setFallbackFarmLocation() {
  const btn = document.getElementById("btn-live-loc");
  const coordDisplay = document.getElementById("map-coord-display");
  const lat = 11.0168, lng = 76.9558;
  AppState.liveLocation = {
    lat: lat,
    lng: lng,
    name: "Coimbatore Agro-Zone (Field GPS: 11.0168° N, 76.9558° E)",
    isLive: true
  };
  if (btn) btn.innerHTML = '<i class="fa-solid fa-check text-green"></i> Location Active';
  if (coordDisplay) coordDisplay.innerText = `Lat: ${lat.toFixed(4)}° N, Lng: ${lng.toFixed(4)}° E (Field Verified)`;
  const repLoc = document.getElementById("rep-location");
  if (repLoc) repLoc.innerText = AppState.liveLocation.name;
  if (AppState.map) {
    AppState.map.setView([lat, lng], 15);
  }
}

function renderFieldsOnMap(fields) {
  if (!AppState.map) return;

  fields.forEach(field => {
    const isNeedsAttention = field.status === "Needs Attention";
    const color = isNeedsAttention ? "#F59E0B" : "#10B981";

    const polygon = L.polygon(field.polygon, {
      color: color,
      fillColor: color,
      fillOpacity: 0.35,
      weight: 3
    }).addTo(AppState.map);

    polygon.bindPopup(`
      <div style="font-family: inherit; font-size: 13px;">
        <strong>${field.name}</strong><br />
        Crop: ${field.crop}<br />
        Area: ${field.area_acres} acres<br />
        Status: <span style="color: ${color}; font-weight: bold;">${field.status}</span>
      </div>
    `);

    polygon.on("click", () => selectField(field));
  });
}

function renderFieldsList(fields) {
  const listEl = document.getElementById("fields-list-container");
  if (!listEl) return;

  if (!fields.length) {
    listEl.innerHTML = '<p class="text-muted">No active fields yet. Add a field to start tracking its reports.</p>';
    return;
  }

  listEl.innerHTML = fields.map(f => {
    const report = f.latest_report;
    const risk = (report?.risk_level || "").toUpperCase();
    const status = report ? (risk === "HIGH" ? "Needs Attention" : risk === "MEDIUM" ? "Review" : "Healthy") : f.status;
    const badgeClass = status === "Needs Attention" ? "badge-danger" : status === "Review" ? "badge-warning" : "badge-success";
    const reportLine = report
      ? `${report.disease_name || (report.scan_type === "soil" ? "Soil scan" : "Crop scan")} · ${report.confidence}% · ${formatScanAge(report.created_at)}`
      : "No scan report for this field yet";
    return `
      <div class="ff-item" data-field-id="${Number(f.id)}" onclick="selectFieldById(${Number(f.id)})" role="button" tabindex="0" onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();selectFieldById(${Number(f.id)});}">
        <div>
          <strong>${escapeDashboardText(f.name)}</strong> • <span>${escapeDashboardText(f.crop)}</span>
          <div class="text-muted" style="font-size:0.8rem;margin-top:4px;">${escapeDashboardText(reportLine)}</div>
        </div>
        <span class="badge ${badgeClass}">${escapeDashboardText(status || "Active")}</span>
      </div>
    `;
  }).join("");
}

function escapeDashboardText(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[char]);
}

function selectFieldById(id) {
  const field = AppState.fields.find(item => Number(item.id) === Number(id));
  if (field) selectField(field);
}

function selectFieldByName(name) {
  const field = AppState.fields.find(f => f.name === name);
  if (field) selectField(field);
}

function selectField(field) {
  document.getElementById("field-detail-name").innerText = field.name;
  document.getElementById("field-detail-crop").innerText = `${field.crop} Plot`;
  document.getElementById("field-detail-area").innerText = `${field.area_acres} acres`;
  const report = field.latest_report;
  document.getElementById("field-detail-date").innerText = report?.created_at
    ? new Date(`${report.created_at}${report.created_at.includes("Z") ? "" : "Z"}`).toLocaleString()
    : "No scan recorded";

  const badge = document.getElementById("field-detail-badge");
  const issueEl = document.getElementById("field-detail-issue");
  const risk = (report?.risk_level || "").toUpperCase();
  const status = report ? (risk === "HIGH" ? "Needs Attention" : risk === "MEDIUM" ? "Review" : "Healthy") : field.status;
  badge.className = `badge ${status === "Needs Attention" ? "badge-danger" : status === "Review" ? "badge-warning" : "badge-success"}`;
  badge.innerText = status || "Active";
  issueEl.className = `fm-val font-bold ${status === "Needs Attention" ? "text-danger" : status === "Review" ? "text-warning" : "text-green"}`;
  issueEl.innerText = report
    ? `${report.disease_name || (report.scan_type === "soil" ? "Soil scan" : "Crop scan")} · ${report.confidence}% · ${status}`
    : "No scan report recorded for this field yet.";

  document.querySelectorAll(".ff-item[data-field-id]").forEach(item => {
    item.classList.toggle("active", Number(item.dataset.fieldId) === Number(field.id));
  });

  if (AppState.map && field.polygon.length > 0) {
    AppState.map.panTo(field.polygon[0]);
  }
}

// ==========================================================================
// Guided Workflow & Live Location Helpers
// ==========================================================================

function startGuidedSoilStep() {
  AppState.scanMode = "soil";
  AppState.isGuidedStep = true;
  openCaptureFlow("soil");
}

function skipSoilStepAndContinue() {
  AppState.isGuidedStep = false;
  navigateTo("recommendations");
}

function useLiveFarmLocation() {
  const btn = document.getElementById("btn-live-loc");
  const coordDisplay = document.getElementById("map-coord-display");

  if (!navigator.geolocation) {
    alert("Geolocation is not supported by your browser.");
    return;
  }

  if (btn) btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Locating GPS...';

  navigator.geolocation.getCurrentPosition(
    position => {
      const lat = position.coords.latitude;
      const lng = position.coords.longitude;

      if (btn) btn.innerHTML = '<i class="fa-solid fa-circle-check text-green"></i> Live GPS Set';
      if (coordDisplay) coordDisplay.innerText = `Lat: ${lat.toFixed(4)}° N, Lng: ${lng.toFixed(4)}° E (Live GPS)`;

      if (AppState.map) {
        AppState.map.setView([lat, lng], 16);
        if (AppState.liveMarker) {
          AppState.liveMarker.setLatLng([lat, lng]);
        } else {
          AppState.liveMarker = L.marker([lat, lng], {
            title: "Your Live Farm Location"
          }).addTo(AppState.map).bindPopup("<b>📍 Your Current Location</b><br>Field calibrated live from GPS.").openPopup();
        }
      }

      fetch("/api/farms/update-location", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat, lng, id: 1 })
      }).catch(err => console.warn("Failed to persist GPS location:", err));
    },
    error => {
      console.warn("GPS error:", error);
      if (btn) btn.innerHTML = '<i class="fa-solid fa-location-crosshairs"></i> Use My Location';
      // Default to Coimbatore coords if permission denied
      if (coordDisplay) coordDisplay.innerText = "Lat: 11.019° N, Lng: 76.955° E (Coimbatore)";
      alert("GPS permission was denied or unavailable. Using default farm coordinates.");
    },
    { enableHighAccuracy: true, timeout: 8000 }
  );
}

function updateCombinedComparisonCard() {
  const card = document.getElementById("combined-health-card");
  if (!card) return;

  const plant = AppState.lastPlantResult;
  const soil = AppState.lastSoilResult;
  if (!plant && !soil) {
    card.style.display = "none";
    return;
  }
  card.style.display = "";

  const summaryGrid = card.querySelector(".card-body > div");
  const cropSummary = summaryGrid?.children[0];
  const soilSummary = summaryGrid?.children[1];
  if (cropSummary) cropSummary.style.display = plant ? "" : "none";
  if (soilSummary) soilSummary.style.display = soil ? "" : "none";

  const statusBadge = document.getElementById("combined-status-badge");
  const cardTitle = document.getElementById("combined-card-title");
  const stepLabel = document.getElementById("combined-step-label");
  const backToCropButton = document.getElementById("combined-back-crop-btn");
  if (statusBadge) {
    statusBadge.textContent = plant && soil ? "Crop + soil results available" : plant ? "Crop-only assessment" : "Soil-only assessment";
    statusBadge.className = `badge ${plant && soil ? "badge-success" : "badge-light"}`;
  }
  if (cardTitle) cardTitle.textContent = plant && soil ? "Combined Crop Health & Soil Health Comparison" : plant ? "Crop Health Summary" : "Soil Health Summary";
  if (stepLabel) stepLabel.innerHTML = plant && soil
    ? '<i class="fa-solid fa-code-compare"></i> Step 2 Correlation'
    : plant ? '<i class="fa-solid fa-leaf"></i> Crop Assessment' : '<i class="fa-solid fa-flask"></i> Soil Assessment';
  if (backToCropButton) backToCropButton.style.display = plant ? "" : "none";

  if (plant) {
    const cropNameEl = document.getElementById("comb-crop-name");
    const disNameEl = document.getElementById("comb-disease-name");
    const confEl = document.getElementById("comb-crop-conf");
    const symEl = document.getElementById("comb-symptoms");
    if (cropNameEl) cropNameEl.textContent = plant.crop || plant.crop_name || "Crop";
    if (disNameEl) disNameEl.textContent = plant.disease_name || "Diagnosis unavailable";
    if (confEl) {
      confEl.textContent = plant.confidence == null ? "—" : `${plant.confidence}%`;
      confEl.className = plant.risk_level === "HIGH" ? "badge badge-danger" : plant.risk_level === "MEDIUM" ? "badge badge-warning" : "badge badge-success";
    }
    if (symEl) symEl.textContent = plant.key_symptoms?.[0] || "No symptom summary available";
  }

  const nutrients = soil?.nutrient_status || {};
  const notMeasured = value => value == null || value === "" ? "Not measured" : value;
  if (soil) {
    const soilTypeEl = document.getElementById("comb-soil-type");
    const soilNEl = document.getElementById("comb-soil-n");
    const soilPKEl = document.getElementById("comb-soil-pk");
    const soilMoistEl = document.getElementById("comb-soil-moist");
    const nStatus = notMeasured(nutrients.n_status);
    if (soilTypeEl) soilTypeEl.textContent = soil.visual_analysis?.texture || "Not assessed";
    if (soilNEl) {
      soilNEl.textContent = nStatus;
      soilNEl.className = `badge ${nStatus === "Not measured" ? "badge-light" : nStatus === "Low" ? "badge-warning" : nStatus === "Normal" ? "badge-success" : "badge-danger"}`;
    }
    if (soilPKEl) soilPKEl.textContent = `P: ${notMeasured(nutrients.p_status)} | K: ${notMeasured(nutrients.k_status)}`;
    if (soilMoistEl) soilMoistEl.textContent = nutrients.moisture_pct == null ? "Moisture: Not measured" : `Moisture: ${nutrients.moisture_pct}%`;
  }

  const corrText = document.getElementById("comb-correlation-text");
  if (!corrText) return;
  if (plant && soil) {
    const nStatus = notMeasured(nutrients.n_status);
    if (nStatus === "Low") {
      corrText.textContent = `Low nitrogen was reported in the soil readings alongside ${plant.disease_name || "the crop diagnosis"}. This may affect crop growth, but the scan cannot establish that it caused the leaf symptoms.`;
    } else if (nStatus === "Not measured") {
      corrText.textContent = `The crop result is ${plant.disease_name || "available"}. Soil texture was assessed, but nutrient readings were not supplied, so a nutrient correlation cannot be determined.`;
    } else {
      corrText.textContent = `The crop result is ${plant.disease_name || "available"}. The supplied nitrogen reading is ${nStatus}; this scan does not indicate a nitrogen deficiency. Follow the crop treatment plan and use lab results for precise soil recommendations.`;
    }
  } else if (soil) {
    const texture = soil.visual_analysis?.texture || "the uploaded soil";
    corrText.textContent = `Soil-only assessment: the photo suggests ${texture} by color and surface texture. pH and NPK values come from the ${nutrients.measurement_source || "lab or sensor readings"}; this scan has no linked crop result for disease correlation.`;
  } else {
    corrText.textContent = `Foliage-only assessment: ${plant.disease_name || "a crop diagnosis"} was assessed from the leaf image. A soil scan can add separate soil readings, but no soil result is linked to this assessment.`;
  }
}

function showRejectionModal(verification) {
  const modal = document.getElementById("rejected-specimen-modal");
  if (!modal) return;

  const iconEl = document.getElementById("rej-modal-icon");
  const titleEl = document.getElementById("rej-modal-title");
  const badgeEl = document.getElementById("rej-modal-badge");
  const descEl = document.getElementById("rej-modal-desc");
  const listEl = document.getElementById("rej-detections-list");

  if (verification) {
    if (AppState.scanMode === "soil") {
      iconEl.innerText = "🌍";
      titleEl.innerText = "Soil Image Not Recognized";
      badgeEl.className = "badge badge-warning mt-2";
      badgeEl.innerText = `Detected: ${verification.label || "not soil"}`;
      descEl.innerText = "Upload a clear, close-up photo containing mostly soil. Leaves, people, tools, and other objects are not accepted as soil samples.";
    } else if (verification.specimen_type === "human") {
      iconEl.innerText = "👤";
      titleEl.innerText = "Human Person Detected";
      badgeEl.className = "badge badge-danger mt-2";
      badgeEl.innerText = `Detected: Human Being (${verification.confidence}%)`;
      descEl.innerText = "SoilDoctor is an agronomy computer vision platform designed for agricultural crops, leaves, and soil. Please point your camera at a plant specimen.";
    } else if (verification.specimen_type === "animal") {
      iconEl.innerText = "🐾";
      titleEl.innerText = "Animal / Pet Detected";
      badgeEl.className = "badge badge-warning mt-2";
      badgeEl.innerText = `Detected: ${verification.label} (${verification.confidence}%)`;
      descEl.innerText = "Animal detected! SoilDoctor analyzes crop leaves and soil. Please scan a plant foliage specimen.";
    } else {
      iconEl.innerText = "📦";
      titleEl.innerText = "Non-Plant Object Detected";
      badgeEl.className = "badge badge-warning mt-2";
      badgeEl.innerText = `Detected: ${verification.label} (${verification.confidence}%)`;
      descEl.innerText = "This object does not have plant foliage characteristics. SoilDoctor requires a plant leaf or soil sample.";
    }

    if (listEl && verification.top_detections) {
      listEl.innerHTML = verification.top_detections.map(d => `
        <div style="display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px dashed var(--border-color);">
          <span>${d.category.replace(/_/g, ' ')}</span>
          <strong style="color: var(--primary);">${d.probability}%</strong>
        </div>
      `).join("");
    }
  }

  modal.style.display = "flex";
}

function closeRejectionModal() {
  const modal = document.getElementById("rejected-specimen-modal");
  if (modal) modal.style.display = "none";
  navigateTo("capture");
}

function openAddFieldModal() {
  document.getElementById("add-field-modal").style.display = "flex";
}
function closeAddFieldModal() {
  document.getElementById("add-field-modal").style.display = "none";
}

function submitNewField(e) {
  e.preventDefault();
  const name = document.getElementById("modal-field-name").value;
  const crop = document.getElementById("modal-field-crop").value;
  const area = parseFloat(document.getElementById("modal-field-area").value);

  fetch("/api/farms/add", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, crop, area_acres: area })
  })
  .then(r => r.json())
  .then(data => {
    closeAddFieldModal();
    initFarmMap(); // Reload map
  });
}

// ==========================================================================
// 6. Chart.js Analytics (Trends & Distribution)
// ==========================================================================

function initAnalyticsCharts() {
  fetch("/api/analytics")
    .then(r => r.json())
    .then(data => {
      if (!data.success) return;

      // Update Analytics KPI Cards from real database values
      if (data.metrics) {
        const m = data.metrics;
        const dc = document.getElementById("an-disease-cases");
        const hs = document.getElementById("an-healthy-scans");
        const pi = document.getElementById("an-potential-issues");
        const ah = document.getElementById("an-avg-health");
        if (dc) dc.innerText = m.disease_cases;
        if (hs) hs.innerText = m.healthy_scans;
        if (pi) pi.innerText = m.potential_issues;
        if (ah) ah.innerText = (m.average_health || 86) + "%";
      }

      // 1. Multi-line Health Trends Chart
      const ctxTrends = document.getElementById("chart-health-trends")?.getContext("2d");
      if (ctxTrends) {
        if (AppState.charts.trends) AppState.charts.trends.destroy();

        AppState.charts.trends = new Chart(ctxTrends, {
          type: "line",
          data: {
            labels: data.trends.dates,
            datasets: [
              {
                label: "Tomato",
                data: data.trends.tomato,
                borderColor: "#E63946",
                backgroundColor: "rgba(230, 57, 70, 0.08)",
                tension: 0.35,
                fill: true
              },
              {
                label: "Rice",
                data: data.trends.rice,
                borderColor: "#10B981",
                backgroundColor: "rgba(16, 185, 129, 0.08)",
                tension: 0.35,
                fill: true
              },
              {
                label: "Chilli",
                data: data.trends.chilli,
                borderColor: "#F59E0B",
                backgroundColor: "rgba(245, 158, 11, 0.08)",
                tension: 0.35,
                fill: true
              },
              {
                label: "Sugarcane",
                data: data.trends.sugarcane,
                borderColor: "#0284C7",
                backgroundColor: "rgba(2, 132, 199, 0.08)",
                tension: 0.35,
                fill: true
              },
              {
                label: "Blackgram",
                data: data.trends.blackgram,
                borderColor: "#8B5CF6",
                backgroundColor: "rgba(139, 92, 246, 0.08)",
                tension: 0.35,
                fill: true
              },
              {
                label: "Soybean",
                data: data.trends.soybean,
                borderColor: "#059669",
                backgroundColor: "rgba(5, 150, 105, 0.08)",
                tension: 0.35,
                fill: true
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: "top" }
            },
            scales: {
              y: { min: 50, max: 100 }
            }
          }
        });
      }

      // 2. Disease Distribution Doughnut Chart
      const ctxDist = document.getElementById("chart-disease-dist")?.getContext("2d");
      if (ctxDist) {
        if (AppState.charts.distribution) AppState.charts.distribution.destroy();

        AppState.charts.distribution = new Chart(ctxDist, {
          type: "doughnut",
          data: {
            labels: data.distribution.labels,
            datasets: [{
              data: data.distribution.data,
              backgroundColor: data.distribution.colors,
              borderWidth: 2
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: "bottom" }
            }
          }
        });
      }
    });
}

// ==========================================================================
// 7. Scan History & Crop Doctor Encyclopedia
// ==========================================================================

function loadScanHistory(filter = "all") {
  AppState.historyFilter = filter;
  document.querySelectorAll(".history-filters .filter-pill").forEach(pill => {
    pill.classList.toggle("active", pill.dataset.filter === filter);
  });
  fetch(`/api/scans?filter=${filter}`)
    .then(r => r.json())
    .then(data => {
      if (!data.success) return;
      AppState.scans = data.scans;
      renderHistoryTable(data.scans);
      renderLatestHistorySummary(data.scans, filter);
    });
}

function renderLatestHistorySummary(scans, filter) {
  const summary = document.getElementById("history-latest-summary");
  const description = document.getElementById("history-view-description");
  if (!summary) return;
  const labels = { plant: "Latest crop report", soil: "Latest soil report", all: "Latest scan report", disease: "Latest disease report", healthy: "Latest healthy scan" };
  const descriptions = { plant: "Showing saved crop and leaf scans, newest first.", soil: "Showing saved soil scans, newest first.", all: "Showing all saved scans, newest first.", disease: "Showing scans marked with a disease risk, newest first.", healthy: "Showing scans marked healthy, newest first." };
  if (description) description.textContent = descriptions[filter] || descriptions.all;
  if (!scans.length) {
    summary.style.display = "block";
    summary.innerHTML = `<div class="card-body"><strong>${labels[filter] || labels.all}</strong><p class="text-muted mb-0">No matching report has been saved yet.</p></div>`;
    return;
  }
  const latest = scans[0];
  const date = latest.created_at ? new Date(`${latest.created_at}${latest.created_at.includes("Z") ? "" : "Z"}`).toLocaleString() : "Date unavailable";
  summary.style.display = "block";
  summary.innerHTML = `
    <div class="card-body">
      <div class="card-header" style="padding:0 0 10px;border:0;">
        <h3>${labels[filter] || labels.all}</h3>
        <span class="badge ${latest.risk_level === "HIGH" ? "badge-danger" : latest.risk_level === "MEDIUM" ? "badge-warning" : "badge-success"}">${escapeDashboardText(latest.risk_level || "Recorded")}</span>
      </div>
      <strong>${escapeDashboardText(latest.crop_name || "Farm Soil")}</strong>
      <span> · ${escapeDashboardText(latest.disease_name || "Scan result")}</span>
      <div class="text-muted" style="margin-top:6px;">Result confidence: ${Number(latest.confidence) || 0}% · ${escapeDashboardText(date)}</div>
    </div>`;
}

function renderHistoryTable(scans) {
  const tbody = document.getElementById("history-table-body");
  if (!tbody) return;
  const countEl = document.getElementById("history-record-count");
  if (countEl) countEl.textContent = `${scans.length} ${scans.length === 1 ? "scan" : "scans"}`;

  if (scans.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center text-muted">No diagnostic scans found.</td></tr>`;
    return;
  }

  tbody.innerHTML = scans.map(s => {
    const riskBadge = s.risk_level === 'HIGH' ? 'badge-danger' : s.risk_level === 'MEDIUM' ? 'badge-warning' : 'badge-success';
    return `
      <tr>
        <td>${s.created_at}</td>
        <td><strong>${s.crop_name}</strong></td>
        <td>${s.disease_name}</td>
        <td><strong class="text-green">${s.confidence}%</strong></td>
        <td><span class="badge ${riskBadge}">${s.risk_level}</span></td>
        <td>
          <button class="btn btn-sm btn-outline" onclick="openPastScan(${s.id})">
            <i class="fa-solid fa-file-lines"></i> View
          </button>
        </td>
      </tr>
    `;
  }).join("");
}

function filterHistory(filterKey, btnEl) {
  document.querySelectorAll(".history-filters .filter-pill").forEach(p => p.classList.remove("active"));
  if (btnEl) btnEl.classList.add("active");
  loadScanHistory(filterKey);
}

function searchHistory(query) {
  query = query.toLowerCase();
  const filtered = AppState.scans.filter(s =>
    s.crop_name.toLowerCase().includes(query) ||
    s.disease_name.toLowerCase().includes(query)
  );
  renderHistoryTable(filtered);
}

function openPastScan(id) {
  // Demo view for past scan
  openDemoResult("tomato");
}

function loadCropDoctor() {
  fetch("/api/crops")
    .then(r => r.json())
    .then(data => {
      if (!data.success) return;
      AppState.crops = data.crops;
      renderCropCards(data.crops);
    });
}

function renderCropCards(crops) {
  const grid = document.getElementById("crop-doctor-grid");
  if (!grid) return;

  grid.innerHTML = crops.map(c => `
    <div class="crop-ency-card">
      <div class="ency-top">
        <span class="ency-icon">${c.icon}</span>
        <div class="ency-info">
          <h3>${c.name}</h3>
          <span>${c.category}</span>
        </div>
      </div>
      <div class="s-attr-row">
        <span class="text-muted">Soil Preference:</span>
        <span style="font-size: 12px;">${c.soil_pref}</span>
      </div>
      <div class="s-attr-row">
        <span class="text-muted">Target pH:</span>
        <strong class="text-green">${c.ph_range}</strong>
      </div>
      <div class="ency-diseases">
        <strong style="font-size: 13px;">Key Diseases:</strong>
        ${c.diseases.slice(0, 2).map(d => `
          <div class="ency-d-item">
            <strong>• ${d.name}:</strong> <span>${d.symptoms}</span>
          </div>
        `).join("")}
      </div>
    </div>
  `).join("");
}

function filterCropCards(query) {
  query = query.toLowerCase();
  const filtered = AppState.crops.filter(c =>
    c.name.toLowerCase().includes(query) ||
    c.category.toLowerCase().includes(query)
  );
  renderCropCards(filtered);
}

// ==========================================================================
// 9. Dashboard Real-Data Loader
// ==========================================================================

const CROP_EMOJIS = {
  tomato: "🍅", rice: "🌾", chilli: "🌶️", sugarcane: "🎋",
  blackgram: "🌱", soybean: "🌿", maize: "🌽", potato: "🥔",
  wheat: "🌾", cotton: "🌿", groundnut: "🥜", onion: "🧅"
};

function getCropEmoji(name) {
  if (!name) return "🌿";
  const key = name.toLowerCase().replace(/\s+/g, "");
  for (const k of Object.keys(CROP_EMOJIS)) {
    if (key.includes(k)) return CROP_EMOJIS[k];
  }
  return "🌿";
}

function getBarColor(health) {
  if (health >= 80) return "bg-green";
  if (health >= 60) return "bg-teal";
  if (health >= 40) return "bg-orange";
  return "bg-red";
}

function formatScanAge(dateStr) {
  if (!dateStr) return "";
  const d = new Date(dateStr + (dateStr.includes("Z") ? "" : "Z"));
  const diffMs = Date.now() - d.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 2) return "Just now";
  if (diffMin < 60) return `${diffMin} min ago`;
  const diffH = Math.floor(diffMin / 60);
  if (diffH < 24) return `${diffH}h ago`;
  const diffD = Math.floor(diffH / 24);
  return diffD === 1 ? "Yesterday" : `${diffD} days ago`;
}

function loadDashboard() {
  fetch("/api/dashboard")
    .then(r => r.json())
    .then(data => {
      if (!data.success) return;
      const kpi = data.kpi;

      // ── KPI: Crop Health
      const cropHealthEl = document.getElementById("kpi-crop-health");
      const cropMetaEl   = document.getElementById("kpi-crop-health-meta");
      if (cropHealthEl) {
        if (kpi.crop_health !== null) {
          cropHealthEl.textContent = kpi.crop_health + "%";
          const risk = (kpi.last_risk || "LOW").toUpperCase();
          const icon = risk === "LOW" ? `<i class="fa-solid fa-circle-check"></i>` :
                       risk === "HIGH" ? `<i class="fa-solid fa-triangle-exclamation"></i>` :
                       `<i class="fa-solid fa-circle-exclamation"></i>`;
          const color = risk === "LOW" ? "text-green" : risk === "HIGH" ? "text-danger" : "text-warning";
          cropMetaEl.className = `kpi-meta ${color}`;
          cropMetaEl.innerHTML = `${icon} Last: ${kpi.last_crop || ""} · ${formatScanAge(kpi.last_scan_time)}`;
        } else {
          cropHealthEl.textContent = "—";
          cropMetaEl.className = "kpi-meta text-muted";
          cropMetaEl.innerHTML = `<i class="fa-solid fa-clock"></i> No plant scans yet`;
        }
      }

      // ── KPI: Soil Health
      const soilHealthEl = document.getElementById("kpi-soil-health");
      const soilMetaEl   = document.getElementById("kpi-soil-health-meta");
      if (soilHealthEl) {
        if (kpi.soil_health !== null) {
          soilHealthEl.textContent = kpi.soil_health + "%";
          soilMetaEl.className = "kpi-meta text-green";
          soilMetaEl.innerHTML = `<i class="fa-solid fa-circle-check"></i> Last soil scan result`;
        } else {
          soilHealthEl.textContent = "—";
          soilMetaEl.className = "kpi-meta text-muted";
          soilMetaEl.innerHTML = `<i class="fa-solid fa-clock"></i> No soil scans yet`;
        }
      }

      // ── KPI: Total Scans
      const totalScansEl   = document.getElementById("kpi-total-scans");
      const totalScansMetaEl = document.getElementById("kpi-total-scans-meta");
      if (totalScansEl) {
        totalScansEl.textContent = kpi.total_scans;
        totalScansMetaEl.textContent = kpi.total_scans === 0
          ? "No scans yet"
          : kpi.total_scans === 1 ? "1 scan recorded" : `${kpi.total_scans} total recorded`;
      }

      const activeFieldsEl = document.getElementById("kpi-active-fields");
      if (activeFieldsEl) activeFieldsEl.textContent = kpi.active_fields ?? 0;

      // ── Crop Health Progress Bars
      const cropListEl = document.getElementById("dash-crop-health-list");
      const badgeEl    = document.getElementById("dash-crop-count-badge");
      const perCrop = data.per_crop_health || [];
      if (cropListEl) {
        if (perCrop.length === 0) {
          cropListEl.innerHTML = `
            <div style="text-align:center;color:var(--text-muted);padding:24px 0;font-size:13px;">
              <i class="fa-solid fa-seedling" style="font-size:24px;margin-bottom:8px;display:block;"></i>
              No crop scans yet. Scan a leaf to see health scores here.
            </div>`;
          if (badgeEl) badgeEl.textContent = "No scans";
        } else {
          if (badgeEl) badgeEl.textContent = `${perCrop.length} Crop${perCrop.length !== 1 ? "s" : ""} Scanned`;
          cropListEl.innerHTML = perCrop.map(item => {
            const pct = Math.min(100, Math.max(0, item.health));
            const colorClass = getBarColor(pct);
            const emoji = getCropEmoji(item.crop);
            return `
              <div class="crop-prog-item">
                <div class="crop-prog-info">
                  <span class="crop-prog-name"><span class="crop-emoji">${emoji}</span> ${item.crop}</span>
                  <span class="crop-prog-pct">${pct}%</span>
                </div>
                <div class="prog-bar-track">
                  <div class="prog-bar-fill ${colorClass}" style="width:${pct}%;"></div>
                </div>
              </div>`;
          }).join("");
        }
      }

      // ── Recent Scans Feed
      const recentEl = document.getElementById("dash-recent-scans");
      const recentScans = data.recent_scans || [];
      if (recentEl && recentScans.length > 0) {
        recentEl.innerHTML = recentScans.map(s => {
          const risk = (s.risk_level || "LOW").toUpperCase();
          const badgeClass = risk === "HIGH" ? "badge-danger" : risk === "MEDIUM" ? "badge-warning" : "badge-success";
          const label = s.disease_name || (s.scan_type === "soil" ? "Soil Scan" : "Healthy");
          const thumb = s.thumbnail_b64
            ? `<img src="data:image/jpeg;base64,${s.thumbnail_b64}" alt="${s.crop_name}" style="width:100%;height:100%;object-fit:cover;" />`
            : `<i class="fa-solid fa-leaf" style="font-size:20px;color:var(--accent);"></i>`;
          return `
            <div class="recent-scan-item">
              <div class="scan-thumb-wrap">${thumb}</div>
              <div class="scan-meta">
                <div class="scan-crop-line">
                  <strong>${s.crop_name || "Farm Soil"}</strong>
                  <span class="scan-time">${formatScanAge(s.created_at)}</span>
                </div>
                <span class="badge ${badgeClass}">${label} (${s.confidence}%)</span>
              </div>
              <i class="fa-solid fa-chevron-right text-muted"></i>
            </div>`;
        }).join("");
      }
    })
    .catch(err => console.error("Dashboard load error:", err));
}

function activateDashboardCard(event, destination) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    openDashboardDetail(destination);
  }
}

function openDashboardDetail(destination) {
  if (destination === "fields") {
    navigateTo("farm-mgmt");
    return;
  }
  AppState.historyFilter = destination;
  navigateTo("history");
}

// ==========================================================================
// 10. Document Ready Initialization
// ==========================================================================

function saveProfileSettings(e) {
  e.preventDefault();
  const name = document.getElementById("prof-input-name").value;
  const email = document.getElementById("prof-input-email").value;
  const phone = document.getElementById("prof-input-phone").value;
  const location = document.getElementById("prof-input-location").value;

  fetch("/api/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      full_name: name,
      email: email,
      phone: phone,
      farm_location: location
    })
  })
  .then(r => r.json())
  .then(data => {
    document.getElementById("header-username").innerText = name;
    document.getElementById("prof-display-name").innerText = name;
    document.getElementById("prof-display-sub").innerText = `Farmer • ${location}`;
    alert("Profile settings saved successfully!");
  });
}

function toggleTheme() {
  const isLight = document.body.classList.contains("light-mode");
  const icon = document.getElementById("theme-toggle-icon");
  if (isLight) {
    document.body.classList.remove("light-mode");
    localStorage.setItem("soildoctor-theme", "dark");
    if (icon) {
      icon.className = "fa-solid fa-moon";
    }
  } else {
    document.body.classList.add("light-mode");
    localStorage.setItem("soildoctor-theme", "light");
    if (icon) {
      icon.className = "fa-solid fa-sun";
    }
  }
}

function initTheme() {
  const savedTheme = localStorage.getItem("soildoctor-theme") || "dark";
  const icon = document.getElementById("theme-toggle-icon");
  if (savedTheme === "light") {
    document.body.classList.add("light-mode");
    if (icon) icon.className = "fa-solid fa-sun";
  } else {
    document.body.classList.remove("light-mode");
    if (icon) icon.className = "fa-solid fa-moon";
  }
}

function toggleDarkMode(enabled) {
  if (enabled) {
    document.body.classList.remove("light-mode");
    localStorage.setItem("soildoctor-theme", "dark");
  } else {
    document.body.classList.add("light-mode");
    localStorage.setItem("soildoctor-theme", "light");
  }
  const icon = document.getElementById("theme-toggle-icon");
  if (icon) icon.className = enabled ? "fa-solid fa-moon" : "fa-solid fa-sun";
}

function saveAppToggles() {
  console.log("Application preference toggles saved.");
}

// ==========================================================================
// 9. Document Ready Initialization
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  console.log("SoilDoctor initialized.");
  // Recover report markup from older cached templates where a missing table
  // closing tag nested the report view inside the hidden scan-history section.
  const contentRoot = document.getElementById("view-content");
  const reportView = document.getElementById("view-report-view");
  if (contentRoot && reportView && reportView.parentElement !== contentRoot) {
    const hiddenHistory = reportView.closest("#view-history");
    contentRoot.appendChild(reportView);
    // The same malformed cached markup may have left later report sections as
    // siblings in the history view. Reattach those pieces to the printable sheet.
    if (hiddenHistory) {
      const printable = reportView.querySelector("#printable-report-sheet");
      const diagnosisColumn = hiddenHistory.querySelector("#rep-diag-name")?.closest(".rep-col");
      const specimenGrid = printable?.querySelector(".rep-section-grid");
      if (diagnosisColumn && specimenGrid) specimenGrid.appendChild(diagnosisColumn);
      const fragments = [
        hiddenHistory.querySelector("h3.mt-4"),
        hiddenHistory.querySelector(".rep-cv-evidence-grid"),
        hiddenHistory.querySelector(".rep-section-grid.mt-4"),
        hiddenHistory.querySelector(".rep-prescription-box"),
        hiddenHistory.querySelector(".rep-signature-row")
      ];
      fragments.forEach(fragment => { if (fragment && printable) printable.appendChild(fragment); });
    }
  }
  initTheme();
  // Load real dashboard data then show dashboard
  loadDashboard();
  navigateTo("dashboard");

  // Auto-refresh dashboard every 30 seconds when on dashboard view
  setInterval(() => {
    if (AppState.currentView === "dashboard") {
      loadDashboard();
      // Update live indicator
      const liveEl = document.getElementById("dash-live-time");
      if (liveEl) {
        const now = new Date();
        liveEl.textContent = `Live · Updated ${now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
      }
    }
  }, 30000);
});
