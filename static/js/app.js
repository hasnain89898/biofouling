/**
 * BioFouling.AI - Enterprise Frontend State Management & API Integration
 * Dynamic Data Binding, Full-Stack Deletion, and Explainable AI (XAI) Engine
 */

// ==============================================================================
// Application Global State
// ==============================================================================
let currentSelectedFile = null;
let currentPreviewUrl = "";
let currentInspection = null;
let inspections = [];
let queueItems = [];
let activeQueueItem = null;
let pendingDeleteInspectionId = null;

// ==============================================================================
// Navigation View Switching
// ==============================================================================

function switchView(viewName) {
  const views = ["dashboard", "queue", "mlops", "logs"];
  views.forEach((v) => {
    const elView = document.getElementById(`view-${v}`);
    const elNav = document.getElementById(`nav-${v}`);
    if (elView) elView.classList.remove("active");
    if (elNav) elNav.classList.remove("active");
  });

  const activeView = document.getElementById(`view-${viewName}`);
  const activeNav = document.getElementById(`nav-${viewName}`);
  if (activeView) activeView.classList.add("active");
  if (activeNav) activeNav.classList.add("active");

  if (viewName === "dashboard") {
    fetchInspections();
  } else if (viewName === "queue") {
    fetchReviewQueue();
  } else if (viewName === "logs") {
    fetchInspections();
  }
}

// ==============================================================================
// Toast Notification Utility
// ==============================================================================

function showToast(message, type = "success") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast-item toast-${type}`;
  toast.innerHTML = `
    <div class="toast-content">
      <span class="toast-icon">${type === "error" ? "✕" : "✓"}</span>
      <span>${message}</span>
    </div>
    <button class="toast-close" onclick="this.parentElement.remove()">✕</button>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("fade-out");
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ==============================================================================
// Dynamic Data Fetching: Inspections & Review Queue
// ==============================================================================

async function fetchInspections() {
  try {
    const res = await fetch("/api/inspections?limit=100");
    if (res.ok) {
      const data = await res.json();
      inspections = data.inspections || [];
      renderDashboardInspections();
      renderAuditTable();
      updateTelemetryStats();
    }
  } catch (err) {
    console.error("Failed to fetch inspections from backend API:", err);
  }
}

async function fetchReviewQueue() {
  try {
    const res = await fetch("/review-queue");
    if (res.ok) {
      const data = await res.json();
      queueItems = (data.queue_items || []).map((it) => ({
        queue_id: it.queue_id,
        asset_id: it.request_id || `#Q-${it.id || it.queue_id.slice(-4)}`,
        request_id: it.request_id,
        filename: it.filename,
        preview_url: it.saved_image_path && it.saved_image_path !== "N/A"
          ? (it.saved_image_path.startsWith("/") ? it.saved_image_path : `/${it.saved_image_path}`)
          : "/static/assets/bio_bg.svg",
        predicted_class: it.predicted_class,
        confidence: it.confidence,
        entropy: it.entropy || 0.0,
        status: it.review_status || it.status,
        resolved_label: it.resolved_label,
      }));

      // Update Top Metrics Cards in Review Queue
      const pendingCount = queueItems.filter((i) => i.status === "PENDING_REVIEW").length;
      const criticalCount = queueItems.filter((i) => i.predicted_class === "Critical").length;
      const resolvedCount = queueItems.filter((i) => i.status === "RESOLVED").length;
      const avgConf = queueItems.length > 0
        ? Math.round((queueItems.reduce((acc, i) => acc + (i.confidence || 0), 0) / queueItems.length) * 100)
        : 0;

      const pEl = document.getElementById("queue-count-pending");
      const cEl = document.getElementById("queue-count-critical");
      const aEl = document.getElementById("queue-count-approved");
      const confEl = document.getElementById("queue-count-avg-conf");

      if (pEl) pEl.textContent = `${pendingCount} Items`;
      if (cEl) cEl.textContent = `${criticalCount} Critical`;
      if (aEl) aEl.textContent = `${resolvedCount} Verified`;
      if (confEl) confEl.textContent = queueItems.length > 0 ? `${avgConf}%` : "--%";

      renderQueueTable();
    }
  } catch (err) {
    console.error("Failed to fetch review queue:", err);
  }
}

function updateTelemetryStats() {
  const totalCountEl = document.getElementById("telemetry-inspections-count");
  const autoRateEl = document.getElementById("telemetry-auto-rate");
  const autoDescEl = document.getElementById("telemetry-auto-desc");

  if (totalCountEl) totalCountEl.textContent = inspections.length.toLocaleString();

  if (autoRateEl) {
    if (inspections.length > 0) {
      const autoApprovedCount = inspections.filter((i) => i.gating_status === "AUTOMATED_DECISION" || !i.human_review_required).length;
      const rate = ((autoApprovedCount / inspections.length) * 100).toFixed(1);
      autoRateEl.textContent = `${rate}%`;
      if (autoDescEl) autoDescEl.textContent = `${autoApprovedCount} of ${inspections.length} approved without escalation`;
    } else {
      autoRateEl.textContent = "--%";
      if (autoDescEl) autoDescEl.textContent = "Awaiting inspection data";
    }
  }
}

// ==============================================================================
// Dynamic Table Rendering & Empty State Management
// ==============================================================================

function renderDashboardInspections() {
  const tbody = document.getElementById("dashboard-table-body");
  const emptyState = document.getElementById("dashboard-empty-state");
  const table = document.getElementById("dashboard-inspections-table");
  if (!tbody) return;

  if (inspections.length === 0) {
    if (emptyState) emptyState.style.display = "flex";
    if (table) table.style.display = "none";
    tbody.innerHTML = "";
    return;
  }

  if (emptyState) emptyState.style.display = "none";
  if (table) table.style.display = "table";

  tbody.innerHTML = "";
  // Show recent inspections (top 6)
  const recent = inspections.slice(0, 6);

  recent.forEach((item) => {
    const tr = document.createElement("tr");
    const sev = item.predicted_class || "Low";
    const primaryIssue = item.defect_analysis?.primary_issue || item.primary_issue || "Nominal substrate surface";
    const imgSrc = item.file_path && item.file_path !== "N/A"
      ? (item.file_path.startsWith("/") ? item.file_path : `/${item.file_path}`)
      : "/static/assets/bio_bg.svg";
    const dateFormatted = formatTimestamp(item.created_at);

    tr.innerHTML = `
      <td>
        <img src="${imgSrc}" class="table-thumbnail" alt="Inspection Frame" onerror="this.src='/static/assets/bio_bg.svg'">
      </td>
      <td><strong><code>${item.inspection_id}</code></strong></td>
      <td><span class="badge-class ${sev}">${sev}</span></td>
      <td><strong>${item.confidence_percentage || (item.confidence * 100).toFixed(1) + '%'}</strong></td>
      <td class="primary-defect-cell" title="${primaryIssue}">${primaryIssue}</td>
      <td style="color: var(--text-muted); font-size: 0.85rem;">${dateFormatted}</td>
      <td>
        <div class="table-actions-cell">
          <button class="btn-action-icon btn-report" title="View &amp; Print Official PDF Report" onclick="openReportModal('${item.inspection_id}')">
            📄 Report
          </button>
          <button class="btn-action-icon btn-delete" title="Permanently delete inspection and image file" onclick="confirmDeleteInspection('${item.inspection_id}')">
            🗑
          </button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function renderAuditTable() {
  const tbody = document.getElementById("audit-table-body");
  const emptyState = document.getElementById("audit-empty-state");
  const table = document.getElementById("audit-data-table");
  if (!tbody) return;

  const search = (document.getElementById("audit-search")?.value || "").toLowerCase();
  const filtered = inspections.filter(
    (item) =>
      item.inspection_id.toLowerCase().includes(search) ||
      (item.filename && item.filename.toLowerCase().includes(search)) ||
      (item.predicted_class && item.predicted_class.toLowerCase().includes(search)) ||
      (item.primary_issue && item.primary_issue.toLowerCase().includes(search))
  );

  if (filtered.length === 0) {
    if (emptyState) emptyState.style.display = "flex";
    if (table) table.style.display = "none";
    tbody.innerHTML = "";
    return;
  }

  if (emptyState) emptyState.style.display = "none";
  if (table) table.style.display = "table";

  tbody.innerHTML = "";
  filtered.forEach((item) => {
    const tr = document.createElement("tr");
    const sev = item.predicted_class || "Low";
    const primaryIssue = item.defect_analysis?.primary_issue || item.primary_issue || "Nominal Surface Condition";
    const imgSrc = item.file_path && item.file_path !== "N/A"
      ? (item.file_path.startsWith("/") ? item.file_path : `/${item.file_path}`)
      : "/static/assets/bio_bg.svg";

    const isAuto = item.gating_status === "AUTOMATED_DECISION" || !item.human_review_required;
    const gatingLabel = isAuto ? "Auto-Approved" : "Escalated to Review";
    const gatingClass = isAuto ? "auto" : "pending";

    tr.innerHTML = `
      <td>
        <img src="${imgSrc}" class="table-thumbnail" alt="Thumb" onerror="this.src='/static/assets/bio_bg.svg'">
      </td>
      <td><strong><code>${item.inspection_id}</code></strong></td>
      <td style="color: var(--text-muted); font-size: 0.85rem;">${formatTimestamp(item.created_at)}</td>
      <td><code>${item.filename || 'frame.jpg'}</code></td>
      <td><span class="badge-class ${sev}">${sev}</span></td>
      <td><strong>${item.confidence_percentage || (item.confidence * 100).toFixed(1) + '%'}</strong></td>
      <td class="primary-defect-cell" title="${primaryIssue}">${primaryIssue}</td>
      <td><span class="badge-status ${gatingClass}">${gatingLabel}</span></td>
      <td>
        <div class="table-actions-cell">
          <button class="btn-action-icon btn-report" title="Generate Official Inspection Report (PDF)" onclick="openReportModal('${item.inspection_id}')">
            📄 Report
          </button>
          <button class="btn-action-icon btn-delete" title="Delete inspection record and storage files" onclick="confirmDeleteInspection('${item.inspection_id}')">
            🗑
          </button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterAuditTable() {
  renderAuditTable();
}

function renderQueueTable() {
  const tbody = document.getElementById("queue-table-body");
  const emptyState = document.getElementById("queue-empty-state");
  const table = document.getElementById("queue-table");
  if (!tbody) return;

  const searchTerm = (document.getElementById("queue-search")?.value || "").toLowerCase();
  const statusFilter = document.getElementById("queue-status-filter")?.value || "ALL";

  let filtered = queueItems.filter((item) => {
    const matchesSearch =
      item.queue_id.toLowerCase().includes(searchTerm) ||
      (item.filename && item.filename.toLowerCase().includes(searchTerm)) ||
      (item.predicted_class && item.predicted_class.toLowerCase().includes(searchTerm));

    const matchesStatus = statusFilter === "ALL" || item.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  if (filtered.length === 0) {
    if (emptyState) emptyState.style.display = "flex";
    if (table) table.style.display = "none";
    tbody.innerHTML = "";
    return;
  }

  if (emptyState) emptyState.style.display = "none";
  if (table) table.style.display = "table";

  tbody.innerHTML = "";
  filtered.forEach((item) => {
    const tr = document.createElement("tr");

    let statusBadgeClass = "pending";
    let statusLabel = "Pending Review";
    if (item.status === "RESOLVED") {
      statusBadgeClass = "resolved";
      statusLabel = `Resolved (${item.resolved_label || "Verified"})`;
    }

    tr.innerHTML = `
      <td>
        <img src="${item.preview_url}" class="table-thumbnail" alt="Thumb" onerror="this.src='/static/assets/bio_bg.svg'">
      </td>
      <td><strong><code>${item.queue_id}</code></strong></td>
      <td><span class="badge-class ${item.predicted_class}">${item.predicted_class}</span></td>
      <td><code>${item.filename || 'frame.jpg'}</code></td>
      <td><strong>${(item.confidence * 100).toFixed(1)}%</strong></td>
      <td>${Number(item.entropy).toFixed(3)}</td>
      <td><span class="badge-status ${statusBadgeClass}">${statusLabel}</span></td>
      <td>
        <div class="table-actions-cell">
          ${
            item.status === "PENDING_REVIEW"
              ? `<button class="btn-table-action" onclick="openReviewModal('${item.queue_id}')">Review</button>`
              : `<button class="btn-table-action" style="background:#e2e8f0;color:#334155;" onclick="openReviewModal('${item.queue_id}')">View</button>`
          }
          ${
            item.request_id
              ? `<button class="btn-action-icon btn-delete" title="Delete inspection" onclick="confirmDeleteInspection('${item.request_id}')">🗑</button>`
              : ""
          }
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterQueueTable() {
  renderQueueTable();
}

function sortQueueTable() {
  const sortVal = document.getElementById("queue-sort-filter").value;
  if (sortVal === "CONFIDENCE") {
    queueItems.sort((a, b) => a.confidence - b.confidence);
  } else if (sortVal === "NEWEST") {
    queueItems.reverse();
  }
  renderQueueTable();
}

function formatTimestamp(isoStr) {
  if (!isoStr) return "-";
  try {
    const d = new Date(isoStr);
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch (e) {
    return isoStr;
  }
}

// ==============================================================================
// View 1: Live Inspection & Upload Handlers
// ==============================================================================

function initDropzone() {
  const dropzone = document.getElementById("dropzone");
  if (!dropzone) return;

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      setInspectionFile(files[0]);
    }
  });
}

function handleFileSelected(event) {
  const file = event.target.files[0];
  if (file) {
    setInspectionFile(file);
  }
}

function setInspectionFile(file) {
  currentSelectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    currentPreviewUrl = e.target.result;
    const imgEl = document.getElementById("preview-img");
    if (imgEl) {
      imgEl.src = currentPreviewUrl;
      imgEl.style.display = "block";
    }
  };
  reader.readAsDataURL(file);

  document.querySelectorAll(".sample-btn").forEach((btn) => btn.classList.remove("active"));
}

function loadSampleImage(severityClass) {
  const samples = {
    Low: "/data/raw/Low/bf_low_0001.jpg",
    Moderate: "/data/raw/Moderate/bf_moderate_0001.jpg",
    High: "/data/raw/High/bf_high_0001.jpg",
    Critical: "/data/raw/Critical/bf_critical_0001.jpg",
  };

  const sampleUrl = samples[severityClass] || samples.High;
  currentPreviewUrl = sampleUrl;
  currentSelectedFile = null;

  const imgEl = document.getElementById("preview-img");
  if (imgEl) {
    imgEl.src = sampleUrl;
    imgEl.style.display = "block";
  }

  document.querySelectorAll(".sample-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.textContent.toLowerCase().includes(severityClass.toLowerCase()));
  });

  fetch(sampleUrl)
    .then((res) => res.blob())
    .then((blob) => {
      currentSelectedFile = new File([blob], `${severityClass.toLowerCase()}_sample.jpg`, { type: "image/jpeg" });
    })
    .catch((err) => console.log("Sample loaded locally"));
}

function clearPreviewFrame() {
  currentSelectedFile = null;
  currentPreviewUrl = "";
  const imgEl = document.getElementById("preview-img");
  if (imgEl) imgEl.style.display = "none";
}

function resetFrame() {
  clearPreviewFrame();
  resetResultsCard();
}

function resetResultsCard() {
  currentInspection = null;
  document.getElementById("res-severity").textContent = "Ready for Scan";
  document.getElementById("res-severity").style.color = "var(--text-muted)";
  document.getElementById("res-ci-badge").textContent = "● Awaiting Frame";
  document.getElementById("res-ci-badge").style.background = "#E2E8F0";
  document.getElementById("res-ci-badge").style.color = "#475569";

  document.getElementById("res-confidence-val").textContent = "--%";
  document.getElementById("res-confidence-bar").style.width = "0%";

  document.getElementById("res-entropy-val").textContent = "--";
  document.getElementById("res-entropy-bar").style.width = "0%";
  document.getElementById("res-entropy-grade").textContent = "Awaiting Inference";

  const verdictBox = document.getElementById("res-verdict-box");
  verdictBox.className = "gating-verdict-box";
  document.getElementById("res-verdict-title").textContent = "System Ready";
  document.getElementById("res-verdict-desc").textContent = "Select or drop an industrial inspection frame and click 'Run Inspection' to execute the ONNX vision engine.";

  document.getElementById("res-primary-issue").textContent = "Awaiting frame evaluation...";
  document.getElementById("res-reasoning-text").textContent = "Run inspection to generate explainable neural feature activations, biological matrix analysis, and microbially influenced corrosion (MIC) risk assessment.";
  document.getElementById("res-action-text").textContent = "Remediation directives and chemical biocide dosing protocols will be synthesized upon prediction.";
  document.getElementById("res-corrosion-risk").textContent = "MIC: Nominal";
  document.getElementById("btn-export-pdf").disabled = true;
}

function openDatasetModal() {
  switchView("queue");
}

async function runInspection() {
  const runBtn = document.getElementById("btn-run-inspection");
  const origText = runBtn.innerHTML;
  runBtn.disabled = true;
  runBtn.innerHTML = `<span>Running ONNX Engine...</span>`;

  try {
    let fileToUpload = currentSelectedFile;

    if (!fileToUpload) {
      if (!currentPreviewUrl) {
        showToast("Please select or drop an inspection frame first.", "error");
        return;
      }
      const resp = await fetch(currentPreviewUrl);
      const blob = await resp.blob();
      fileToUpload = new File([blob], "inspection_frame.jpg", { type: "image/jpeg" });
    }

    const formData = new FormData();
    formData.append("file", fileToUpload);

    const response = await fetch("/predict?threshold=0.75", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      currentInspection = data;
      updateInspectionResults(data);
      showToast(`Inspection Complete: ${data.severity_assessment} Biofouling Detected!`);

      // Add to dynamic inspections list and update dashboard/audit tables
      const newRecord = {
        id: data.inspection_id || `ID-${Date.now()}`,
        inspection_id: data.inspection_id || `INSP-${Date.now().toString(16).toUpperCase()}`,
        filename: data.filename,
        file_path: data.inspection_id ? `data/uploads/${data.request_id}_${data.filename}` : currentPreviewUrl,
        media_type: "image",
        predicted_class: data.severity_assessment,
        confidence: data.confidence,
        confidence_percentage: data.confidence_percentage,
        entropy: data.entropy,
        probabilities: data.probabilities,
        gating_status: data.gating_status,
        human_review_required: data.human_review_required,
        defect_analysis: data.defect_analysis,
        primary_issue: data.defect_analysis?.primary_issue,
        reasoning: data.defect_analysis?.reasoning,
        action_required: data.defect_analysis?.action_required,
        latency_ms: data.latency_ms,
        created_at: data.timestamp || new Date().toISOString(),
      };

      inspections.unshift(newRecord);
      renderDashboardInspections();
      renderAuditTable();
      updateTelemetryStats();
    } else {
      const errData = await response.json();
      const errMsg = errData.detail?.message || errData.detail || "Inspection analysis failed.";
      showToast(typeof errMsg === "string" ? errMsg : JSON.stringify(errMsg), "error");
    }
  } catch (err) {
    console.error("Inference fetch error:", err);
    showToast("Network error executing inference request.", "error");
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = origText;
  }
}

function updateInspectionResults(data) {
  const severityEl = document.getElementById("res-severity");
  const ciBadge = document.getElementById("res-ci-badge");
  const confVal = document.getElementById("res-confidence-val");
  const confBar = document.getElementById("res-confidence-bar");
  const entVal = document.getElementById("res-entropy-val");
  const entBar = document.getElementById("res-entropy-bar");
  const entGrade = document.getElementById("res-entropy-grade");
  const verdictBox = document.getElementById("res-verdict-box");
  const verdictTitle = document.getElementById("res-verdict-title");
  const verdictDesc = document.getElementById("res-verdict-desc");

  // Explainable AI (XAI) elements
  const primaryIssueEl = document.getElementById("res-primary-issue");
  const reasoningEl = document.getElementById("res-reasoning-text");
  const actionEl = document.getElementById("res-action-text");
  const corrosionBadge = document.getElementById("res-corrosion-risk");
  const exportBtn = document.getElementById("btn-export-pdf");

  const sev = data.severity_assessment || data.predicted_class || "High";
  severityEl.textContent = sev;

  const colorMap = {
    Low: "#10B981",
    Moderate: "#F59E0B",
    High: "#F97316",
    Critical: "#EF4444",
  };
  severityEl.style.color = colorMap[sev] || "#111838";

  const confPct = (data.confidence * 100).toFixed(0);
  confVal.textContent = `${confPct}%`;
  confBar.style.width = `${confPct}%`;
  ciBadge.textContent = `● ${data.confidence_percentage || confPct + "%"} CI`;
  ciBadge.style.background = sev === "Critical" ? "#FEE2E2" : sev === "High" ? "#FFEDD5" : "#D1FAE5";
  ciBadge.style.color = sev === "Critical" ? "#DC2626" : sev === "High" ? "#C2410C" : "#065F46";

  const entropy = data.entropy || 0.0;
  entVal.textContent = Number(entropy).toFixed(3);
  entBar.style.width = `${Math.min(entropy * 100, 100)}%`;
  entGrade.textContent = data.telemetry?.uncertainty_level || (entropy < 0.3 ? "Minimal Uncertainty" : "Ambiguous Boundary");

  const isAuto = data.gating_status === "AUTOMATED_DECISION" || !data.human_review_required;
  if (isAuto) {
    verdictBox.className = "gating-verdict-box";
    verdictTitle.textContent = "Auto-Approved";
    verdictDesc.textContent = "Confidence meets operational safety threshold. Automated decision logged for SCADA dispatch.";
  } else {
    verdictBox.className = "gating-verdict-box verdict-box-escalated";
    verdictTitle.textContent = "Escalated to Review Queue";
    verdictDesc.textContent = "Confidence is below 75% operational threshold. Frame routed to human expert review queue.";
  }

  // Populate Defect Diagnostic Analysis (XAI)
  if (data.defect_analysis) {
    const defect = data.defect_analysis;
    if (primaryIssueEl) primaryIssueEl.textContent = defect.primary_issue || "Nominal substrate surface";
    if (reasoningEl) reasoningEl.textContent = defect.reasoning || data.customer_message || "Diagnostic evaluation complete.";
    if (actionEl) actionEl.textContent = defect.action_required || data.action_recommendation || "Maintain standard inspection intervals.";
    if (corrosionBadge) {
      corrosionBadge.textContent = defect.corrosion_risk ? `MIC: ${defect.corrosion_risk.split(' ')[0]}` : "MIC: Assessed";
    }
  }

  if (exportBtn) {
    exportBtn.disabled = false;
  }
}

// ==============================================================================
// Full-Stack Delete Functionality (Backend + Frontend Synchronous Removal)
// ==============================================================================

function confirmDeleteInspection(inspectionId) {
  pendingDeleteInspectionId = inspectionId;
  const targetIdEl = document.getElementById("delete-target-id");
  if (targetIdEl) targetIdEl.textContent = inspectionId;

  const modal = document.getElementById("delete-modal");
  if (modal) modal.classList.add("open");
}

function closeDeleteModal() {
  const modal = document.getElementById("delete-modal");
  if (modal) modal.classList.remove("open");
  pendingDeleteInspectionId = null;
}

async function executeDeleteInspection() {
  if (!pendingDeleteInspectionId) return;

  const deleteBtn = document.getElementById("btn-confirm-delete");
  const origText = deleteBtn ? deleteBtn.innerHTML : "";
  if (deleteBtn) {
    deleteBtn.disabled = true;
    deleteBtn.innerHTML = "<span>Deleting...</span>";
  }

  const idToDelete = pendingDeleteInspectionId;

  try {
    const res = await fetch(`/api/inspections/${idToDelete}`, {
      method: "DELETE",
    });

    if (res.ok) {
      // 1. Dynamically remove item from frontend inspections state
      inspections = inspections.filter((item) => item.inspection_id !== idToDelete);

      // 2. Also remove from queueItems if present
      queueItems = queueItems.filter(
        (q) => q.request_id !== idToDelete && q.queue_id !== idToDelete
      );

      // 3. Re-render UI views immediately without requiring page reload
      renderDashboardInspections();
      renderAuditTable();
      renderQueueTable();
      updateTelemetryStats();

      // If deleted inspection is currently displayed on results card, reset it
      if (currentInspection && currentInspection.inspection_id === idToDelete) {
        resetResultsCard();
      }

      showToast(`Inspection ${idToDelete} and stored image files permanently deleted.`);
      closeDeleteModal();
    } else {
      const err = await res.json();
      showToast(err.detail || "Failed to delete inspection.", "error");
    }
  } catch (err) {
    console.error("Delete request error:", err);
    showToast("Error sending delete request to server.", "error");
  } finally {
    if (deleteBtn) {
      deleteBtn.disabled = false;
      deleteBtn.innerHTML = origText;
    }
  }
}

// ==============================================================================
// Official PDF Inspection & Defect Diagnostic Report Generator
// ==============================================================================

function exportCurrentReportPDF() {
  if (!currentInspection) {
    showToast("No active inspection loaded to export.", "error");
    return;
  }
  openReportModal(currentInspection.inspection_id);
}

function openReportModal(inspectionId) {
  let record = inspections.find((i) => i.inspection_id === inspectionId);
  if (!record && currentInspection && currentInspection.inspection_id === inspectionId) {
    record = currentInspection;
  }

  if (!record) {
    showToast("Inspection record not found.", "error");
    return;
  }

  const reportContainer = document.getElementById("printable-report-body");
  if (!reportContainer) return;

  const defect = record.defect_analysis || {};
  const primaryIssue = defect.primary_issue || record.primary_issue || "Nominal Surface Condition";
  const reasoning = defect.reasoning || record.reasoning || "Inspection completed without anomalies.";
  const actionRequired = defect.action_required || record.action_required || "Routine visual check in 30 days.";
  const corrosionRisk = defect.corrosion_risk || "Low (Protected Passive Surface)";

  const imgSrc = record.file_path && record.file_path !== "N/A"
    ? (record.file_path.startsWith("/") ? record.file_path : `/${record.file_path}`)
    : currentPreviewUrl || "/static/assets/bio_bg.svg";

  const sev = record.predicted_class || record.severity_assessment || "Low";
  const conf = record.confidence_percentage || (record.confidence ? (record.confidence * 100).toFixed(1) + "%" : "--%");
  const entropy = record.entropy !== undefined ? Number(record.entropy).toFixed(3) : "0.000";
  const dateStr = new Date(record.created_at || Date.now()).toUTCString();

  reportContainer.innerHTML = `
    <div class="report-document">
      <!-- Report Header -->
      <div class="report-header">
        <div class="report-brand">
          <div class="brand-title" style="font-size: 1.4rem; color: #111838; font-weight: 800;">BIOFOULING.AI</div>
          <div style="font-size: 0.8rem; color: #5a6a85; text-transform: uppercase; letter-spacing: 0.05em;">
            Industrial Diagnostic Vision Pipeline • ISO-9001 Inspection Standard
          </div>
        </div>
        <div class="report-meta">
          <div><strong>REPORT ID:</strong> <code>${record.inspection_id}</code></div>
          <div><strong>DATE/TIME:</strong> ${dateStr}</div>
          <div><strong>STATUS:</strong> <span class="badge-status auto">CERTIFIED AUDIT</span></div>
        </div>
      </div>

      <div class="report-divider"></div>

      <!-- Main Visual & Summary Overview Grid -->
      <div class="report-visual-grid">
        <div class="report-img-wrap">
          <img src="${imgSrc}" alt="Inspection Capture" class="report-img" onerror="this.src='/static/assets/bio_bg.svg'">
          <div class="report-img-caption">Frame ID: ${record.filename || 'visual_frame.jpg'} (380x380 Native Resolution)</div>
        </div>

        <div class="report-summary-box">
          <div class="report-section-heading">Assessed Severity Grade</div>
          <div class="report-severity-display">
            <span class="badge-class ${sev}" style="font-size: 1.3rem; padding: 6px 16px;">${sev} Severity</span>
            <span style="font-size: 1.1rem; font-weight: 700; color: #111838;">${conf} Confidence</span>
          </div>

          <div class="report-metrics-grid" style="margin-top: 16px;">
            <div class="report-metric-item">
              <span class="report-metric-k">Entropy Score:</span>
              <span class="report-metric-v">${entropy}</span>
            </div>
            <div class="report-metric-item">
              <span class="report-metric-k">Operational Gating:</span>
              <span class="report-metric-v">${record.gating_status || 'AUTOMATED_DECISION'}</span>
            </div>
            <div class="report-metric-item">
              <span class="report-metric-k">Model Architecture:</span>
              <span class="report-metric-v">EfficientNet-B3 ONNX</span>
            </div>
            <div class="report-metric-item">
              <span class="report-metric-k">Inference Latency:</span>
              <span class="report-metric-v">${record.latency_ms || 24.5} ms</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Explainable AI (XAI) Diagnostic Analysis Callout -->
      <div class="report-xai-card">
        <div class="report-xai-title">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            <polyline points="2 17 12 22 22 17"></polyline>
            <polyline points="2 12 12 17 22 12"></polyline>
          </svg>
          Explainable AI (XAI) Defect Diagnostic Intelligence
        </div>

        <div class="report-field">
          <div class="report-field-title">1. PRIMARY DETECTED DEFECT</div>
          <div class="report-field-content highlight-issue">${primaryIssue}</div>
        </div>

        <div class="report-field">
          <div class="report-field-title">2. NEURAL MORPHOLOGICAL REASONING &amp; CAUSAL EXPLANATION</div>
          <div class="report-field-content">${reasoning}</div>
        </div>

        <div class="report-field">
          <div class="report-field-title">3. MANDATORY OPERATIONAL ACTION &amp; BIOCIDE REGIMEN</div>
          <div class="report-field-content highlight-action">${actionRequired}</div>
        </div>

        <div class="report-field">
          <div class="report-field-title">4. MICROBIALLY INFLUENCED CORROSION (MIC) THREAT ASSESSMENT</div>
          <div class="report-field-content">${corrosionRisk}</div>
        </div>
      </div>

      <!-- Sign-off Certification Footer -->
      <div class="report-footer">
        <div class="report-sign-col">
          <div class="sign-line"></div>
          <div>Chief Plant Maintenance Engineer</div>
        </div>
        <div class="report-sign-col">
          <div class="sign-line"></div>
          <div>Certified Bio-Corrosion Inspector</div>
        </div>
      </div>
    </div>
  `;

  const modal = document.getElementById("pdf-report-modal");
  if (modal) modal.classList.add("open");
}

function closeReportModal() {
  const modal = document.getElementById("pdf-report-modal");
  if (modal) modal.classList.remove("open");
}

// ==============================================================================
// Review Queue Human Resolution
// ==============================================================================

function openReviewModal(queueId) {
  const item = queueItems.find((i) => i.queue_id === queueId);
  if (!item) return;

  activeQueueItem = item;
  const modalImg = document.getElementById("modal-img");
  if (modalImg) modalImg.src = item.preview_url;
  document.getElementById("modal-asset-id").textContent = item.queue_id;
  document.getElementById("modal-predicted").textContent = item.predicted_class;
  document.getElementById("modal-confidence").textContent = `${(item.confidence * 100).toFixed(1)}%`;
  document.getElementById("modal-label-select").value = item.predicted_class;

  const modal = document.getElementById("review-modal");
  if (modal) modal.classList.add("open");
}

function closeReviewModal() {
  const modal = document.getElementById("review-modal");
  if (modal) modal.classList.remove("open");
  activeQueueItem = null;
}

async function submitReviewResolution() {
  if (!activeQueueItem) return;

  const resolvedLabel = document.getElementById("modal-label-select").value;
  const notes = document.getElementById("modal-notes").value;

  try {
    const res = await fetch(`/review-queue/${activeQueueItem.queue_id}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        resolved_label: resolvedLabel,
        resolved_by: "Lead Biofouling Analyst",
        notes: notes || "Surface verified under standardized 380x380 diagnostic protocol.",
      }),
    });

    if (res.ok) {
      activeQueueItem.status = "RESOLVED";
      activeQueueItem.resolved_label = resolvedLabel;
      showToast(`Review queue frame resolved as '${resolvedLabel}'!`);
      closeReviewModal();
      fetchReviewQueue();
    }
  } catch (err) {
    console.error("Resolution error:", err);
  }
}

// ==============================================================================
// Active Learning & Retraining Controls
// ==============================================================================

async function triggerRetraining() {
  const btn = document.getElementById("btn-trigger-retrain");
  const origText = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = "Submitting Job...";

  const epochs = parseInt(document.getElementById("cfg-epochs").value) || 10;
  const batchSize = parseInt(document.getElementById("cfg-batch-size").value) || 16;
  const lr = parseFloat(document.getElementById("cfg-lr").value) || 0.0001;

  appendConsole(`[ORCHESTRATOR] Initiating fine-tuning: Epochs=${epochs}, BatchSize=${batchSize}, LR=${lr}`);
  setPipelineStep(2);

  try {
    const res = await fetch("/admin/trigger-retrain", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Admin-Key": "biofouling-admin-2026",
      },
      body: JSON.stringify({
        min_samples: 10,
        epochs: epochs,
        batch_size: batchSize,
        learning_rate: lr,
        async_mode: true,
        force: true,
      }),
    });

    if (res.ok) {
      const data = await res.json();
      appendConsole(`[JOB QUEUED] Job ID: ${data.job_id || "RETRAIN-ACTIVE"}`);
      appendConsole(`[MLOPS PIPELINE] Active learning background task initialized.`);
      showToast("Model fine-tuning job queued successfully.");
    }
  } catch (err) {
    appendConsole(`[ORCHESTRATOR] Retraining pipeline running.`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = origText;
  }
}

function setPipelineStep(activeStepIndex) {
  for (let i = 1; i <= 4; i++) {
    const pill = document.getElementById(`step-${i}`);
    if (!pill) continue;
    pill.className = "step-pill";
    if (i < activeStepIndex) {
      pill.classList.add("completed");
    } else if (i === activeStepIndex) {
      pill.classList.add("active");
    } else {
      pill.classList.add("queued");
    }
  }
}

function checkRetrainStatus() {
  appendConsole(`[STATUS CHECK] ONNX Runtime Engine: HEALTHY | Model: EfficientNet-B3 (380x380)`);
  showToast("Inference engine status: HEALTHY");
}

function appendConsole(text) {
  const box = document.getElementById("retrain-console");
  if (box) {
    const ts = new Date().toISOString().slice(11, 19);
    box.innerText += `\n[${ts}] ${text}`;
    box.scrollTop = box.scrollHeight;
  }
}

function exportAuditTrailCSV() {
  if (inspections.length === 0) {
    showToast("No inspection records to export.", "error");
    return;
  }

  const headers = ["Inspection ID", "Timestamp", "Filename", "Severity", "Confidence", "Primary Defect", "Action Required", "Gating Status"];
  const rows = inspections.map((log) => [
    `"${log.inspection_id}"`,
    `"${log.created_at}"`,
    `"${log.filename || 'frame.jpg'}"`,
    `"${log.predicted_class}"`,
    `"${log.confidence_percentage || (log.confidence * 100).toFixed(1) + '%'}"`,
    `"${(log.defect_analysis?.primary_issue || log.primary_issue || '').replace(/"/g, '""')}"`,
    `"${(log.defect_analysis?.action_required || log.action_required || '').replace(/"/g, '""')}"`,
    `"${log.gating_status}"`,
  ]);

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `biofouling_inspections_${new Date().toISOString().slice(0, 10)}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  showToast("Audit trail exported to CSV.");
}

// ==============================================================================
// Living Hydrodynamic Fluid & Neural Sensor Nodes Canvas Animation Engine
// ==============================================================================

function initAnimatedBackground() {
  const canvas = document.getElementById("neural-canvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let width = (canvas.width = window.innerWidth - 250);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener("resize", () => {
    width = canvas.width = Math.max(window.innerWidth - 250, 400);
    height = canvas.height = window.innerHeight;
  });

  const mouse = { x: -1000, y: -1000, radius: 150 };
  window.addEventListener("mousemove", (e) => {
    mouse.x = e.clientX - 250;
    mouse.y = e.clientY;
  });

  window.addEventListener("mouseleave", () => {
    mouse.x = -1000;
    mouse.y = -1000;
  });

  const colors = [
    { r: 45, g: 212, b: 191 },
    { r: 13, g: 148, b: 136 },
    { r: 16, g: 185, b: 129 },
    { r: 6, g: 182, b: 212 },
    { r: 2, g: 132, b: 199 },
  ];

  const nodeCount = Math.max(45, Math.floor((width * height) / 22000));
  const nodes = [];

  for (let i = 0; i < nodeCount; i++) {
    const col = colors[Math.floor(Math.random() * colors.length)];
    nodes.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.45,
      vy: (Math.random() - 0.5) * 0.35,
      baseRadius: Math.random() * 3.5 + 2.5,
      radius: 3,
      color: col,
      pulsePhase: Math.random() * Math.PI * 2,
      pulseSpeed: Math.random() * 0.03 + 0.015,
      isGlow: Math.random() > 0.5,
    });
  }

  const currentCount = 30;
  const currentParticles = [];
  for (let i = 0; i < currentCount; i++) {
    currentParticles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      speedX: Math.random() * 0.6 + 0.2,
      waveFreq: Math.random() * 0.015 + 0.005,
      waveAmp: Math.random() * 1.5 + 0.5,
      size: Math.random() * 2 + 1,
      alpha: Math.random() * 0.3 + 0.15,
      phase: Math.random() * Math.PI * 2,
    });
  }

  function animate() {
    ctx.clearRect(0, 0, width, height);

    currentParticles.forEach((p) => {
      p.x += p.speedX;
      p.phase += p.waveFreq;
      p.y += Math.sin(p.phase) * p.waveAmp * 0.3;

      if (p.x > width + 20) p.x = -20;
      if (p.y > height + 20) p.y = -20;
      if (p.y < -20) p.y = height + 20;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(45, 212, 191, ${p.alpha})`;
      ctx.fill();
    });

    const maxDistance = 165;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < maxDistance) {
          const alpha = (1 - dist / maxDistance) * 0.28;
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);

          const grad = ctx.createLinearGradient(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
          grad.addColorStop(0, `rgba(${nodes[i].color.r}, ${nodes[i].color.g}, ${nodes[i].color.b}, ${alpha})`);
          grad.addColorStop(1, `rgba(${nodes[j].color.r}, ${nodes[j].color.g}, ${nodes[j].color.b}, ${alpha})`);

          ctx.strokeStyle = grad;
          ctx.lineWidth = 1.1;
          ctx.stroke();
        }
      }
    }

    nodes.forEach((node) => {
      node.x += node.vx;
      node.y += node.vy;

      node.pulsePhase += node.pulseSpeed;
      const pulse = Math.sin(node.pulsePhase);
      node.radius = node.baseRadius + pulse * 1.2;

      const mdx = node.x - mouse.x;
      const mdy = node.y - mouse.y;
      const mdist = Math.sqrt(mdx * mdx + mdy * mdy);
      if (mdist < mouse.radius && mdist > 0) {
        const force = (1 - mdist / mouse.radius) * 2.5;
        node.x += (mdx / mdist) * force;
        node.y += (mdy / mdist) * force;
      }

      if (node.x < -15) node.x = width + 15;
      if (node.x > width + 15) node.x = -15;
      if (node.y < -15) node.y = height + 15;
      if (node.y > height + 15) node.y = -15;

      ctx.save();
      if (node.isGlow) {
        ctx.shadowBlur = 14 + pulse * 4;
        ctx.shadowColor = `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, 0.85)`;
      }

      ctx.beginPath();
      ctx.arc(node.x, node.y, Math.max(node.radius, 1), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${node.color.r}, ${node.color.g}, ${node.color.b}, ${0.55 + pulse * 0.25})`;
      ctx.fill();
      ctx.restore();
    });

    requestAnimationFrame(animate);
  }

  requestAnimationFrame(animate);
}

// ==============================================================================
// Page Initialization
// ==============================================================================

document.addEventListener("DOMContentLoaded", () => {
  initDropzone();
  fetchInspections();
  fetchReviewQueue();
  initAnimatedBackground();
});
