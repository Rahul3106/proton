/**
 * PROTON Web Interface Controller
 * Full point-and-click interaction layer, Rule 14 Explain-Before-Apply cleaning guardrail,
 * dataset schema explorer, single-row interactive inference, model leaderboard,
 * SSE live-streaming progress checklists, and session persistence.
 */

// ── GLOBAL STATE ──────────────────────────────────────────────────────────────
let currentSessionId = getInitialSessionId();
let uploadRole = "train"; // 'train' or 'test'
let activeDatasetColumns = [];
let activeDatasetDtypes = {};
let sampleViewerPage = 1;

const BUILTIN_COMMANDS = [
  "analyze dataset",
  "audit data quality",
  "detect missing values",
  "detect duplicates",
  "inspect data types",
  "clean dataset",
  "train model",
  "train decision tree",
  "train random forest",
  "train gradient boosting",
  "train logistic regression",
  "compare models",
  "predict",
  "explain model",
  "export model",
  "export predictions",
  "plot correlation matrix",
  "plot feature importance",
  "generate report",
  "help"
];

// ── DOM ELEMENTS ──────────────────────────────────────────────────────────────
const sessionTag = document.getElementById("sessionTag");
const llmPill = document.getElementById("llmPill");
const llmDot = document.getElementById("llmDot");
const llmText = document.getElementById("llmText");
const trainDatasetStatus = document.getElementById("trainDatasetStatus");
const testDatasetStatus = document.getElementById("testDatasetStatus");
const btnViewData = document.getElementById("btnViewData");
const btnRevertData = document.getElementById("btnRevertData");
const promptInput = document.getElementById("promptInput");
const sendBtn = document.getElementById("sendBtn");
const terminalFeed = document.getElementById("terminalFeed");
const modalOverlay = document.getElementById("modalOverlay");
const chipsBar = document.getElementById("chipsBar");

const mlTargetSelect = document.getElementById("mlTargetSelect");
const vizColSelect = document.getElementById("vizColSelect");


// ── INITIALIZATION ────────────────────────────────────────────────────────────

function getInitialSessionId() {
  const urlParams = new URLSearchParams(window.location.search);
  const qSession = urlParams.get("session");
  if (qSession) {
    localStorage.setItem("proton_session_id", qSession);
    return qSession;
  }
  return localStorage.getItem("proton_session_id") || "";
}

async function initSession() {
  if (!currentSessionId) {
    try {
      const res = await fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" }
      });
      if (res.ok) {
        const data = await res.json();
        currentSessionId = data.session_id;
        localStorage.setItem("proton_session_id", currentSessionId);
      }
    } catch (err) {
      console.warn("Could not create session, using default:", err);
      currentSessionId = "default";
    }
  }

  if (sessionTag) {
    sessionTag.textContent = `Session: ${currentSessionId || "default"}`;
  }

  renderChips(BUILTIN_COMMANDS.slice(0, 8));
  await updateSystemStatus();

  // Attach listener to mlTargetSelect to immediately sync selected target with backend
  if (mlTargetSelect) {
    mlTargetSelect.addEventListener("change", async (e) => {
      await syncTargetColumn(e.target.value);
    });
  }
}

/**
 * Syncs the selected target column to the backend session state.
 */
async function syncTargetColumn(targetCol) {
  if (!currentSessionId) return;
  try {
    await fetch(`/api/session/${currentSessionId}/target-column`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_column: targetCol || null })
    });
  } catch (err) {
    console.warn("Could not sync target column:", err);
  }
}

/**
 * Fetch and update live system status from API
 */
async function updateSystemStatus() {
  try {
    const configRes = await fetch("/api/config");
    if (configRes.ok) {
      const config = await configRes.json();
      const hasKey = config.api_key_status && !config.api_key_status.includes("Not Configured");
      
      if (hasKey) {
        llmPill.className = "status-pill pill-online";
        llmDot.className = "status-dot dot-green";
        llmText.textContent = `● ONLINE ${config.gemini_model || "Gemini 1.5 Flash"} (AI Active)`;
      } else {
        llmPill.className = "status-pill pill-heuristic";
        llmDot.className = "status-dot dot-yellow";
        llmText.textContent = "● HEURISTIC MODE (Rule Engine Active)";
      }
    }

    if (currentSessionId) {
      const sessRes = await fetch(`/api/session/${currentSessionId}`);
      if (sessRes.ok) {
        const sess = await sessRes.json();

        if (sess.has_dataset && sess.current_dataset) {
          const ds = sess.current_dataset;
          activeDatasetColumns = Array.isArray(ds.columns) ? ds.columns : [];
          activeDatasetDtypes = ds.dtypes || {};
          populateColumnDropdowns(activeDatasetColumns, sess.target_column);

          if (btnViewData) btnViewData.disabled = false;
          if (btnRevertData) btnRevertData.disabled = false;

          trainDatasetStatus.innerHTML = `
            <span class="status-pill pill-loaded">
              <span class="status-dot dot-green"></span>
              <span><strong>✓ Loaded:</strong> ${escapeHtml(ds.name)} <span style="color: var(--text-dim); font-size: 11px;">(${ds.rows.toLocaleString()} rows × ${ds.columns_count} cols)</span></span>
            </span>
          `;
        } else {
          if (btnViewData) btnViewData.disabled = true;
          if (btnRevertData) btnRevertData.disabled = true;

          trainDatasetStatus.innerHTML = `
            <span class="status-pill pill-empty">
              <span class="status-dot dot-yellow"></span>
              <span>○ None <span style="color: var(--text-muted); font-weight: normal;">(Drop file or click Ingestion below)</span></span>
            </span>
          `;
        }

        if (sess.has_test_dataset && sess.test_dataset) {
          const tds = sess.test_dataset;
          testDatasetStatus.innerHTML = `
            <span class="status-pill pill-loaded">
              <span class="status-dot dot-green"></span>
              <span><strong>✓ Loaded:</strong> ${escapeHtml(tds.name)} <span style="color: var(--text-dim); font-size: 11px;">(${tds.rows.toLocaleString()} rows × ${tds.columns_count} cols)</span></span>
            </span>
          `;
        } else {
          testDatasetStatus.innerHTML = `
            <span class="status-pill pill-empty">
              <span>○ None</span>
            </span>
          `;
        }
      }
    }
  } catch (err) {
    console.error("Failed to update status:", err);
  }
}

function populateColumnDropdowns(columns, selectedTarget = null) {
  if (!columns || !Array.isArray(columns) || !columns.length) return;
  activeDatasetColumns = columns;

  if (mlTargetSelect) {
    const prevVal = selectedTarget || mlTargetSelect.value;
    mlTargetSelect.innerHTML = '<option value="">-- Auto-Detect Target Column --</option>';
    columns.forEach(col => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      mlTargetSelect.appendChild(opt);
    });

    // Auto-select likely target column if found
    const likelyTargets = ["target", "churn", "species", "label", "outcome", "price", "class", "survived", "status"];
    const matched = columns.find(c => likelyTargets.includes(c.toLowerCase()));
    if (prevVal && columns.includes(prevVal)) {
      mlTargetSelect.value = prevVal;
    } else if (matched) {
      mlTargetSelect.value = matched;
    }
  }

  if (vizColSelect) {
    const prevVal = vizColSelect.value;
    vizColSelect.innerHTML = '<option value="">Select a column...</option>';
    columns.forEach(col => {
      const opt = document.createElement("option");
      opt.value = col;
      opt.textContent = col;
      vizColSelect.appendChild(opt);
    });
    if (prevVal && columns.includes(prevVal)) {
      vizColSelect.value = prevVal;
    }
  }
}


// ── MODAL MANAGEMENT ──────────────────────────────────────────────────────────

async function openModal(modalId) {
  closeAllModals();
  modalOverlay.classList.add("active");
  const modal = document.getElementById(modalId);
  if (modal) modal.style.display = "flex";

  // Ensure dropdowns are fully updated when ML/Viz modals open
  if (modalId === "modalMl" || modalId === "modalViz") {
    if (!activeDatasetColumns.length) {
      await updateSystemStatus();
    } else {
      populateColumnDropdowns(activeDatasetColumns);
    }
  }

  // If opening Machine Learning modal, fetch top 5 tailored recommendations
  if (modalId === "modalMl") {
    await fetchModelRecommendations(mlTargetSelect?.value);
  }
}


function closeAllModals() {
  if (modalOverlay) modalOverlay.classList.remove("active");
  document.querySelectorAll(".guided-modal").forEach(m => m.style.display = "none");
}

if (modalOverlay) {
  modalOverlay.addEventListener("click", (e) => {
    if (e.target === modalOverlay) closeAllModals();
  });
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeAllModals();
});

// Card Click Bindings
document.getElementById("btnOpenIngest")?.addEventListener("click", () => openModal("modalIngest"));
document.getElementById("btnOpenEda")?.addEventListener("click", () => openModal("modalEda"));
document.getElementById("btnOpenClean")?.addEventListener("click", () => openModal("modalClean"));
document.getElementById("btnOpenMl")?.addEventListener("click", () => openModal("modalMl"));
document.getElementById("btnOpenViz")?.addEventListener("click", () => openModal("modalViz"));
document.getElementById("btnOpenReport")?.addEventListener("click", () => openModal("modalReport"));


// ── DATASET INGESTION & UPLOAD ────────────────────────────────────────────────

function setUploadRole(role) {
  uploadRole = role;
  const btnTrain = document.getElementById("btnRoleTrain");
  const btnTest = document.getElementById("btnRoleTest");
  if (role === "train") {
    btnTrain.style.borderColor = "var(--cyan-primary)";
    btnTrain.style.color = "#ffffff";
    btnTest.style.borderColor = "var(--border-subtle)";
    btnTest.style.color = "var(--text-dim)";
  } else {
    btnTest.style.borderColor = "var(--cyan-primary)";
    btnTest.style.color = "#ffffff";
    btnTrain.style.borderColor = "var(--border-subtle)";
    btnTrain.style.color = "var(--text-dim)";
  }
}

async function handleFileSelect(event) {
  const file = event.target.files[0];
  if (!file) return;
  await uploadFile(file);
}

const dropzone = document.getElementById("dropzoneArea");
if (dropzone) {
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", async (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      await uploadFile(e.dataTransfer.files[0]);
    }
  });
}

async function uploadFile(file) {
  closeAllModals();
  const formData = new FormData();
  formData.append("file", file);
  formData.append("dataset_type", uploadRole);

  const cardId = appendFeedCard(`Ingesting ${escapeHtml(file.name)}...`, `Uploading ${uploadRole} dataset with multi-format and encoding analysis...`, "running");

  try {
    const res = await fetch(`/api/session/${currentSessionId}/load`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (res.ok && data.success) {
      const colsList = data.columns_list || (data.sample_preview && data.sample_preview.length ? Object.keys(data.sample_preview[0]) : []);
      if (colsList.length) {
        populateColumnDropdowns(colsList);
      }

      updateFeedCard(
        cardId,
        `${uploadRole.toUpperCase()} Dataset Loaded: ${data.name}`,
        `
        <div style="margin-bottom: 8px; display: flex; gap: 12px; flex-wrap: wrap;">
          <span class="status-pill pill-loaded"><strong>Rows:</strong> ${data.rows.toLocaleString()}</span>
          <span class="status-pill pill-loaded"><strong>Columns:</strong> ${data.columns}</span>
          <span class="status-pill pill-loaded"><strong>Encoding:</strong> ${data.encoding_used || "utf-8"}</span>
          <span class="status-pill ${data.total_missing > 0 ? 'pill-heuristic' : 'pill-online'}"><strong>Missing Cells:</strong> ${data.total_missing || 0}</span>
        </div>
        ${renderSampleTableHtml(data.sample_preview)}
        `,
        "success"
      );
      await updateSystemStatus();
    } else {
      updateFeedCard(cardId, `Ingestion Failed`, data.detail || data.error || "Failed to load file.", "error");
    }
  } catch (err) {
    updateFeedCard(cardId, `Ingestion Error`, String(err), "error");
  }
}

async function handleManualPathLoad() {
  const pathInput = document.getElementById("manualFilePath");
  const path = pathInput.value.trim();
  if (!path) return;

  closeAllModals();
  const cardId = appendFeedCard(`Ingesting Dataset from Path`, `Reading <code>${escapeHtml(path)}</code>...`, "running");

  try {
    const res = await fetch(`/api/session/${currentSessionId}/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filepath: path, dataset_type: uploadRole })
    });
    const data = await res.json();
    if (res.ok && data.success) {
      const colsList = data.columns_list || (data.sample_preview && data.sample_preview.length ? Object.keys(data.sample_preview[0]) : []);
      if (colsList.length) {
        populateColumnDropdowns(colsList);
      }

      updateFeedCard(
        cardId,
        `${uploadRole.toUpperCase()} Dataset Loaded: ${data.name}`,
        `
        <div style="margin-bottom: 8px; display: flex; gap: 12px; flex-wrap: wrap;">
          <span class="status-pill pill-loaded"><strong>Rows:</strong> ${data.rows.toLocaleString()}</span>
          <span class="status-pill pill-loaded"><strong>Columns:</strong> ${data.columns}</span>
          <span class="status-pill pill-loaded"><strong>Encoding:</strong> ${data.encoding_used || "utf-8"}</span>
        </div>
        ${renderSampleTableHtml(data.sample_preview)}
        `,
        "success"
      );
      pathInput.value = "";
      await updateSystemStatus();
    } else {
      updateFeedCard(cardId, `Load Error`, data.detail || "Could not read file.", "error");
    }
  } catch (err) {
    updateFeedCard(cardId, `Error`, String(err), "error");
  }
}


// ── PHASE 4: DATASET & SCHEMA EXPLORER ────────────────────────────────────────

async function openDataViewer(page = 1) {
  sampleViewerPage = page;
  openModal("modalDataViewer");
  await loadSamplePage(page);
}

async function changeSamplePage(delta) {
  sampleViewerPage = Math.max(1, sampleViewerPage + delta);
  await loadSamplePage(sampleViewerPage);
}

async function loadSamplePage(page) {
  const wrap = document.getElementById("viewerTableWrap");
  const statsHeader = document.getElementById("viewerStatsHeader");
  const pageLabel = document.getElementById("viewerPageLabel");
  if (!wrap) return;

  wrap.innerHTML = '<div style="color: var(--text-dim); padding: 12px;">Loading sample records...</div>';

  try {
    const res = await fetch(`/api/session/${currentSessionId}/sample?page=${page}&page_size=10`);
    if (res.ok) {
      const data = await res.json();
      statsHeader.textContent = `${data.dataset_name} — Total: ${data.total_rows.toLocaleString()} rows × ${data.columns.length} columns`;
      pageLabel.textContent = `Page ${data.page} of ${data.total_pages}`;
      wrap.innerHTML = renderSampleTableHtml(data.records, data.dtypes);
    } else {
      wrap.innerHTML = '<div style="color: var(--yellow-warn); padding: 12px;">No active dataset loaded.</div>';
    }
  } catch (err) {
    wrap.innerHTML = `<div style="color: var(--red-alert); padding: 12px;">Failed to load sample: ${err}</div>`;
  }
}


// ── PHASE 5: RULE 14 CLEANING GUARDRAIL (PREVIEW-BEFORE-APPLY & REVERT) ───────

async function triggerCleanPreview() {
  closeAllModals();
  const cardId = appendFeedCard(`Inspecting Cleaning Transformations (Dry-Run)`, `Evaluating missing values, duplicate records, and outliers without mutating data...`, "running");

  try {
    const res = await fetch(`/api/session/${currentSessionId}/tool/clean_dataset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params: { dry_run: true } })
    });
    const result = await res.json();

    if (result.success) {
      const d = result.data;
      const previewHtml = `
        <div class="diff-card">
          <div class="diff-header">
            <span>[RULE 14 GUARDRAIL] Proposed Cleaning Diff</span>
            <span>STATUS: PENDING USER CONFIRMATION</span>
          </div>
          <div style="font-size: 12.5px; color: var(--text-white);">
            <strong>Initial Shape:</strong> ${d.initial_shape?.[0]} rows × ${d.initial_shape?.[1]} cols &rarr; 
            <strong>Cleaned Shape:</strong> ${d.cleaned_shape?.[0]} rows × ${d.cleaned_shape?.[1]} cols
          </div>
          <div style="color: var(--text-dim); font-size: 12px;">
            ${escapeHtml(d.message || "Proposed automated cleaning pipeline.")}
          </div>
          <div class="diff-actions-bar">
            <button class="action-btn-primary" style="background: linear-gradient(135deg, #059669, #10b981);" onclick="applyCleanChanges('${cardId}')">
              Apply These Changes
            </button>
            <button class="action-btn-secondary" style="border-color: var(--red-alert); color: var(--red-alert);" onclick="cancelCleanPreview('${cardId}')">
              Skip / Dismiss
            </button>
          </div>
        </div>
      `;
      updateFeedCard(cardId, `Proposed Dataset Cleaning Changes`, previewHtml, "info");
    } else {
      updateFeedCard(cardId, `Preview Failed`, result.error || "Could not analyze cleaning diff.", "error");
    }
  } catch (err) {
    updateFeedCard(cardId, `Error`, String(err), "error");
  }
}

async function applyCleanChanges(cardId) {
  updateFeedCard(cardId, `Applying Cleaning Transformations...`, `Mutating active dataset in memory...`, "running");

  try {
    const res = await fetch(`/api/session/${currentSessionId}/tool/clean_dataset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params: { dry_run: false } })
    });
    const result = await res.json();

    if (result.success) {
      updateFeedCard(cardId, `Dataset Cleaned Successfully`, `
        <div style="color: var(--green-online); font-weight: 600; margin-bottom: 6px;">
          ${escapeHtml(result.data.message || "Cleaned dataset applied to active session.")}
        </div>
        <div style="color: var(--text-dim); font-size: 12px;">
          Dataset is updated. You can click <strong>Revert to Original</strong> at any time to restore the initial raw dataset.
        </div>
      `, "success");
      await updateSystemStatus();
    } else {
      updateFeedCard(cardId, `Cleaning Failed`, result.error || "Error applying transformations.", "error");
    }
  } catch (err) {
    updateFeedCard(cardId, `Error`, String(err), "error");
  }
}

function cancelCleanPreview(cardId) {
  updateFeedCard(cardId, `Cleaning Skipped`, `<div style="color: var(--text-muted);">No modifications were made to the dataset.</div>`, "info");
}

async function handleRevertDataset() {
  closeAllModals();
  const cardId = appendFeedCard(`Reverting Dataset...`, `Restoring original unmodified file state...`, "running");

  try {
    const res = await fetch(`/api/session/${currentSessionId}/revert`, { method: "POST" });
    const data = await res.json();
    if (res.ok && data.success) {
      updateFeedCard(cardId, `Dataset Reverted`, `
        <div style="color: var(--green-online); font-weight: 600;">
          ${escapeHtml(data.message)}
        </div>
        <div style="color: var(--text-dim); font-size: 12px; margin-top: 4px;">
          Active dataset restored to <strong>${data.rows.toLocaleString()} rows × ${data.columns} cols</strong>.
        </div>
      `, "success");
      await updateSystemStatus();
    } else {
      updateFeedCard(cardId, `Revert Failed`, data.detail || "Could not revert dataset.", "error");
    }
  } catch (err) {
    updateFeedCard(cardId, `Error`, String(err), "error");
  }
}


// ── PHASE 6: MACHINE LEARNING & SINGLE-ROW PREDICTOR ──────────────────────────

async function handleTargetColumnChange() {
  const targetCol = mlTargetSelect?.value || "";
  await syncTargetColumn(targetCol);
  await fetchModelRecommendations(targetCol);
}

async function fetchModelRecommendations(targetCol = null) {
  if (!currentSessionId) return;
  const col = targetCol || (mlTargetSelect ? mlTargetSelect.value : "");
  const top5Container = document.getElementById("mlTop5Chips");
  const problemBadge = document.getElementById("mlProblemBadge");
  const optgroupSuggested = document.getElementById("optgroupSuggested");
  const mlSelect = document.getElementById("mlModelSelect");

  try {
    const url = `/api/session/${currentSessionId}/suggest-models${col ? `?target_column=${encodeURIComponent(col)}` : ''}`;
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();
    if (!data.has_dataset) return;

    if (problemBadge) {
      const isClf = data.problem_type === "classification";
      problemBadge.textContent = isClf ? "● Classification Task" : "● Regression Task";
      problemBadge.className = isClf ? "status-pill pill-online" : "status-pill pill-heuristic";
    }

    if (top5Container && data.top_5_models && data.top_5_models.length) {
      top5Container.innerHTML = data.top_5_models.map((m, idx) => `
        <div class="action-btn-secondary" style="display: flex; flex-direction: column; align-items: flex-start; text-align: left; padding: 7px 9px; cursor: pointer; border-color: rgba(168, 85, 247, 0.4); background: rgba(168, 85, 247, 0.08); transition: all 0.15s ease;" onclick="selectRecommendedModel('${m.model_type}')" title="${escapeHtml(m.rationale)}">
          <div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
            <span style="font-weight: 700; color: #ffffff; font-size: 11.5px;">${idx + 1}. ${escapeHtml(m.display_name.replace(' Classifier', '').replace(' Regressor', ''))}</span>
            <span style="font-size: 9.5px; color: #c084fc; font-weight: 700;">${escapeHtml(m.badge)}</span>
          </div>
          <div style="font-size: 10px; color: var(--text-dim); margin-top: 2px; line-height: 1.25;">${escapeHtml(m.rationale)}</div>
        </div>
      `).join("");
    }

    if (optgroupSuggested && data.top_5_models && data.top_5_models.length) {
      const currentSelected = mlSelect ? mlSelect.value : "";
      optgroupSuggested.innerHTML = data.top_5_models.map(m => `
        <option value="${m.model_type}">★ [Recommended] ${escapeHtml(m.display_name)} (${escapeHtml(m.tag)})</option>
      `).join("");
      if (currentSelected && mlSelect) {
        mlSelect.value = currentSelected;
      }
    }
  } catch (err) {
    console.warn("Failed to fetch model recommendations:", err);
  }
}

function selectRecommendedModel(modelType) {
  const mlSelect = document.getElementById("mlModelSelect");
  if (mlSelect) {
    mlSelect.value = modelType;
    mlSelect.style.borderColor = "var(--purple-accent)";
    setTimeout(() => { mlSelect.style.borderColor = "var(--border-subtle)"; }, 1200);
  }
}


function handleMlSubmit() {
  const targetCol = mlTargetSelect?.value || "";
  const modelType = document.getElementById("mlModelSelect")?.value || "compare_models";
  const tune = document.getElementById("mlTuneCheckbox")?.checked || false;

  closeAllModals();

  if (modelType === "compare_models") {
    runToolAction("compare_models", { target_column: targetCol });
  } else {
    runToolAction("train_model", {
      target_column: targetCol,
      model_type: modelType,
      tune: tune,
    });
  }
}

function openSingleRowPredictionModal() {
  closeAllModals();
  openModal("modalSinglePredict");

  const container = document.getElementById("singleRowFormContainer");
  const resultBox = document.getElementById("singleRowResultBox");
  if (!container) return;

  container.innerHTML = "";
  if (resultBox) resultBox.style.display = "none";

  const targetCol = mlTargetSelect?.value || "";
  const features = activeDatasetColumns.filter(c => c !== targetCol);

  if (!features.length) {
    container.innerHTML = '<div style="color: var(--yellow-warn); grid-column: 1 / -1; padding: 12px;">Please load a dataset and train a model first.</div>';
    return;
  }

  features.forEach(feat => {
    const group = document.createElement("div");
    group.className = "form-group";
    group.innerHTML = `
      <label class="form-label" for="field_${escapeHtml(feat)}">${escapeHtml(feat)}:</label>
      <input type="text" id="field_${escapeHtml(feat)}" class="form-input single-row-field" data-feature="${escapeHtml(feat)}" placeholder="Enter ${escapeHtml(feat)} value...">
    `;
    container.appendChild(group);
  });
}

async function populateRandomSingleRowData() {
  try {
    const res = await fetch(`/api/session/${currentSessionId}/sample?page=1&page_size=1`);
    if (res.ok) {
      const data = await res.json();
      if (data.records && data.records.length > 0) {
        const row = data.records[0];
        document.querySelectorAll(".single-row-field").forEach(input => {
          const feat = input.getAttribute("data-feature");
          if (row[feat] !== undefined) {
            input.value = row[feat];
          }
        });
      }
    }
  } catch (err) {
    console.error("Could not fetch sample data:", err);
  }
}

let lastPredictedFeatures = {};

async function submitSingleRowPrediction() {
  const fields = document.querySelectorAll(".single-row-field");
  const featuresObj = {};
  fields.forEach(input => {
    const feat = input.getAttribute("data-feature");
    const val = input.value.trim();
    featuresObj[feat] = isNaN(val) || val === "" ? val : Number(val);
  });
  lastPredictedFeatures = { ...featuresObj };

  const resultBox = document.getElementById("singleRowResultBox");
  if (resultBox) {
    resultBox.style.display = "block";
    resultBox.innerHTML = '<div style="color: var(--cyan-light);">Running model inference...</div>';
  }

  try {
    const res = await fetch(`/api/session/${currentSessionId}/predict-row`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: featuresObj })
    });
    const data = await res.json();

    if (res.ok && data.success) {
      let probsHtml = "";
      if (data.probabilities) {
        probsHtml = `
          <div style="margin-top: 6px; font-size: 11px; color: var(--text-dim);">
            <strong>Probabilities:</strong> ${Object.entries(data.probabilities).map(([c, p]) => `<span style="color: var(--cyan-light); margin-left: 6px;">${escapeHtml(c)}: ${(p * 100).toFixed(1)}%</span>`).join(" | ")}
          </div>
        `;
      }

      resultBox.innerHTML = `
        <div style="color: var(--cyan-light); font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px;">
          ▲ Model Prediction (${escapeHtml(data.model_name)})
        </div>
        <div style="font-size: 20px; font-weight: 700; color: var(--green-online); margin: 4px 0;">
          ${escapeHtml(data.prediction)}
        </div>
        <div style="font-size: 11.5px; color: var(--text-dim);">
          Target: <strong>${escapeHtml(data.target_column)}</strong>
        </div>
        ${probsHtml}
        <div style="margin-top: 10px; display: flex; gap: 8px;">
          <button type="button" id="explainPredictionBtn" onclick="explainCurrentPredictionRow()" style="font-size: 11px; font-family: inherit; font-weight: 600; padding: 5px 12px; background: rgba(0, 229, 255, 0.12); border: 1px solid var(--cyan-light); color: var(--cyan-light); border-radius: 4px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; transition: all 0.2s ease;">
            🔍 Why? (Explain Prediction)
          </button>
        </div>
        <div id="predictionExplanationBox" style="display: none; margin-top: 12px; padding: 12px; background: rgba(0,0,0,0.4); border: 1px solid rgba(0, 229, 255, 0.25); border-radius: 6px;"></div>
      `;
    } else {
      resultBox.innerHTML = `<div style="color: var(--red-alert);">Prediction error: ${escapeHtml(data.detail || "Inference failed.")}</div>`;
    }
  } catch (err) {
    if (resultBox) resultBox.innerHTML = `<div style="color: var(--red-alert);">Network error: ${err}</div>`;
  }
}

async function explainCurrentPredictionRow() {
  const expBox = document.getElementById("predictionExplanationBox");
  const btn = document.getElementById("explainPredictionBtn");
  if (!expBox) return;

  expBox.style.display = "block";
  expBox.innerHTML = '<div style="color: var(--cyan-light); font-size: 11px;">Computing SHAP feature attributions...</div>';
  if (btn) btn.disabled = true;

  try {
    const res = await fetch(`/api/session/${currentSessionId}/explain-prediction`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: lastPredictedFeatures })
    });
    const data = await res.json();

    if (res.ok && data.success) {
      const topFeats = (data.top_contributions || []).slice(0, 6);
      const maxAbs = Math.max(...topFeats.map(f => Math.abs(f.contribution) || 0.01), 0.01);

      let barsHtml = topFeats.map(f => {
        const isPos = f.contribution >= 0;
        const pct = Math.min(100, Math.round((Math.abs(f.contribution) / maxAbs) * 100));
        const color = isPos ? "var(--green-online, #00e676)" : "var(--orange-warn, #ff9100)";
        const sign = isPos ? "+" : "";
        return `
          <div style="margin-bottom: 9px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 2px;">
              <span style="color: var(--text-main); font-weight: 600;">${escapeHtml(f.feature)} = <span style="color: var(--cyan-light);">${escapeHtml(String(f.value))}</span></span>
              <span style="color: ${color}; font-weight: 700;">${sign}${f.contribution.toFixed(4)}</span>
            </div>
            <div style="height: 6px; width: 100%; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: hidden; display: flex;">
              <div style="width: ${pct}%; background: ${color}; height: 100%; border-radius: 3px; transition: width 0.3s ease;"></div>
            </div>
            <div style="font-size: 10px; color: var(--text-dim); margin-top: 3px; line-height: 1.3;">
              ${escapeHtml(f.summary || "")}
            </div>
          </div>
        `;
      }).join("");

      expBox.innerHTML = `
        <div style="font-size: 11px; font-weight: 700; color: var(--cyan-light); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
          <span>Instance Feature Attribution</span>
          <span style="font-size: 9.5px; color: var(--text-dim); font-weight: normal; background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 3px;">${escapeHtml(data.explanation_method || "SHAP")}</span>
        </div>
        ${barsHtml}
      `;
    } else {
      expBox.innerHTML = `<div style="color: var(--red-alert); font-size: 11px;">Explanation error: ${escapeHtml(data.detail || "Failed to generate explanation")}</div>`;
    }
  } catch (err) {
    expBox.innerHTML = `<div style="color: var(--red-alert); font-size: 11px;">Network error: ${err}</div>`;
  } finally {
    if (btn) btn.disabled = false;
  }
}



// ── PHASE 7: SESSION EXPORT, IMPORT & SHARE ───────────────────────────────────

async function exportSessionJson() {
  try {
    const res = await fetch(`/api/session/${currentSessionId}/export-state`);
    if (res.ok) {
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `proton_session_${currentSessionId}_backup.json`;
      a.click();
      URL.revokeObjectURL(url);
    }
  } catch (err) {
    alert("Failed to export session: " + err);
  }
}

function handleShareSession() {
  const shareUrl = `${window.location.origin}/?session=${currentSessionId}`;
  navigator.clipboard.writeText(shareUrl).then(() => {
    alert(`✓ Shareable Session URL copied to clipboard:\n${shareUrl}`);
  }).catch(() => {
    prompt("Copy this session URL:", shareUrl);
  });
}

function handlePlotDistribution() {
  const col = vizColSelect?.value;
  if (!col) return;
  runToolAction("plot_distribution", { column: col });
}

function handleExportModel() {
  const name = document.getElementById("exportModelName")?.value.trim() || "";
  runToolAction("export_model", { model_name: name });
}

function handleExportPredictions() {
  const name = document.getElementById("exportPredsName")?.value.trim() || "predictions.csv";
  runToolAction("export_predictions", { filename: name });
}


// ── GENERIC TOOL EXECUTION (POINT-AND-CLICK) ──────────────────────────────────

async function runToolAction(toolName, params = {}) {
  closeAllModals();
  const cardId = appendFeedCard(`Executing ${toolName.replace(/_/g, " ").toUpperCase()}`, `Running against active session...`, "running");

  try {
    const res = await fetch(`/api/session/${currentSessionId}/tool/${toolName}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params })
    });
    const result = await res.json();

    if (result.success) {
      updateFeedCard(cardId, `${toolName.replace(/_/g, " ").toUpperCase()} Complete`, formatToolDataHtml(toolName, result.data), "success");
      await updateSystemStatus();
    } else {
      updateFeedCard(cardId, `${toolName.replace(/_/g, " ").toUpperCase()} Error`, `<div style="color: var(--red-alert);">${escapeHtml(result.error || "Tool execution failed.")}</div>`, "error");
    }
  } catch (err) {
    updateFeedCard(cardId, `Execution Error`, String(err), "error");
  }
}


// ── SSE LIVE STREAMING AGENT EXECUTION ────────────────────────────────────────

function runAgentQuery(queryText) {
  closeAllModals();
  const trimmed = queryText.trim();
  if (!trimmed) return;

  const cardId = "stream-" + Date.now();
  const card = document.createElement("div");
  card.className = "result-card";
  card.id = cardId;
  card.innerHTML = `
    <div class="result-card-header">
      <div class="result-title">
        <span class="status-dot dot-green"></span>
        <span>▲ PROTON ❯ ${escapeHtml(trimmed)}</span>
      </div>
      <div class="result-timestamp">${new Date().toLocaleTimeString()}</div>
    </div>
    <div class="checklist-container" id="${cardId}-checklist">
      <div style="color: var(--text-dim); font-size: 11.5px;">Planning execution steps...</div>
    </div>
    <div id="${cardId}-results" style="display: flex; flex-direction: column; gap: 10px; margin-top: 4px;"></div>
  `;
  terminalFeed.appendChild(card);
  card.scrollIntoView({ behavior: "smooth" });

  const checklistEl = document.getElementById(`${cardId}-checklist`);
  const resultsEl = document.getElementById(`${cardId}-results`);

  const eventSource = new EventSource(`/api/session/${currentSessionId}/stream?query=${encodeURIComponent(trimmed)}`);

  eventSource.addEventListener("plan_created", (e) => {
    const data = JSON.parse(e.data);
    checklistEl.innerHTML = `
      <div style="color: var(--cyan-light); font-weight: 700; margin-bottom: 6px; font-size: 12px;">
        Goal: ${escapeHtml(data.goal)} (${data.total_steps} steps)
      </div>
      ${data.steps.map(s => `
        <div class="checklist-step" id="${cardId}-step-${s.step_number}">
          <span class="step-badge pending">${s.step_number}</span>
          <span>${escapeHtml(s.description)}</span>
        </div>
      `).join("")}
    `;
  });

  eventSource.addEventListener("step_start", (e) => {
    const data = JSON.parse(e.data);
    const stepEl = document.getElementById(`${cardId}-step-${data.step_number}`);
    if (stepEl) {
      stepEl.querySelector(".step-badge").className = "step-badge running";
    }
  });

  eventSource.addEventListener("step_complete", (e) => {
    const data = JSON.parse(e.data);
    const stepEl = document.getElementById(`${cardId}-step-${data.step_number}`);
    if (stepEl) {
      if (data.success) {
        stepEl.className = "checklist-step completed";
        stepEl.querySelector(".step-badge").className = "step-badge success";
        stepEl.querySelector(".step-badge").textContent = "✓";

        if (data.data && Object.keys(data.data).length > 0) {
          const stepResultDiv = document.createElement("div");
          stepResultDiv.style.borderTop = "1px solid var(--border-dim)";
          stepResultDiv.style.paddingTop = "8px";
          stepResultDiv.innerHTML = formatToolDataHtml(data.tool_name, data.data);
          resultsEl.appendChild(stepResultDiv);
        }
      } else {
        stepEl.querySelector(".step-badge").className = "step-badge error";
        stepEl.querySelector(".step-badge").textContent = "✗";
        if (data.error) {
          const errMsg = data.error || "";
          const isDataQualityErr = /convert|float|string|text|url|clean/i.test(errMsg);
          const errorDiv = document.createElement("div");
          errorDiv.style.marginTop = "6px";

          // Main error message
          const errText = document.createElement("div");
          errText.style.color = "var(--red-alert)";
          errText.style.fontSize = "11.5px";
          errText.style.whiteSpace = "pre-wrap";
          errText.textContent = `Error: ${errMsg}`;
          errorDiv.appendChild(errText);

          // If data quality issue → show a "Clean Dataset" action button
          if (isDataQualityErr) {
            const cleanNote = document.createElement("div");
            cleanNote.style.marginTop = "8px";
            cleanNote.innerHTML = `
              <span style="color: var(--text-muted); font-size: 11px;">⚠ This looks like a data quality issue.</span>
              <button id="err-clean-btn-${Date.now()}" style="
                display: inline-block; margin-left: 8px;
                background: linear-gradient(135deg, #ff6b35, #ff4081);
                color: #fff; border: none; border-radius: 5px;
                padding: 3px 10px; font-size: 11px; font-weight: 600;
                cursor: pointer; transition: opacity 0.2s;" 
                onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'"
                onclick="(function(){
                  const inp = document.getElementById('promptInput');
                  if(inp){ inp.value = 'clean dataset'; }
                  const btn = document.getElementById('sendBtn');
                  if(btn){ btn.click(); }
                })()">
                🧹 Clean Dataset
              </button>`;
            errorDiv.appendChild(cleanNote);
          }

          resultsEl.appendChild(errorDiv);
        }
      }
    }
  });

  eventSource.addEventListener("qa_response", (e) => {
    const data = JSON.parse(e.data);
    checklistEl.innerHTML = `<div style="color: var(--text-white); white-space: pre-wrap;">${escapeHtml(data.explanation)}</div>`;
  });

  eventSource.addEventListener("plan_complete", (e) => {
    const data = JSON.parse(e.data);
    eventSource.close();
    if (data.summary) {
      const summaryDiv = document.createElement("div");
      summaryDiv.style.background = "rgba(0, 229, 255, 0.05)";
      summaryDiv.style.border = "1px solid var(--cyan-dim)";
      summaryDiv.style.borderRadius = "var(--radius-sm)";
      summaryDiv.style.padding = "10px";
      summaryDiv.style.marginTop = "8px";
      summaryDiv.innerHTML = `
        <div style="color: var(--cyan-light); font-weight: 700; margin-bottom: 4px;">▲ Executive Summary:</div>
        <div style="color: #ffffff; white-space: pre-wrap; font-size: 12px; line-height: 1.6;">${escapeHtml(data.summary)}</div>
      `;
      resultsEl.appendChild(summaryDiv);
    }
    updateSystemStatus();
  });

  eventSource.onerror = () => {
    eventSource.close();
  };
}


// ── OUTPUT FORMATTERS ─────────────────────────────────────────────────────────

function appendFeedCard(title, bodyHtml, type = "info") {
  const cardId = "card-" + Date.now();
  const card = document.createElement("div");
  card.className = "result-card";
  card.id = cardId;
  if (type === "error") card.style.borderLeftColor = "var(--red-alert)";
  if (type === "success") card.style.borderLeftColor = "var(--green-online)";

  card.innerHTML = `
    <div class="result-card-header">
      <div class="result-title">${title}</div>
      <div class="result-timestamp">${new Date().toLocaleTimeString()}</div>
    </div>
    <div class="result-body" id="${cardId}-body">${bodyHtml}</div>
  `;
  terminalFeed.appendChild(card);
  card.scrollIntoView({ behavior: "smooth" });
  return cardId;
}

function updateFeedCard(cardId, title, bodyHtml, type = "success") {
  const card = document.getElementById(cardId);
  if (!card) return;
  if (type === "error") card.style.borderLeftColor = "var(--red-alert)";
  if (type === "success") card.style.borderLeftColor = "var(--green-online)";

  card.querySelector(".result-title").textContent = title;
  card.querySelector(".result-body").innerHTML = bodyHtml;
  card.scrollIntoView({ behavior: "smooth" });
}

function renderSampleTableHtml(records, dtypes = null) {
  if (!records || !records.length) return "<div style='color: var(--text-muted);'>No records found.</div>";
  const cols = Object.keys(records[0]);
  return `
    <div class="feed-table-wrap">
      <table class="feed-table">
        <thead>
          <tr>
            ${cols.map(c => `
              <th>
                <div>${escapeHtml(c)}</div>
                ${dtypes && dtypes[c] ? `<div style="font-size: 9.5px; color: var(--text-dim); font-weight: normal;">${escapeHtml(dtypes[c])}</div>` : ''}
              </th>
            `).join("")}
          </tr>
        </thead>
        <tbody>
          ${records.map(r => `
            <tr>${cols.map(c => `<td>${escapeHtml(String(r[c] !== undefined ? r[c] : ""))}</td>`).join("")}</tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function formatToolDataHtml(toolName, data) {
  if (!data) return "<div>No data returned.</div>";

  // 0. Detect Target & Problem Type (with Generative AI Recommendation)
  if (toolName === "detect_problem_type" || (data.target_column && data.problem_type && data.unique_values_count !== undefined)) {
    const isClf = data.problem_type === "classification";
    return `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
          <span class="status-pill pill-online"><strong>Target:</strong> ${escapeHtml(data.target_column)}</span>
          <span class="status-pill ${isClf ? 'pill-loaded' : 'pill-heuristic'}"><strong>Task:</strong> ${escapeHtml(data.problem_type.toUpperCase())}</span>
          <span class="status-pill pill-loaded"><strong>Data Type:</strong> <code>${escapeHtml(data.target_dtype || '-')}</code></span>
          <span class="status-pill pill-loaded"><strong>${isClf ? 'Distinct Classes' : 'Unique Values'}:</strong> ${data.unique_values_count || 0}</span>
        </div>
        ${data.detail ? `<div style="color: #ffffff; font-size: 12px; margin-top: 2px;">${escapeHtml(data.detail)}</div>` : ''}
        ${data.ai_recommendation ? `
          <div style="background: rgba(168, 85, 247, 0.08); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: var(--radius-sm); padding: 8px 12px; margin-top: 4px;">
            <div style="color: #c084fc; font-weight: 700; font-size: 11.5px; margin-bottom: 2px;">🤖 Gemini Model Recommendation:</div>
            <div style="color: #e9d5ff; font-size: 12px; line-height: 1.5;">${escapeHtml(data.ai_recommendation)}</div>
          </div>
        ` : ''}
      </div>
    `;
  }

  // 1. Audit Data Quality & Health Score (Comprehensive Assessment)
  if (toolName === "audit_data_quality" || data.quality_score !== undefined) {
    const score = data.quality_score !== undefined ? data.quality_score : 0;
    const status = data.status || (score >= 90 ? "Excellent" : (score >= 75 ? "Good" : (score >= 50 ? "Fair" : "Poor")));
    const statusClass = score >= 80 ? "pill-online" : (score >= 50 ? "pill-heuristic" : "pill-heuristic");
    const statusColor = score >= 80 ? "var(--green-online)" : (score >= 50 ? "var(--yellow-warn)" : "#ef4444");
    const warnings = Array.isArray(data.warnings) ? data.warnings : [];

    return `
      <div style="display: flex; flex-direction: column; gap: 10px;">
        <div class="quality-meter-wrap" style="display: flex; align-items: center; gap: 16px; background: rgba(0, 229, 255, 0.04); border: 1px solid var(--border-subtle); padding: 12px 16px; border-radius: var(--radius-md);">
          <div class="quality-score-circle" style="display: flex; flex-direction: column; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 50%; border: 3px solid ${statusColor}; background: rgba(0,0,0,0.4); flex-shrink: 0;">
            <span style="font-size: 18px; font-weight: 800; color: ${statusColor}; line-height: 1;">${score}</span>
            <span class="label" style="font-size: 9px; color: var(--text-dim); text-transform: uppercase;">/ 100</span>
          </div>
          <div style="flex: 1;">
            <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
              <div style="color: var(--text-white); font-weight: 700; font-size: 14px;">
                Data Quality Health: <span style="color: ${statusColor}; font-weight: 800;">${escapeHtml(status.toUpperCase())}</span>
              </div>
              <span class="status-pill ${statusClass}" style="font-size: 11px;">Score: ${score}%</span>
            </div>
            <div style="color: var(--text-dim); font-size: 11.5px; margin-top: 4px;">
              Audited across completeness, row uniqueness, feature variance, and identifier cardinality.
            </div>
          </div>
        </div>

        <div class="feed-table-wrap">
          <table class="feed-table">
            <thead>
              <tr>
                <th>Audit Dimension</th>
                <th>Observed Value</th>
                <th>Health Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style="font-weight: 600;">Dataset Dimensions</td>
                <td>${(data.total_rows || 0).toLocaleString()} rows × ${(data.total_columns || 0)} columns</td>
                <td><span class="status-pill pill-loaded" style="font-size: 10px;">Dimensions</span></td>
              </tr>
              <tr>
                <td style="font-weight: 600;">Missing Values</td>
                <td>${(data.total_missing || 0).toLocaleString()} cells (${data.missing_pct || 0}%)</td>
                <td>
                  <span class="status-pill ${(data.total_missing || 0) === 0 ? 'pill-online' : 'pill-heuristic'}" style="font-size: 10px;">
                    ${(data.total_missing || 0) === 0 ? '✓ Complete' : '▲ Has Missing'}
                  </span>
                </td>
              </tr>
              <tr>
                <td style="font-weight: 600;">Duplicate Rows</td>
                <td>${(data.duplicate_rows || 0).toLocaleString()} rows (${data.duplicate_pct || 0}%)</td>
                <td>
                  <span class="status-pill ${(data.duplicate_rows || 0) === 0 ? 'pill-online' : 'pill-heuristic'}" style="font-size: 10px;">
                    ${(data.duplicate_rows || 0) === 0 ? '✓ 100% Unique' : '▲ Duplicates Found'}
                  </span>
                </td>
              </tr>
              <tr>
                <td style="font-weight: 600;">Zero-Variance Features</td>
                <td>${(data.zero_variance_columns && data.zero_variance_columns.length) ? escapeHtml(data.zero_variance_columns.join(', ')) : 'None (All have variance)'}</td>
                <td>
                  <span class="status-pill ${(!data.zero_variance_columns || !data.zero_variance_columns.length) ? 'pill-online' : 'pill-heuristic'}" style="font-size: 10px;">
                    ${(!data.zero_variance_columns || !data.zero_variance_columns.length) ? '✓ Valid' : '▲ Constant Feature'}
                  </span>
                </td>
              </tr>
              <tr>
                <td style="font-weight: 600;">High-Cardinality IDs</td>
                <td>${(data.high_cardinality_columns && data.high_cardinality_columns.length) ? escapeHtml(data.high_cardinality_columns.join(', ')) : 'None (No pure IDs)'}</td>
                <td>
                  <span class="status-pill ${(!data.high_cardinality_columns || !data.high_cardinality_columns.length) ? 'pill-online' : 'pill-heuristic'}" style="font-size: 10px;">
                    ${(!data.high_cardinality_columns || !data.high_cardinality_columns.length) ? '✓ Clean' : '▲ Potential ID'}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        ${warnings.length ? `
          <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: var(--radius-sm); padding: 8px 12px;">
            <div style="color: var(--yellow-warn); font-weight: 700; font-size: 11.5px; margin-bottom: 4px;">▲ Quality Warnings & Recommendations:</div>
            <ul style="margin: 0; padding-left: 18px; color: var(--text-dim); font-size: 11px; line-height: 1.4;">
              ${warnings.map(w => `<li style="color: var(--yellow-warn);">${escapeHtml(w)}</li>`).join("")}
            </ul>
          </div>
        ` : ''}
      </div>
    `;
  }

  // 2. Inspect Data Types (Categorization + Full Column Dtypes Table)
  if (toolName === "inspect_data_types" || (data.dtypes && data.numerical_columns !== undefined)) {
    const numCols = data.numerical_columns || [];
    const catCols = data.categorical_columns || [];
    const boolCols = data.boolean_columns || [];
    const dtCols = data.datetime_columns || [];
    const dtypesMap = data.dtypes || {};
    const total = data.total_columns || Object.keys(dtypesMap).length;

    return `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
          <span class="status-pill pill-loaded"><strong>Total Columns:</strong> ${total}</span>
          <span class="status-pill pill-online"><strong>Numerical:</strong> ${numCols.length}</span>
          <span class="status-pill pill-loaded"><strong>Categorical:</strong> ${catCols.length}</span>
          ${boolCols.length ? `<span class="status-pill pill-loaded"><strong>Boolean:</strong> ${boolCols.length}</span>` : ''}
          ${dtCols.length ? `<span class="status-pill pill-loaded"><strong>Datetime:</strong> ${dtCols.length}</span>` : ''}
        </div>
        <div class="feed-table-wrap">
          <table class="feed-table">
            <thead>
              <tr>
                <th style="width: 45px;">#</th>
                <th>Column Name</th>
                <th>Data Type</th>
                <th>Category</th>
              </tr>
            </thead>
            <tbody>
              ${Object.entries(dtypesMap).map(([col, dt], idx) => {
                let cat = "Categorical / Text";
                let badgeColor = "var(--yellow-warn)";
                if (numCols.includes(col)) {
                  cat = "Numerical (Continuous / Discrete)";
                  badgeColor = "var(--cyan-light)";
                } else if (boolCols.includes(col)) {
                  cat = "Boolean (Binary)";
                  badgeColor = "var(--purple-accent)";
                } else if (dtCols.includes(col)) {
                  cat = "Datetime";
                  badgeColor = "var(--green-online)";
                }
                return `
                  <tr>
                    <td style="color: var(--text-dim);">${idx + 1}</td>
                    <td style="font-weight: 700; color: #ffffff;">${escapeHtml(col)}</td>
                    <td><code>${escapeHtml(String(dt))}</code></td>
                    <td style="color: ${badgeColor}; font-weight: 600;">${cat}</td>
                  </tr>
                `;
              }).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // 3. Missing Values Audit (Full breakdown per column)
  if (toolName === "detect_missing_values" || (data.missing_by_column !== undefined && data.quality_score === undefined)) {
    let list = [];
    if (Array.isArray(data.missing_by_column)) {
      list = data.missing_by_column;
    } else if (data.missing_by_column && typeof data.missing_by_column === "object") {
      list = Object.entries(data.missing_by_column).map(([name, val]) => {
        if (typeof val === "object" && val !== null) {
          return {
            name: val.name || name,
            dtype: val.dtype || "-",
            non_null: val.non_null !== undefined ? val.non_null : "-",
            missing: val.missing !== undefined ? val.missing : (val.count || 0),
            missing_pct: val.missing_pct !== undefined ? val.missing_pct : (val.pct || 0)
          };
        }
        return { name: name, dtype: "-", non_null: "-", missing: Number(val) || 0, missing_pct: 0 };
      });
    }

    const totMissing = data.total_missing !== undefined ? data.total_missing : list.reduce((acc, c) => acc + (c.missing || 0), 0);
    const overallPct = data.overall_missing_pct !== undefined ? data.overall_missing_pct : 0;
    const colsWithMissing = data.columns_with_missing_count !== undefined ? data.columns_with_missing_count : list.filter(c => (c.missing || 0) > 0).length;

    return `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
          <div style="font-weight: 700; color: ${totMissing === 0 ? 'var(--green-online)' : 'var(--yellow-warn)'};">
            ${totMissing === 0 ? '✓ Clean Dataset — No Missing Values Detected' : `▲ Missing Values: ${totMissing.toLocaleString()} total missing cells (${overallPct}%) across ${colsWithMissing} column(s)`}
          </div>
          <span class="status-pill ${totMissing === 0 ? 'pill-online' : 'pill-heuristic'}">
            ${totMissing === 0 ? '100% Complete' : `${overallPct}% Missing`}
          </span>
        </div>
        <div class="feed-table-wrap">
          <table class="feed-table">
            <thead>
              <tr>
                <th>Column Name</th>
                <th>Data Type</th>
                <th>Non-Null Rows</th>
                <th>Missing Count</th>
                <th>Missing %</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              ${list.map(col => {
                const hasMiss = (col.missing || 0) > 0;
                return `
                  <tr style="${hasMiss ? 'background: rgba(245, 158, 11, 0.06); font-weight: 600;' : ''}">
                    <td style="font-weight: 700; color: ${hasMiss ? 'var(--yellow-warn)' : '#ffffff'};">${escapeHtml(col.name)}</td>
                    <td><code>${escapeHtml(col.dtype || "-")}</code></td>
                    <td>${col.non_null !== undefined && col.non_null !== '-' ? Number(col.non_null).toLocaleString() : '-'}</td>
                    <td style="color: ${hasMiss ? 'var(--yellow-warn)' : 'var(--green-online)'}; font-weight: 700;">
                      ${col.missing !== undefined ? Number(col.missing).toLocaleString() : '0'}
                    </td>
                    <td style="color: ${hasMiss ? 'var(--yellow-warn)' : 'var(--green-online)'};">
                      ${col.missing_pct !== undefined ? (typeof col.missing_pct === 'number' ? col.missing_pct.toFixed(2) + '%' : col.missing_pct) : '0.00%'}
                    </td>
                    <td>
                      <span class="status-pill ${hasMiss ? 'pill-heuristic' : 'pill-online'}" style="font-size: 10px; padding: 1px 6px;">
                        ${hasMiss ? 'Has Missing' : '✓ Clean'}
                      </span>
                    </td>
                  </tr>
                `;
              }).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // 4. Outliers Detection (IQR Method)
  if (toolName === "detect_outliers" || data.outliers_by_column !== undefined) {
    const map = data.outliers_by_column || {};
    const entries = Object.entries(map);
    const totalOut = data.total_outliers !== undefined ? data.total_outliers : entries.reduce((acc, [_, o]) => acc + (o.outlier_count !== undefined ? o.outlier_count : (o.count || 0)), 0);
    const affected = data.affected_columns ? data.affected_columns.length : entries.filter(([_, o]) => (o.outlier_count !== undefined ? o.outlier_count : (o.count || 0)) > 0).length;

    return `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
          <div style="font-weight: 700; color: ${totalOut === 0 ? 'var(--green-online)' : 'var(--yellow-warn)'};">
            ${totalOut === 0 ? '✓ No Statistical Outliers Detected (IQR Method)' : `▲ Outlier Analysis: ${totalOut.toLocaleString()} total outliers across ${affected} numerical column(s)`}
          </div>
          <span class="status-pill ${totalOut === 0 ? 'pill-online' : 'pill-heuristic'}">
            ${totalOut === 0 ? 'Standard Spread' : `${totalOut} Outliers`}
          </span>
        </div>
        <div class="feed-table-wrap">
          <table class="feed-table">
            <thead>
              <tr>
                <th>Column Name</th>
                <th>Outlier Count</th>
                <th>Outlier %</th>
                <th>IQR Normal Range [Lower, Upper]</th>
                <th>Q1 (25%) / Q3 (75%)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              ${entries.map(([col, o]) => {
                const count = o.outlier_count !== undefined ? o.outlier_count : (o.count !== undefined ? o.count : 0);
                const pct = o.outlier_pct !== undefined ? (typeof o.outlier_pct === 'number' ? o.outlier_pct.toFixed(2) + '%' : o.outlier_pct) : '-';
                const lower = o.lower_bound !== undefined ? (typeof o.lower_bound === 'number' ? o.lower_bound.toFixed(2) : o.lower_bound) : '-';
                const upper = o.upper_bound !== undefined ? (typeof o.upper_bound === 'number' ? o.upper_bound.toFixed(2) : o.upper_bound) : '-';
                const q25 = o.q25 !== undefined ? (typeof o.q25 === 'number' ? o.q25.toFixed(2) : o.q25) : '-';
                const q75 = o.q75 !== undefined ? (typeof o.q75 === 'number' ? o.q75.toFixed(2) : o.q75) : '-';
                const hasOut = count > 0;

                return `
                  <tr style="${hasOut ? 'background: rgba(245, 158, 11, 0.06); font-weight: 600;' : ''}">
                    <td style="font-weight: 700; color: ${hasOut ? 'var(--yellow-warn)' : '#ffffff'};">${escapeHtml(col)}</td>
                    <td style="color: ${hasOut ? 'var(--yellow-warn)' : 'var(--green-online)'}; font-weight: 700;">${count.toLocaleString()}</td>
                    <td style="color: ${hasOut ? 'var(--yellow-warn)' : 'var(--green-online)'};">${pct}</td>
                    <td style="color: var(--cyan-light); font-family: monospace;">[${lower}, ${upper}]</td>
                    <td style="color: var(--text-dim); font-size: 11px;">${q25} / ${q75}</td>
                    <td>
                      <span class="status-pill ${hasOut ? 'pill-heuristic' : 'pill-online'}" style="font-size: 10px; padding: 1px 6px;">
                        ${hasOut ? `${count} Outliers` : '✓ Normal'}
                      </span>
                    </td>
                  </tr>
                `;
              }).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // 5. Duplicate Rows Detection (Dedicated Tool)
  if (toolName === "detect_duplicates" || (data.duplicate_rows !== undefined && data.quality_score === undefined)) {
    const dups = data.duplicate_rows || 0;
    const tot = data.total_rows || 0;
    const pct = data.duplicate_pct !== undefined ? data.duplicate_pct : (tot > 0 ? ((dups / tot) * 100).toFixed(2) : 0);

    return `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div style="font-weight: 700; color: ${dups === 0 ? 'var(--green-online)' : 'var(--yellow-warn)'};">
            ${dups === 0 ? '✓ No Duplicate Rows Found' : `▲ Duplicate Rows Detected: ${dups.toLocaleString()} duplicate rows (${pct}%)`}
          </div>
          <span class="status-pill ${dups === 0 ? 'pill-online' : 'pill-heuristic'}">
            ${dups === 0 ? '100% Unique' : `${pct}% Duplicates`}
          </span>
        </div>
        <div class="feed-table-wrap">
          <table class="feed-table">
            <tbody>
              <tr><td style="font-weight: 600; width: 220px;">Total Rows Checked</td><td>${tot.toLocaleString()}</td></tr>
              <tr><td style="font-weight: 600;">Duplicate Rows Count</td><td style="color: ${dups === 0 ? 'var(--green-online)' : 'var(--yellow-warn)'}; font-weight: 700;">${dups.toLocaleString()}</td></tr>
              <tr><td style="font-weight: 600;">Duplicate Percentage</td><td>${pct}%</td></tr>
              <tr><td style="font-weight: 600;">Dataset Uniqueness</td><td style="color: var(--cyan-light); font-weight: 700;">${(100 - pct).toFixed(2)}% Unique Rows</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // 6. Calculate Statistics (Comprehensive 8-Point Numerical Summary)
  if (toolName === "calculate_statistics" || (data.statistics && Object.keys(data.statistics).length > 0)) {
    const statsMap = data.statistics || {};
    const cols = Object.keys(statsMap);
    return `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
          <div style="font-weight: 700; color: var(--cyan-light); font-size: 13px;">
            📊 Descriptive Statistics (8-Point Summary across ${cols.length} Numerical Columns):
          </div>
          <span class="status-pill pill-online">${cols.length} Numerical Features</span>
        </div>
        <div class="feed-table-wrap">
          <table class="feed-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Count (N)</th>
                <th>Mean ± Std Dev</th>
                <th>Variance (σ²)</th>
                <th>Min</th>
                <th>25% (Q1)</th>
                <th>Median (50%)</th>
                <th>75% (Q3)</th>
                <th>Max</th>
                <th>IQR Range [Q1, Q3]</th>
              </tr>
            </thead>
            <tbody>
              ${Object.entries(statsMap).map(([col, s]) => `
                <tr>
                  <td style="font-weight: 700; color: #ffffff;">${escapeHtml(col)}</td>
                  <td>${s.count !== undefined ? Number(s.count).toLocaleString() : '-'}</td>
                  <td style="color: var(--cyan-light); font-weight: 600;">
                    ${s.mean !== undefined ? Number(s.mean).toFixed(2) : '-'} ± ${s.std !== undefined ? Number(s.std).toFixed(2) : '-'}
                  </td>
                  <td style="color: var(--text-dim); font-size: 11px;">${s.variance !== undefined ? Number(s.variance).toFixed(2) : '-'}</td>
                  <td>${s.min !== undefined ? Number(s.min).toFixed(2) : '-'}</td>
                  <td>${s.q25 !== undefined ? Number(s.q25).toFixed(2) : '-'}</td>
                  <td style="font-weight: 700; color: #ffffff;">${s.median !== undefined ? Number(s.median).toFixed(2) : '-'}</td>
                  <td>${s.q75 !== undefined ? Number(s.q75).toFixed(2) : '-'}</td>
                  <td>${s.max !== undefined ? Number(s.max).toFixed(2) : '-'}</td>
                  <td style="color: var(--green-online); font-family: monospace; font-size: 11px;">
                    ${s.iqr !== undefined ? `[${s.q25 || 0} - ${s.q75 || 0}] (Δ=${Number(s.iqr).toFixed(2)})` : '-'}
                  </td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // 7. Inspect Dataset Overview & Schema Preview
  if (toolName === "inspect_dataset" || (data.preview && data.rows !== undefined)) {
    const colInfo = Array.isArray(data.columns) ? data.columns : [];
    return `
      <div style="display: flex; flex-direction: column; gap: 10px;">
        <div style="display: flex; gap: 10px; flex-wrap: wrap; align-items: center;">
          <span class="status-pill pill-loaded"><strong>Dataset:</strong> ${escapeHtml(data.name || "Active Dataset")}</span>
          <span class="status-pill pill-loaded"><strong>Dimensions:</strong> ${Number(data.rows).toLocaleString()} rows × ${data.columns_count || colInfo.length} cols</span>
          <span class="status-pill pill-loaded"><strong>Memory:</strong> ${data.memory_kb ? data.memory_kb + ' KB' : '-'}</span>
          <span class="status-pill ${(data.total_missing || 0) > 0 ? 'pill-heuristic' : 'pill-online'}"><strong>Total Missing:</strong> ${data.total_missing || 0}</span>
        </div>
        ${colInfo.length ? `
          <div style="font-size: 12px; font-weight: 700; color: var(--cyan-light); margin-top: 4px;">Column Schema Overview:</div>
          <div class="feed-table-wrap" style="max-height: 220px; overflow-y: auto;">
            <table class="feed-table">
              <thead>
                <tr><th>Column</th><th>Data Type</th><th>Non-Null</th><th>Missing</th><th>Missing %</th></tr>
              </thead>
              <tbody>
                ${colInfo.map(c => `
                  <tr>
                    <td style="font-weight: 700;">${escapeHtml(c.name)}</td>
                    <td><code>${escapeHtml(c.dtype)}</code></td>
                    <td>${c.non_null !== undefined ? Number(c.non_null).toLocaleString() : '-'}</td>
                    <td style="color: ${c.missing > 0 ? 'var(--yellow-warn)' : 'var(--green-online)'}; font-weight: 700;">${c.missing}</td>
                    <td>${c.missing_pct !== undefined ? (typeof c.missing_pct === 'number' ? c.missing_pct.toFixed(2) + '%' : c.missing_pct) : '-'}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        ` : ''}
        ${data.preview && data.preview.length ? `
          <div style="font-size: 12px; font-weight: 700; color: var(--cyan-light); margin-top: 4px;">First 5 Sample Records:</div>
          ${renderSampleTableHtml(data.preview)}
        ` : ''}
      </div>
    `;
  }

  // 8. Describe Dataset (Full pandas describe matrix)
  if (toolName === "describe_dataset" || (data.numerical_summary && data.categorical_summary)) {
    const numSum = data.numerical_summary || {};
    const cols = Object.keys(numSum);
    return `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="font-weight: 700; color: var(--cyan-light); font-size: 13px;">
          📋 Pandas Summary Statistics Matrix (df.describe):
        </div>
        <div class="feed-table-wrap">
          <table class="feed-table">
            <thead>
              <tr><th>Metric</th>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr>
            </thead>
            <tbody>
              ${['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].map(stat => `
                <tr>
                  <td style="font-weight: 700; color: var(--cyan-light);">${stat}</td>
                  ${cols.map(c => `<td>${numSum[c] && numSum[c][stat] !== undefined ? (typeof numSum[c][stat] === 'number' ? Number(numSum[c][stat]).toFixed(2) : numSum[c][stat]) : '-'}</td>`).join("")}
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // 9. Visual Plots
  if (data.plot_path) {
    const cleanPath = data.plot_path.replace(/\\/g, "/");
    return `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="color: var(--green-online); font-weight: 600;">${escapeHtml(data.message || "Plot generated successfully.")}</div>
        <div class="artifact-img-wrap">
          <img src="/api/artifacts/${cleanPath}" alt="Visualization Plot" onclick="window.open(this.src, '_blank')" style="cursor: zoom-in;" title="Click to view full size">
        </div>
      </div>
    `;
  }

  // 10. Model comparison / Leaderboard (Phase 6)
  if (data.comparison_matrix) {
    return `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div style="color: var(--cyan-light); font-weight: 700; font-size: 13px;">▲ Model Performance Leaderboard</div>
          <span class="status-pill pill-online">[TOP MODEL] ${escapeHtml(data.best_model)}</span>
        </div>
        <div class="feed-table-wrap">
          <table class="feed-table">
            <thead>
              <tr><th>Algorithm</th><th>Metrics</th><th>Action</th></tr>
            </thead>
            <tbody>
              ${data.comparison_matrix.map(m => `
                <tr style="${m.model_name === data.best_model ? 'background: rgba(0, 229, 255, 0.08); font-weight: 700; color: #fff;' : ''}">
                  <td>${escapeHtml(m.model_name)}</td>
                  <td>${Object.entries(m.metrics || {}).map(([k, v]) => `<span style="color: var(--cyan-light); margin-right: 6px;">${k}: ${v}</span>`).join("")}</td>
                  <td>
                    <button class="action-btn-secondary" style="padding: 2px 6px; font-size: 10px;" onclick="runToolAction('predict', {model_name: '${escapeHtml(m.model_name)}', n_samples: 3})">Predict</button>
                  </td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
        ${data.selection_rationale ? `<div style="color: var(--yellow-warn); font-style: italic; font-size: 11.5px;">${escapeHtml(data.selection_rationale)}</div>` : ''}
      </div>
    `;
  }

  // 11. Single Model Training
  if (data.metrics && data.model_name) {
    return `
      <div style="display: flex; flex-direction: column; gap: 6px;">
        <div style="color: var(--green-online); font-weight: 600;">${escapeHtml(data.message || "Model trained successfully.")}</div>
        <div><strong>Model:</strong> ${escapeHtml(data.model_name)} | <strong>Target:</strong> ${escapeHtml(data.target_column || "")}</div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 4px;">
          ${Object.entries(data.metrics).map(([k, v]) => `<span class="status-pill pill-loaded">${escapeHtml(k)}: ${v}</span>`).join("")}
        </div>
        ${data.best_params && Object.keys(data.best_params).length ? `
          <div style="color: var(--yellow-warn); margin-top: 4px;">
            <strong>[TUNED HYPERPARAMETERS]:</strong> <code>${escapeHtml(JSON.stringify(data.best_params))}</code>
          </div>
        ` : ''}
      </div>
    `;
  }

  // 12. Correlation Top Pairs Table
  if (data.top_correlations && Array.isArray(data.top_correlations)) {
    return `
      <div style="display: flex; flex-direction: column; gap: 6px;">
        <div style="color: var(--green-online); font-weight: 600;">Top Feature Correlations:</div>
        <div class="feed-table-wrap">
          <table class="feed-table">
            <thead><tr><th>Feature A</th><th>Feature B</th><th>Pearson Correlation</th></tr></thead>
            <tbody>
              ${data.top_correlations.slice(0, 10).map(c => `
                <tr><td>${escapeHtml(c.feature_1)}</td><td>${escapeHtml(c.feature_2)}</td><td style="color: var(--cyan-light); font-weight: 600;">${c.correlation > 0 ? '+' : ''}${Number(c.correlation).toFixed(4)}</td></tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // 13. Distribution Profiling & Categorical Value Frequencies (Dedicated analyze_distribution Tool)
  if (toolName === "analyze_distribution" || (data.numerical_distributions && Object.keys(data.numerical_distributions).length > 0)) {
    const numDist = data.numerical_distributions || {};
    const catDist = data.categorical_distributions || {};
    const numEntries = Object.entries(numDist);
    const catEntries = Object.entries(catDist);

    return `
      <div style="display: flex; flex-direction: column; gap: 12px;">
        <!-- Numerical Distribution Shapes -->
        <div>
          <div style="font-weight: 700; color: var(--cyan-light); font-size: 13px; margin-bottom: 6px;">
            📈 Numerical Distribution Shapes, Skewness & Kurtosis (${numEntries.length} Features):
          </div>
          <div class="feed-table-wrap">
            <table class="feed-table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Distribution Shape</th>
                  <th>Skewness</th>
                  <th>Kurtosis (Tails)</th>
                  <th>Central Tendency Drift (Mean vs Median)</th>
                </tr>
              </thead>
              <tbody>
                ${numEntries.map(([col, d]) => {
                  const skew = d.skewness !== undefined ? Number(d.skewness) : 0;
                  const isSymm = Math.abs(skew) < 0.5;
                  const isRight = skew >= 0.5;
                  const shapeColor = isSymm ? "var(--green-online)" : (isRight ? "var(--yellow-warn)" : "var(--purple-accent)");
                  const badgeClass = isSymm ? "pill-online" : "pill-heuristic";

                  return `
                    <tr>
                      <td style="font-weight: 700; color: #ffffff;">${escapeHtml(col)}</td>
                      <td>
                        <span class="status-pill ${badgeClass}" style="font-size: 10px; color: ${shapeColor}; font-weight: 700;">
                          ${escapeHtml(d.shape || (isSymm ? 'Symmetric' : 'Skewed'))}
                        </span>
                      </td>
                      <td style="color: ${shapeColor}; font-weight: 700;">${skew.toFixed(2)}</td>
                      <td style="color: var(--text-dim); font-size: 11.5px;">${d.kurtosis !== undefined ? Number(d.kurtosis).toFixed(2) : '-'}</td>
                      <td>
                        <span style="color: var(--cyan-light);">Mean: ${d.mean !== undefined ? Number(d.mean).toFixed(2) : '-'}</span>
                        <span style="color: var(--text-dim); margin: 0 4px;">|</span>
                        <span style="color: #ffffff; font-weight: 600;">Median: ${d.median !== undefined ? Number(d.median).toFixed(2) : '-'}</span>
                      </td>
                    </tr>
                  `;
                }).join("")}
              </tbody>
            </table>
          </div>
        </div>

        <!-- Categorical Frequencies Breakdown -->
        ${catEntries.length ? `
          <div>
            <div style="font-weight: 700; color: #c084fc; font-size: 13px; margin-bottom: 6px;">
              📊 Categorical Column Value Counts & Frequencies (${catEntries.length} Features):
            </div>
            <div class="feed-table-wrap">
              <table class="feed-table">
                <thead>
                  <tr>
                    <th>Categorical Feature</th>
                    <th>Unique Classes</th>
                    <th>Non-Null Rows</th>
                    <th>Top Classes (% Share)</th>
                  </tr>
                </thead>
                <tbody>
                  ${catEntries.map(([col, cd]) => `
                    <tr>
                      <td style="font-weight: 700; color: #ffffff;">${escapeHtml(col)}</td>
                      <td><span class="status-pill pill-loaded">${cd.unique_values || 0} classes</span></td>
                      <td>${(cd.total_non_null || 0).toLocaleString()}</td>
                      <td>
                        <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                          ${(cd.top_categories || []).map(cat => `
                            <span style="background: rgba(168, 85, 247, 0.12); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 4px; padding: 2px 6px; font-size: 10.5px; color: #e9d5ff;">
                              <strong>${escapeHtml(cat.category)}:</strong> ${cat.count} (${cat.percentage}%)
                            </span>
                          `).join("")}
                        </div>
                      </td>
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            </div>
          </div>
        ` : ''}
      </div>
    `;
  }

  // 14. Sample Predictions Table
  if (data.predictions) {
    return `
      <div style="display: flex; flex-direction: column; gap: 8px;">
        <div style="color: var(--green-online); font-weight: 600;">${escapeHtml(data.message || "Generated sample predictions.")}</div>
        <div class="feed-table-wrap">
          <table class="feed-table">
            <thead><tr><th>Row #</th><th>Predicted Output</th></tr></thead>
            <tbody>
              ${data.predictions.map(p => `<tr><td>${p.row}</td><td style="color: var(--cyan-light); font-weight: 700;">${escapeHtml(String(p.prediction))}</td></tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  // Fallback: Generic key-value table
  return `
    <div style="display: flex; flex-direction: column; gap: 6px;">
      ${data.message ? `<div style="color: var(--green-online); font-weight: 600;">✓ ${escapeHtml(data.message)}</div>` : ''}
      <div class="feed-table-wrap">
        <table class="feed-table">
          <tbody>
            ${Object.entries(data).filter(([k]) => k !== 'message' && typeof data[k] !== 'object').map(([k, v]) => `
              <tr><td style="font-weight: 600; width: 200px;">${escapeHtml(k.replace(/_/g, " "))}</td><td>${escapeHtml(String(v))}</td></tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    </div>
  `;
}


// ── AUTOCOMPLETE CHIPS & PROMPT INPUT ─────────────────────────────────────────

function renderChips(commands) {
  if (!chipsBar) return;
  chipsBar.innerHTML = "";
  commands.forEach(cmd => {
    const chip = document.createElement("button");
    chip.className = "autocomplete-chip";
    chip.type = "button";
    chip.textContent = cmd;
    chip.onclick = () => {
      promptInput.value = cmd;
      handlePromptSubmit();
    };
    chipsBar.appendChild(chip);
  });
}

if (promptInput) {
  promptInput.addEventListener("input", (e) => {
    const val = e.target.value.toLowerCase().trim();
    if (!val) {
      renderChips(BUILTIN_COMMANDS.slice(0, 8));
    } else {
      const filtered = BUILTIN_COMMANDS.filter(c => c.toLowerCase().includes(val));
      renderChips(filtered.length ? filtered : [val]);
    }
  });

  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      handlePromptSubmit();
    }
  });
}

function handlePromptSubmit() {
  const query = promptInput.value.trim();
  if (!query) return;
  promptInput.value = "";
  runAgentQuery(query);
}

if (sendBtn) {
  sendBtn.addEventListener("click", handlePromptSubmit);
}

function escapeHtml(str) {
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ── DOM READY BOOTSTRAP ───────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  initSession();
  setInterval(updateSystemStatus, 8000);
});
