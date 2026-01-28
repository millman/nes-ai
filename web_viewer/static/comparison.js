let hasRenderedPlot = false;
let currentXAxisMode = "steps";
let lastPreviewStep = null;
let pendingInitialPreviewStep = null;
let currentPreviewMap = {};
let currentStepsByExp = {};
let currentExperiments = [];
let currentSpecsByExp = {};
let compareStickyInitialized = false;
let folderSelector = null;
let availableFolderValues = new Set();

// Store original data for x-axis toggling
let originalSteps = [];
let cumulativeFlops = [];
let elapsedSeconds = [];
let hasElapsedSeconds = false;

// Uses IMAGE_FOLDER_SPECS, IMAGE_FOLDER_DESCRIPTIONS, getImageOption,
// getImageDisplayTitle, getImageDescription, getResolvedImageSpec,
// buildImagePreviewCard, initializeImageCardTooltips from image_card_common.js
// Uses createFolderSelector from folder_selector.js

// Helper to get current selected folders
function getSelectedFolders() {
  return folderSelector ? folderSelector.getSelectedFolders() : ["vis_fixed_0"];
}

function resolveImageSpec(folderValue, expId) {
  const options = currentSpecsByExp?.[expId];
  const resolved = getResolvedImageSpec(folderValue, options);
  return { stepsKey: folderValue, folderPath: resolved.folder, prefix: resolved.prefix };
}

function computeAvailableFolderValues(stepsByExp, experiments) {
  const available = new Set();
  (experiments || []).forEach((exp) => {
    const map = stepsByExp?.[exp.id];
    if (!map) return;
    Object.entries(map).forEach(([key, steps]) => {
      if (key === "__fallback") return;
      if (Array.isArray(steps) && steps.length) {
        available.add(key);
      }
    });
  });
  return available;
}

function getStepsForFolder(stepsMap, expId, folderValue) {
  const map = stepsMap?.[expId];
  if (!map) {
    return [];
  }
  const direct = map[folderValue];
  if (Array.isArray(direct) && direct.length) {
    return direct;
  }
  const fallback = map.__fallback;
  if (Array.isArray(fallback)) {
    return fallback;
  }
  return [];
}

function nearestStepAtOrBelow(target, list) {
  if (!Array.isArray(list) || list.length === 0) {
    return null;
  }
  let best = null;
  for (let i = 0; i < list.length; i++) {
    const step = list[i];
    if (step > target) {
      continue;
    }
    if (best === null || step > best) {
      best = step;
    }
  }
  return best;
}

function computeInitialPreviewStep(stepsMap, folderValue) {
  const maxStepsPerExp = Object.entries(stepsMap || {}).map(([expId, map]) => {
    const steps = getStepsForFolder(stepsMap, expId, folderValue);
    if (!Array.isArray(steps) || !steps.length) {
      return null;
    }
    return steps[steps.length - 1];
  }).filter((v) => v !== null && v !== undefined);
  if (!maxStepsPerExp.length) {
    return null;
  }
  return Math.min(...maxStepsPerExp);
}

function computeInitialPreviewStepForFolders(stepsMap, folderValues) {
  const candidates = (folderValues || [])
    .map((folderValue) => computeInitialPreviewStep(stepsMap, folderValue))
    .filter((value) => value !== null && value !== undefined);
  if (!candidates.length) {
    return null;
  }
  return Math.min(...candidates);
}

function renderPreviewsAtStep(step, previews, stepsMap) {
  const targetStep = step !== null && step !== undefined ? Math.max(0, Math.round(step)) : null;
  Object.values(previews || {}).forEach((preview) => {
    const expId = preview.expId;
    const folderValue = preview.folderValue;
    if (!expId || !folderValue) {
      return;
    }
    const spec = resolveImageSpec(folderValue, expId);
    const available = getStepsForFolder(stepsMap, expId, folderValue);
    const matched = targetStep !== null ? nearestStepAtOrBelow(targetStep, available) : (available.length ? available[available.length - 1] : null);
    if (matched === null) {
      if (preview.stepLabel) {
        preview.stepLabel.textContent = "—";
      }
      preview.img.classList.add("d-none");
      preview.missing.textContent = "No images available.";
      preview.missing.classList.remove("d-none");
      preview.img.removeAttribute("data-step");
      return;
    }
    const filename = `${spec.prefix}${matched.toString().padStart(7, "0")}.png`;
    const src = `/assets/${expId}/${spec.folderPath}/${filename}`;
    if (preview.stepLabel) {
      preview.stepLabel.textContent = matched;
    }
    preview.img.src = src;
    preview.img.alt = `${spec.folderPath} ${matched}`;
    preview.img.dataset.step = String(matched);
    preview.img.classList.remove("d-none");
    preview.img.onerror = () => {
      preview.img.classList.add("d-none");
      preview.missing.textContent = `No image for step ${matched}`;
      preview.missing.classList.remove("d-none");
    };
    preview.img.onload = () => {
      preview.missing.classList.add("d-none");
    };
    preview.missing.classList.add("d-none");
  });
}

document.addEventListener("DOMContentLoaded", () => {
  // Initialize folder selector using shared component
  folderSelector = createFolderSelector({
    menuId: "folder-menu",
    onSelectionChange: updatePreviewRowsForSelection,
  });

  // Restore x-axis mode from URL
  const urlParams = new URLSearchParams(window.location.search);
  const xAxisParam = urlParams.get("xaxis");
  if (xAxisParam === "flops" || xAxisParam === "elapsed") {
    currentXAxisMode = xAxisParam;
    const xAxisSelect = document.getElementById("comparison-xaxis-select");
    if (xAxisSelect) {
      xAxisSelect.value = xAxisParam;
    }
  }

  const ids = getIdsFromDataset();
  if (ids.length >= 2) {
    // Update URL to reflect selected IDs
    const currentUrl = new URL(window.location.href);
    const normalizedUrl = applyIdsToUrl(currentUrl, ids);
    if (normalizedUrl !== window.location.href) {
      window.history.replaceState({}, "", normalizedUrl);
    }
    runComparison(ids);
  }
});

function runComparison(ids) {
  const plot = document.getElementById("comparison-plot");
  const grid = document.getElementById("comparison-grid");
  showPlotLoading(plot);
  grid.innerHTML = "";
  fetch("/comparison/data", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ids }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error("Failed to load comparison");
      }
      return response.json();
    })
    .then((payload) => renderComparison(payload))
    .catch((error) => {
      hasRenderedPlot = false;
      plot.innerHTML = `<p class="text-muted fst-italic">${error.message}</p>`;
    });
}

function renderComparison(payload) {
  const plot = document.getElementById("comparison-plot");
  const grid = document.getElementById("comparison-grid");
  const xAxisSelect = document.getElementById("comparison-xaxis-select");
  if (!payload || !Array.isArray(payload.experiments) || payload.experiments.length === 0) {
    plot.innerHTML = "<p class='text-muted fst-italic'>No comparison data available.</p>";
    grid.innerHTML = "";
    return;
  }
  const experiments = payload.experiments;
  const availableStepsByExp = {};
  const specsByExp = {};
  experiments.forEach((exp) => {
    const map = exp.visualization_steps && typeof exp.visualization_steps === "object" ? { ...exp.visualization_steps } : {};
    const rollout = Array.isArray(exp.rollout_steps) ? exp.rollout_steps : [];
    if (rollout.length) {
      map.vis_fixed_0 = map.vis_fixed_0 || rollout;
    }
    if (rollout.length && !map.__fallback) {
      map.__fallback = rollout;
    }
    availableStepsByExp[exp.id] = map;
    specsByExp[exp.id] = Array.isArray(exp.image_folder_specs) ? exp.image_folder_specs : [];
  });
  currentStepsByExp = availableStepsByExp;
  currentSpecsByExp = specsByExp;
  currentExperiments = experiments;
  availableFolderValues = computeAvailableFolderValues(availableStepsByExp, experiments);
  if (folderSelector) {
    folderSelector.setAvailableFolderValues(availableFolderValues);
  }
  grid.innerHTML = "";
  grid.appendChild(buildExperimentGrid(experiments));
  refreshPreviewImages(grid, availableStepsByExp);
  initializeImageCardTooltips(grid);
  pendingInitialPreviewStep = computeInitialPreviewStepForFolders(availableStepsByExp, getSelectedFolders());
  lastPreviewStep = null;
  const figure = payload.figure;
  if (figure) {
    // Store original x-axis data for toggling
    originalSteps = figure.data.map((trace) => (trace.x ? [...trace.x] : []));
    cumulativeFlops = figure.data.map((trace) => {
      // Extract cumulative_flops from meta (stored by plots.py)
      if (trace.meta && Array.isArray(trace.meta.cumulative_flops)) {
        return [...trace.meta.cumulative_flops];
      }
      // Fallback: use customdata if available (cumulative_flops is at index 1)
      if (trace.customdata && Array.isArray(trace.customdata)) {
        return trace.customdata.map((cd) => (Array.isArray(cd) ? cd[1] : null));
      }
      return trace.x ? [...trace.x] : [];  // Fallback to steps
    });
    hasElapsedSeconds = false;
    elapsedSeconds = figure.data.map((trace) => {
      const metaElapsed =
        trace.meta && Array.isArray(trace.meta.elapsed_seconds) ? trace.meta.elapsed_seconds : null;
      if (metaElapsed) {
        if (trace.meta && trace.meta.has_elapsed_seconds) {
          hasElapsedSeconds = true;
        }
        return [...metaElapsed];
      }
      if (trace.customdata && Array.isArray(trace.customdata)) {
        const values = trace.customdata.map((cd) => (Array.isArray(cd) ? cd[2] : null));
        if (values.some((v) => v !== null && v !== undefined)) {
          return values;
        }
      }
      return trace.x ? [...trace.x] : [];
    });

    const applyXAxisMode = (mode, { updateUrl = true } = {}) => {
      let nextMode = mode;
      if (nextMode === "elapsed" && !hasElapsedSeconds) {
        nextMode = "steps";
      }

      if (xAxisSelect) {
        const elapsedOption = xAxisSelect.querySelector('option[value="elapsed"]');
        if (elapsedOption) {
          elapsedOption.disabled = !hasElapsedSeconds;
        }
        xAxisSelect.value = nextMode;
      }

      if (updateUrl || nextMode !== mode) {
        const url = new URL(window.location.href);
        url.searchParams.set("xaxis", nextMode);
        // Clear zoom state when switching axes
        url.searchParams.delete("xmin");
        url.searchParams.delete("xmax");
        window.history.replaceState({}, "", normalizeIdsInUrl(url));
      }

      let xData = originalSteps;
      let xTitle = "Step";
      if (nextMode === "flops") {
        xData = cumulativeFlops;
        xTitle = "Cumulative FLOPs";
      } else if (nextMode === "elapsed") {
        xData = elapsedSeconds;
        xTitle = "Elapsed time (s)";
      }

      Plotly.restyle(plot, { x: xData });
      Plotly.relayout(plot, { "xaxis.title": xTitle, "xaxis.autorange": true });
      currentXAxisMode = nextMode;
    };

    // Parse view state from URL
    const urlParams = new URLSearchParams(window.location.search);
    const xMin = urlParams.get("xmin");
    const xMax = urlParams.get("xmax");
    const yMin = urlParams.get("ymin");
    const yMax = urlParams.get("ymax");

    // Apply saved view state to figure layout
    if (xMin !== null && xMax !== null) {
      figure.layout.xaxis = figure.layout.xaxis || {};
      figure.layout.xaxis.range = [parseFloat(xMin), parseFloat(xMax)];
      figure.layout.xaxis.autorange = false;
    }
    if (yMin !== null && yMax !== null) {
      figure.layout.yaxis = figure.layout.yaxis || {};
      figure.layout.yaxis.range = [parseFloat(yMin), parseFloat(yMax)];
      figure.layout.yaxis.autorange = false;
    }

    const config = buildPlotlyConfig(figure.config);
    Plotly.react(plot, figure.data, figure.layout, config).then(() => {
      applyXAxisMode(currentXAxisMode, { updateUrl: false });
      initializeCompareStickyPlot();
      if (lastPreviewStep === null && pendingInitialPreviewStep !== null) {
        lastPreviewStep = pendingInitialPreviewStep;
        renderPreviewsAtStep(lastPreviewStep, currentPreviewMap, currentStepsByExp);
      }

      // X-axis toggle handler
      if (xAxisSelect) {
        xAxisSelect.addEventListener("change", () => {
          applyXAxisMode(xAxisSelect.value);
        });
      }

      // Save view state to URL on zoom/pan
      plot.on("plotly_relayout", (eventData) => {
        if (!eventData) return;
        const url = new URL(window.location.href);

        if (eventData["xaxis.autorange"] || eventData["yaxis.autorange"]) {
          // Reset to autorange - remove params
          url.searchParams.delete("xmin");
          url.searchParams.delete("xmax");
          url.searchParams.delete("ymin");
          url.searchParams.delete("ymax");
        } else {
          if (eventData["xaxis.range[0]"] !== undefined) {
            url.searchParams.set("xmin", eventData["xaxis.range[0]"]);
            url.searchParams.set("xmax", eventData["xaxis.range[1]"]);
          }
          if (eventData["yaxis.range[0]"] !== undefined) {
            url.searchParams.set("ymin", eventData["yaxis.range[0]"]);
            url.searchParams.set("ymax", eventData["yaxis.range[1]"]);
          }
        }

        window.history.replaceState({}, "", normalizeIdsInUrl(url));
      });

      wireExperimentVisibilitySync(plot);
    });
    attachComparisonHover(plot);
    hasRenderedPlot = true;
  } else {
    hasRenderedPlot = false;
    plot.innerHTML = "<p class='text-muted fst-italic'>No overlapping CSV metrics found.</p>";
  }
  wireTitleForms(grid);
  wireTagsForms(grid);
}

function initializeCompareStickyPlot() {
  if (compareStickyInitialized) {
    return;
  }
  const wrapper = document.querySelector(".compare-plot-wrapper");
  const plot = document.getElementById("comparison-plot");
  if (!wrapper || !plot) {
    return;
  }
  compareStickyInitialized = true;
  let isSticky = false;
  let ticking = false;
  const shrinkHeight = 240;
  const placeholder = document.createElement("div");
  placeholder.style.display = "none";
  wrapper.parentNode.insertBefore(placeholder, wrapper);

  const updateSticky = () => {
    ticking = false;
    const anchor = isSticky ? placeholder : wrapper;
    const rect = anchor.getBoundingClientRect();
    const shouldStick = rect.bottom <= shrinkHeight;
    if (shouldStick !== isSticky) {
      // Capture height BEFORE changing sticky state
      const heightBeforeChange = isSticky ? null : wrapper.getBoundingClientRect().height;
      isSticky = shouldStick;
      wrapper.classList.toggle("sticky-active", isSticky);
      if (isSticky) {
        placeholder.style.height = `${heightBeforeChange}px`;
        placeholder.style.display = "block";
      } else {
        placeholder.style.display = "none";
        placeholder.style.height = "";
        wrapper.style.left = "";
        wrapper.style.width = "";
      }
      if (window.Plotly && Plotly.Plots && typeof Plotly.Plots.resize === "function") {
        requestAnimationFrame(() => Plotly.Plots.resize(plot));
      }
    }
    if (isSticky) {
      wrapper.style.left = `${rect.left}px`;
      wrapper.style.width = `${rect.width}px`;
    }
    wrapper.classList.toggle("sticky-shrink", isSticky);
  };

  const onScroll = () => {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(updateSticky);
    }
  };

  updateSticky();
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
}

function updatePreviewRowsForSelection() {
  if (!currentExperiments.length) {
    return;
  }
  const grid = document.getElementById("comparison-grid");
  if (!grid) {
    return;
  }
  const previewRows = grid.querySelector("#comparison-preview-rows");
  if (!previewRows) {
    return;
  }
  const nextRows = buildPreviewRows(currentExperiments, getSelectedFolders());
  previewRows.replaceWith(nextRows);
  refreshPreviewImages(grid, currentStepsByExp);
  initializeImageCardTooltips(grid);
  if (lastPreviewStep === null || lastPreviewStep === undefined) {
    pendingInitialPreviewStep = computeInitialPreviewStepForFolders(currentStepsByExp, getSelectedFolders());
    if (pendingInitialPreviewStep !== null) {
      lastPreviewStep = pendingInitialPreviewStep;
    }
  }
  if (lastPreviewStep !== null && lastPreviewStep !== undefined) {
    renderPreviewsAtStep(lastPreviewStep, currentPreviewMap, currentStepsByExp);
  }
}

function buildExperimentGrid(experiments) {
  const container = document.createElement("div");
  container.className = "d-flex flex-column gap-4";
  const previewRows = buildPreviewRows(experiments, getSelectedFolders());
  container.appendChild(previewRows);

  // Row 2: Title/path/git section
  const infoRow = buildSectionRow(
    "Experiment Info",
    experiments,
    (exp, index) => buildInfoCell(exp, index)
  );
  container.appendChild(infoRow);

  // Row 3: Loss curves section
  const lossRow = buildSectionRow(
    "Loss Curves",
    experiments,
    (exp) => buildLossImageCell(exp)
  );
  container.appendChild(lossRow);

  // Row 4: Metadata section
  const metaRow = buildSectionRow(
    "Metadata",
    experiments,
    (exp) => buildMetadataCell(exp)
  );
  container.appendChild(metaRow);

  // Row 5: Git metadata section
  const gitMetaRow = buildSectionRow(
    "Git Metadata",
    experiments,
    (exp) => buildGitMetadataCell(exp),
    { rowClass: "compare-git-row", colClass: "compare-git-col" }
  );
  container.appendChild(gitMetaRow);

  return container;
}

function buildPreviewRows(experiments, folderValues) {
  const container = document.createElement("div");
  container.id = "comparison-preview-rows";
  container.className = "d-flex flex-column gap-3";

  (folderValues || []).forEach((folderValue) => {
    const label = getImageOption(folderValue)?.label || folderValue;
    const row = buildSectionRow(
      label,
      experiments,
      (exp) => buildPreviewCell(exp, folderValue)
    );
    container.appendChild(row);
  });

  return container;
}

function buildSectionRow(title, experiments, cellBuilder, options = {}) {
  const row = document.createElement("div");
  row.className = `row g-3 ${options.rowClass || ""}`.trim();

  experiments.forEach((exp, index) => {
    const col = document.createElement("div");
    col.className = `col ${options.colClass || ""}`.trim();
    col.dataset.expColumn = exp.id;
    const cell = cellBuilder(exp, index);
    col.appendChild(cell);
    row.appendChild(col);
  });

  return row;
}

function buildPreviewCell(exp, folderValue) {
  // Use title field when available; fall back to name.
  const displayTitle = exp.title && exp.title !== "Untitled" ? exp.title : exp.name;

  // Build card using the shared buildImagePreviewCard helper
  const { card, stepLabelEl, img, missing } = buildImagePreviewCard({
    folderValue,
    showStepLabel: true,
    initialStep: "—",
  });

  // Add rollout-preview class and data attributes for hover tracking
  card.classList.add("rollout-preview");
  card.dataset.expId = exp.id;
  card.dataset.folderValue = folderValue;
  card.dataset.expTitle = displayTitle;

  // Insert experiment title between folder label and step/info controls
  const cardHeader = card.querySelector(".card-header");
  if (cardHeader) {
    const headerControls = cardHeader.querySelector(".image-preview-card-controls");
    const expTitleSpan = document.createElement("span");
    expTitleSpan.className = "rollout-exp-title text-muted small";
    expTitleSpan.textContent = displayTitle;
    if (headerControls) {
      cardHeader.insertBefore(expTitleSpan, headerControls);
    } else {
      cardHeader.appendChild(expTitleSpan);
    }
  }

  // Mark elements with rollout-specific classes for backward compatibility
  img.classList.add("rollout-img");
  missing.classList.add("rollout-missing");

  // Add step label reference class
  if (stepLabelEl) {
    stepLabelEl.classList.add("rollout-step-label");
  }

  return card;
}

function buildInfoCell(exp, index) {
  const container = document.createElement("div");

  const form = document.createElement("form");
  form.className = "title-form inline-field-form inline-field-group mb-2";
  form.dataset.expId = exp.id;

  const group = document.createElement("div");
  group.className = "input-group input-group-sm title-input-group inline-field-group w-100";

  const input = document.createElement("input");
  input.type = "text";
  input.className = "form-control form-control-sm exp-title-input inline-field";
  const titleValue = exp.title && exp.title !== "Untitled" ? exp.title : "";
  input.placeholder = titleValue || "Untitled";
  if (titleValue) {
    input.value = titleValue;
  }

  const status = document.createElement("span");
  status.className = "title-status small text-muted";
  status.setAttribute("aria-live", "polite");

  group.appendChild(input);
  form.appendChild(group);
  form.appendChild(status);

  const tagsForm = document.createElement("form");
  tagsForm.className = "tags-form inline-field-form inline-field-group mb-2";
  tagsForm.dataset.expId = exp.id;

  const tagsGroup = document.createElement("div");
  tagsGroup.className = "input-group input-group-sm tags-input-group inline-field-group w-100";

  const tagsLabel = document.createElement("span");
  tagsLabel.className = "input-group-text";
  tagsLabel.textContent = "Tags:";

  const tagsInput = document.createElement("input");
  tagsInput.type = "text";
  tagsInput.className = "form-control form-control-sm exp-tags-input inline-field";
  const tagsValue = exp.tags || "";
  if (tagsValue) {
    tagsInput.value = tagsValue;
  }

  const tagsStatus = document.createElement("span");
  tagsStatus.className = "tags-status small text-muted";
  tagsStatus.setAttribute("aria-live", "polite");

  tagsGroup.appendChild(tagsLabel);
  tagsGroup.appendChild(tagsInput);
  tagsForm.appendChild(tagsGroup);
  tagsForm.appendChild(tagsStatus);

  const name = document.createElement("div");
  name.className = "fw-semibold";
  name.textContent = exp.name;

  const commit = document.createElement("div");
  commit.className = "font-monospace text-muted small mt-1";
  commit.textContent = exp.git_commit || "Unknown commit";

  const params = document.createElement("div");
  params.className = "text-muted small mt-1";
  params.innerHTML = `Parameters: <span class="font-monospace">${formatParamCount(exp.total_params)}</span>`;

  const flops = document.createElement("div");
  flops.className = "text-muted small";
  flops.innerHTML = `FLOPs/step: <span class="font-monospace">${formatFlops(exp.flops_per_step)}</span>`;

  container.appendChild(form);
  container.appendChild(tagsForm);
  container.appendChild(name);
  container.appendChild(commit);
  container.appendChild(params);
  container.appendChild(flops);
  return container;
}

function buildLossImageCell(exp) {
  const container = document.createElement("div");

  if (exp.loss_image) {
    const link = document.createElement("a");
    link.href = `/experiment/${exp.id}`;
    link.className = "d-block";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = exp.loss_image;
    img.alt = `Loss curves for ${exp.name}`;
    img.className = "img-fluid rounded border";
    link.appendChild(img);
    container.appendChild(link);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "text-muted fst-italic";
    placeholder.textContent = "metrics/loss_curves.png missing";
    container.appendChild(placeholder);
  }

  return container;
}

function collectPreviewMap(root) {
  const map = {};
  root.querySelectorAll(".rollout-preview").forEach((preview) => {
    const expId = preview.dataset.expId;
    const folderValue = preview.dataset.folderValue;
    if (!expId || !folderValue) {
      return;
    }
    const displayTitle = preview.dataset.expTitle || expId;
    const stepLabel = preview.querySelector(".rollout-step-label");
    const img = preview.querySelector(".rollout-img");
    const missing = preview.querySelector(".rollout-missing");
    if (img && missing) {
      const key = `${expId}:${folderValue}`;
      map[key] = { stepLabel, img, missing, displayTitle, expId, folderValue };
    }
  });
  return map;
}

function refreshPreviewImages(container, stepsMap) {
  const previews = collectPreviewMap(container);
  currentPreviewMap = previews;
  Object.values(previews).forEach((preview) => {
    const expId = preview.expId;
    const folderValue = preview.folderValue;
    if (!expId || !folderValue) {
      return;
    }
    const steps = getStepsForFolder(stepsMap, expId, folderValue);
    const latestStep = steps.length ? steps[steps.length - 1] : null;
    if (latestStep === null) {
      preview.img.classList.add("d-none");
      preview.missing.classList.remove("d-none");
      if (preview.stepLabel) {
        preview.stepLabel.textContent = "—";
      }
      return;
    }
    const spec = resolveImageSpec(folderValue, expId);
    preview.img.src = `/assets/${expId}/${spec.folderPath}/${spec.prefix}${latestStep}.png`;
    preview.img.alt = `Rollout ${latestStep} for ${preview.displayTitle}`;
    preview.img.classList.remove("d-none");
    preview.missing.classList.add("d-none");
    if (preview.stepLabel) {
      preview.stepLabel.textContent = latestStep;
    }
  });
}

function attachComparisonHover(plotEl) {
  if (!plotEl || typeof Plotly === "undefined") {
    return;
  }
  plotEl.on("plotly_hover", (event) => {
    const points = event?.points || [];
    const point = points.find((p) => p && typeof p.x !== "undefined");
    if (!point) {
      return;
    }
    lastPreviewStep = point.x;
    renderPreviewsAtStep(point.x, currentPreviewMap, currentStepsByExp);
  });
}

function showPlotLoading(plot) {
  if (hasRenderedPlot && typeof Plotly !== "undefined") {
    Plotly.purge(plot);
    hasRenderedPlot = false;
  }
  plot.innerHTML = "<p>Loading…</p>";
}

function getIdsFromDataset() {
  const container = document.querySelector("[data-selected-ids]");
  if (!container) {
    return [];
  }
  return container.dataset.selectedIds.split(",").filter(Boolean);
}

function buildMetadataCell(exp) {
  const container = document.createElement("div");
  const pre = document.createElement("pre");
  pre.className = "bg-dark text-light p-2 rounded overflow-auto mb-0 small";
  pre.style.maxHeight = "280px";
  pre.textContent = exp.metadata || "(no metadata)";
  container.appendChild(pre);
  return container;
}

function buildGitMetadataCell(exp) {
  const pre = document.createElement("pre");
  pre.className = "bg-dark text-light p-2 rounded mb-0 small";
  pre.style.maxHeight = "280px";
  pre.style.overflow = "auto";
  pre.style.whiteSpace = "pre";
  pre.style.maxWidth = "100%";
  pre.style.boxSizing = "border-box";
  pre.textContent = exp.git_metadata || "(no git metadata)";
  return pre;
}

function wireExperimentVisibilitySync(plotEl) {
  if (!plotEl) {
    return;
  }
  const syncVisibility = () => {
    const visibleIds = collectVisibleExperimentIds(plotEl);
    applyExperimentVisibility(visibleIds);
  };
  syncVisibility();
  plotEl.on("plotly_restyle", () => {
    window.requestAnimationFrame(syncVisibility);
  });
}

function collectVisibleExperimentIds(plotEl) {
  const visible = new Set();
  if (!plotEl || !Array.isArray(plotEl.data) || plotEl.data.length === 0) {
    document.querySelectorAll("[data-exp-column]").forEach((col) => {
      if (col.dataset.expColumn) {
        visible.add(col.dataset.expColumn);
      }
    });
    return visible;
  }
  plotEl.data.forEach((trace) => {
    const expId = extractExperimentId(trace);
    if (!expId) {
      return;
    }
    const visibility = trace.visible;
    const isVisible = visibility === undefined || visibility === true;
    if (visibility !== "legendonly" && visibility !== false && isVisible) {
      visible.add(expId);
    }
  });
  return visible;
}

function extractExperimentId(trace) {
  if (!trace || !trace.customdata) {
    return null;
  }
  const custom = trace.customdata;
  if (!Array.isArray(custom) || custom.length === 0) {
    return null;
  }
  const first = custom[0];
  if (Array.isArray(first) && first.length > 0 && first[0]) {
    return first[0];
  }
  return null;
}

function applyExperimentVisibility(visibleIds) {
  document.querySelectorAll("[data-exp-column]").forEach((col) => {
    const expId = col.dataset.expColumn;
    const shouldShow = visibleIds.has(expId);
    col.classList.toggle("d-none", !shouldShow);
  });
}
