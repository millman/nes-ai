// Common comparison page logic shared between compare_z.js, compare_h.js, and compare_s.js
//
// Usage: Each page-specific file should define:
// - COMPARE_CONFIG.defaultImageFolder: string (e.g., "pca_z")
// - COMPARE_CONFIG.apiEndpoint: string (e.g., "/compare_z/data")
// - COMPARE_CONFIG.imageFolderOptions: array of image folder option objects
//
// Then call: initializeComparePage(COMPARE_CONFIG);

// Global state
let hasRenderedPlot = false;
let currentImageFolder = null;
let currentXAxisMode = "steps";
let lastPreviewStep = null;
let pendingInitialPreviewStep = null;
let IMAGE_FOLDER_OPTIONS = [];
let availableStepsByExp = {};

// Store original data for x-axis toggling
let originalSteps = [];
let cumulativeFlops = [];
let elapsedSeconds = [];
let hasElapsedSeconds = false;

// Helper to build URL string without encoding commas in ids parameter
function buildUrlString(url) {
  const str = url.toString();
  // Decode %2C back to comma for the ids parameter
  return str.replace(/ids=([^&]+)/g, (match, p1) => `ids=${decodeURIComponent(p1)}`);
}

function getImageOption(folderValue) {
  return IMAGE_FOLDER_OPTIONS.find((opt) => opt.value === folderValue);
}

function resolveImageSpec(folderValue, stepsMap, expId) {
  const option = getImageOption(folderValue);
  if (!option) {
    return { stepsKey: folderValue, folderPath: folderValue, prefix: "rollout_" };
  }
  const map = stepsMap?.[expId];
  const directSteps = map?.[folderValue];
  if (
    option.legacyFolder
    && (!Array.isArray(directSteps) || !directSteps.length)
    && Array.isArray(map?.[option.legacyFolder])
    && map[option.legacyFolder].length
  ) {
    return {
      stepsKey: option.legacyFolder,
      folderPath: option.legacyFolder,
      prefix: option.legacyPrefix || option.prefix,
    };
  }
  return {
    stepsKey: folderValue,
    folderPath: option.folder || option.value,
    prefix: option.prefix,
  };
}

function getStepsForFolder(stepsMap, expId, folderValue) {
  const map = stepsMap?.[expId];
  if (!map) {
    return [];
  }
  const spec = resolveImageSpec(folderValue, stepsMap, expId);
  const direct = map[spec.stepsKey];
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

function renderPreviewsAtStep(step, previews, stepsMap) {
  const targetStep = step !== null && step !== undefined ? Math.max(0, Math.round(step)) : null;
  Object.entries(previews || {}).forEach(([key, preview]) => {
    const expId = preview.expId || key;
    const folderValue = preview.folderValue || currentImageFolder;
    const spec = resolveImageSpec(folderValue, stepsMap, expId);
    const available = getStepsForFolder(stepsMap, expId, folderValue);
    const matched = targetStep !== null ? nearestStepAtOrBelow(targetStep, available) : (available.length ? available[available.length - 1] : null);
    if (matched === null) {
      preview.path.textContent = "No images available.";
      preview.img.classList.add("d-none");
      preview.missing.textContent = "No images available.";
      preview.missing.classList.remove("d-none");
      preview.img.removeAttribute("data-step");
      return;
    }
    const filename = `${spec.prefix}${matched.toString().padStart(7, "0")}.png`;
    const src = `/assets/${expId}/${spec.folderPath}/${filename}`;
    const pathText =
      targetStep !== null
        ? `${expId}/${spec.folderPath}/${filename} (selected ${targetStep}, showing ${matched})`
        : `${expId}/${spec.folderPath}/${filename} (showing ${matched})`;
    preview.path.textContent = pathText;
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

// Pin functionality (matching Diagnostics page style)
function setComparePinned(pinned) {
  const wrapper = document.getElementById("compare-loss-wrapper");
  const pinButton = document.getElementById("compare-loss-pin");
  if (!wrapper || !pinButton) return;

  wrapper.classList.toggle("sticky-active", pinned);
  pinButton.classList.toggle("active", pinned);
  pinButton.setAttribute("aria-pressed", pinned ? "true" : "false");
  pinButton.setAttribute("title", pinned ? "Unpin comparison plot" : "Pin comparison plot");
}

function runComparison(ids, apiEndpoint) {
  const plot = document.getElementById("comparison-plot");
  const grid = document.getElementById("comparison-grid");
  showPlotLoading(plot);
  grid.innerHTML = "";
  fetch(apiEndpoint, {
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
  // X-axis selector removed per user request
  // const xAxisSelect = document.getElementById("comparison-xaxis-select");
  if (!payload || !Array.isArray(payload.experiments) || payload.experiments.length === 0) {
    plot.innerHTML = "<p class='text-muted fst-italic'>No comparison data available.</p>";
    grid.innerHTML = "";
    return;
  }
  const experiments = payload.experiments;
  availableStepsByExp = {};
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
  });
  grid.innerHTML = "";
  grid.appendChild(buildExperimentGrid(experiments));
  const previewMap = collectPreviewMap(grid);
  pendingInitialPreviewStep = computeInitialPreviewStep(availableStepsByExp, currentImageFolder);
  lastPreviewStep = pendingInitialPreviewStep;
  // Load initial images at the last available step (before plot is rendered)
  if (lastPreviewStep !== null) {
    renderPreviewsAtStep(lastPreviewStep, previewMap, availableStepsByExp);
  } else {
    refreshPreviewImages(grid, availableStepsByExp);
  }
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

      // X-axis selector removed per user request
      // if (xAxisSelect) {
      //   const elapsedOption = xAxisSelect.querySelector('option[value="elapsed"]');
      //   if (elapsedOption) {
      //     elapsedOption.disabled = !hasElapsedSeconds;
      //   }
      //   xAxisSelect.value = nextMode;
      // }

      if (updateUrl || nextMode !== mode) {
        const url = new URL(window.location.href);
        url.searchParams.set("xaxis", nextMode);
        // Clear zoom state when switching axes
        url.searchParams.delete("xmin");
        url.searchParams.delete("xmax");
        window.history.replaceState({}, "", buildUrlString(url));
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

      // X-axis toggle handler - REMOVED per user request
      // if (xAxisSelect) {
      //   xAxisSelect.addEventListener("change", () => {
      //     applyXAxisMode(xAxisSelect.value);
      //   });
      // }

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

        window.history.replaceState({}, "", buildUrlString(url));
      });

      wireExperimentVisibilitySync(plot);
    });
    attachComparisonHover(plot, availableStepsByExp, previewMap);
    hasRenderedPlot = true;
  } else {
    hasRenderedPlot = false;
    plot.innerHTML = "<p class='text-muted fst-italic'>No overlapping CSV metrics found.</p>";
  }
  wireTitleForms(grid);
  wireTagsForms(grid);
}

function buildExperimentGrid(experiments) {
  const container = document.createElement("div");
  container.className = "d-flex flex-column gap-4";

  // Folder selector row - REMOVED per user request
  // const selectorRow = document.createElement("div");
  // selectorRow.className = "d-flex align-items-center gap-2 mb-2";
  //
  // const selectorLabel = document.createElement("label");
  // selectorLabel.className = "form-label mb-0 small text-muted";
  // selectorLabel.textContent = "Image folder:";
  // selectorLabel.htmlFor = "image-folder-select";
  //
  // const selector = document.createElement("select");
  // selector.id = "image-folder-select";
  // selector.className = "form-select form-select-sm";
  // selector.style.width = "auto";
  //
  // IMAGE_FOLDER_OPTIONS.forEach((opt) => {
  //   const option = document.createElement("option");
  //   option.value = opt.value;
  //   option.textContent = opt.label;
  //   if (opt.value === currentImageFolder) {
  //     option.selected = true;
  //   }
  //   selector.appendChild(option);
  // });
  // selector.value = currentImageFolder;
  //
  // selector.addEventListener("change", (e) => {
  //   currentImageFolder = e.target.value;
  //   // Update URL with folder selection
  //   const url = new URL(window.location.href);
  //   url.searchParams.set("folder", currentImageFolder);
  //   window.history.replaceState({}, "", url.toString());
  //   // Refresh rollout previews for the newly selected folder
  //   const previews = collectPreviewMap(container);
  //   if (lastPreviewStep === null || lastPreviewStep === undefined) {
  //     lastPreviewStep = computeInitialPreviewStep(availableStepsByExp, currentImageFolder);
  //   }
  //   renderPreviewsAtStep(lastPreviewStep, previews, availableStepsByExp);
  // });
  //
  // selectorRow.appendChild(selectorLabel);
  // selectorRow.appendChild(selector);
  // container.appendChild(selectorRow);

  // For each experiment, create a column with rows of images
  const gridRow = document.createElement("div");
  gridRow.className = "row g-3";

  experiments.forEach((exp) => {
    const col = document.createElement("div");
    col.className = "col";
    col.dataset.expColumn = exp.id;

    const colContainer = document.createElement("div");
    colContainer.className = "d-flex flex-column gap-3";

    // Add each image type as a row in the column
    IMAGE_FOLDER_OPTIONS.forEach((opt) => {
      const imageRow = buildImageRow(exp, opt, availableStepsByExp);
      colContainer.appendChild(imageRow);
    });

    col.appendChild(colContainer);
    gridRow.appendChild(col);
  });

  container.appendChild(gridRow);

  return container;
}

function buildImageRow(exp, imageOption, stepsMap) {
  const container = document.createElement("div");
  container.className = "image-row-preview";
  container.dataset.expId = exp.id;
  container.dataset.folderValue = imageOption.value;

  const path = document.createElement("div");
  path.className = "small text-muted mb-1 image-row-path";
  path.textContent = "Hover to preview";

  const img = document.createElement("img");
  img.className = "rounded border image-row-img d-none";
  img.alt = `${imageOption.label} preview`;
  img.style.maxWidth = "100%";
  img.style.height = "auto";

  const missing = document.createElement("div");
  missing.className = "text-muted fst-italic image-row-missing";
  missing.textContent = "No image available";

  container.appendChild(path);
  container.appendChild(img);
  container.appendChild(missing);
  return container;
}

function buildSectionRow(title, experiments, cellBuilder) {
  const row = document.createElement("div");
  row.className = "row g-3";

  experiments.forEach((exp, index) => {
    const col = document.createElement("div");
    col.className = "col";
    col.dataset.expColumn = exp.id;
    const cell = cellBuilder(exp, index);
    col.appendChild(cell);
    row.appendChild(col);
  });

  return row;
}

function buildPreviewCell(exp) {
  const container = document.createElement("div");
  container.className = "rollout-preview";
  container.dataset.expId = exp.id;
   // Use title field when available; fall back to name.
  const displayTitle = exp.title && exp.title !== "Untitled" ? exp.title : exp.name;
  container.dataset.expTitle = displayTitle;

  const title = document.createElement("div");
  title.className = "rollout-title small mb-1";
  title.textContent = displayTitle || "Rollout preview";

  const path = document.createElement("div");
  path.className = "small text-muted mb-1 rollout-path";
  path.textContent = "Hover a point to preview rollout.";

  const img = document.createElement("img");
  img.className = "rounded border rollout-img d-none";
  img.alt = "Rollout preview";

  const missing = document.createElement("div");
  missing.className = "text-muted fst-italic rollout-missing d-none";
  missing.textContent = "No rollout image available.";

  container.appendChild(title);
  container.appendChild(path);
  container.appendChild(img);
  container.appendChild(missing);
  return container;
}

function collectPreviewMap(root) {
  const map = {};
  // Collect all image-row-preview elements for each experiment and folder
  root.querySelectorAll(".image-row-preview").forEach((preview) => {
    const expId = preview.dataset.expId;
    const folderValue = preview.dataset.folderValue;
    if (!expId || !folderValue) {
      return;
    }
    const key = `${expId}:${folderValue}`;
    const path = preview.querySelector(".image-row-path");
    const img = preview.querySelector(".image-row-img");
    const missing = preview.querySelector(".image-row-missing");
    if (path && img && missing) {
      map[key] = { path, img, missing, expId, folderValue };
    }
  });
  return map;
}

function refreshPreviewImages(container, stepsMap) {
  const previews = collectPreviewMap(container);
  Object.entries(previews).forEach(([key, preview]) => {
    const { expId, folderValue } = preview;
    const steps = getStepsForFolder(stepsMap, expId, folderValue);
    const latestStep = steps.length ? steps[steps.length - 1] : null;
    if (latestStep === null) {
      preview.img.classList.add("d-none");
      preview.missing.classList.remove("d-none");
      preview.path.textContent = "No image available.";
      return;
    }
    const spec = resolveImageSpec(folderValue, stepsMap, expId);
    const filename = `${spec.prefix}${latestStep.toString().padStart(7, "0")}.png`;
    preview.img.src = `/assets/${expId}/${spec.folderPath}/${filename}`;
    preview.img.alt = `${folderValue} ${latestStep}`;
    preview.img.classList.remove("d-none");
    preview.missing.classList.add("d-none");
    preview.path.textContent = `${spec.folderPath}/${filename}`;
  });
}

function attachComparisonHover(plotEl, stepsMap, previews) {
  if (!plotEl || !previews || typeof Plotly === "undefined") {
    return;
  }
  plotEl.on("plotly_hover", (event) => {
    const points = event?.points || [];
    const point = points.find((p) => p && typeof p.x !== "undefined");
    if (!point) {
      return;
    }
    lastPreviewStep = point.x;

    // Update all image rows based on hover
    Object.entries(previews).forEach(([key, preview]) => {
      const { expId, folderValue } = preview;
      const spec = resolveImageSpec(folderValue, stepsMap, expId);
      const available = getStepsForFolder(stepsMap, expId, folderValue);
      const matched = nearestStepAtOrBelow(point.x, available);

      if (matched === null) {
        preview.img.classList.add("d-none");
        preview.missing.classList.remove("d-none");
        preview.path.textContent = "No image available.";
        return;
      }

      const filename = `${spec.prefix}${matched.toString().padStart(7, "0")}.png`;
      const src = `/assets/${expId}/${spec.folderPath}/${filename}`;
      const pathText = `${expId}/${spec.folderPath}/${filename} (selected ${point.x}, showing ${matched})`;

      preview.path.textContent = pathText;
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

// Main initialization function
function initializeComparePage(config) {
  // Set configuration
  currentImageFolder = config.defaultImageFolder;
  IMAGE_FOLDER_OPTIONS = config.imageFolderOptions;
  const apiEndpoint = config.apiEndpoint;

  document.addEventListener("DOMContentLoaded", () => {
    // Initialize pin functionality
    const pinButton = document.getElementById("compare-loss-pin");
    if (pinButton) {
      setComparePinned(true); // Pinned by default
      pinButton.addEventListener("click", () => {
        const wrapper = document.getElementById("compare-loss-wrapper");
        const isPinned = wrapper && wrapper.classList.contains("sticky-active");
        setComparePinned(!isPinned);
      });
    }

    // Restore folder selection from URL
    const urlParams = new URLSearchParams(window.location.search);
    const folderParam = urlParams.get("folder");
    if (folderParam && IMAGE_FOLDER_OPTIONS.some((opt) => opt.value === folderParam)) {
      currentImageFolder = folderParam;
    }

    // Restore x-axis mode from URL - REMOVED per user request
    // const xAxisParam = urlParams.get("xaxis");
    // if (xAxisParam === "flops" || xAxisParam === "elapsed") {
    //   currentXAxisMode = xAxisParam;
    //   const xAxisSelect = document.getElementById("comparison-xaxis-select");
    //   if (xAxisSelect) {
    //     xAxisSelect.value = xAxisParam;
    //   }
    // }

    const ids = getIdsFromDataset();
    if (ids.length >= 2) {
      // Update URL to reflect selected IDs
      const currentUrl = new URL(window.location.href);
      const urlIds = currentUrl.searchParams.get("ids");
      if (urlIds !== ids.join(",")) {
        // Build URL manually to avoid encoding commas
        currentUrl.searchParams.delete("ids");
        const baseUrl = `${currentUrl.origin}${currentUrl.pathname}`;
        const newUrl = `${baseUrl}?ids=${ids.join(",")}`;
        window.history.replaceState({}, "", newUrl);
      }
      runComparison(ids, apiEndpoint);
    }
  });
}
