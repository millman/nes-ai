let hasRenderedPlot = false;
let currentImageFolder = "vis_fixed_0";
let currentXAxisMode = "steps";
let lastPreviewStep = null;
let pendingInitialPreviewStep = null;

// Store original data for x-axis toggling
let originalSteps = [];
let cumulativeFlops = [];
let elapsedSeconds = [];
let hasElapsedSeconds = false;

const IMAGE_FOLDER_OPTIONS = [
  { value: "vis_fixed_0", label: "vis_fixed_0", prefix: "rollout_", folder: "vis_fixed_0" },
  { value: "vis_fixed_1", label: "vis_fixed_1", prefix: "rollout_", folder: "vis_fixed_1" },
  { value: "vis_rolling_0", label: "vis_rolling_0", prefix: "rollout_", folder: "vis_rolling_0" },
  { value: "vis_rolling_1", label: "vis_rolling_1", prefix: "rollout_", folder: "vis_rolling_1" },
  { value: "embeddings", label: "embeddings", prefix: "embeddings_", folder: "embeddings" },
  { value: "samples_hard", label: "samples_hard", prefix: "hard_", folder: "samples_hard" },
  { value: "vis_self_distance_z", label: "vis_self_distance_z", prefix: "self_distance_z_", folder: "vis_self_distance_z" },
  { value: "vis_self_distance_s", label: "vis_self_distance_s", prefix: "self_distance_s_", folder: "vis_self_distance_s" },
  { value: "vis_delta_z_pca", label: "vis_delta_z_pca", prefix: "delta_z_pca_", folder: "vis_delta_z_pca" },
  { value: "vis_delta_s_pca", label: "vis_delta_s_pca", prefix: "delta_s_pca_", folder: "vis_delta_s_pca" },
  {
    value: "vis_action_alignment_detail",
    label: "vis_action_alignment_z:detail",
    prefix: "action_alignment_detail_",
    folder: "vis_action_alignment_z",
  },
  {
    value: "vis_action_alignment_detail_s",
    label: "vis_action_alignment_s:detail",
    prefix: "action_alignment_detail_",
    folder: "vis_action_alignment_s",
  },
  { value: "vis_cycle_error", label: "vis_cycle_error_z", prefix: "cycle_error_", folder: "vis_cycle_error_z" },
  { value: "vis_cycle_error_s", label: "vis_cycle_error_s", prefix: "cycle_error_", folder: "vis_cycle_error_s" },
  { value: "vis_adjacency", label: "vis_adjacency", prefix: "adjacency_", folder: "vis_adjacency" },
  { value: "vis_graph_rank1_cdf_z", label: "vis_graph_rank1_cdf_z", prefix: "rank1_cdf_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_rank2_cdf_z", label: "vis_graph_rank2_cdf_z", prefix: "rank2_cdf_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_neff_violin_z", label: "vis_graph_neff_violin_z", prefix: "neff_violin_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_in_degree_hist_z", label: "vis_graph_in_degree_hist_z", prefix: "in_degree_hist_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_edge_consistency_z", label: "vis_graph_edge_consistency_z", prefix: "edge_consistency_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_metrics_history_z", label: "vis_graph_metrics_history_z", prefix: "metrics_history_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_rank1_cdf_s", label: "vis_graph_rank1_cdf_s", prefix: "rank1_cdf_", folder: "graph_diagnostics_s" },
  { value: "vis_graph_rank2_cdf_s", label: "vis_graph_rank2_cdf_s", prefix: "rank2_cdf_", folder: "graph_diagnostics_s" },
  { value: "vis_graph_neff_violin_s", label: "vis_graph_neff_violin_s", prefix: "neff_violin_", folder: "graph_diagnostics_s" },
  { value: "vis_graph_in_degree_hist_s", label: "vis_graph_in_degree_hist_s", prefix: "in_degree_hist_", folder: "graph_diagnostics_s" },
  { value: "vis_graph_edge_consistency_s", label: "vis_graph_edge_consistency_s", prefix: "edge_consistency_", folder: "graph_diagnostics_s" },
  { value: "vis_graph_metrics_history_s", label: "vis_graph_metrics_history_s", prefix: "metrics_history_", folder: "graph_diagnostics_s" },
].sort((a, b) => a.value.localeCompare(b.value));

function getImageOption(folderValue) {
  return IMAGE_FOLDER_OPTIONS.find((opt) => opt.value === folderValue);
}

function getImagePrefixForFolder(folderValue) {
  const option = getImageOption(folderValue);
  return option ? option.prefix : "rollout_";
}

function getImageFolderPath(folderValue) {
  const option = getImageOption(folderValue);
  return option ? option.folder || option.value : folderValue;
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

function renderPreviewsAtStep(step, previews, stepsMap) {
  const targetStep = step !== null && step !== undefined ? Math.max(0, Math.round(step)) : null;
  const folderValue = currentImageFolder;
  const folderPath = getImageFolderPath(folderValue);
  const prefix = getImagePrefixForFolder(folderValue);
  Object.entries(previews || {}).forEach(([expId, preview]) => {
    const displayTitle = preview.displayTitle || expId;
    const available = getStepsForFolder(stepsMap, expId, folderValue);
    const matched = targetStep !== null ? nearestStepAtOrBelow(targetStep, available) : (available.length ? available[available.length - 1] : null);
    if (matched === null) {
      preview.title.textContent = displayTitle;
      preview.path.textContent = "No images available.";
      preview.img.classList.add("d-none");
      preview.missing.textContent = "No images available.";
      preview.missing.classList.remove("d-none");
      preview.img.removeAttribute("data-step");
      return;
    }
    const filename = `${prefix}${matched.toString().padStart(7, "0")}.png`;
    const src = `/assets/${expId}/${folderPath}/${filename}`;
    const pathText =
      targetStep !== null
        ? `${expId}/${folderPath}/${filename} (selected ${targetStep}, showing ${matched})`
        : `${expId}/${folderPath}/${filename} (showing ${matched})`;
    preview.title.textContent = displayTitle;
    preview.path.textContent = pathText;
    preview.img.src = src;
    preview.img.alt = `${folderPath} ${matched}`;
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
  // Restore folder selection from URL
  const urlParams = new URLSearchParams(window.location.search);
  const folderParam = urlParams.get("folder");
  if (folderParam && IMAGE_FOLDER_OPTIONS.some((opt) => opt.value === folderParam)) {
    currentImageFolder = folderParam;
  }

  // Restore x-axis mode from URL
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
  lastPreviewStep = null;
  refreshPreviewImages(grid, availableStepsByExp);
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
        window.history.replaceState({}, "", url.toString());
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
      if (lastPreviewStep === null && pendingInitialPreviewStep !== null) {
        lastPreviewStep = pendingInitialPreviewStep;
        renderPreviewsAtStep(lastPreviewStep, previewMap, availableStepsByExp);
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

        window.history.replaceState({}, "", url.toString());
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

  // Folder selector row
  const selectorRow = document.createElement("div");
  selectorRow.className = "d-flex align-items-center gap-2 mb-2";

  const selectorLabel = document.createElement("label");
  selectorLabel.className = "form-label mb-0 small text-muted";
  selectorLabel.textContent = "Image folder:";
  selectorLabel.htmlFor = "image-folder-select";

  const selector = document.createElement("select");
  selector.id = "image-folder-select";
  selector.className = "form-select form-select-sm";
  selector.style.width = "auto";

  IMAGE_FOLDER_OPTIONS.forEach((opt) => {
    const option = document.createElement("option");
    option.value = opt.value;
    option.textContent = opt.label;
    if (opt.value === currentImageFolder) {
      option.selected = true;
    }
    selector.appendChild(option);
  });
  selector.value = currentImageFolder;

  selector.addEventListener("change", (e) => {
    currentImageFolder = e.target.value;
    // Update URL with folder selection
    const url = new URL(window.location.href);
    url.searchParams.set("folder", currentImageFolder);
    window.history.replaceState({}, "", url.toString());
    // Refresh rollout previews for the newly selected folder
    const previews = collectPreviewMap(container);
    if (lastPreviewStep === null || lastPreviewStep === undefined) {
      lastPreviewStep = computeInitialPreviewStep(availableStepsByExp, currentImageFolder);
    }
    renderPreviewsAtStep(lastPreviewStep, previews, availableStepsByExp);
  });

  selectorRow.appendChild(selectorLabel);
  selectorRow.appendChild(selector);
  container.appendChild(selectorRow);

  // Row 1: Preview section
  const previewRow = buildSectionRow(
    "Rollout Preview",
    experiments,
    (exp) => buildPreviewCell(exp)
  );
  container.appendChild(previewRow);

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
    (exp, index) => buildMetadataCell(exp, index)
  );
  container.appendChild(metaRow);

  // Row 5: Git metadata section
  const gitMetaRow = buildSectionRow(
    "Git Metadata",
    experiments,
    (exp) => buildGitMetadataCell(exp)
  );
  container.appendChild(gitMetaRow);

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
    if (!expId) {
      return;
    }
    const displayTitle = preview.dataset.expTitle || expId;
    const title = preview.querySelector(".rollout-title");
    const path = preview.querySelector(".rollout-path");
    const img = preview.querySelector(".rollout-img");
    const missing = preview.querySelector(".rollout-missing");
    if (title && path && img && missing) {
      map[expId] = { title, path, img, missing, displayTitle };
    }
  });
  return map;
}

function refreshPreviewImages(container, stepsMap) {
  const previews = collectPreviewMap(container);
  Object.entries(previews).forEach(([expId, preview]) => {
    const steps = stepsMap?.[expId]?.[currentImageFolder] || stepsMap?.[expId]?.__fallback || [];
    const latestStep = steps.length ? steps[steps.length - 1] : null;
    if (latestStep === null) {
      preview.img.classList.add("d-none");
      preview.missing.classList.remove("d-none");
      preview.path.textContent = "No rollout image available.";
      return;
    }
    const folderPath = getImageFolderPath(currentImageFolder);
    const prefix = getImagePrefixForFolder(currentImageFolder);
    preview.img.src = `/assets/${expId}/${folderPath}/${prefix}${latestStep}.png`;
    preview.img.alt = `Rollout ${latestStep} for ${preview.displayTitle}`;
    preview.img.classList.remove("d-none");
    preview.missing.classList.add("d-none");
    preview.path.textContent = `${folderPath}/${prefix}${latestStep}.png`;
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
    renderPreviewsAtStep(point.x, previews, stepsMap);
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

function buildMetadataCell(exp, index) {
  const container = document.createElement("div");
  const pre = document.createElement("pre");
  pre.className = "bg-dark text-light p-2 rounded overflow-auto mb-0 small";
  pre.style.maxHeight = "280px";
  if (index === 0) {
    pre.textContent = exp.metadata;
  } else {
    pre.textContent = exp.metadata_diff || "(no diff)";
  }
  container.appendChild(pre);
  return container;
}

function buildGitMetadataCell(exp) {
  const container = document.createElement("div");
  container.style.overflow = "hidden";
  const pre = document.createElement("pre");
  pre.className = "bg-dark text-light p-2 rounded mb-0 small";
  pre.style.maxHeight = "280px";
  pre.style.overflowX = "auto";
  pre.style.overflowY = "auto";
  pre.style.whiteSpace = "pre";
  pre.textContent = exp.git_metadata || "(no git metadata)";
  container.appendChild(pre);
  return container;
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
