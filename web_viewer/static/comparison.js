let hasRenderedPlot = false;
let currentImageFolder = "vis_fixed_0";
let currentXAxisMode = "steps";

// Store original data for x-axis toggling
let originalSteps = [];
let cumulativeFlops = [];

const IMAGE_FOLDER_OPTIONS = [
  { value: "vis_fixed_0", label: "vis_fixed_0", prefix: "rollout_" },
  { value: "vis_fixed_1", label: "vis_fixed_1", prefix: "rollout_" },
  { value: "vis_rolling_0", label: "vis_rolling_0", prefix: "rollout_" },
  { value: "vis_rolling_1", label: "vis_rolling_1", prefix: "rollout_" },
  { value: "embeddings", label: "embeddings", prefix: "embeddings_" },
  { value: "samples_hard", label: "samples_hard", prefix: "hard_" },
];

function getImagePrefixForFolder(folder) {
  const option = IMAGE_FOLDER_OPTIONS.find((opt) => opt.value === folder);
  return option ? option.prefix : "rollout_";
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
  if (xAxisParam === "flops") {
    currentXAxisMode = "flops";
    const xAxisSelect = document.getElementById("comparison-xaxis-select");
    if (xAxisSelect) {
      xAxisSelect.value = "flops";
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
  if (!payload || !Array.isArray(payload.experiments) || payload.experiments.length === 0) {
    plot.innerHTML = "<p class='text-muted fst-italic'>No comparison data available.</p>";
    grid.innerHTML = "";
    return;
  }
  const experiments = payload.experiments;
  const availableStepsByExp = {};
  experiments.forEach((exp) => {
    availableStepsByExp[exp.id] = Array.isArray(exp.rollout_steps) ? exp.rollout_steps : [];
  });
  grid.innerHTML = "";
  grid.appendChild(buildExperimentGrid(experiments));
  const previewMap = collectPreviewMap(grid);
  const figure = payload.figure;
  if (figure) {
    // Store original x-axis data for toggling
    originalSteps = figure.data.map(trace => trace.x ? [...trace.x] : []);
    cumulativeFlops = figure.data.map(trace => {
      // Extract cumulative_flops from meta (stored by plots.py)
      if (trace.meta && trace.meta.cumulative_flops) {
        return trace.meta.cumulative_flops;
      }
      // Fallback: use customdata if available (cumulative_flops is at index 1)
      if (trace.customdata && Array.isArray(trace.customdata)) {
        return trace.customdata.map(cd => Array.isArray(cd) ? cd[1] : null);
      }
      return trace.x ? [...trace.x] : [];  // Fallback to steps
    });

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

    const config = buildPlotConfig(figure.config);
    Plotly.react(plot, figure.data, figure.layout, config).then(() => {
      // Apply x-axis mode if set to flops
      if (currentXAxisMode === "flops") {
        Plotly.restyle(plot, { x: cumulativeFlops });
        Plotly.relayout(plot, { "xaxis.title": "Cumulative FLOPs", "xaxis.autorange": true });
      }

      // X-axis toggle handler
      const xAxisSelect = document.getElementById("comparison-xaxis-select");
      if (xAxisSelect) {
        xAxisSelect.addEventListener("change", () => {
          const mode = xAxisSelect.value;
          currentXAxisMode = mode;
          const url = new URL(window.location.href);
          url.searchParams.set("xaxis", mode);
          // Clear zoom state when switching axes
          url.searchParams.delete("xmin");
          url.searchParams.delete("xmax");
          window.history.replaceState({}, "", url.toString());

          if (mode === "flops") {
            Plotly.restyle(plot, { x: cumulativeFlops });
            Plotly.relayout(plot, { "xaxis.title": "Cumulative FLOPs", "xaxis.autorange": true });
          } else {
            Plotly.restyle(plot, { x: originalSteps });
            Plotly.relayout(plot, { "xaxis.title": "Step", "xaxis.autorange": true });
          }
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
    });
    attachComparisonHover(plot, availableStepsByExp, previewMap);
    hasRenderedPlot = true;
  } else {
    hasRenderedPlot = false;
    plot.innerHTML = "<p class='text-muted fst-italic'>No overlapping CSV metrics found.</p>";
  }
  wireTitleForms(grid);
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

  selector.addEventListener("change", (e) => {
    currentImageFolder = e.target.value;
    // Update URL with folder selection
    const url = new URL(window.location.href);
    url.searchParams.set("folder", currentImageFolder);
    window.history.replaceState({}, "", url.toString());
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

  const path = document.createElement("div");
  path.className = "small text-muted mb-1 rollout-path";
  path.textContent = "Hover a point to preview rollout.";

  const img = document.createElement("img");
  img.className = "rounded border rollout-img d-none";
  img.alt = "Rollout preview";

  const missing = document.createElement("div");
  missing.className = "text-muted fst-italic rollout-missing d-none";
  missing.textContent = "No rollout image available.";

  container.appendChild(path);
  container.appendChild(img);
  container.appendChild(missing);
  return container;
}

function buildInfoCell(exp, index) {
  const container = document.createElement("div");

  const form = document.createElement("form");
  form.className = "title-form mb-2";
  form.dataset.expId = exp.id;

  const group = document.createElement("div");
  group.className = "input-group input-group-sm title-input-group";

  const input = document.createElement("input");
  input.type = "text";
  input.className = "form-control form-control-sm exp-title-input";
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
    link.href = `/experiments/${exp.id}`;
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
    const path = preview.querySelector(".rollout-path");
    const img = preview.querySelector(".rollout-img");
    const missing = preview.querySelector(".rollout-missing");
    if (path && img && missing) {
      map[expId] = { path, img, missing };
    }
  });
  return map;
}

function attachComparisonHover(plotEl, stepsMap, previews) {
  if (!plotEl || !previews || typeof Plotly === "undefined") {
    return;
  }
  const nearestStep = (target, list) => {
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
  };
  const renderAll = (step) => {
    const target = Math.max(0, Math.round(step));
    const folder = currentImageFolder;
    const prefix = getImagePrefixForFolder(folder);
    Object.entries(previews).forEach(([expId, preview]) => {
      const available = stepsMap?.[expId] || [];
      const matched = nearestStep(target, available);
      if (matched === null) {
        preview.path.textContent = "No images available.";
        preview.img.classList.add("d-none");
        preview.missing.textContent = "No images available.";
        preview.missing.classList.remove("d-none");
        return;
      }
      const filename = `${prefix}${matched.toString().padStart(7, "0")}.png`;
      const src = `/assets/${expId}/${folder}/${filename}`;
      preview.path.textContent = `${expId}/${folder}/${filename} (hovered ${target}, showing ${matched})`;
      preview.img.src = src;
      preview.img.alt = `${folder} ${matched}`;
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
    });
  };
  plotEl.on("plotly_hover", (event) => {
    const points = event?.points || [];
    const point = points.find((p) => p && typeof p.x !== "undefined");
    if (!point) {
      return;
    }
    renderAll(point.x);
  });
}

function showPlotLoading(plot) {
  if (hasRenderedPlot && typeof Plotly !== "undefined") {
    Plotly.purge(plot);
    hasRenderedPlot = false;
  }
  plot.innerHTML = "<p>Loading…</p>";
}

function buildPlotConfig(baseConfig = {}) {
  const clone = { ...baseConfig };
  const customButtons = [
    {
      name: "Show all traces",
      icon: PLOTLY_SHOW_ICON,
      click: (gd) => toggleTraceVisibility(gd, true),
    },
    {
      name: "Hide all traces",
      icon: PLOTLY_HIDE_ICON,
      click: (gd) => toggleTraceVisibility(gd, "legendonly"),
    },
    {
      name: "Toggle loss_recon",
      icon: PLOTLY_RECON_ICON,
      click: (gd) => toggleTracesByPattern(gd, "loss_recon"),
    },
  ];
  const existingButtons = Array.isArray(clone.modeBarButtonsToAdd) ? clone.modeBarButtonsToAdd : [];
  clone.modeBarButtonsToAdd = [...existingButtons, ...customButtons];
  clone.responsive = true;
  return clone;
}

function toggleTraceVisibility(gd, visibility) {
  if (!gd || !gd.data || !gd.data.length) {
    return;
  }
  const indices = gd.data.map((_, idx) => idx);
  Plotly.restyle(gd, { visible: visibility }, indices);
}

function toggleTracesByPattern(gd, pattern) {
  if (!gd || !gd.data || !gd.data.length) {
    return;
  }
  const indices = [];
  const visibilities = [];
  gd.data.forEach((trace, idx) => {
    if (trace.name && trace.name.includes(pattern)) {
      indices.push(idx);
      // Toggle: if currently visible (true or undefined), hide; otherwise show
      const currentlyVisible = trace.visible === true || trace.visible === undefined;
      visibilities.push(currentlyVisible ? "legendonly" : true);
    }
  });
  if (indices.length > 0) {
    Plotly.restyle(gd, { visible: visibilities }, indices);
  }
}

function getIdsFromDataset() {
  const container = document.querySelector("[data-selected-ids]");
  if (!container) {
    return [];
  }
  return container.dataset.selectedIds.split(",").filter(Boolean);
}

const PLOTLY_SHOW_ICON = {
  width: 1000,
  height: 1000,
  path: "M500 200 C250 200 50 500 50 500 C50 500 250 800 500 800 C750 800 950 500 950 500 C950 500 750 200 500 200 Z M500 350 C583 350 650 417 650 500 C650 583 583 650 500 650 C417 650 350 583 350 500 C350 417 417 350 500 350 Z",
};

const PLOTLY_HIDE_ICON = {
  width: 1000,
  height: 1000,
  path: "M150 150 L850 850 M500 200 C250 200 50 500 50 500 C50 500 250 800 500 800 C750 800 950 500 950 500 C950 500 750 200 500 200 Z",
};

const PLOTLY_RECON_ICON = {
  width: 1000,
  height: 1000,
  path: "M200 200 L800 200 L800 800 L200 800 Z M350 350 L650 650 M650 350 L350 650",
};

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
