let hasRenderedPlot = false;

document.addEventListener("DOMContentLoaded", () => {
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
    const config = buildPlotConfig(figure.config);
    Plotly.react(plot, figure.data, figure.layout, config);
    attachComparisonHover(plot, availableStepsByExp, previewMap);
    hasRenderedPlot = true;
  } else {
    hasRenderedPlot = false;
    plot.innerHTML = "<p class='text-muted fst-italic'>No overlapping CSV metrics found.</p>";
  }
  wireTitleForms(grid);
}

function buildExperimentGrid(experiments) {
  const row = document.createElement("div");
  row.className = "row g-3";
  experiments.forEach((exp, index) => {
    const col = document.createElement("div");
    col.className = "col";
    col.appendChild(buildExperimentColumn(exp, index));
    row.appendChild(col);
  });
  return row;
}

function buildExperimentColumn(exp, index) {
  const card = document.createElement("div");
  card.className = "card h-100";
  // Preview section
  const previewBody = document.createElement("div");
  previewBody.className = "card-body p-2 rollout-preview";
  previewBody.dataset.expId = exp.id;
  const path = document.createElement("div");
  path.className = "small text-muted mb-1 rollout-path";
  path.textContent = "Hover a point to preview rollout.";
  const img = document.createElement("img");
  img.className = "img-fluid rounded border rollout-img d-none";
  img.alt = "Rollout preview";
  const missing = document.createElement("div");
  missing.className = "text-muted fst-italic rollout-missing d-none";
  missing.textContent = "No rollout image available.";
  previewBody.appendChild(path);
  previewBody.appendChild(img);
  previewBody.appendChild(missing);
  // Title/path/git section
  const infoBody = document.createElement("div");
  infoBody.className = "card-body";
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
  infoBody.appendChild(form);
  infoBody.appendChild(name);
  infoBody.appendChild(commit);
  // Loss image section
  const imageBody = document.createElement("div");
  imageBody.className = "card-body";
  if (exp.loss_image) {
    const lossImg = document.createElement("img");
    lossImg.loading = "lazy";
    lossImg.src = exp.loss_image;
    lossImg.alt = `Loss curves for ${exp.name}`;
    lossImg.className = "img-fluid rounded border";
    imageBody.appendChild(lossImg);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "text-muted fst-italic";
    placeholder.textContent = "metrics/loss_curves.png missing";
    imageBody.appendChild(placeholder);
  }
  // Metadata section
  const metaBody = document.createElement("div");
  metaBody.className = "card-body";
  const pre = document.createElement("pre");
  pre.className = "bg-dark text-light p-2 rounded overflow-auto mb-0 small";
  pre.style.maxHeight = "280px";
  if (index === 0) {
    pre.textContent = exp.metadata;
  } else {
    pre.textContent = exp.metadata_diff || "(no diff)";
  }
  metaBody.appendChild(pre);
  card.appendChild(previewBody);
  card.appendChild(infoBody);
  card.appendChild(imageBody);
  card.appendChild(metaBody);
  return card;
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
    Object.entries(previews).forEach(([expId, preview]) => {
      const available = stepsMap?.[expId] || [];
      const matched = nearestStep(target, available);
      if (matched === null) {
        preview.path.textContent = "No rollout images available.";
        preview.img.classList.add("d-none");
        preview.missing.textContent = "No rollout images available.";
        preview.missing.classList.remove("d-none");
        return;
      }
      const filename = `rollout_${matched.toString().padStart(7, "0")}.png`;
      const src = `/assets/${expId}/vis_fixed_0/${filename}`;
      preview.path.textContent = `${expId}/vis_fixed_0/${filename} (hovered ${target}, showing ${matched})`;
      preview.img.src = src;
      preview.img.alt = `Rollout ${matched}`;
      preview.img.dataset.step = String(matched);
      preview.img.classList.remove("d-none");
      preview.img.onerror = () => {
        preview.img.classList.add("d-none");
        preview.missing.textContent = `No rollout image for step ${matched}`;
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

function getIdsFromDataset() {
  const container = document.querySelector("[data-selected-ids]");
  if (!container) {
    return [];
  }
  return container.dataset.selectedIds.split(",").filter(Boolean);
}

const PLOTLY_SHOW_ICON = {
  width: 512,
  height: 512,
  path: "M256 64 L256 448 M64 256 L448 256",
};

const PLOTLY_HIDE_ICON = {
  width: 512,
  height: 512,
  path: "M64 256 L448 256",
};

function buildImageCell(exp) {
  const card = document.createElement("div");
  card.className = "card h-100";
  const cardBody = document.createElement("div");
  cardBody.className = "card-body";
  if (exp.loss_image) {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = exp.loss_image;
    img.alt = `Loss curves for ${exp.name}`;
    img.className = "img-fluid rounded border";
    cardBody.appendChild(img);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "text-muted fst-italic";
    placeholder.textContent = "metrics/loss_curves.png missing";
    cardBody.appendChild(placeholder);
  }
  card.appendChild(cardBody);
  return card;
}

function buildMetadataCell(exp, index) {
  const card = document.createElement("div");
  card.className = "card h-100";
  const cardBody = document.createElement("div");
  cardBody.className = "card-body";
  const pre = document.createElement("pre");
  pre.className = "bg-dark text-light p-2 rounded overflow-auto mb-0 small";
  pre.style.maxHeight = "280px";
  if (index === 0) {
    pre.textContent = exp.metadata;
  } else {
    pre.textContent = exp.metadata_diff || "(no diff)";
  }
  cardBody.appendChild(pre);
  card.appendChild(cardBody);
  return card;
}
