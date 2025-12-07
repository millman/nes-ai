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
  const figure = payload.figure;
  if (figure) {
    const config = buildPlotConfig(figure.config);
    Plotly.react(plot, figure.data, figure.layout, config);
    hasRenderedPlot = true;
  } else {
    hasRenderedPlot = false;
    plot.innerHTML = "<p class='text-muted fst-italic'>No overlapping CSV metrics found.</p>";
  }
  const experiments = payload.experiments;
  grid.innerHTML = "";
  grid.appendChild(buildRow("names", experiments, buildNameCell));
  grid.appendChild(buildRow("images", experiments, buildImageCell));
  grid.appendChild(buildRow("metadata", experiments, buildMetadataCell));
  wireTitleForms(grid);
}

function buildRow(name, experiments, cellFactory) {
  const row = document.createElement("div");
  row.className = "row g-3 mb-3";
  experiments.forEach((exp, index) => {
    const col = document.createElement("div");
    col.className = "col";
    const cell = cellFactory(exp, index);
    col.appendChild(cell);
    row.appendChild(col);
  });
  return row;
}

function buildNameCell(exp) {
  const card = document.createElement("div");
  card.className = "card h-100";
  const cardBody = document.createElement("div");
  cardBody.className = "card-body";
  const form = document.createElement("form");
  form.className = "title-form mb-2";
  form.dataset.expId = exp.id;
  const titleGroup = document.createElement("div");
  titleGroup.className = "input-group input-group-sm title-input-group";
  const input = document.createElement("input");
  input.type = "text";
  input.className = "form-control form-control-sm exp-title-input";
  input.placeholder = "Untitled";
  if (exp.title && exp.title !== "Untitled") {
    input.value = exp.title;
  }
  input.readOnly = true;
  const status = document.createElement("span");
  status.className = "title-status small text-muted";
  status.setAttribute("aria-live", "polite");
  form.appendChild(titleGroup);
  form.appendChild(status);
  const name = document.createElement("div");
  name.className = "fw-semibold";
  name.textContent = exp.name;
  const commit = document.createElement("div");
  commit.className = "font-monospace text-muted small mt-1";
  commit.textContent = exp.git_commit || "Unknown commit";
  cardBody.appendChild(form);
  cardBody.appendChild(name);
  cardBody.appendChild(commit);
  card.appendChild(cardBody);
  return card;
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
