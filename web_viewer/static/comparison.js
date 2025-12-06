let hasRenderedPlot = false;

document.addEventListener("DOMContentLoaded", () => {
  const select = document.getElementById("comparison-select");
  const trigger = document.getElementById("comparison-run");
  if (!select || !trigger) {
    return;
  }
  trigger.addEventListener("click", () => {
    const ids = Array.from(select.selectedOptions).map((option) => option.value);
    if (ids.length < 2) {
      alert("Select at least two experiments.");
      return;
    }
    runComparison(ids);
  });
  const preset = (select.dataset.selected || "")
    .split(",")
    .filter((id) => id && Array.from(select.options).some((opt) => opt.value === id));
  if (preset.length >= 2) {
    runComparison(preset);
  } else if (select.options.length >= 2) {
    const defaults = [select.options[0].value, select.options[1].value];
    defaults.forEach((value) => {
      const option = Array.from(select.options).find((opt) => opt.value === value);
      if (option) option.selected = true;
    });
    runComparison(defaults);
  }
});

function runComparison(ids) {
  const plot = document.getElementById("comparison-plot");
  const grid = document.getElementById("comparison-grid");
  syncQuery(ids);
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
      plot.innerHTML = `<p class="empty-state">${error.message}</p>`;
    });
}

function renderComparison(payload) {
  const plot = document.getElementById("comparison-plot");
  const grid = document.getElementById("comparison-grid");
  if (!payload || !Array.isArray(payload.experiments) || payload.experiments.length === 0) {
    plot.innerHTML = "<p class='empty-state'>No comparison data available.</p>";
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
    plot.innerHTML = "<p class='empty-state'>No overlapping CSV metrics found.</p>";
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
  row.className = `comparison-row ${name}`;
  experiments.forEach((exp, index) => {
    const cell = cellFactory(exp, index);
    row.appendChild(cell);
  });
  return row;
}

function buildNameCell(exp) {
  const card = document.createElement("div");
  card.className = "comparison-card";
  const form = document.createElement("form");
  form.className = "title-form";
  form.dataset.expId = exp.id;
  const input = document.createElement("input");
  input.type = "text";
  input.className = "exp-title-input";
  input.placeholder = "Untitled";
  if (exp.title && exp.title !== "Untitled") {
    input.value = exp.title;
  }
  const status = document.createElement("span");
  status.className = "title-status";
  status.setAttribute("aria-live", "polite");
  form.appendChild(input);
  form.appendChild(status);
  const name = document.createElement("div");
  name.className = "exp-name";
  name.textContent = exp.name;
  const commit = document.createElement("div");
  commit.className = "exp-commit";
  commit.textContent = exp.git_commit || "Unknown commit";
  card.appendChild(form);
  card.appendChild(name);
  card.appendChild(commit);
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

function syncQuery(ids) {
  const url = new URL(window.location.href);
  if (ids.length >= 1) {
    url.searchParams.set("ids", ids.join(","));
  } else {
    url.searchParams.delete("ids");
  }
  window.history.replaceState(null, "", url.toString());
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
  card.className = "comparison-card";
  if (exp.loss_image) {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = exp.loss_image;
    img.alt = `Loss curves for ${exp.name}`;
    card.appendChild(img);
  } else {
    const placeholder = document.createElement("div");
    placeholder.className = "empty-state";
    placeholder.textContent = "metrics/loss_curves.png missing";
    card.appendChild(placeholder);
  }
  return card;
}

function buildMetadataCell(exp, index) {
  const card = document.createElement("div");
  card.className = "comparison-card";
  const pre = document.createElement("pre");
  if (index === 0) {
    pre.textContent = exp.metadata;
  } else {
    pre.textContent = exp.metadata_diff || "(no diff)";
  }
  card.appendChild(pre);
  return card;
}
