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
});

function runComparison(ids) {
  const plot = document.getElementById("comparison-plot");
  const grid = document.getElementById("comparison-grid");
  plot.innerHTML = "<p>Loading…</p>";
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
    Plotly.react(plot, figure.data, figure.layout, figure.config || {});
  } else {
    plot.innerHTML = "<p class='empty-state'>No overlapping CSV metrics found.</p>";
  }
  const experiments = payload.experiments;
  grid.innerHTML = "";
  grid.appendChild(buildRow("names", experiments, buildNameCell));
  grid.appendChild(buildRow("images", experiments, buildImageCell));
  grid.appendChild(buildRow("metadata", experiments, buildMetadataCell));
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
  const title = document.createElement("div");
  title.className = "exp-name";
  title.textContent = exp.name;
  const commit = document.createElement("div");
  commit.className = "exp-commit";
  commit.textContent = exp.git_commit || "Unknown commit";
  card.appendChild(title);
  card.appendChild(commit);
  return card;
}

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
