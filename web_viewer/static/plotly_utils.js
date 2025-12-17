// Shared Plotly utilities for trace visibility controls

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

const DEFAULT_EXPERIMENT_KEY = "__single__";

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

function extractTraceMetric(name = "") {
  const parts = name.split(":").map((p) => p.trim());
  return parts.length > 1 ? parts[1] : parts[0];
}

function extractTraceExperiment(name = "") {
  const parts = name.split(":").map((p) => p.trim());
  return parts.length > 1 ? parts[0] : null;
}

function toggleTracesByMetricName(gd, metricName) {
  if (!gd || !gd.data || !gd.data.length) {
    return;
  }
  if (!metricName) {
    return;
  }
  const indices = [];
  const visibilities = [];
  gd.data.forEach((trace, idx) => {
    const traceMetric = extractTraceMetric(trace.name);
    if (traceMetric === metricName) {
      indices.push(idx);
      const currentlyVisible = trace.visible === true || trace.visible === undefined;
      visibilities.push(currentlyVisible ? "legendonly" : true);
    }
  });
  if (indices.length > 0) {
    Plotly.restyle(gd, { visible: visibilities }, indices);
  }
}

function pickLossReconMetricsByExperiment(gd) {
  if (!gd || !gd.data || !gd.data.length) {
    return new Map();
  }
  const exact = new Map();
  const fallback = new Map();
  gd.data.forEach((trace) => {
    const metric = extractTraceMetric(trace.name);
    if (!metric) {
      return;
    }
    const experiment = extractTraceExperiment(trace.name) || DEFAULT_EXPERIMENT_KEY;
    if (metric === "loss_recon") {
      exact.set(experiment, metric);
      return;
    }
    if (!fallback.has(experiment) && metric.startsWith("loss_recon_")) {
      fallback.set(experiment, metric);
    }
  });
  const chosen = new Map();
  fallback.forEach((metric, experiment) => {
    chosen.set(experiment, metric);
  });
  exact.forEach((metric, experiment) => {
    chosen.set(experiment, metric);
  });
  return chosen;
}

function toggleLossReconPerExperiment(gd) {
  if (!gd || !gd.data || !gd.data.length) {
    return;
  }
  const selectedByExperiment = pickLossReconMetricsByExperiment(gd);
  if (!selectedByExperiment.size) {
    return;
  }
  const indices = [];
  const visibilities = [];
  gd.data.forEach((trace, idx) => {
    const metric = extractTraceMetric(trace.name);
    if (!metric) {
      return;
    }
    const experiment = extractTraceExperiment(trace.name) || DEFAULT_EXPERIMENT_KEY;
    const targetMetric = selectedByExperiment.get(experiment);
    if (targetMetric && metric === targetMetric) {
      indices.push(idx);
      const currentlyVisible = trace.visible === true || trace.visible === undefined;
      visibilities.push(currentlyVisible ? "legendonly" : true);
    }
  });
  if (indices.length > 0) {
    Plotly.restyle(gd, { visible: visibilities }, indices);
  }
}

function buildPlotlyConfig(baseConfig = {}) {
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
      click: (gd) => toggleLossReconPerExperiment(gd),
    },
  ];
  const existingButtons = Array.isArray(clone.modeBarButtonsToAdd) ? clone.modeBarButtonsToAdd : [];
  clone.modeBarButtonsToAdd = [...existingButtons, ...customButtons];
  clone.responsive = true;
  return clone;
}
