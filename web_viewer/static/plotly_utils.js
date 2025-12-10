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
      click: (gd) => toggleTracesByPattern(gd, "loss_recon"),
    },
  ];
  const existingButtons = Array.isArray(clone.modeBarButtonsToAdd) ? clone.modeBarButtonsToAdd : [];
  clone.modeBarButtonsToAdd = [...existingButtons, ...customButtons];
  clone.responsive = true;
  return clone;
}
