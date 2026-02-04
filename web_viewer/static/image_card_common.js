// Common definitions and utilities for image preview cards
// Used across Detail, Compare, and Diagnostics pages

const IMAGE_FOLDER_SPECS = window.IMAGE_FOLDER_SPECS || [];

// Descriptions for tooltips - explains how to read each plot and what "good" looks like
const IMAGE_FOLDER_DESCRIPTIONS = {
  vis_fixed_0: "Rollout visualization from fixed initial state 0. Read: observe agent behavior over time from a consistent starting position.",
  vis_fixed_1: "Rollout visualization from fixed initial state 1. Read: observe agent behavior over time from a second consistent starting position.",
  vis_rolling_0: "Rollout visualization from rolling initial state 0. Read: observe agent behavior from dynamically sampled starting positions.",
  vis_rolling_1: "Rollout visualization from rolling initial state 1. Read: observe agent behavior from a second set of dynamically sampled positions.",
  pca_z: "PCA of encoder latent (Z). Read: clusters indicate distinct states; spread shows representation diversity. Good: well-separated clusters with smooth transitions.",
  pca_p: "PCA of world model pose (P). Read: clusters indicate distinct internal states; structure shows learned dynamics. Good: organized clusters reflecting game states.",
  pca_h: "PCA of GRU hidden state (H). Read: clusters indicate memory states; trajectories show temporal patterns. Good: smooth trajectories with clear structure.",
  samples_hard: "Hard samples from the dataset. Read: examples that the model finds challenging. Useful for diagnosing model weaknesses.",
  vis_self_distance_z: "Self-distance matrix for Z embeddings. Read: bright diagonals indicate temporal coherence; off-diagonal blocks show recurring states. Good: clear diagonal with periodic structure.",
  vis_self_distance_p: "Self-distance matrix for P embeddings. Read: bright diagonals indicate temporal coherence; off-diagonal blocks show recurring states. Good: clear diagonal with periodic structure.",
  vis_self_distance_h: "Self-distance matrix for H embeddings. Read: bright diagonals indicate temporal coherence; off-diagonal blocks show recurring states. Good: clear diagonal with periodic structure.",
  vis_delta_z_pca: "PCA of delta-Z steps across the batch. Read: spokes or clusters correspond to action directions; separation shows distinct motions. Good: clear, separated clusters with minimal overlap.",
  vis_delta_p_pca: "PCA of delta-P steps across the batch. Read: spokes or clusters correspond to action directions; separation shows distinct motions. Good: clear, separated clusters with minimal overlap.",
  vis_delta_h_pca: "PCA of delta-H steps across the batch. Read: spokes or clusters correspond to action directions; separation shows distinct motions. Good: clear, separated clusters with minimal overlap.",
  vis_odometry_current_z: "Cumulative sum of Δz using PCA/ICA/t-SNE. Read: trajectory shows integrated motion; coherent paths indicate consistent action effects. Good: smooth, interpretable trajectories.",
  vis_odometry_current_p: "Cumulative sum of Δp using PCA/ICA/t-SNE. Read: trajectory shows integrated motion; coherent paths indicate consistent action effects. Good: smooth, interpretable trajectories.",
  vis_odometry_current_h: "Cumulative sum of Δh using PCA/ICA/t-SNE. Read: trajectory shows integrated motion; coherent paths indicate consistent action effects. Good: smooth, interpretable trajectories.",
  vis_odometry_z_vs_z_hat: "||z - z_hat|| comparison with scatter plot. Read: low error means accurate predictions; scatter shows error distribution. Good: tight clustering near origin.",
  vis_odometry_p_vs_p_hat: "||p - p_hat|| comparison with scatter plot. Read: low error means accurate predictions; scatter shows error distribution. Good: tight clustering near origin.",
  vis_odometry_h_vs_h_hat: "||h - h_hat|| comparison with scatter plot. Read: low error means accurate predictions; scatter shows error distribution. Good: tight clustering near origin.",
  vis_action_alignment_detail_z: "Per-action cosine alignment of delta-Z directions. Read: values near 1 mean consistent action effects; near 0 means noisy alignment. Good: high alignment for common actions.",
  vis_action_alignment_detail_raw_z: "Per-action alignment of raw delta-Z (unprocessed). Read: raw action effects before any normalization. Useful for debugging.",
  vis_action_alignment_detail_centered_z: "Per-action alignment of centered delta-Z. Read: action effects after mean-centering. Good: clearer action-specific patterns.",
  vis_action_alignment_detail_p: "Per-action cosine alignment of delta-P directions. Read: values near 1 mean consistent action effects; near 0 means noisy alignment. Good: high alignment for common actions.",
  vis_action_alignment_detail_raw_p: "Per-action alignment of raw delta-P (unprocessed). Read: raw action effects before any normalization. Useful for debugging.",
  vis_action_alignment_detail_centered_p: "Per-action alignment of centered delta-P. Read: action effects after mean-centering. Good: clearer action-specific patterns.",
  vis_action_alignment_detail_h: "Per-action cosine alignment of delta-H directions. Read: values near 1 mean consistent action effects; near 0 means noisy alignment. Good: high alignment for common actions.",
  vis_action_alignment_detail_raw_h: "Per-action alignment of raw delta-H (unprocessed). Read: raw action effects before any normalization. Useful for debugging.",
  vis_action_alignment_detail_centered_h: "Per-action alignment of centered delta-H. Read: action effects after mean-centering. Good: clearer action-specific patterns.",
  vis_ctrl_smoothness_z: "Local smoothness of Z representations. Read: measures how smoothly representations change. Good: low smoothness error indicates continuous representations.",
  vis_ctrl_smoothness_p: "Local smoothness of P representations. Read: measures how smoothly representations change. Good: low smoothness error indicates continuous representations.",
  vis_ctrl_smoothness_h: "Local smoothness of H representations. Read: measures how smoothly representations change. Good: low smoothness error indicates continuous representations.",
  vis_ctrl_composition_z: "Two-step composition error for Z. Read: measures if two consecutive actions compose correctly. Good: low error means action effects are additive.",
  vis_ctrl_composition_p: "Two-step composition error for P. Read: measures if two consecutive actions compose correctly. Good: low error means action effects are additive.",
  vis_ctrl_composition_h: "Two-step composition error for H. Read: measures if two consecutive actions compose correctly. Good: low error means action effects are additive.",
  vis_composability_z: "Two-step composability for Z. Read: compares model rollout vs additive deltas and action-pair strip errors. Good: low errors across timesteps and pairs.",
  vis_composability_p: "Two-step composability for P. Read: compares model rollout vs additive deltas and action-pair strip errors. Good: low errors across timesteps and pairs.",
  vis_composability_h: "Two-step composability for H (actual rollout projected to Z). Read: compares model rollout vs additive deltas and action-pair strip errors. Good: low errors across timesteps and pairs.",
  vis_off_manifold: "Off-manifold diagnostic for rollouts. Read: enc(dec(z_roll)) drift by rollout index and histogram. Good: low errors clustered near zero.",
  vis_ctrl_stability_z: "Neighborhood stability for Z. Read: measures if similar states stay similar after actions. Good: high stability means predictable dynamics.",
  vis_ctrl_stability_p: "Neighborhood stability for P. Read: measures if similar states stay similar after actions. Good: high stability means predictable dynamics.",
  vis_ctrl_stability_h: "Neighborhood stability for H. Read: measures if similar states stay similar after actions. Good: high stability means predictable dynamics.",
  vis_cycle_error: "Cycle error for Z embeddings. Read: measures if opposite actions cancel out. Good: low error means reversible action effects.",
  vis_cycle_error_p: "Cycle error for P embeddings. Read: measures if opposite actions cancel out. Good: low error means reversible action effects.",
  vis_cycle_error_h: "Cycle error for H embeddings. Read: measures if opposite actions cancel out. Good: low error means reversible action effects.",
  vis_rollout_divergence: "Rollout divergence curves. Read: pixel vs latent errors over rollout horizon. Good: flat or slowly growing curves.",
  vis_rollout_divergence_z: "Rollout divergence curves for Z. Read: pixel vs Z errors over rollout horizon. Good: flat or slowly growing curves.",
  vis_rollout_divergence_h: "Rollout divergence curves for H. Read: pixel vs H errors over rollout horizon. Good: flat or slowly growing curves.",
  vis_rollout_divergence_p: "Rollout divergence curves for P. Read: pixel vs P errors over rollout horizon. Good: flat or slowly growing curves.",
  vis_z_consistency: "Same-frame consistency for Z under small noise. Read: tight histograms imply stable z. Good: narrow distance spread, high cosine.",
  vis_z_monotonicity: "Z distance vs pixel shift. Read: smooth monotonic increase indicates perceptual alignment. Good: steadily rising curve.",
  vis_path_independence: "Path independence for Z and P. Read: compare end-state distances for alternate paths. Good: low distances.",
  vis_zp_distance_scatter: "Z/P distance scatter. Read: small z-dist should imply small p-dist; look for monotonic trend near origin.",
  vis_straightline_p: "Straight-line action rays in P. Read: repeated actions trace straight paths; opposite actions mirror. Good: clean rays.",
  vis_straightline_z: "Straight-line action rays in Z. Read: repeated actions trace straight paths; opposite actions mirror. Good: clean rays.",
  vis_straightline_h: "Straight-line action rays in H. Read: repeated actions trace straight paths; opposite actions mirror. Good: clean rays.",
  vis_h_ablation: "H ablation divergence. Read: compare rollout errors with normal vs zeroed H. Good: minimal gap in fully observable settings.",
  vis_h_drift_by_action: "H drift by action. Read: mean ||h_{t+1}-h_t|| grouped by action. Good: low, consistent drift.",
  vis_norm_timeseries: "Norm stability for Z/H/P. Read: mean and p95 norms over steps. Good: stable, bounded lines.",
  vis_planning_action_stats_p: "Planning action deltas summary (P). Read: per-action delta distributions and medians in pose space. Good: clean, separated action clusters.",
  vis_planning_action_stats_strip_p: "Planning action delta strip (P). Read: compact view of action delta directions and magnitudes. Good: consistent, separated bands.",
  vis_planning_pca_test1: "Planning PCA path (test1). Read: PCA projection of pose latents with planned path overlay. Good: smooth path aligned with latent structure.",
  vis_planning_pca_test2: "Planning PCA path (test2). Read: PCA projection of pose latents with planned path overlay. Good: smooth path aligned with latent structure.",
  vis_planning_exec_test1_p: "Planning execution trace (test1). Read: grid trace of executed plan in the environment. Good: path reaches the goal with few detours.",
  vis_planning_exec_test2_p: "Planning execution trace (test2). Read: grid trace of executed plan in the environment. Good: path reaches the goal with few detours.",
  vis_planning_reachable_h: "Planning reachable fraction (H). Read: distribution of graph reachability in H clusters. Good: healthy spread without many isolated nodes.",
  vis_planning_reachable_p: "Planning reachable fraction (P). Read: distribution of graph reachability in P clusters. Good: healthy spread without many isolated nodes.",
  vis_planning_graph_h: "Planning graph (H). Read: PCA projection of samples, cluster centers, and edges in H space. Good: connected structure without excessive cross-links.",
  vis_planning_graph_p: "Planning graph (P). Read: PCA projection of samples, cluster centers, and edges in P space. Good: connected structure without excessive cross-links.",
  vis_graph_rank1_cdf_z: "Rank-1 CDF for Z graph. Read: distribution of top-1 neighbor accuracy. Good: high values at low thresholds.",
  vis_graph_rank2_cdf_z: "Rank-2 CDF for Z graph. Read: distribution of top-2 neighbor accuracy. Good: high values at low thresholds.",
  vis_graph_neff_violin_z: "Neighborhood size (N_eff) for Z. Read: violin plot of effective neighbor counts. Good: consistent, moderate neighborhood sizes.",
  vis_graph_in_degree_hist_z: "In-degree histogram for Z graph. Read: distribution of incoming edges per node. Good: balanced distribution without outliers.",
  vis_graph_edge_consistency_z: "Edge consistency for Z graph. Read: measures if similar nodes share edges. Good: high consistency indicates reliable graph structure.",
  vis_graph_metrics_history_z: "Graph metrics history for Z. Read: how graph quality metrics evolve over training. Good: improving or stable metrics.",
  vis_graph_rank1_cdf_h: "Rank-1 CDF for H graph. Read: distribution of top-1 neighbor accuracy. Good: high values at low thresholds.",
  vis_graph_rank2_cdf_h: "Rank-2 CDF for H graph. Read: distribution of top-2 neighbor accuracy. Good: high values at low thresholds.",
  vis_graph_neff_violin_h: "Neighborhood size (N_eff) for H. Read: violin plot of effective neighbor counts. Good: consistent, moderate neighborhood sizes.",
  vis_graph_in_degree_hist_h: "In-degree histogram for H graph. Read: distribution of incoming edges per node. Good: balanced distribution without outliers.",
  vis_graph_edge_consistency_h: "Edge consistency for H graph. Read: measures if similar nodes share edges. Good: high consistency indicates reliable graph structure.",
  vis_graph_metrics_history_h: "Graph metrics history for H. Read: how graph quality metrics evolve over training. Good: improving or stable metrics.",
  vis_graph_rank1_cdf_p: "Rank-1 CDF for P graph. Read: distribution of top-1 neighbor accuracy. Good: high values at low thresholds.",
  vis_graph_rank2_cdf_p: "Rank-2 CDF for P graph. Read: distribution of top-2 neighbor accuracy. Good: high values at low thresholds.",
  vis_graph_neff_violin_p: "Neighborhood size (N_eff) for P. Read: violin plot of effective neighbor counts. Good: consistent, moderate neighborhood sizes.",
  vis_graph_in_degree_hist_p: "In-degree histogram for P graph. Read: distribution of incoming edges per node. Good: balanced distribution without outliers.",
  vis_graph_edge_consistency_p: "Edge consistency for P graph. Read: measures if similar nodes share edges. Good: high consistency indicates reliable graph structure.",
  vis_graph_metrics_history_p: "Graph metrics history for P. Read: how graph quality metrics evolve over training. Good: improving or stable metrics.",
};

function getResolvedImageSpec(folderValue, optionsList) {
  const options = Array.isArray(optionsList) ? optionsList : IMAGE_FOLDER_SPECS;
  const option = options.find((opt) => opt.value === folderValue);
  if (option && option.folder && option.prefix !== undefined) {
    return { folder: option.folder, prefix: option.prefix };
  }
  return { folder: folderValue, prefix: "rollout_" };
}

// Get option by folder value
function getImageOption(folderValue) {
  return IMAGE_FOLDER_SPECS.find((opt) => opt.value === folderValue);
}

// Get display title without group prefix (e.g., "Diagnostics:Delta-z PCA" -> "Delta-z PCA")
function getImageDisplayTitle(folderValue) {
  const option = getImageOption(folderValue);
  if (!option) return folderValue;
  const label = option.label;
  const colonIndex = label.indexOf(":");
  return colonIndex >= 0 ? label.substring(colonIndex + 1).trim() : label;
}

// Get description for tooltip
function getImageDescription(folderValue) {
  return IMAGE_FOLDER_DESCRIPTIONS[folderValue] || "";
}

// Build an image preview card element
// Options:
//   folderValue: string - the folder value key
//   showStepLabel: boolean - whether to show step label (default true)
//   initialStep: string|number - initial step to display (default "—")
// Returns: { card, stepLabelEl, img, missing }
function buildImagePreviewCard(options = {}) {
  const { folderValue, showStepLabel = true, initialStep = "—" } = options;

  const card = document.createElement("div");
  card.className = "card h-100 image-preview-card";

  // Card header
  const cardHeader = document.createElement("div");
  cardHeader.className = "card-header py-2 image-preview-card-header";

  const titleSpan = document.createElement("span");
  titleSpan.className = "image-preview-card-title";
  titleSpan.textContent = getImageDisplayTitle(folderValue);

  const headerRight = document.createElement("div");
  headerRight.className = "image-preview-card-controls";

  let stepLabelEl = null;
  if (showStepLabel) {
    const stepWrapper = document.createElement("span");
    stepWrapper.className = "image-preview-card-step";

    const stepPrefix = document.createElement("span");
    stepPrefix.className = "image-preview-card-step-prefix";
    stepPrefix.textContent = "Step:";

    stepLabelEl = document.createElement("span");
    stepLabelEl.className = "image-preview-card-step-value";
    stepLabelEl.textContent = typeof initialStep === "number" ? initialStep.toLocaleString() : initialStep;

    stepWrapper.appendChild(stepPrefix);
    stepWrapper.appendChild(document.createTextNode("\u00A0"));
    stepWrapper.appendChild(stepLabelEl);
    headerRight.appendChild(stepWrapper);
  }

  const description = getImageDescription(folderValue);
  if (description) {
    const infoIcon = document.createElement("span");
    infoIcon.className = "image-preview-card-info";
    infoIcon.setAttribute("data-bs-toggle", "tooltip");
    infoIcon.setAttribute("data-bs-placement", "left");
    infoIcon.setAttribute("title", description);
    infoIcon.textContent = "i";
    headerRight.appendChild(infoIcon);
  }

  cardHeader.appendChild(titleSpan);
  cardHeader.appendChild(headerRight);

  // Card body
  const cardBody = document.createElement("div");
  cardBody.className = "card-body image-preview-card-body";

  const img = document.createElement("img");
  img.className = "img-fluid rounded border image-preview-card-img d-none";
  img.alt = "Preview";

  const missing = document.createElement("div");
  missing.className = "text-muted fst-italic small image-preview-card-missing d-none";
  missing.textContent = "No images available.";

  cardBody.appendChild(img);
  cardBody.appendChild(missing);

  card.appendChild(cardHeader);
  card.appendChild(cardBody);

  return { card, stepLabelEl, img, missing };
}

// Initialize Bootstrap tooltips for image cards
function initializeImageCardTooltips(container) {
  const root = container || document;
  const tooltipTriggerList = Array.from(root.querySelectorAll('.image-preview-card-info[data-bs-toggle="tooltip"]'));
  tooltipTriggerList.forEach((el) => {
    if (window.bootstrap && !el._tooltipInitialized) {
      new bootstrap.Tooltip(el);
      el._tooltipInitialized = true;
    }
  });
}
