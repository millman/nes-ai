// Common definitions and utilities for image preview cards
// Used across Detail, Compare, and Diagnostics pages

const IMAGE_FOLDER_OPTIONS = [
  { value: "vis_fixed_0", label: "Rollouts:Fixed 0", prefix: "rollout_", folder: "vis_fixed_0" },
  { value: "vis_fixed_1", label: "Rollouts:Fixed 1", prefix: "rollout_", folder: "vis_fixed_1" },
  { value: "vis_rolling_0", label: "Rollouts:Rolling 0", prefix: "rollout_", folder: "vis_rolling_0" },
  { value: "vis_rolling_1", label: "Rollouts:Rolling 1", prefix: "rollout_", folder: "vis_rolling_1" },
  { value: "pca_z", label: "Diagnostics:PCA (Z)", prefix: "pca_z_", folder: "pca_z", legacyPrefix: "embeddings_", legacyFolder: "embeddings" },
  { value: "pca_s", label: "Diagnostics:PCA (S)", prefix: "pca_s_", folder: "pca_s" },
  { value: "pca_h", label: "Diagnostics:PCA (H)", prefix: "pca_h_", folder: "pca_h" },
  { value: "samples_hard", label: "Samples:Hard", prefix: "hard_", folder: "samples_hard" },
  { value: "vis_self_distance_z", label: "Self-distance:Distance (Z)", prefix: "self_distance_z_", folder: "vis_self_distance_z" },
  { value: "vis_self_distance_s", label: "Self-distance:Distance (S)", prefix: "self_distance_s_", folder: "vis_self_distance_s" },
  { value: "vis_self_distance_h", label: "Self-distance:Distance (H)", prefix: "self_distance_h_", folder: "vis_self_distance_h" },
  { value: "vis_delta_z_pca", label: "Diagnostics:Delta-z PCA", prefix: "delta_z_pca_", folder: "vis_delta_z_pca" },
  { value: "vis_delta_s_pca", label: "Diagnostics:Delta-s PCA", prefix: "delta_s_pca_", folder: "vis_delta_s_pca" },
  { value: "vis_delta_h_pca", label: "Diagnostics:Delta-h PCA", prefix: "delta_h_pca_", folder: "vis_delta_h_pca" },
  { value: "vis_odometry_current_z", label: "Odometry:Cumulative sum of Δz PCA/ICA/t-SNE", prefix: "odometry_z_", folder: "vis_odometry" },
  { value: "vis_odometry_current_s", label: "Odometry:Cumulative sum of Δs PCA/ICA/t-SNE", prefix: "odometry_s_", folder: "vis_odometry" },
  { value: "vis_odometry_current_h", label: "Odometry:Cumulative sum of Δh PCA/ICA/t-SNE", prefix: "odometry_h_", folder: "vis_odometry" },
  { value: "vis_odometry_z_vs_z_hat", label: "Odometry:||z - z_hat|| + scatter", prefix: "z_vs_z_hat_", folder: "vis_odometry" },
  { value: "vis_odometry_s_vs_s_hat", label: "Odometry:||s - s_hat|| + scatter", prefix: "s_vs_s_hat_", folder: "vis_odometry" },
  { value: "vis_odometry_h_vs_h_hat", label: "Odometry:||h - h_hat|| + scatter", prefix: "h_vs_h_hat_", folder: "vis_odometry" },
  { value: "vis_action_alignment_detail_z", label: "Diagnostics:Action alignment of PCA (Z)", prefix: "action_alignment_detail_", folder: "vis_action_alignment_z" },
  { value: "vis_action_alignment_detail_raw_z", label: "Diagnostics:Action alignment of raw delta (Z)", prefix: "action_alignment_detail_", folder: "vis_action_alignment_z_raw" },
  { value: "vis_action_alignment_detail_centered_z", label: "Diagnostics:Action alignment of centered delta (Z)", prefix: "action_alignment_detail_", folder: "vis_action_alignment_z_centered" },
  { value: "vis_action_alignment_detail_s", label: "Diagnostics:Action alignment of PCA (S)", prefix: "action_alignment_detail_", folder: "vis_action_alignment_s" },
  { value: "vis_action_alignment_detail_raw_s", label: "Diagnostics:Action alignment of raw delta (S)", prefix: "action_alignment_detail_", folder: "vis_action_alignment_s_raw" },
  { value: "vis_action_alignment_detail_centered_s", label: "Diagnostics:Action alignment of centered delta (S)", prefix: "action_alignment_detail_", folder: "vis_action_alignment_s_centered" },
  { value: "vis_action_alignment_detail_h", label: "Diagnostics:Action alignment of PCA (H)", prefix: "action_alignment_detail_", folder: "vis_action_alignment_h" },
  { value: "vis_action_alignment_detail_raw_h", label: "Diagnostics:Action alignment of raw delta (H)", prefix: "action_alignment_detail_", folder: "vis_action_alignment_h_raw" },
  { value: "vis_action_alignment_detail_centered_h", label: "Diagnostics:Action alignment of centered delta (H)", prefix: "action_alignment_detail_", folder: "vis_action_alignment_h_centered" },
  { value: "vis_ctrl_smoothness_z", label: "Vis v Ctrl:Local smoothness (Z)", prefix: "smoothness_z_", folder: "vis_vis_ctrl" },
  { value: "vis_ctrl_smoothness_s", label: "Vis v Ctrl:Local smoothness (S)", prefix: "smoothness_s_", folder: "vis_vis_ctrl" },
  { value: "vis_ctrl_smoothness_h", label: "Vis v Ctrl:Local smoothness (H)", prefix: "smoothness_h_", folder: "vis_vis_ctrl" },
  { value: "vis_ctrl_composition_z", label: "Vis v Ctrl:Two-step composition error (Z)", prefix: "composition_error_z_", folder: "vis_vis_ctrl" },
  { value: "vis_ctrl_composition_s", label: "Vis v Ctrl:Two-step composition error (S)", prefix: "composition_error_s_", folder: "vis_vis_ctrl" },
  { value: "vis_ctrl_composition_h", label: "Vis v Ctrl:Two-step composition error (H)", prefix: "composition_error_h_", folder: "vis_vis_ctrl" },
  { value: "vis_composability_z", label: "Composability:Two-step (Z)", prefix: "composability_z_", folder: "vis_composability_z" },
  { value: "vis_composability_s", label: "Composability:Two-step (S)", prefix: "composability_s_", folder: "vis_composability_s" },
  { value: "vis_composability_h", label: "Composability:Two-step (H)", prefix: "composability_h_", folder: "vis_composability_h" },
  { value: "vis_off_manifold", label: "Diagnostics:Off-manifold (rollout)", prefix: "off_manifold_", folder: "vis_off_manifold" },
  { value: "vis_ctrl_stability_z", label: "Vis v Ctrl:Neighborhood stability (Z)", prefix: "stability_z_", folder: "vis_vis_ctrl" },
  { value: "vis_ctrl_stability_s", label: "Vis v Ctrl:Neighborhood stability (S)", prefix: "stability_s_", folder: "vis_vis_ctrl" },
  { value: "vis_ctrl_stability_h", label: "Vis v Ctrl:Neighborhood stability (H)", prefix: "stability_h_", folder: "vis_vis_ctrl" },
  { value: "vis_cycle_error", label: "Diagnostics:Cycle error (Z)", prefix: "cycle_error_", folder: "vis_cycle_error_z" },
  { value: "vis_cycle_error_s", label: "Diagnostics:Cycle error (S)", prefix: "cycle_error_", folder: "vis_cycle_error_s" },
  { value: "vis_cycle_error_h", label: "Diagnostics:Cycle error (H)", prefix: "cycle_error_", folder: "vis_cycle_error_h" },
  { value: "vis_rollout_divergence", label: "Diagnostics:Rollout divergence", prefix: "rollout_divergence_", folder: "vis_rollout_divergence" },
  { value: "vis_z_consistency", label: "Diagnostics:Z consistency", prefix: "z_consistency_", folder: "vis_z_consistency" },
  { value: "vis_z_monotonicity", label: "Diagnostics:Z monotonicity", prefix: "z_monotonicity_", folder: "vis_z_monotonicity" },
  { value: "vis_path_independence", label: "Diagnostics:Path independence", prefix: "path_independence_", folder: "vis_path_independence" },
  { value: "vis_straightline_s", label: "Diagnostics:Straight-line S", prefix: "straightline_s_", folder: "vis_straightline_s" },
  { value: "vis_h_ablation", label: "Diagnostics:H ablation divergence", prefix: "h_ablation_", folder: "vis_h_ablation" },
  { value: "vis_h_drift_by_action", label: "Diagnostics:H drift by action", prefix: "h_drift_by_action_", folder: "vis_h_drift_by_action" },
  { value: "vis_norm_timeseries", label: "Diagnostics:Norm stability", prefix: "norm_timeseries_", folder: "vis_norm_timeseries" },
  { value: "vis_graph_rank1_cdf_z", label: "Graph Diagnostics:Rank-1 CDF (Z)", prefix: "rank1_cdf_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_rank2_cdf_z", label: "Graph Diagnostics:Rank-2 CDF (Z)", prefix: "rank2_cdf_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_neff_violin_z", label: "Graph Diagnostics:Neighborhood size (Z)", prefix: "neff_violin_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_in_degree_hist_z", label: "Graph Diagnostics:In-degree (Z)", prefix: "in_degree_hist_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_edge_consistency_z", label: "Graph Diagnostics:Edge consistency (Z)", prefix: "edge_consistency_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_metrics_history_z", label: "Graph Diagnostics:Metrics history (Z)", prefix: "metrics_history_", folder: "graph_diagnostics_z" },
  { value: "vis_graph_rank1_cdf_h", label: "Graph Diagnostics:Rank-1 CDF (H)", prefix: "rank1_cdf_", folder: "graph_diagnostics_h" },
  { value: "vis_graph_rank2_cdf_h", label: "Graph Diagnostics:Rank-2 CDF (H)", prefix: "rank2_cdf_", folder: "graph_diagnostics_h" },
  { value: "vis_graph_neff_violin_h", label: "Graph Diagnostics:Neighborhood size (H)", prefix: "neff_violin_", folder: "graph_diagnostics_h" },
  { value: "vis_graph_in_degree_hist_h", label: "Graph Diagnostics:In-degree (H)", prefix: "in_degree_hist_", folder: "graph_diagnostics_h" },
  { value: "vis_graph_edge_consistency_h", label: "Graph Diagnostics:Edge consistency (H)", prefix: "edge_consistency_", folder: "graph_diagnostics_h" },
  { value: "vis_graph_metrics_history_h", label: "Graph Diagnostics:Metrics history (H)", prefix: "metrics_history_", folder: "graph_diagnostics_h" },
  { value: "vis_graph_rank1_cdf_s", label: "Graph Diagnostics:Rank-1 CDF (S)", prefix: "rank1_cdf_", folder: "graph_diagnostics_s" },
  { value: "vis_graph_rank2_cdf_s", label: "Graph Diagnostics:Rank-2 CDF (S)", prefix: "rank2_cdf_", folder: "graph_diagnostics_s" },
  { value: "vis_graph_neff_violin_s", label: "Graph Diagnostics:Neighborhood size (S)", prefix: "neff_violin_", folder: "graph_diagnostics_s" },
  { value: "vis_graph_in_degree_hist_s", label: "Graph Diagnostics:In-degree (S)", prefix: "in_degree_hist_", folder: "graph_diagnostics_s" },
  { value: "vis_graph_edge_consistency_s", label: "Graph Diagnostics:Edge consistency (S)", prefix: "edge_consistency_", folder: "graph_diagnostics_s" },
  { value: "vis_graph_metrics_history_s", label: "Graph Diagnostics:Metrics history (S)", prefix: "metrics_history_", folder: "graph_diagnostics_s" },
].sort((a, b) => {
  const groupA = a.label.split(":")[0].trim();
  const groupB = b.label.split(":")[0].trim();
  const groupCompare = groupA.localeCompare(groupB);
  if (groupCompare !== 0) return groupCompare;
  const labelCompare = a.label.localeCompare(b.label);
  if (labelCompare !== 0) return labelCompare;
  return a.value.localeCompare(b.value);
});

// Descriptions for tooltips - explains how to read each plot and what "good" looks like
const IMAGE_FOLDER_DESCRIPTIONS = {
  vis_fixed_0: "Rollout visualization from fixed initial state 0. Read: observe agent behavior over time from a consistent starting position.",
  vis_fixed_1: "Rollout visualization from fixed initial state 1. Read: observe agent behavior over time from a second consistent starting position.",
  vis_rolling_0: "Rollout visualization from rolling initial state 0. Read: observe agent behavior from dynamically sampled starting positions.",
  vis_rolling_1: "Rollout visualization from rolling initial state 1. Read: observe agent behavior from a second set of dynamically sampled positions.",
  pca_z: "PCA of encoder latent (Z). Read: clusters indicate distinct states; spread shows representation diversity. Good: well-separated clusters with smooth transitions.",
  pca_s: "PCA of world model state (S). Read: clusters indicate distinct internal states; structure shows learned dynamics. Good: organized clusters reflecting game states.",
  pca_h: "PCA of GRU hidden state (H). Read: clusters indicate memory states; trajectories show temporal patterns. Good: smooth trajectories with clear structure.",
  samples_hard: "Hard samples from the dataset. Read: examples that the model finds challenging. Useful for diagnosing model weaknesses.",
  vis_self_distance_z: "Self-distance matrix for Z embeddings. Read: bright diagonals indicate temporal coherence; off-diagonal blocks show recurring states. Good: clear diagonal with periodic structure.",
  vis_self_distance_s: "Self-distance matrix for S embeddings. Read: bright diagonals indicate temporal coherence; off-diagonal blocks show recurring states. Good: clear diagonal with periodic structure.",
  vis_self_distance_h: "Self-distance matrix for H embeddings. Read: bright diagonals indicate temporal coherence; off-diagonal blocks show recurring states. Good: clear diagonal with periodic structure.",
  vis_delta_z_pca: "PCA of delta-Z steps across the batch. Read: spokes or clusters correspond to action directions; separation shows distinct motions. Good: clear, separated clusters with minimal overlap.",
  vis_delta_s_pca: "PCA of delta-S steps across the batch. Read: spokes or clusters correspond to action directions; separation shows distinct motions. Good: clear, separated clusters with minimal overlap.",
  vis_delta_h_pca: "PCA of delta-H steps across the batch. Read: spokes or clusters correspond to action directions; separation shows distinct motions. Good: clear, separated clusters with minimal overlap.",
  vis_odometry_current_z: "Cumulative sum of Δz using PCA/ICA/t-SNE. Read: trajectory shows integrated motion; coherent paths indicate consistent action effects. Good: smooth, interpretable trajectories.",
  vis_odometry_current_s: "Cumulative sum of Δs using PCA/ICA/t-SNE. Read: trajectory shows integrated motion; coherent paths indicate consistent action effects. Good: smooth, interpretable trajectories.",
  vis_odometry_current_h: "Cumulative sum of Δh using PCA/ICA/t-SNE. Read: trajectory shows integrated motion; coherent paths indicate consistent action effects. Good: smooth, interpretable trajectories.",
  vis_odometry_z_vs_z_hat: "||z - z_hat|| comparison with scatter plot. Read: low error means accurate predictions; scatter shows error distribution. Good: tight clustering near origin.",
  vis_odometry_s_vs_s_hat: "||s - s_hat|| comparison with scatter plot. Read: low error means accurate predictions; scatter shows error distribution. Good: tight clustering near origin.",
  vis_odometry_h_vs_h_hat: "||h - h_hat|| comparison with scatter plot. Read: low error means accurate predictions; scatter shows error distribution. Good: tight clustering near origin.",
  vis_action_alignment_detail_z: "Per-action cosine alignment of delta-Z directions. Read: values near 1 mean consistent action effects; near 0 means noisy alignment. Good: high alignment for common actions.",
  vis_action_alignment_detail_raw_z: "Per-action alignment of raw delta-Z (unprocessed). Read: raw action effects before any normalization. Useful for debugging.",
  vis_action_alignment_detail_centered_z: "Per-action alignment of centered delta-Z. Read: action effects after mean-centering. Good: clearer action-specific patterns.",
  vis_action_alignment_detail_s: "Per-action cosine alignment of delta-S directions. Read: values near 1 mean consistent action effects; near 0 means noisy alignment. Good: high alignment for common actions.",
  vis_action_alignment_detail_raw_s: "Per-action alignment of raw delta-S (unprocessed). Read: raw action effects before any normalization. Useful for debugging.",
  vis_action_alignment_detail_centered_s: "Per-action alignment of centered delta-S. Read: action effects after mean-centering. Good: clearer action-specific patterns.",
  vis_action_alignment_detail_h: "Per-action cosine alignment of delta-H directions. Read: values near 1 mean consistent action effects; near 0 means noisy alignment. Good: high alignment for common actions.",
  vis_action_alignment_detail_raw_h: "Per-action alignment of raw delta-H (unprocessed). Read: raw action effects before any normalization. Useful for debugging.",
  vis_action_alignment_detail_centered_h: "Per-action alignment of centered delta-H. Read: action effects after mean-centering. Good: clearer action-specific patterns.",
  vis_ctrl_smoothness_z: "Local smoothness of Z representations. Read: measures how smoothly representations change. Good: low smoothness error indicates continuous representations.",
  vis_ctrl_smoothness_s: "Local smoothness of S representations. Read: measures how smoothly representations change. Good: low smoothness error indicates continuous representations.",
  vis_ctrl_smoothness_h: "Local smoothness of H representations. Read: measures how smoothly representations change. Good: low smoothness error indicates continuous representations.",
  vis_ctrl_composition_z: "Two-step composition error for Z. Read: measures if two consecutive actions compose correctly. Good: low error means action effects are additive.",
  vis_ctrl_composition_s: "Two-step composition error for S. Read: measures if two consecutive actions compose correctly. Good: low error means action effects are additive.",
  vis_ctrl_composition_h: "Two-step composition error for H. Read: measures if two consecutive actions compose correctly. Good: low error means action effects are additive.",
  vis_composability_z: "Two-step composability for Z. Read: compares model rollout vs additive deltas and action-pair strip errors. Good: low errors across timesteps and pairs.",
  vis_composability_s: "Two-step composability for S. Read: compares model rollout vs additive deltas and action-pair strip errors. Good: low errors across timesteps and pairs.",
  vis_composability_h: "Two-step composability for H (actual rollout projected to Z). Read: compares model rollout vs additive deltas and action-pair strip errors. Good: low errors across timesteps and pairs.",
  vis_off_manifold: "Off-manifold diagnostic for rollouts. Read: enc(dec(z_roll)) drift by rollout index and histogram. Good: low errors clustered near zero.",
  vis_ctrl_stability_z: "Neighborhood stability for Z. Read: measures if similar states stay similar after actions. Good: high stability means predictable dynamics.",
  vis_ctrl_stability_s: "Neighborhood stability for S. Read: measures if similar states stay similar after actions. Good: high stability means predictable dynamics.",
  vis_ctrl_stability_h: "Neighborhood stability for H. Read: measures if similar states stay similar after actions. Good: high stability means predictable dynamics.",
  vis_cycle_error: "Cycle error for Z embeddings. Read: measures if opposite actions cancel out. Good: low error means reversible action effects.",
  vis_cycle_error_s: "Cycle error for S embeddings. Read: measures if opposite actions cancel out. Good: low error means reversible action effects.",
  vis_cycle_error_h: "Cycle error for H embeddings. Read: measures if opposite actions cancel out. Good: low error means reversible action effects.",
  vis_rollout_divergence: "Rollout divergence curves. Read: pixel vs latent errors over rollout horizon. Good: flat or slowly growing curves.",
  vis_z_consistency: "Same-frame consistency for Z under small noise. Read: tight histograms imply stable z. Good: narrow distance spread, high cosine.",
  vis_z_monotonicity: "Z distance vs pixel shift. Read: smooth monotonic increase indicates perceptual alignment. Good: steadily rising curve.",
  vis_path_independence: "Path independence for Z and S. Read: compare end-state distances for alternate paths. Good: low distances.",
  vis_straightline_s: "Straight-line action rays in S. Read: repeated actions trace straight paths; opposite actions mirror. Good: clean rays.",
  vis_h_ablation: "H ablation divergence. Read: compare rollout errors with normal vs zeroed H. Good: minimal gap in fully observable settings.",
  vis_h_drift_by_action: "H drift by action. Read: mean ||h_{t+1}-h_t|| grouped by action. Good: low, consistent drift.",
  vis_norm_timeseries: "Norm stability for Z/H/S. Read: mean and p95 norms over steps. Good: stable, bounded lines.",
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
  vis_graph_rank1_cdf_s: "Rank-1 CDF for S graph. Read: distribution of top-1 neighbor accuracy. Good: high values at low thresholds.",
  vis_graph_rank2_cdf_s: "Rank-2 CDF for S graph. Read: distribution of top-2 neighbor accuracy. Good: high values at low thresholds.",
  vis_graph_neff_violin_s: "Neighborhood size (N_eff) for S. Read: violin plot of effective neighbor counts. Good: consistent, moderate neighborhood sizes.",
  vis_graph_in_degree_hist_s: "In-degree histogram for S graph. Read: distribution of incoming edges per node. Good: balanced distribution without outliers.",
  vis_graph_edge_consistency_s: "Edge consistency for S graph. Read: measures if similar nodes share edges. Good: high consistency indicates reliable graph structure.",
  vis_graph_metrics_history_s: "Graph metrics history for S. Read: how graph quality metrics evolve over training. Good: improving or stable metrics.",
};

// Get option by folder value
function getImageOption(folderValue) {
  return IMAGE_FOLDER_OPTIONS.find((opt) => opt.value === folderValue);
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
