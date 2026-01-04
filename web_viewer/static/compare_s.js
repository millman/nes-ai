// Compare S page configuration
// This file configures the common comparison logic for S-related visualizations

const COMPARE_CONFIG = {
  defaultImageFolder: "pca_s",
  apiEndpoint: "/compare_s/data",
  imageFolderOptions: [
    {
      value: "pca_s",
      label: "Diagnostics:PCA (S)",
      prefix: "pca_s_",
      folder: "pca_s",
      legacyPrefix: "embeddings_",
      legacyFolder: "embeddings",
    },
    { value: "vis_self_distance_s", label: "Self-distance:Distance (S)", prefix: "self_distance_s_", folder: "vis_self_distance_s" },
    { value: "vis_delta_s_pca", label: "Diagnostics:Delta-s PCA", prefix: "delta_s_pca_", folder: "vis_delta_s_pca" },
    { value: "vis_odometry_current_s", label: "Odometry:Cumulative sum of Δs PCA/ICA/t-SNE", prefix: "odometry_s_", folder: "vis_odometry" },
    { value: "vis_odometry_s_vs_s_hat", label: "Odometry:||s - s_hat|| + scatter", prefix: "s_vs_s_hat_", folder: "vis_odometry" },
    {
      value: "vis_action_alignment_detail_s",
      label: "Diagnostics:Action alignment of PCA (S)",
      prefix: "action_alignment_detail_",
      folder: "vis_action_alignment_s",
    },
    {
      value: "vis_action_alignment_detail_raw_s",
      label: "Diagnostics:Action alignment of raw delta (S)",
      prefix: "action_alignment_detail_",
      folder: "vis_action_alignment_s_raw",
    },
    {
      value: "vis_action_alignment_detail_centered_s",
      label: "Diagnostics:Action alignment of centered delta (S)",
      prefix: "action_alignment_detail_",
      folder: "vis_action_alignment_s_centered",
    },
    { value: "vis_ctrl_smoothness_s", label: "Vis v Ctrl:Local smoothness (S)", prefix: "smoothness_s_", folder: "vis_vis_ctrl" },
    {
      value: "vis_ctrl_composition_s",
      label: "Vis v Ctrl:Two-step composition error (S)",
      prefix: "composition_error_s_",
      folder: "vis_vis_ctrl",
    },
    { value: "vis_ctrl_stability_s", label: "Vis v Ctrl:Neighborhood stability (S)", prefix: "stability_s_", folder: "vis_vis_ctrl" },
    { value: "vis_cycle_error", label: "Diagnostics:Cycle error (S)", prefix: "cycle_error_", folder: "vis_cycle_error_s" },
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
    if (groupCompare !== 0) {
      return groupCompare;
    }
    const labelCompare = a.label.localeCompare(b.label);
    if (labelCompare !== 0) {
      return labelCompare;
    }
    return a.value.localeCompare(b.value);
  }),
};

// Initialize the comparison page with S configuration
initializeComparePage(COMPARE_CONFIG);
