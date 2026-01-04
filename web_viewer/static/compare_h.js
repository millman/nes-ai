// Compare H page configuration
// This file configures the common comparison logic for H-related visualizations

const COMPARE_CONFIG = {
  defaultImageFolder: "pca_h",
  apiEndpoint: "/compare_h/data",
  imageFolderOptions: [
    {
      value: "pca_h",
      label: "Diagnostics:PCA (H)",
      prefix: "pca_h_",
      folder: "pca_h",
      legacyPrefix: "embeddings_",
      legacyFolder: "embeddings",
    },
    { value: "vis_self_distance_h", label: "Self-distance:Distance (H)", prefix: "self_distance_h_", folder: "vis_self_distance_h" },
    { value: "vis_delta_h_pca", label: "Diagnostics:Delta-h PCA", prefix: "delta_h_pca_", folder: "vis_delta_h_pca" },
    { value: "vis_odometry_current_h", label: "Odometry:Cumulative sum of Δh PCA/ICA/t-SNE", prefix: "odometry_h_", folder: "vis_odometry" },
    { value: "vis_odometry_h_vs_h_hat", label: "Odometry:||h - h_hat|| + scatter", prefix: "h_vs_h_hat_", folder: "vis_odometry" },
    {
      value: "vis_action_alignment_detail",
      label: "Diagnostics:Action alignment (H)",
      prefix: "action_alignment_detail_",
      folder: "vis_action_alignment_h",
    },
    { value: "vis_ctrl_smoothness_h", label: "Vis v Ctrl:Local smoothness (H)", prefix: "smoothness_h_", folder: "vis_vis_ctrl" },
    {
      value: "vis_ctrl_composition_h",
      label: "Vis v Ctrl:Two-step composition error (H)",
      prefix: "composition_error_h_",
      folder: "vis_vis_ctrl",
    },
    { value: "vis_ctrl_stability_h", label: "Vis v Ctrl:Neighborhood stability (H)", prefix: "stability_h_", folder: "vis_vis_ctrl" },
    { value: "vis_cycle_error", label: "Diagnostics:Cycle error (H)", prefix: "cycle_error_", folder: "vis_cycle_error_h" },
    { value: "vis_graph_rank1_cdf_h", label: "Graph Diagnostics:Rank-1 CDF (H)", prefix: "rank1_cdf_", folder: "graph_diagnostics_h" },
    { value: "vis_graph_rank2_cdf_h", label: "Graph Diagnostics:Rank-2 CDF (H)", prefix: "rank2_cdf_", folder: "graph_diagnostics_h" },
    { value: "vis_graph_neff_violin_h", label: "Graph Diagnostics:Neighborhood size (H)", prefix: "neff_violin_", folder: "graph_diagnostics_h" },
    { value: "vis_graph_in_degree_hist_h", label: "Graph Diagnostics:In-degree (H)", prefix: "in_degree_hist_", folder: "graph_diagnostics_h" },
    { value: "vis_graph_edge_consistency_h", label: "Graph Diagnostics:Edge consistency (H)", prefix: "edge_consistency_", folder: "graph_diagnostics_h" },
    { value: "vis_graph_metrics_history_h", label: "Graph Diagnostics:Metrics history (H)", prefix: "metrics_history_", folder: "graph_diagnostics_h" },
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

// Initialize the comparison page with H configuration
initializeComparePage(COMPARE_CONFIG);
