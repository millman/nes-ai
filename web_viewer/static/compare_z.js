// Compare Z page configuration
// This file configures the common comparison logic for Z-related visualizations

const COMPARE_CONFIG = {
  defaultImageFolder: "pca_z",
  apiEndpoint: "/compare_z/data",
  imageFolderOptions: [
    {
      value: "pca_z",
      label: "Diagnostics:PCA (Z)",
      prefix: "pca_z_",
      folder: "pca_z",
      legacyPrefix: "embeddings_",
      legacyFolder: "embeddings",
    },
    { value: "vis_self_distance_z", label: "Self-distance:Distance (Z)", prefix: "self_distance_z_", folder: "vis_self_distance_z" },
    { value: "vis_delta_z_pca", label: "Diagnostics:Delta-z PCA", prefix: "delta_z_pca_", folder: "vis_delta_z_pca" },
    { value: "vis_odometry_current_z", label: "Odometry:Cumulative sum of Δz PCA/ICA/t-SNE", prefix: "odometry_z_", folder: "vis_odometry" },
    { value: "vis_odometry_z_vs_z_hat", label: "Odometry:||z - z_hat|| + scatter", prefix: "z_vs_z_hat_", folder: "vis_odometry" },
    {
      value: "vis_action_alignment_detail_z",
      label: "Diagnostics:Action alignment of PCA (Z)",
      prefix: "action_alignment_detail_",
      folder: "vis_action_alignment_z",
    },
    {
      value: "vis_action_alignment_detail_raw_z",
      label: "Diagnostics:Action alignment of raw delta (Z)",
      prefix: "action_alignment_detail_",
      folder: "vis_action_alignment_z_raw",
    },
    {
      value: "vis_action_alignment_detail_centered_z",
      label: "Diagnostics:Action alignment of centered delta (Z)",
      prefix: "action_alignment_detail_",
      folder: "vis_action_alignment_z_centered",
    },
    { value: "vis_ctrl_smoothness_z", label: "Vis v Ctrl:Local smoothness (Z)", prefix: "smoothness_z_", folder: "vis_vis_ctrl" },
    {
      value: "vis_ctrl_composition_z",
      label: "Vis v Ctrl:Two-step composition error (Z)",
      prefix: "composition_error_z_",
      folder: "vis_vis_ctrl",
    },
    { value: "vis_ctrl_stability_z", label: "Vis v Ctrl:Neighborhood stability (Z)", prefix: "stability_z_", folder: "vis_vis_ctrl" },
    { value: "vis_cycle_error", label: "Diagnostics:Cycle error (Z)", prefix: "cycle_error_", folder: "vis_cycle_error_z" },
    { value: "vis_graph_rank1_cdf_z", label: "Graph Diagnostics:Rank-1 CDF (Z)", prefix: "rank1_cdf_", folder: "graph_diagnostics_z" },
    { value: "vis_graph_rank2_cdf_z", label: "Graph Diagnostics:Rank-2 CDF (Z)", prefix: "rank2_cdf_", folder: "graph_diagnostics_z" },
    { value: "vis_graph_neff_violin_z", label: "Graph Diagnostics:Neighborhood size (Z)", prefix: "neff_violin_", folder: "graph_diagnostics_z" },
    { value: "vis_graph_in_degree_hist_z", label: "Graph Diagnostics:In-degree (Z)", prefix: "in_degree_hist_", folder: "graph_diagnostics_z" },
    { value: "vis_graph_edge_consistency_z", label: "Graph Diagnostics:Edge consistency (Z)", prefix: "edge_consistency_", folder: "graph_diagnostics_z" },
    { value: "vis_graph_metrics_history_z", label: "Graph Diagnostics:Metrics history (Z)", prefix: "metrics_history_", folder: "graph_diagnostics_z" },
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

// Initialize the comparison page with Z configuration
initializeComparePage(COMPARE_CONFIG);
