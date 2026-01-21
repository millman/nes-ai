from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

from jepa_world_model.diagnostics_prepare import DiagnosticsBatchState
from jepa_world_model.plots.build_motion_subspace import MotionSubspace
from jepa_world_model.plots.plot_action_alignment_debug import build_action_alignment_debug
from jepa_world_model.plots.plot_action_alignment_stats import compute_action_alignment_stats
from jepa_world_model.plots.plot_variance_report import write_variance_report
from jepa_world_model.plots.plot_variance_spectrum import save_variance_spectrum_plot
from jepa_world_model.plots.plot_delta_pca import save_delta_pca_plot
from jepa_world_model.plots.write_action_alignment_crosscheck import write_action_alignment_crosscheck
from jepa_world_model.plots.write_action_alignment_csv import write_action_alignment_csv
from jepa_world_model.plots.write_action_alignment_full_csv import write_action_alignment_full_csv
from jepa_world_model.plots.write_action_alignment_overview_txt import write_action_alignment_overview_txt
from jepa_world_model.plots.write_action_alignment_pairwise_csv import write_action_alignment_pairwise_csv
from jepa_world_model.plots.write_action_alignment_report import write_action_alignment_report
from jepa_world_model.plots.write_action_alignment_strength import write_action_alignment_strength
from jepa_world_model.plots.write_cycle_error_summary_csv import write_cycle_error_summary_csv
from jepa_world_model.plots.write_cycle_error_values_csv import write_cycle_error_values_csv
from jepa_world_model.plots.write_delta_samples_csv import write_delta_samples_csv
from jepa_world_model.plots.write_delta_variance_csv import write_delta_variance_csv
from jepa_world_model.vis_action_alignment import save_action_alignment_detail_plot
from jepa_world_model.vis_cycle_error import compute_cycle_errors, save_cycle_error_plot


def write_motion_pca_artifacts(
    *,
    diagnostics_cfg,
    global_step: int,
    name: str,
    motion: MotionSubspace,
    delta_dir: Path,
) -> Tuple[
    Dict[str, Dict[int, Dict[str, float]]],
    Dict[str, Dict[int, Dict[str, float]]],
]:
    save_delta_pca_plot(
        delta_dir / f"delta_{name}_pca_{global_step:07d}.png",
        motion.variance_ratio,
        motion.delta_proj,
        motion.proj_flat,
        motion.action_ids,
        motion.action_dim,
        name,
    )
    save_variance_spectrum_plot(
        motion.variance_ratio,
        delta_dir,
        global_step,
        name,
    )
    write_variance_report(
        motion.variance_ratio,
        delta_dir,
        global_step,
        name,
    )
    stats = compute_action_alignment_stats(
        motion.delta_proj,
        motion.action_ids,
        diagnostics_cfg.min_action_count,
        diagnostics_cfg.cosine_high_threshold,
    )
    motion_raw = replace(motion, delta_proj=motion.delta_embed)
    stats_raw = compute_action_alignment_stats(
        motion_raw.delta_proj,
        motion_raw.action_ids,
        diagnostics_cfg.min_action_count,
        diagnostics_cfg.cosine_high_threshold,
    )
    motion_centered = replace(motion, delta_proj=motion.delta_centered)
    stats_centered = compute_action_alignment_stats(
        motion_centered.delta_proj,
        motion_centered.action_ids,
        diagnostics_cfg.min_action_count,
        diagnostics_cfg.cosine_high_threshold,
    )
    write_delta_variance_csv(
        delta_dir,
        global_step,
        motion.variance_ratio,
        name,
    )
    write_delta_samples_csv(
        delta_dir,
        global_step,
        motion.paths,
        name,
    )
    return {"pca": stats, "raw": stats_raw, "centered": stats_centered}


def write_alignment_artifacts(
    *,
    diagnostics_cfg,
    global_step: int,
    name: str,
    motion: MotionSubspace,
    inverse_map: Dict[int, Dict[int, int]],
    alignment_dir: Path,
    alignment_raw_dir: Path,
    alignment_centered_dir: Path,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    stats = compute_action_alignment_stats(
        motion.delta_proj,
        motion.action_ids,
        diagnostics_cfg.min_action_count,
        diagnostics_cfg.cosine_high_threshold,
    )
    debug = build_action_alignment_debug(
        stats,
        motion.delta_proj,
        motion.action_ids,
    )
    save_action_alignment_detail_plot(
        alignment_dir / f"action_alignment_detail_{global_step:07d}.png",
        debug,
        diagnostics_cfg.cosine_high_threshold,
        motion.action_dim,
        alignment_label="PCA",
    )
    write_action_alignment_report(
        stats,
        motion.action_dim,
        inverse_map,
        alignment_dir,
        global_step,
    )
    write_action_alignment_strength(
        stats,
        motion.action_dim,
        alignment_dir,
        global_step,
    )
    write_action_alignment_crosscheck(
        stats,
        motion,
        alignment_dir,
        global_step,
    )
    motion_raw = replace(motion, delta_proj=motion.delta_embed)
    stats_raw = compute_action_alignment_stats(
        motion_raw.delta_proj,
        motion_raw.action_ids,
        diagnostics_cfg.min_action_count,
        diagnostics_cfg.cosine_high_threshold,
    )
    debug_raw = build_action_alignment_debug(
        stats_raw,
        motion_raw.delta_proj,
        motion_raw.action_ids,
    )
    save_action_alignment_detail_plot(
        alignment_raw_dir / f"action_alignment_detail_{global_step:07d}.png",
        debug_raw,
        diagnostics_cfg.cosine_high_threshold,
        motion_raw.action_dim,
        alignment_label="raw delta",
    )
    write_action_alignment_crosscheck(
        stats_raw,
        motion_raw,
        alignment_raw_dir,
        global_step,
    )
    motion_centered = replace(motion, delta_proj=motion.delta_centered)
    stats_centered = compute_action_alignment_stats(
        motion_centered.delta_proj,
        motion_centered.action_ids,
        diagnostics_cfg.min_action_count,
        diagnostics_cfg.cosine_high_threshold,
    )
    debug_centered = build_action_alignment_debug(
        stats_centered,
        motion_centered.delta_proj,
        motion_centered.action_ids,
    )
    save_action_alignment_detail_plot(
        alignment_centered_dir / f"action_alignment_detail_{global_step:07d}.png",
        debug_centered,
        diagnostics_cfg.cosine_high_threshold,
        motion_centered.action_dim,
        alignment_label="centered delta",
    )
    write_action_alignment_crosscheck(
        stats_centered,
        motion_centered,
        alignment_centered_dir,
        global_step,
    )
    for dir_entry, stats_entry, debug_entry in [
        (alignment_dir, stats, debug),
        (alignment_raw_dir, stats_raw, debug_raw),
        (alignment_centered_dir, stats_centered, debug_centered),
    ]:
        write_action_alignment_csv(
            dir_entry,
            global_step,
            motion.action_dim,
            stats_entry,
        )
        write_action_alignment_full_csv(
            dir_entry,
            global_step,
            motion.action_dim,
            stats_entry,
        )
        write_action_alignment_pairwise_csv(
            dir_entry,
            global_step,
            motion.action_dim,
            debug_entry,
        )
        write_action_alignment_overview_txt(
            dir_entry,
            global_step,
            diagnostics_cfg.cosine_high_threshold,
            debug_entry,
        )
    return {"pca": stats, "raw": stats_raw, "centered": stats_centered}


def write_cycle_error_artifacts(
    *,
    diagnostics_cfg,
    global_step: int,
    name: str,
    motion: MotionSubspace,
    inverse_map: Dict[int, Dict[int, int]],
    cycle_dir: Path,
) -> Tuple[List[Tuple[int, float]], Dict[int, float]]:
    cycle_errors, cycle_per_action = compute_cycle_errors(
        motion.proj_sequences,
        motion.actions_seq,
        inverse_map,
        include_synthetic=diagnostics_cfg.synthesize_cycle_samples,
    )
    save_cycle_error_plot(
        cycle_dir / f"cycle_error_{global_step:07d}.png",
        [e[1] for e in cycle_errors],
        cycle_per_action,
        motion.action_dim,
    )
    write_cycle_error_values_csv(
        cycle_dir,
        global_step,
        motion.action_dim,
        cycle_errors,
    )
    write_cycle_error_summary_csv(
        cycle_dir,
        global_step,
        motion.action_dim,
        cycle_per_action,
    )
    return cycle_errors, cycle_per_action

