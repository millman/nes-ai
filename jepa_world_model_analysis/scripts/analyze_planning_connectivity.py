#!/usr/bin/env python3
"""Summarize planning-graph center/connectivity health from run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional


def _to_float(value: str) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise AssertionError(f"Expected CSV file does not exist: {path}")
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"CSV file is empty: {path}")
    return rows


def _summarize_numeric(rows: List[Dict[str, str]], column: str) -> Dict[str, float]:
    values = [_to_float(row[column]) for row in rows if column in row]
    values = [value for value in values if value is not None]
    if not values:
        raise AssertionError(f"No finite values found in column {column!r}")
    ordered = sorted(values)
    return {
        "min": float(ordered[0]),
        "median": float(median(ordered)),
        "max": float(ordered[-1]),
    }


def _first_step_at_or_below(rows: List[Dict[str, str]], column: str, threshold: float) -> Optional[int]:
    for row in rows:
        value = _to_float(row[column])
        if value is not None and value <= threshold:
            step = _to_float(row.get("step", ""))
            return int(step) if step is not None else None
    return None


def summarize_run(run_dir: Path) -> Dict[str, object]:
    metrics_dir = run_dir / "metrics"
    planning_rows = _read_csv(metrics_dir / "planning_metrics.csv")
    anchor_rows = _read_csv(metrics_dir / "planning_anchor_metrics.csv")

    last_plan = planning_rows[-1]
    last_anchor = anchor_rows[-1]

    c_add_floor_step = _first_step_at_or_below(anchor_rows, "c_add_after", 1.0)
    reach_below_0_2_step = _first_step_at_or_below(planning_rows, "reach_h_median", 0.2)

    output = {
        "run_dir": str(run_dir),
        "planning_metrics": {
            "rows": len(planning_rows),
            "num_nodes_h": _summarize_numeric(planning_rows, "num_nodes_h"),
            "reach_h_median": _summarize_numeric(planning_rows, "reach_h_median"),
            "final": {
                "step": int(float(last_plan["step"])),
                "num_nodes_h": float(last_plan["num_nodes_h"]),
                "reach_h_median": float(last_plan["reach_h_median"]),
            },
        },
        "planning_anchor_metrics": {
            "rows": len(anchor_rows),
            "anchor_h": _summarize_numeric(anchor_rows, "anchor_h"),
            "r_add": _summarize_numeric(anchor_rows, "r_add"),
            "r_edge": _summarize_numeric(anchor_rows, "r_edge"),
            "c_add_after": _summarize_numeric(anchor_rows, "c_add_after"),
            "first_c_add_floor_step": c_add_floor_step,
            "first_reach_h_median_le_0_2_step": reach_below_0_2_step,
            "final": {
                "step": int(float(last_anchor["step"])),
                "anchor_h": float(last_anchor["anchor_h"]),
                "c_add_before": float(last_anchor["c_add_before"]),
                "c_add_after": float(last_anchor["c_add_after"]),
                "c_edge": float(last_anchor["c_edge"]),
                "r_add": float(last_anchor["r_add"]),
                "r_edge": float(last_anchor["r_edge"]),
                "num_nodes_h": float(last_anchor["num_nodes_h"]),
                "reach_h_median": float(last_anchor["reach_h_median"]),
            },
        },
        "signals": {
            "controller_saturated_at_min_c_add": c_add_floor_step is not None,
            "low_reachability_final": float(last_plan["reach_h_median"]) < 0.2,
            "high_node_count_final": float(last_plan["num_nodes_h"]) > 28,
        },
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory (out.jepa_world_model_trainer/<run>)")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    summary = summarize_run(args.run_dir)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    final = summary["planning_metrics"]["final"]
    anchor_final = summary["planning_anchor_metrics"]["final"]
    print(f"run_dir: {summary['run_dir']}")
    print(
        "final: "
        f"step={final['step']} "
        f"num_nodes_h={final['num_nodes_h']:.0f} "
        f"reach_h_median={final['reach_h_median']:.4f} "
        f"anchor_h={anchor_final['anchor_h']:.5f} "
        f"r_edge={anchor_final['r_edge']:.5f} "
        f"c_add_after={anchor_final['c_add_after']:.2f}"
    )
    print(
        "events: "
        f"first_c_add_floor_step={summary['planning_anchor_metrics']['first_c_add_floor_step']} "
        f"first_reach_h_median_le_0_2_step={summary['planning_anchor_metrics']['first_reach_h_median_le_0_2_step']}"
    )


if __name__ == "__main__":
    main()
