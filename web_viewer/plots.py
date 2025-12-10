from __future__ import annotations

from typing import Dict, Optional

import plotly.graph_objects as go

from .experiments import LossCurveData


def build_overlay(
    curves: Dict[str, LossCurveData],
    include_experiment_in_trace: bool = True,
    trace_ids: Optional[Dict[str, str]] = None,
) -> Optional[Dict]:
    """Build a Plotly figure with loss curves.

    The figure includes both step-based and cumulative_flops-based x values,
    allowing the UI to toggle between them via JavaScript.
    """
    if not curves:
        return None
    fig = go.Figure()
    for exp_name, curve in curves.items():
        exp_id = trace_ids.get(exp_name) if trace_ids else None
        customdata_base = [exp_id] * len(curve.steps) if exp_id is not None else None
        for metric_name, values in curve.series.items():
            # Store both x options in customdata for toggle functionality
            # customdata[i] = [exp_id, cumulative_flops[i]]
            if customdata_base is not None:
                customdata = [[exp_id, flops] for flops in curve.cumulative_flops]
            else:
                customdata = [[None, flops] for flops in curve.cumulative_flops]
            fig.add_trace(
                go.Scatter(
                    x=curve.steps,
                    y=values,
                    mode="lines",
                    name=f"{exp_name}: {metric_name}" if include_experiment_in_trace else metric_name,
                    customdata=customdata,
                    hovertemplate="%{y}<extra></extra>",
                    meta={"cumulative_flops": curve.cumulative_flops},
                )
            )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Step"),
        yaxis=dict(
            title="Loss",
            type="log",
            dtick=1,
            exponentformat="power",
            showexponent="all",
            minor=dict(showgrid=False),
        ),
        margin=dict(t=40, b=40, l=60, r=20),
        hovermode="x unified",
        hoverlabel=dict(namelength=-1, align="left"),
        legend_title_text="Experiment:Metric",
    )
    return fig.to_dict()
