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

    The figure includes step-based, cumulative_flops-based, and elapsed time x values,
    allowing the UI to toggle between them via JavaScript.
    """
    if not curves:
        return None
    fig = go.Figure()
    for exp_name, curve in curves.items():
        exp_id = trace_ids.get(exp_name) if trace_ids else None
        customdata_base = [exp_id] * len(curve.steps) if exp_id is not None else None
        elapsed_axis = curve.elapsed_seconds if curve.elapsed_seconds else curve.steps
        for metric_name, values in curve.series.items():
            # Store both x options in customdata for toggle functionality
            # customdata[i] = [exp_id, cumulative_flops[i], elapsed_axis[i]]
            if customdata_base is not None:
                customdata = [
                    [exp_id, flops, elapsed]
                    for flops, elapsed in zip(curve.cumulative_flops, elapsed_axis)
                ]
            else:
                customdata = [
                    [None, flops, elapsed]
                    for flops, elapsed in zip(curve.cumulative_flops, elapsed_axis)
                ]
            fig.add_trace(
                go.Scatter(
                    x=curve.steps,
                    y=values,
                    mode="lines",
                    name=f"{exp_name}: {metric_name}" if include_experiment_in_trace else metric_name,
                    customdata=customdata,
                    hovertemplate="%{y}<extra></extra>",
                    meta={
                        "cumulative_flops": curve.cumulative_flops,
                        "elapsed_seconds": elapsed_axis,
                        "has_elapsed_seconds": bool(curve.elapsed_seconds),
                    },
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


def build_ranking_accuracy_plot(curves: LossCurveData) -> Optional[Dict]:
    if not curves.series:
        return None
    accuracy = curves.series.get("geometry_rank_p_accuracy")
    loss = curves.series.get("loss_geometry_rank_p")
    if not accuracy and not loss:
        return None
    fig = go.Figure()
    if accuracy:
        fig.add_trace(
            go.Scatter(
                x=curves.steps,
                y=accuracy,
                mode="lines",
                name="geometry_rank_p_accuracy",
                hovertemplate="%{y:.3f}<extra></extra>",
            )
        )
    if loss:
        fig.add_trace(
            go.Scatter(
                x=curves.steps,
                y=loss,
                mode="lines",
                name="loss_geometry_rank_p",
                yaxis="y2",
                hovertemplate="%{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(title="Step"),
        yaxis=dict(title="Ranking accuracy", range=[0, 1]),
        yaxis2=dict(title="Ranking loss", overlaying="y", side="right", type="log"),
        margin=dict(t=40, b=40, l=60, r=60),
        hovermode="x unified",
        legend_title_text="Metric",
    )
    return fig.to_dict()
