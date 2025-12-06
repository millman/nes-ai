from __future__ import annotations

from typing import Dict, Optional

import plotly.graph_objects as go

from .experiments import LossCurveData


def build_overlay(curves: Dict[str, LossCurveData], include_experiment_in_trace: bool = True) -> Optional[Dict]:
    if not curves:
        return None
    fig = go.Figure()
    for exp_name, curve in curves.items():
        for metric_name, values in curve.series.items():
            fig.add_trace(
                go.Scatter(
                    x=curve.steps,
                    y=values,
                    mode="lines",
                    name=f"{exp_name}: {metric_name}" if include_experiment_in_trace else metric_name,
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
