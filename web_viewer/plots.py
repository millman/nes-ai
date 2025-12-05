from __future__ import annotations

from typing import Dict, Optional

import plotly.graph_objects as go

from .experiments import LossCurveData


def build_overlay(curves: Dict[str, LossCurveData]) -> Optional[Dict]:
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
                    name=f"{exp_name}: {metric_name}",
                )
            )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Steps",
        yaxis_title="Metric value",
        hovermode="x unified",
        legend_title_text="Experiment:Metric",
    )
    return fig.to_dict()
