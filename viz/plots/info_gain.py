"""Layer-wise information gain (§5.1.4 of the thesis).

Bar chart of incremental F1 added by each layer (Emb→L0, L0→L1, …,
L10→L11), averaged across the 23 emotions. Story: layer L0 absorbs
roughly 61% of the final linear separability in a single jump
(F1 mean L0 ≈ 0.349, F1 mean L11 ≈ 0.569). After that, each layer
contributes only marginal gains. L8 shows a small second peak —
contextual disambiguation.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import style as st


CSV_PATH = (pathlib.Path(__file__).resolve().parents[2]
            / "results" / "csvs" / "notebook4" / "probe_results.csv")


LAYERS = ["Emb", "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7",
          "L8", "L9", "L10", "L11"]


LANG = {
    "es": {
        "title":         "Ganancia informativa por capa (probing lineal)",
        "y_left":        "Δ F1 macro vs capa anterior",
        "y_right":       "F1 macro acumulado",
        "xlabel":        "Capa",
        "annotation_l0": "<b>L0 absorbe el 61 %</b><br>de la separabilidad final",
        "annotation_l8": "L8: rebote por<br>desambiguación contextual",
        "trace_delta":   "Ganancia incremental",
        "trace_cum":     "F1 macro acumulado",
    },
    "en": {
        "title":         "Layer-wise information gain (linear probing)",
        "y_left":        "Δ F1 macro vs previous layer",
        "y_right":       "Cumulative F1 macro",
        "xlabel":        "Layer",
        "annotation_l0": "<b>L0 absorbs 61 %</b><br>of final separability",
        "annotation_l8": "L8: rebound from<br>contextual disambiguation",
        "trace_delta":   "Incremental gain",
        "trace_cum":     "Cumulative F1 macro",
    },
}


def build_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    df = pd.read_csv(CSV_PATH)

    # Mean F1 across emotions per layer
    f1_per_layer = [df[col].mean() for col in LAYERS]
    deltas = [f1_per_layer[0]] + [
        f1_per_layer[i] - f1_per_layer[i - 1]
        for i in range(1, len(LAYERS))
    ]

    # Color: highlight L0 (biggest jump) and L8 (second peak)
    colors = []
    for i, lbl in enumerate(LAYERS):
        if lbl == "L0":
            colors.append(st.TERRA)
        elif lbl == "L8":
            colors.append(st.SAND)
        else:
            colors.append(st.BLUE)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=LAYERS, y=deltas,
        name=L["trace_delta"],
        marker=dict(color=colors, line=dict(color=st.INK, width=0.5)),
        customdata=[[f] for f in f1_per_layer],
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{L['trace_delta']}: %{{y:.3f}}<br>"
            f"{L['trace_cum']}: %{{customdata[0]:.3f}}"
            "<extra></extra>"
        ),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=LAYERS, y=f1_per_layer,
        mode="lines+markers",
        name=L["trace_cum"],
        line=dict(color=st.INK, width=1.6),
        marker=dict(size=5, color=st.INK),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{L['trace_cum']}: %{{y:.3f}}"
            "<extra></extra>"
        ),
    ), secondary_y=True)

    # Annotations highlighting story
    fig.add_annotation(
        x="L0", y=deltas[1], yshift=22,
        text=L["annotation_l0"],
        showarrow=True, arrowhead=2, arrowsize=0.7,
        arrowwidth=0.6, arrowcolor=st.TERRA,
        font=dict(size=10.5, color=st.TERRA),
        align="center",
    )
    fig.add_annotation(
        x="L8", y=deltas[9], yshift=24,
        text=L["annotation_l8"],
        showarrow=True, arrowhead=2, arrowsize=0.7,
        arrowwidth=0.6, arrowcolor=st.SAND,
        font=dict(size=10.5, color=st.SAND),
        align="center",
    )

    layout = st.thesis_layout(height=520)
    fig.update_layout(**layout)
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.16,
            xanchor="center", x=0.5,
            font=dict(size=11, color=st.INK_2),
        ),
    )
    st.style_axes(fig, xtitle=L["xlabel"], ytitle=L["y_left"])
    fig.update_yaxes(title_text=L["y_left"], secondary_y=False, range=[-0.02, 0.4])
    fig.update_yaxes(
        title_text=L["y_right"], secondary_y=True,
        range=[-0.02, 0.65],
        showgrid=False,
        tickfont=dict(size=11, color=st.INK_3),
        title_font=dict(size=13, color=st.INK_2),
        showline=True, linecolor=st.SPINE, linewidth=0.5,
    )
    fig.update_xaxes(showgrid=False)
    return fig


if __name__ == "__main__":
    fig = build_figure(lang="es")
    out = pathlib.Path(__file__).resolve().parents[2] / "viz" / "output" / "info_gain.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"✓ {out}")
