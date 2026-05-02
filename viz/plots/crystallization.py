"""Hero plot 3 — Crystallization heatmap, the flagship image of Cap. 5.1.

23 emotions × 13 layers (embedding + 12 encoder) of REAL probe F1 from the
user's notebook 4 (`probe_results_long.csv`). Rows ordered by crystallization
layer. Crystallization markers (the first layer where probe F1 hits 80% of
max) are pinned with annotated diamonds. Cluster membership is shown as a
colored ribbon on the left.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import thesis_data as td
from viz import style as st
from viz.data.load_results import (
    load_probing, load_informed, probe_f1_matrix, crystallization_dict,
    f1_baseline_per_emotion, EMOTIONS_23,
)


def _layer_labels() -> list[str]:
    return ["Emb"] + [f"L{i}" for i in range(12)]


def build_crystallization_figure() -> go.Figure:
    # Load REAL data from the user's notebook 4
    probing = load_probing()
    informed = load_informed()
    f1_grid, _, _ = probe_f1_matrix(probing)         # (23, 13) — REAL probe F1
    crystals = crystallization_dict(probing)         # {emotion: dict}
    f1_baseline = f1_baseline_per_emotion(informed)  # {emotion: F1}

    # Order emotions by crystallization layer ascending, then by F1 baseline desc.
    emotions = sorted(
        EMOTIONS_23,
        key=lambda e: (crystals[e]["crystal_layer"], -f1_baseline.get(e, td.F1_BASELINE.get(e, 0))),
    )

    # Reorder f1_grid to match
    emotion_to_row = {e: i for i, e in enumerate(EMOTIONS_23)}
    z = np.array([f1_grid[emotion_to_row[e]] for e in emotions])

    # Cluster ribbon column on the left
    cluster_color_map = td.CLUSTER_COLORS
    ribbon_colors = [cluster_color_map[td.EMOTION_TO_CLUSTER[e]] for e in emotions]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.04, 0.96],
        horizontal_spacing=0.005,
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    # Cluster ribbon (small heatmap with one column per emotion)
    fig.add_trace(go.Heatmap(
        z=[[i] for i in range(len(emotions))],
        x=[" "],
        y=emotions,
        colorscale=[
            *[[i / (len(emotions) - 1), ribbon_colors[i]] for i in range(len(emotions))],
        ],
        showscale=False,
        hoverinfo="skip",
        zmin=-0.5, zmax=len(emotions) - 0.5,
        xgap=0, ygap=2,
    ), row=1, col=1)

    # Main crystallization heatmap
    fig.add_trace(go.Heatmap(
        z=z, x=_layer_labels(), y=emotions,
        colorscale=[
            [0.00, "#FFFFFF"],
            [0.15, "#FAE9CB"],
            [0.35, st.SAND_L],
            [0.55, st.SAND],
            [0.75, st.TERRA_L],
            [1.00, st.TERRA],
        ],
        zmin=0, zmax=0.95,
        showscale=True,
        colorbar=dict(
            title=dict(text="Probe F1", font=dict(size=11)),
            thickness=14, x=1.02, len=0.85,
            tickfont=dict(size=10, color=st.INK_3),
        ),
        hovertemplate="<b>%{y}</b><br>Capa: %{x}<br>Probe F1: %{z:.3f}<extra></extra>",
        xgap=0.5, ygap=2,
    ), row=1, col=2)

    # Crystallization markers (layer index in 0..12 already)
    crystal_x = [crystals[e]["crystal_layer"] for e in emotions]
    crystal_y = list(range(len(emotions)))
    fig.add_trace(go.Scatter(
        x=[_layer_labels()[i] for i in crystal_x], y=emotions,
        mode="markers",
        marker=dict(symbol="diamond", size=11,
                    color="white", line=dict(color=st.INK, width=1.6)),
        hovertemplate="<b>%{y}</b><br>Cristaliza en %{x}<extra></extra>",
        name="Capa de cristalización",
        showlegend=False,
    ), row=1, col=2)

    # Annotations: F1 baseline label on the right side (real model F1)
    for i, e in enumerate(emotions):
        fig.add_annotation(
            x=12.65, xref="x2",
            y=i, yref="y2",
            text=f"{f1_baseline.get(e, td.F1_BASELINE.get(e, 0)):.2f}",
            showarrow=False, xanchor="left",
            font=dict(size=9.5, color=st.INK_3, family="serif"),
        )
    # Header for that column
    fig.add_annotation(
        x=12.65, xref="x2", y=-1, yref="y2",
        text="<b>F1<br>head</b>", showarrow=False, xanchor="left",
        font=dict(size=9.5, color=st.INK_2, family="serif"),
    )

    # Top cluster legend
    legend_y = len(emotions) + 1.2
    cluster_x_offset = 0
    for cname, ccolor in cluster_color_map.items():
        fig.add_annotation(
            x=cluster_x_offset, y=legend_y, xref="x2", yref="y2",
            text=f"■ {cname}", showarrow=False, xanchor="left",
            font=dict(size=10.5, color=ccolor, family="serif"),
        )
        cluster_x_offset += 2.05

    fig.update_layout(
        **st.thesis_layout(
            title="Cap. 5.1 — Cristalización emocional capa por capa",
            height=720, width=1200,
        ),
    )

    fig.update_xaxes(
        title=dict(text="", font=dict(size=12, color=st.INK_2)),
        tickfont=dict(size=10, color=st.INK_3),
        showgrid=False, zeroline=False, fixedrange=True,
        row=1, col=1, showticklabels=False,
    )
    fig.update_yaxes(
        title=dict(text="", font=dict(size=12, color=st.INK_2)),
        tickfont=dict(size=10, color=st.INK_3),
        showgrid=False, zeroline=False, autorange="reversed", fixedrange=True,
        row=1, col=1, showticklabels=False,
    )
    fig.update_xaxes(
        title=dict(text="Capa del encoder", font=dict(size=13, color=st.INK_2)),
        tickfont=dict(size=11, color=st.INK_3),
        showgrid=False, zeroline=False,
        row=1, col=2,
    )
    fig.update_yaxes(
        title=dict(text="Emoción (ordenada por cristalización)", font=dict(size=13, color=st.INK_2)),
        tickfont=dict(size=10.5, color=st.INK_2, family="serif"),
        showgrid=False, zeroline=False, autorange="reversed",
        row=1, col=2,
    )

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_crystallization_figure()
    out = out_dir / "03_crystallization.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
