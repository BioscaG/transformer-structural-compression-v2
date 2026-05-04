"""Where the emotional neurons live (§5.5 of the thesis).

Two-panel figure. LEFT: cumulative significant-neuron count per depth
band (Emb–L3, L4–L7, L8–L11) with a stacked-bar style breakdown by
emotion cluster. The thesis finding: 3,061 of the 3,642 significant
neurons (|d|>2.0) live in layers 8–11, that's 84%. RIGHT: top-N
selectivity per emotion, sorted, showing the long tail — gratitude
has 818 significant neurons; *annoyance*, *disappointment* and
*realization* each have zero.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import style as st
from viz import thesis_data as td


CSV_COUNTS = (pathlib.Path(__file__).resolve().parents[2]
              / "results" / "csvs" / "notebook7"
              / "neuron_significant_counts.csv")


LAYER_BANDS = [
    ("Emb–L3", ["layer_0", "layer_1", "layer_2", "layer_3"]),
    ("L4–L7",  ["layer_4", "layer_5", "layer_6", "layer_7"]),
    ("L8–L11", ["layer_8", "layer_9", "layer_10", "layer_11"]),
]


LANG = {
    "es": {
        "title":         "Neuronas emocionalmente selectivas (|d| > 2.0)",
        "panel_left":    "Distribución por bloque de profundidad",
        "panel_right":   "Total de neuronas por emoción",
        "ylabel_left":   "Número de neuronas significativas",
        "ylabel_right":  "Total significativas",
        "xlabel_left":   "Bloque de profundidad",
        "xlabel_right":  "",
        "annot":         "<b>84 %</b> de las neuronas significativas<br>viven en L8–L11",
    },
    "en": {
        "title":         "Emotionally selective neurons (|d| > 2.0)",
        "panel_left":    "Distribution by depth band",
        "panel_right":   "Total neurons per emotion",
        "ylabel_left":   "Significant neurons",
        "ylabel_right":  "Total significant",
        "xlabel_left":   "Depth band",
        "xlabel_right":  "",
        "annot":         "<b>84 %</b> of significant neurons<br>live in L8–L11",
    },
}


def build_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    df = pd.read_csv(CSV_COUNTS)

    # --- Left panel: stacked bars per depth band, coloured by cluster ---
    band_totals = {band: 0 for band, _ in LAYER_BANDS}
    band_per_cluster = {band: {c: 0 for c in td.CLUSTER_DEFS}
                        for band, _ in LAYER_BANDS}
    for _, row in df.iterrows():
        emo = row["emotion"]
        cluster = next((c for c, members in td.CLUSTER_DEFS.items()
                        if emo in members), None)
        if cluster is None:
            continue
        for band, cols in LAYER_BANDS:
            v = sum(int(row[c]) for c in cols)
            band_totals[band] += v
            band_per_cluster[band][cluster] += v

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.42, 0.58],
        horizontal_spacing=0.10,
        subplot_titles=(L["panel_left"], L["panel_right"]),
    )

    band_labels = [b for b, _ in LAYER_BANDS]
    for cluster, color in td.CLUSTER_COLORS.items():
        ys = [band_per_cluster[b][cluster] for b in band_labels]
        fig.add_trace(go.Bar(
            x=band_labels, y=ys,
            name=cluster,
            marker=dict(color=color,
                        line=dict(color=st.INK, width=0.4)),
            customdata=[[cluster, b] for b in band_labels],
            hovertemplate=(
                "<b>%{customdata[1]}</b> · %{customdata[0]}<br>"
                "Neuronas: %{y}"
                "<extra></extra>"
            ),
        ), row=1, col=1)

    # 84% annotation on left panel
    fig.add_annotation(
        x="L8–L11", y=band_totals["L8–L11"],
        yshift=20,
        text=L["annot"],
        showarrow=False,
        font=dict(size=11, color=st.TERRA),
        align="center",
        row=1, col=1,
    )

    # --- Right panel: emotions sorted by total ---
    df_sorted = df.sort_values("total_significant", ascending=False).reset_index(drop=True)
    bar_colors = []
    for emo in df_sorted["emotion"]:
        cluster = next((c for c, members in td.CLUSTER_DEFS.items()
                        if emo in members), None)
        bar_colors.append(td.CLUSTER_COLORS.get(cluster, st.INK_3))

    fig.add_trace(go.Bar(
        x=df_sorted["emotion"], y=df_sorted["total_significant"],
        marker=dict(color=bar_colors, line=dict(color=st.INK, width=0.4)),
        showlegend=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Total significativas: %{y}"
            "<extra></extra>"
        ),
    ), row=1, col=2)

    layout = st.thesis_layout(height=560)
    fig.update_layout(**layout)
    fig.update_layout(
        barmode="stack",
        bargap=0.22,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.20,
            xanchor="center", x=0.21,
            font=dict(size=10.5, color=st.INK_2),
        ),
    )

    fig.update_xaxes(title_text=L["xlabel_left"], showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text=L["ylabel_left"], gridcolor=st.GRID,
                     row=1, col=1)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10, color=st.INK_3),
                     showgrid=False, row=1, col=2)
    fig.update_yaxes(title_text=L["ylabel_right"], gridcolor=st.GRID,
                     row=1, col=2)
    return fig


if __name__ == "__main__":
    fig = build_figure(lang="es")
    out = pathlib.Path(__file__).resolve().parents[2] / "viz" / "output" / "neurons.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"✓ {out}")
