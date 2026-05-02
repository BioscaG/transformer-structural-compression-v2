"""Hero plot 4 — Sunburst of the 6 emergent emotional clusters (§5.4.6).

Inner ring: 6 cluster groups discovered by hierarchical clustering on the
neuron-selectivity vectors. Outer ring: 23 emotions, sized by number of
significant neurons (|d|>2.0) — the "neural footprint" of each emotion.

Companion bar plot to the right: norm of selectivity vector (the best
predictor of SVD vulnerability per §5.4.5, Spearman ρ=0.64).
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import thesis_data as td
from viz import style as st
from viz.data.load_results import load_neurons, neuron_count_per_emotion


def build_sunburst_figure() -> go.Figure:
    # REAL neuron counts from notebook 7
    neurons = load_neurons()
    real_counts = neuron_count_per_emotion(neurons)
    if not real_counts:
        real_counts = td.NEURON_COUNT_PER_EMOTION
    # Real selectivity norm: derived from neuron_catalog by aggregating
    # max-selectivity per emotion. Fall back to td if missing.
    catalog = neurons.get("catalog")
    if catalog is not None:
        # Sum of |selectivity| per emotion as proxy for vector norm
        selectivity_norm = catalog.groupby("emotion")["abs_selectivity"].sum().to_dict()
    else:
        selectivity_norm = td.NEURON_SELECTIVITY_NORM
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        specs=[[{"type": "sunburst"}, {"type": "xy"}]],
        subplot_titles=(
            "Clusters emocionales emergentes (§5.4.6)",
            "Norma de selectividad neuronal (predictor de vulnerabilidad)",
        ),
        horizontal_spacing=0.05,
    )

    # ----- Sunburst data -----
    labels = ["BERT-base"]                # root
    parents = [""]
    values = [0.0]
    colors = ["#FFFFFF"]
    customdata = [["", ""]]

    # NOTE: Plotly requires parent_value == sum(child_values) when
    # branchvalues="total". We pad emotions with zero neurons up to a minimum
    # of 30 for visibility, so the parent cluster value MUST be the sum of the
    # padded values, not the raw real_counts sum, or the sunburst silently
    # refuses to render.
    MIN_VIS = 30
    real_total_per_cluster = {}
    for cluster, members in td.CLUSTER_DEFS.items():
        padded_total = sum(max(real_counts.get(e, 0), MIN_VIS) for e in members)
        real_total = sum(real_counts.get(e, 0) for e in members)
        real_total_per_cluster[cluster] = real_total

        labels.append(cluster)
        parents.append("BERT-base")
        values.append(padded_total)
        colors.append(td.CLUSTER_COLORS[cluster])
        customdata.append([f"{len(members)} emociones",
                           f"{real_total} neuronas significativas reales"])

        for emo in members:
            labels.append(emo)
            parents.append(cluster)
            count = real_counts.get(emo, 0)
            values.append(max(count, MIN_VIS))
            colors.append(td.CLUSTER_COLORS[cluster])
            sel = selectivity_norm.get(emo, 0)
            sel_str = f"{sel:.1f}" if isinstance(sel, float) else str(sel)
            customdata.append([
                f"F1 baseline: {td.F1_BASELINE[emo]:.3f}",
                f"Selectividad agregada: {sel_str} (de {count} neuronas)",
            ])

    # Root value = sum of all cluster values (which themselves are
    # sums of padded child emotions)
    cluster_total = sum(
        max(real_counts.get(e, 0), MIN_VIS)
        for cluster, members in td.CLUSTER_DEFS.items()
        for e in members
    )
    values[0] = cluster_total

    fig.add_trace(go.Sunburst(
        labels=labels, parents=parents, values=values,
        marker=dict(colors=colors, line=dict(color="white", width=1.6)),
        customdata=customdata,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Neuronas significativas: %{value}<br>"
            "%{customdata[0]}<br>%{customdata[1]}<extra></extra>"
        ),
        branchvalues="total",
        insidetextorientation="radial",
        textfont=dict(family="serif", size=11),
    ), row=1, col=1)

    # ----- Bar plot: selectivity norm per emotion, colored by cluster -----
    sorted_emotions = sorted(td.EMOTIONS,
                             key=lambda e: -selectivity_norm.get(e, 0))
    bar_colors = [td.CLUSTER_COLORS[td.EMOTION_TO_CLUSTER[e]]
                  for e in sorted_emotions]

    fig.add_trace(go.Bar(
        x=[selectivity_norm.get(e, 0) for e in sorted_emotions],
        y=sorted_emotions,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="white", width=0.6)),
        customdata=[[real_counts.get(e, 0),
                     td.EMOTION_TO_CLUSTER[e],
                     td.F1_BASELINE[e]] for e in sorted_emotions],
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Selectividad agregada: %{x:.1f}<br>"
            "Neuronas significativas: %{customdata[0]}<br>"
            "Cluster: %{customdata[1]}<br>"
            "F1 baseline: %{customdata[2]:.3f}<extra></extra>"
        ),
        showlegend=False,
    ), row=1, col=2)

    # Vulnerability annotation
    fig.add_annotation(
        x=0.99, y=0.02, xref="paper", yref="paper",
        text=("Norma selectividad ↔ caída F1 bajo SVD<br>"
              "Spearman ρ = 0.64, p = 0.001"),
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=10.5, color=st.INK_3, family="serif"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor=st.SPINE, borderwidth=0.5,
        borderpad=6,
    )

    fig.update_layout(
        **st.thesis_layout(
            title="Cap. 5.4 — Especialización neuronal y geografía emocional",
            height=620, width=1320,
        ),
    )
    fig.update_xaxes(
        title=dict(text="Norma del vector de selectividad",
                   font=dict(size=12, color=st.INK_2)),
        tickfont=dict(size=10, color=st.INK_3),
        showgrid=True, gridcolor=st.GRID, zeroline=False,
        showline=True, linecolor=st.SPINE, ticks="outside", tickcolor=st.INK_3,
        row=1, col=2,
    )
    fig.update_yaxes(
        title=dict(text="", font=dict(size=12, color=st.INK_2)),
        tickfont=dict(size=10, color=st.INK_2, family="serif"),
        showgrid=False, zeroline=False, autorange="reversed",
        row=1, col=2,
    )

    # Re-style subplot titles
    for ann in fig["layout"]["annotations"]:
        if "text" in ann and any(s in ann["text"] for s in ["Clusters emocionales", "Norma de selectividad"]):
            ann["font"] = dict(size=12.5, color=st.INK, family="serif")
            ann["yshift"] = -3

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_sunburst_figure()
    out = out_dir / "04_sunburst_clusters.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
