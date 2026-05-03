"""Probe constellations — the 23 emotion DIRECTIONS in 3D.

The classifier head's 23 weight vectors (one per emotion, each 768-d) ARE
the model's emotion-detection directions. Project them to PCA-3D and render
as arrows radiating from origin. Each arrow is the direction in which the
classifier "looks for" that emotion.

Geometric reading:
  - Vectors pointing in similar directions → emotions the model conflates
    (high cosine similarity in the classifier weights)
  - Vectors near-orthogonal → emotions the model perfectly distinguishes
  - Vectors opposite → emotions on opposite sides of a hyperplane

This is the most direct visualization of §2.7.5 (superposition): the model
encodes emotions as quasi-orthogonal directions in 768-d, and the 3D
projection preserves the inter-emotion geometry.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from transformers import AutoModelForSequenceClassification

from viz import style as st
from viz.data.load_results import EMOTIONS_23, MODEL_CHECKPOINT
from viz.thesis_data import EXTENDED_CLUSTER_MAP, emotion_palette


LANG = {
    "es": {
        "panel_3d":   "Direcciones de detección · PCA del peso del classifier",
        "panel_heat": "Similitud coseno entre emociones (768-d)",
        "origin":     "origen",
        "hover_emo":  "<b>{emo}</b><br>cluster: {cluster}<br>norma 768d: %{{customdata:.2f}}<extra></extra>",
        "heatmap_h":  "%{y} ↔ %{x}<br>cos sim: %{z:.3f}<extra></extra>",
        "cbar":       "cos sim",
        "clusters": {
            "Positivas alta energía":  "Positivas alta energía",
            "Negativas reactivas":     "Negativas reactivas",
            "Negativas internas":      "Negativas internas",
            "Epistémicas":             "Epistémicas",
            "Orientadas al otro":      "Orientadas al otro",
            "Baja especificidad":      "Baja especificidad",
        },
    },
    "en": {
        "panel_3d":   "Detection directions · PCA of classifier weights",
        "panel_heat": "Cosine similarity between emotions (768-d)",
        "origin":     "origin",
        "hover_emo":  "<b>{emo}</b><br>cluster: {cluster}<br>768d norm: %{{customdata:.2f}}<extra></extra>",
        "heatmap_h":  "%{y} ↔ %{x}<br>cos sim: %{z:.3f}<extra></extra>",
        "cbar":       "cos sim",
        "clusters": {
            "Positivas alta energía":  "High-energy positives",
            "Negativas reactivas":     "Reactive negatives",
            "Negativas internas":      "Internal negatives",
            "Epistémicas":             "Epistemic",
            "Orientadas al otro":      "Other-oriented",
            "Baja especificidad":      "Low specificity",
        },
    },
}


def build_constellation_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    W = mdl.classifier.weight.detach().numpy()       # (23, 768)
    # The user's checkpoint stores emotions in EMOTIONS_23 order.

    # PCA on the 23 weight vectors → 3D
    pca = PCA(n_components=3, random_state=42).fit(W)
    W3 = pca.transform(W)
    explained = pca.explained_variance_ratio_

    # Cosine similarity matrix (in the original 768-d space)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    cos_sim = Wn @ Wn.T

    palette = emotion_palette(EMOTIONS_23)
    cluster_for = {e: EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad") for e in EMOTIONS_23}

    # ─── Build figure with 3D constellation + 2D similarity heatmap ───
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.58, 0.42],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(L["panel_3d"], L["panel_heat"]),
        horizontal_spacing=0.13,
    )

    # Arrows from origin to each emotion's projected weight vector
    # We build them as line segments + tip markers
    for i, emo in enumerate(EMOTIONS_23):
        v = W3[i]
        cluster = cluster_for[emo]
        color = palette[emo]
        # Line from origin to vector tip
        fig.add_trace(go.Scatter3d(
            x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
            mode="lines",
            line=dict(color=color, width=4),
            hoverinfo="skip", showlegend=False,
            name=emo,
        ), row=1, col=1)
        # Tip marker + label
        fig.add_trace(go.Scatter3d(
            x=[v[0]], y=[v[1]], z=[v[2]],
            mode="markers+text",
            marker=dict(size=8, color=color, symbol="diamond",
                        line=dict(color=st.INK, width=1)),
            text=[emo], textposition="top center",
            textfont=dict(size=10, color=color, family="serif"),
            hovertemplate=L["hover_emo"].format(emo=emo,
                cluster=L["clusters"].get(cluster, cluster)),
            customdata=[float(np.linalg.norm(W[i]))],
            name=emo, legendgroup=cluster,
            legendgrouptitle=dict(text=L["clusters"].get(cluster, cluster)),
            showlegend=True,
        ), row=1, col=1)

    # Origin marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=6, color=st.INK, symbol="cross"),
        hovertext=[L["origin"]], hoverinfo="text",
        showlegend=False,
    ), row=1, col=1)

    # Cosine similarity heatmap — order emotions by cluster for visual blocks
    cluster_order = ["Positivas alta energía", "Negativas reactivas",
                     "Negativas internas", "Epistémicas",
                     "Orientadas al otro", "Baja especificidad"]
    ordered = []
    for cluster in cluster_order:
        ordered.extend(sorted([e for e in EMOTIONS_23 if cluster_for[e] == cluster]))
    emo_to_idx = {e: i for i, e in enumerate(EMOTIONS_23)}
    reorder = [emo_to_idx[e] for e in ordered]
    cos_reordered = cos_sim[np.ix_(reorder, reorder)]

    fig.add_trace(go.Heatmap(
        z=cos_reordered, x=ordered, y=ordered,
        colorscale=[[0, "#3A0000"], [0.45, "#FFFFFF"], [1, st.TERRA]],
        zmid=0, zmin=-0.5, zmax=1.0,
        showscale=True,
        colorbar=dict(thickness=14, len=0.75, x=1.0,
                      title=dict(text=L["cbar"], font=dict(size=11)),
                      tickfont=dict(size=10, color=st.INK_3)),
        hovertemplate=L["heatmap_h"],
        xgap=0.5, ygap=0.5,
    ), row=1, col=2)

    # Cluster boundary lines on the heatmap. The heatmap subplot on
    # row=1 col=2 uses axes "x" and "y" (not "x2"/"y2") because the
    # scene takes the first axis slot. Use paper-relative refs scoped
    # via the row/col argument.
    cum = 0
    shapes = []
    for cluster in cluster_order:
        n_in = sum(1 for e in EMOTIONS_23 if cluster_for[e] == cluster)
        cum += n_in
        if cum < 23:
            shapes.append(dict(type="line",
                               x0=cum - 0.5, x1=cum - 0.5, y0=-0.5, y1=22.5,
                               line=dict(color=st.INK_3, width=1, dash="dash"),
                               xref="x", yref="y"))
            shapes.append(dict(type="line",
                               x0=-0.5, x1=22.5, y0=cum - 0.5, y1=cum - 0.5,
                               line=dict(color=st.INK_3, width=1, dash="dash"),
                               xref="x", yref="y"))

    fig.update_layout(
        **st.thesis_layout(
            title=("Probe constellations · 23 emociones como direcciones en el espacio del classifier"
                   "<br><sub>Cada flecha es la proyección PCA-3D de uno de los 23 pesos del clasificador. "
                   f"Los 3 ejes explican {explained.sum():.1%} de la varianza total. "
                   "Heatmap = cosine similarity en 768-d (reordenada por cluster).</sub>"),
            height=720, width=1380,
        ),
        scene=dict(
            xaxis=dict(title="PC1", showticklabels=False, backgroundcolor=st.BG,
                       gridcolor=st.GRID, zerolinecolor=st.SPINE),
            yaxis=dict(title="PC2", showticklabels=False, backgroundcolor=st.BG,
                       gridcolor=st.GRID, zerolinecolor=st.SPINE),
            zaxis=dict(title="PC3", showticklabels=False, backgroundcolor=st.BG,
                       gridcolor=st.GRID, zerolinecolor=st.SPINE),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.0)),
            aspectmode="cube",
        ),
        shapes=shapes,
        legend=dict(x=1.02, y=0.5, xanchor="left", yanchor="middle",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                    borderwidth=0.5, font=dict(size=10)),
    )
    fig.update_xaxes(type="category", side="top", tickangle=-45,
                     tickfont=dict(size=9, color=st.INK_2, family="serif"),
                     showgrid=False, linecolor=st.SPINE, automargin=True,
                     row=1, col=2)
    fig.update_yaxes(type="category", autorange="reversed",
                     tickfont=dict(size=9, color=st.INK_2, family="serif"),
                     showgrid=False, linecolor=st.SPINE, automargin=True,
                     row=1, col=2)

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_constellation_figure()
    out = out_dir / "20_probe_constellations.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
