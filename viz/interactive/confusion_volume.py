"""Confusion volume — 23×23 confusion matrices stacked in 3D across 13 layers.

Instead of animating the confusion matrix in 2D, render the full 23×23×13
cube as 3D scatter points. Z-axis = layer. Color = mean sigmoid (gold,
predicted) at that layer. Threshold low values (< 0.15) so only the
significant cells appear — reveals which (gold, pred) pairs persist
across layers.

What you see:
  - Diagonal: a vertical column of cells that BRIGHTENS from L0 (dim) to
    L11 (bright). The diagonal is built layer by layer.
  - Off-diagonal "smudges" that survive late layers: persistent confusions
    like annoyance↔disapproval, fear↔sadness — emotion pairs the model
    NEVER fully separates.
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from transformers import AutoModelForSequenceClassification

from viz import style as st
from viz.data.load_results import EMOTIONS_23, MODEL_CHECKPOINT
from viz.thesis_data import EXTENDED_CLUSTER_MAP


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


def build_volume_figure() -> go.Figure:
    data = np.load(CACHE_DIR / "activations.npz")
    cls = data["cls_per_layer"]                    # (N, 13, 768)
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    labels = np.array(meta["label_names"])

    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    Wp = mdl.bert.pooler.dense.weight.detach().numpy()
    bp = mdl.bert.pooler.dense.bias.detach().numpy()
    W = mdl.classifier.weight.detach().numpy()
    b = mdl.classifier.bias.detach().numpy()

    pooled = np.tanh(np.einsum('slh,ph->slp', cls, Wp) + bp)
    logits = np.einsum('slp,ep->sle', pooled, W) + b
    sigmoids = 1.0 / (1.0 + np.exp(-logits))       # (N, 13, 23)

    n_layers = sigmoids.shape[1]
    layer_labels = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]

    cluster_order = ["Positivas alta energía", "Negativas reactivas",
                     "Negativas internas", "Epistémicas",
                     "Orientadas al otro", "Baja especificidad"]
    ordered_emotions = []
    for cluster in cluster_order:
        members = [e for e in EMOTIONS_23
                   if EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad") == cluster]
        ordered_emotions.extend(sorted(members))
    emo_to_idx = {e: i for i, e in enumerate(EMOTIONS_23)}

    # Compute confusion at each layer
    # M[L, gold, pred] = mean sigmoid for `pred` among sentences with gold == gold
    cube = np.zeros((n_layers, 23, 23), dtype=np.float32)
    for L in range(n_layers):
        for ri, gold in enumerate(ordered_emotions):
            mask = labels == gold
            if not mask.any():
                continue
            for ci, pred in enumerate(ordered_emotions):
                cube[L, ri, ci] = sigmoids[mask, L, emo_to_idx[pred]].mean()

    # ─── Persistence filter ───
    # For each (gold, pred) pair, count how many layers have sigmoid > 0.15.
    # A pair is "persistent" if it appears in at least N_PERSIST layers — these
    # are the structural patterns of the model, not transient noise.
    SIGMOID_THRESHOLD = 0.15
    N_PERSIST = 7   # appears in at least 7/13 layers (>50%)
    appears = (cube > SIGMOID_THRESHOLD)         # (n_layers, 23, 23)
    persistence_count = appears.sum(axis=0)      # (23, 23)

    # Build 3D scatter ONLY for persistent (gold, pred) pairs
    xs, ys, zs, vals, hovers = [], [], [], [], []
    for L in range(n_layers):
        for ri in range(23):
            for ci in range(23):
                v = cube[L, ri, ci]
                if v < SIGMOID_THRESHOLD:
                    continue
                if persistence_count[ri, ci] < N_PERSIST:
                    continue
                xs.append(ci)
                ys.append(ri)
                zs.append(L)
                vals.append(v)
                gold = ordered_emotions[ri]
                pred = ordered_emotions[ci]
                marker = "✓" if ri == ci else "✗"
                hovers.append(f"{marker} <b>{gold} → {pred}</b><br>"
                              f"capa: {layer_labels[L]}<br>"
                              f"σ medio: {v:.3f}<br>"
                              f"persistencia: {persistence_count[ri, ci]}/{n_layers} capas")

    # Build cluster-block separators on the floor (L0)
    cluster_boundaries = []
    cum = 0
    for cluster in cluster_order:
        n_in = sum(1 for e in EMOTIONS_23
                   if EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad") == cluster)
        cum += n_in
        cluster_boundaries.append(cum)

    fig = go.Figure()

    # Diagonal column highlight (a translucent line along the diagonal at every layer)
    fig.add_trace(go.Scatter3d(
        x=list(range(23)) * n_layers,
        y=list(range(23)) * n_layers,
        z=[L for L in range(n_layers) for _ in range(23)],
        mode="lines",
        line=dict(color=st.SAGE, width=2),
        opacity=0.3,
        hoverinfo="skip", showlegend=False,
        name="diagonal",
    ))

    # The actual confusion cells
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(
            size=[6 + v * 14 for v in vals],
            color=vals,
            colorscale=[[0, "#FFFFFF"], [0.3, st.SAND_L], [0.6, st.SAND],
                        [0.85, st.TERRA], [1.0, "#7A2A18"]],
            cmin=SIGMOID_THRESHOLD, cmax=0.85,
            opacity=0.85,
            line=dict(color="white", width=0.5),
            showscale=True,
            colorbar=dict(thickness=14, len=0.7, x=1.0,
                          title=dict(text="σ medio<br>(predicho|gold)",
                                     font=dict(size=11)),
                          tickfont=dict(size=10, color=st.INK_3)),
        ),
        hovertext=hovers, hoverinfo="text",
        showlegend=False,
        name="confusiones",
    ))

    # Cluster-block separators as faint planes (use lines for simplicity)
    for b_idx in cluster_boundaries[:-1]:
        for L in range(n_layers):
            fig.add_trace(go.Scatter3d(
                x=[b_idx - 0.5, b_idx - 0.5], y=[-0.5, 22.5], z=[L, L],
                mode="lines",
                line=dict(color=st.INK_3, width=0.5, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter3d(
                x=[-0.5, 22.5], y=[b_idx - 0.5, b_idx - 0.5], z=[L, L],
                mode="lines",
                line=dict(color=st.INK_3, width=0.5, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))

    # Build a 2D summary heatmap of persistence on the floor — gives a
    # context for where the structural patterns concentrate
    persistent_pairs = []
    for ri in range(23):
        for ci in range(23):
            if persistence_count[ri, ci] >= N_PERSIST:
                persistent_pairs.append((ordered_emotions[ri], ordered_emotions[ci],
                                          int(persistence_count[ri, ci])))
    n_persistent = len(persistent_pairs)
    n_diagonal = sum(1 for g, p, _ in persistent_pairs if g == p)
    n_offdiag = n_persistent - n_diagonal

    fig.update_layout(
        **st.thesis_layout(
            title=(f"Confusion volume · solo pares persistentes ({N_PERSIST}+/{n_layers} capas)"
                   f"<br><sub>{n_persistent} pares (gold, predicho) sobreviven el filtro: "
                   f"{n_diagonal} en la diagonal (aciertos estructurales) + "
                   f"{n_offdiag} fuera de diagonal (confusiones que el modelo NUNCA arregla). "
                   "Las columnas verticales que ves son patrones intrínsecos del modelo.</sub>"),
            height=820, width=1320,
        ),
        scene=dict(
            xaxis=dict(title="predicho",
                       tickmode="array",
                       tickvals=list(range(23)),
                       ticktext=ordered_emotions,
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       zerolinecolor=st.SPINE, color=st.INK_2),
            yaxis=dict(title="gold",
                       tickmode="array",
                       tickvals=list(range(23)),
                       ticktext=ordered_emotions,
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       zerolinecolor=st.SPINE, color=st.INK_2,
                       autorange="reversed"),
            zaxis=dict(title="capa",
                       tickmode="array",
                       tickvals=list(range(n_layers)),
                       ticktext=layer_labels,
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       zerolinecolor=st.SPINE, color=st.INK_2),
            camera=dict(eye=dict(x=2.0, y=-2.0, z=1.0)),
            aspectmode="manual", aspectratio=dict(x=1.0, y=1.0, z=0.85),
        ),
    )

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_volume_figure()
    out = out_dir / "22_confusion_volume.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
