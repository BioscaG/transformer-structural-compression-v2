"""Confusion matrix evolution — 23×23 confusion across 13 layers.

Apply pooler+classifier to each layer's CLS (logit lens). For each (gold, predicted)
pair, compute mean sigmoid of the predicted emotion across sentences whose
gold is `gold`. Animate across layers.

Storyline: at L0 the matrix is uniform (model hasn't decided), middle layers
collapse near zero (transition valley), late layers reveal a clean diagonal
with informative off-diagonal smudges (pairs the model confuses — exactly
the §5.4.6 cluster structure made visible).
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


LANG = {
    "es": {
        "cbar":      "sigmoid medio<br>(predicho | gold)",
        "hover":     "<b>gold: %{y}</b><br>predicho: %{x}<br>sigmoid medio: %{z:.3f}<extra></extra>",
        "stage_pref": "Capa: ",
        "play":      "▶ Play",
        "pause":     "⏸ Pause",
        "reset":     "↺ Reset",
        "trans":     "transición",
        "captions": {
            0: "L0 (Embedding) — saturación inicial: el modelo activa muchas emociones a la vez para CADA gold. Matriz casi uniforme — no hay decisión.",
            4: "L4 — el valle de transición empieza. Los valores caen, las filas se vuelven más oscuras.",
            7: "L7 — fondo del valle. Casi todo cerca de 0: el modelo está 'pensando' pero no decide.",
            9: "L9 — la diagonal empieza a emerger. Para algunas emociones (gratitude, fear) la celda diagonal se separa del resto.",
            11: "L11 — diagonal limpia + smudges fuera de diagonal. Las celdas off-diagonal cuentan QUÉ EMOCIONES SE CONFUNDEN: annoyance↔disapproval, fear↔sadness — exactamente las vecinas en los 6 clusters psicológicos.",
            12: "L12 (capa final) — el modelo confiable. Donde no logra limpiar la diagonal, se ve la estructura de la dificultad: approval/realization siguen difusas.",
        },
    },
    "en": {
        "cbar":      "mean sigmoid<br>(predicted | gold)",
        "hover":     "<b>gold: %{y}</b><br>predicted: %{x}<br>mean sigmoid: %{z:.3f}<extra></extra>",
        "stage_pref": "Layer: ",
        "play":      "▶ Play",
        "pause":     "⏸ Pause",
        "reset":     "↺ Reset",
        "trans":     "transition",
        "captions": {
            0: "L0 (Embedding) — initial saturation: the model fires many emotions at once for EVERY gold. Matrix is nearly uniform — no decision.",
            4: "L4 — the transition valley begins. Values drop, rows go darker.",
            7: "L7 — bottom of the valley. Almost everything near 0: the model is \"thinking\" but not deciding.",
            9: "L9 — the diagonal starts emerging. For some emotions (gratitude, fear) the diagonal cell pulls ahead.",
            11: "L11 — clean diagonal + off-diagonal smudges. The off-diagonal cells reveal WHICH EMOTIONS GET CONFUSED: annoyance↔disapproval, fear↔sadness — exactly the neighbours in the 6 psychological clusters.",
            12: "L12 (final layer) — the reliable model. Where the diagonal can't be cleaned, the difficulty structure shows: approval/realization stay diffuse.",
        },
    },
}


def build_confusion_figure(lang: str = "es") -> go.Figure:
    _L = LANG[lang]
    data = np.load(CACHE_DIR / "activations.npz")
    cls = data["cls_per_layer"]                      # (N, 13, 768)
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    labels = np.array(meta["label_names"])

    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    Wp = mdl.bert.pooler.dense.weight.detach().numpy()
    bp = mdl.bert.pooler.dense.bias.detach().numpy()
    W = mdl.classifier.weight.detach().numpy()
    b = mdl.classifier.bias.detach().numpy()

    pooled = np.tanh(np.einsum('slh,ph->slp', cls, Wp) + bp)
    logits = np.einsum('slp,ep->sle', pooled, W) + b
    sigmoids = 1.0 / (1.0 + np.exp(-logits))         # (N, 13, 23)

    # Order emotions by cluster for visual coherence
    cluster_order = ["Positivas alta energía", "Negativas reactivas",
                     "Negativas internas", "Epistémicas",
                     "Orientadas al otro", "Baja especificidad"]
    ordered_emotions = []
    for cluster in cluster_order:
        members = [e for e in EMOTIONS_23
                   if EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad") == cluster]
        ordered_emotions.extend(sorted(members))
    emo_to_idx = {e: i for i, e in enumerate(EMOTIONS_23)}

    n_layers = sigmoids.shape[1]
    layer_labels = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]

    # Build confusion-style matrices: rows = gold emotion (in cluster order),
    # cols = predicted emotion (same order). Cell value = mean sigmoid for the
    # predicted emotion among sentences whose gold is the row emotion.
    matrices = []
    for L in range(n_layers):
        M = np.zeros((23, 23), dtype=np.float32)
        for ri, gold in enumerate(ordered_emotions):
            mask = labels == gold
            if not mask.any():
                continue
            for ci, pred in enumerate(ordered_emotions):
                pred_idx = emo_to_idx[pred]
                M[ri, ci] = sigmoids[mask, L, pred_idx].mean()
        matrices.append(M)

    # Cluster boundaries for grid overlay
    cluster_boundaries = []
    cum = 0
    for cluster in cluster_order:
        members = [e for e in EMOTIONS_23
                   if EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad") == cluster]
        cum += len(members)
        cluster_boundaries.append(cum)

    init = matrices[0]

    fig = go.Figure(go.Heatmap(
        z=init, x=ordered_emotions, y=ordered_emotions,
        colorscale=[[0, "#FFFFFF"], [0.20, st.SAND_L], [0.55, st.SAND],
                    [0.85, st.TERRA], [1.0, "#7A2A18"]],
        zmin=0, zmax=0.85,
        showscale=True, colorbar=dict(thickness=14, len=0.85,
                                       title=dict(text=_L["cbar"],
                                                  font=dict(size=11)),
                                       tickfont=dict(size=10, color=st.INK_3)),
        hovertemplate=_L["hover"],
        xgap=0.5, ygap=0.5,
    ))

    # Cluster division lines
    shapes = []
    for b_idx in cluster_boundaries[:-1]:
        shapes.append(dict(type="line",
                           x0=b_idx - 0.5, x1=b_idx - 0.5, y0=-0.5, y1=22.5,
                           line=dict(color=st.INK_3, width=1.0, dash="dash")))
        shapes.append(dict(type="line",
                           x0=-0.5, x1=22.5, y0=b_idx - 0.5, y1=b_idx - 0.5,
                           line=dict(color=st.INK_3, width=1.0, dash="dash")))

    # Frames
    captions = _L["captions"]

    frames = []
    for L in range(n_layers):
        cap = captions.get(L, f"{layer_labels[L]} — {_L['trans']}")
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=matrices[L], x=ordered_emotions, y=ordered_emotions,
                colorscale=[[0, "#FFFFFF"], [0.20, st.SAND_L], [0.55, st.SAND],
                            [0.85, st.TERRA], [1.0, "#7A2A18"]],
                zmin=0, zmax=0.85,
                hovertemplate=("<b>gold: %{y}</b><br>"
                               "predicho: %{x}<br>"
                               "sigmoid medio: %{z:.3f}<extra></extra>"),
                xgap=0.5, ygap=0.5, showscale=True,
                colorbar=dict(thickness=14, len=0.85,
                              title=dict(text="sigmoid medio<br>(predicho | gold)",
                                         font=dict(size=11)),
                              tickfont=dict(size=10, color=st.INK_3)),
            )],
            name=layer_labels[L],
            layout=dict(annotations=[
                dict(text=f"<b>{layer_labels[L]}</b> · {cap}",
                     x=0.5, y=-0.18, xref="paper", yref="paper", showarrow=False,
                     align="center", xanchor="center",
                     font=dict(size=11.5, color=st.INK_2, family="serif"),
                     bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                     borderwidth=0.5, borderpad=10),
            ]),
        ))

    fig.frames = frames

    steps = [dict(
        method="animate",
        args=[[layer_labels[L]],
              dict(mode="immediate", frame=dict(duration=0, redraw=True),
                   transition=dict(duration=400, easing="cubic-in-out"))],
        label=layer_labels[L],
    ) for L in range(n_layers)]

    fig.update_layout(
        **st.thesis_layout(
            title=("Confusion matrix evolution · 23×23 a través de las 13 capas"
                   "<br><sub>Filas = emoción gold (por cluster). Columnas = "
                   "emoción predicha. Cada celda = sigmoid medio del modelo "
                   "para PRED dado GOLD. Líneas discontinuas = fronteras de "
                   "cluster psicológico. Diagonal = aciertos.</sub>"),
            height=820, width=1100,
        ),
        shapes=shapes,
        annotations=[
            dict(text=f"<b>{layer_labels[0]}</b> · {captions[0]}",
                 x=0.5, y=-0.18, xref="paper", yref="paper", showarrow=False,
                 align="center", xanchor="center",
                 font=dict(size=11.5, color=st.INK_2, family="serif"),
                 bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                 borderwidth=0.5, borderpad=10),
        ],
        sliders=[dict(
            active=0, x=0.05, y=-0.05, len=0.85,
            currentvalue=dict(prefix=_L["stage_pref"],
                              font=dict(size=14, color=st.INK, family="serif"),
                              xanchor="left"),
            steps=steps,
            pad=dict(t=20, b=10),
            tickcolor=st.INK_3,
            font=dict(size=10, color=st.INK_3),
            transition=dict(duration=400, easing="cubic-in-out"),
        )],
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.05, y=-0.20, xanchor="left", yanchor="top",
            buttons=[
                dict(label=_L["play"], method="animate",
                     args=[None, dict(frame=dict(duration=900, redraw=True),
                                      transition=dict(duration=400, easing="cubic-in-out"),
                                      fromcurrent=True, mode="immediate")]),
                dict(label=_L["pause"], method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
                dict(label=_L["reset"], method="animate",
                     args=[[layer_labels[0]], dict(mode="immediate",
                                                    frame=dict(duration=0, redraw=True))]),
            ],
            font=dict(size=11, color=st.INK_2),
            bgcolor="white", bordercolor=st.SPINE,
        )],
    )

    fig.update_xaxes(
        side="top", tickangle=-45, tickfont=dict(size=10, color=st.INK_2, family="serif"),
        showgrid=False, linecolor=st.SPINE,
    )
    fig.update_yaxes(
        autorange="reversed", tickfont=dict(size=10, color=st.INK_2, family="serif"),
        showgrid=False, linecolor=st.SPINE,
    )

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_confusion_figure()
    out = out_dir / "17_confusion_evolution.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
