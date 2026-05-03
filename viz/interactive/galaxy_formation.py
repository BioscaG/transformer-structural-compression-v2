"""Galaxy formation — emotional clusters emerging across BERT's 13 layers.

Uses the CORRECT pipeline (after the tokenizer + pooler fix):

  CLS[layer] → pooler.dense + tanh → LDA fitted on L12 → 3 supervised axes

LDA is fitted on the L12 pooled representation, then the SAME projection is
applied to every layer. Axes don't move; the points do. With 23 single-label
sentences per emotion (2300 total) and the corrected embeddings, the LDA-3D
projection shows separation ratio ≈ 4.3 and nearest-centroid classification ≈
40% (vs ~4% random) — meaning emotions cluster meaningfully in this 3-axis
view of the model's actual decision space.
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from transformers import AutoModelForSequenceClassification

from viz import thesis_data as td
from viz import style as st
from viz.data.load_results import EMOTIONS_23, MODEL_CHECKPOINT
from viz.thesis_data import EXTENDED_CLUSTER_MAP, emotion_palette as _emotion_palette


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


def _layer_label(idx: int) -> str:
    return "Embedding" if idx == 0 else f"Encoder L{idx-1}"


def load_real_activations(dim: int = 3) -> tuple[np.ndarray, list[str], list[str], list[str]] | None:
    npz = CACHE_DIR / "activations.npz"
    meta = CACHE_DIR / "meta.json"
    if not (npz.exists() and meta.exists()):
        return None

    print(f"Loading real activations from {CACHE_DIR}")
    data = np.load(npz)
    cls = data["cls_per_layer"]                    # (N, 13, 768)
    info = json.loads(meta.read_text())
    label_names = info["label_names"]
    sentences = info["sentences"]
    cluster_labels = [EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad") for e in label_names]

    # Apply the pooler (Linear + tanh) — this is what the classifier actually sees.
    # Skipping the pooler was one of the bugs in the original galaxy attempt.
    print("  applying pooler (Linear + tanh) to every layer's CLS...")
    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    Wp = mdl.bert.pooler.dense.weight.detach().numpy()    # (768, 768)
    bp = mdl.bert.pooler.dense.bias.detach().numpy()      # (768,)
    pooled = np.tanh(np.einsum('slh,ph->slp', cls, Wp) + bp)   # (N, 13, 768)

    # Supervised LDA on POOLED L12 — find the `dim` directions that maximally
    # separate the 6 PSYCHOLOGICAL CLUSTERS (rather than the 23 individual
    # emotions). Cluster-supervision gives much cleaner visual separation
    # of the macro structure while individual emotions still occupy distinct
    # regions inside each cluster.
    cluster_to_int = {c: i for i, c in enumerate(sorted(set(cluster_labels)))}
    y = np.array([cluster_to_int[c] for c in cluster_labels])
    print(f"  fitting LDA on pooled L12 (cluster-supervised, n_components={dim})...")
    lda = LinearDiscriminantAnalysis(n_components=dim)
    lda.fit(pooled[:, -1, :], y)
    var_ratio = lda.explained_variance_ratio_

    n_layers = pooled.shape[1]
    coords = np.zeros((n_layers, pooled.shape[0], dim))
    for L in range(n_layers):
        coords[L] = lda.transform(pooled[:, L, :])
    print(f"  {pooled.shape[0]} sentences, {n_layers} layers, "
          f"LDA class variance explained: {var_ratio.sum():.2%}")
    return coords, label_names, cluster_labels, sentences


LANG = {
    "es": {
        "centroid":   "centroides",
        "centroid_h": "centroide en",
        "stage_pref": "Capa: ",
        "play":       "▶ Play",
        "pause":      "⏸ Pause",
        "reset":      "↺ Reset",
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
        "centroid":   "centroids",
        "centroid_h": "centroid at",
        "stage_pref": "Layer: ",
        "play":       "▶ Play",
        "pause":      "⏸ Pause",
        "reset":      "↺ Reset",
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


def build_galaxy_figure(dim: int = 3, lang: str = "es") -> go.Figure:
    _L = LANG[lang]
    """Build the galaxy formation figure in 2D or 3D LDA projection.

    `dim`: 2 or 3. With 2, uses go.Scatter (xaxis/yaxis). With 3, uses
    go.Scatter3d (scene). Both share the same animation by layer.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    real = load_real_activations(dim=dim)
    if real is None:
        raise RuntimeError("No cached activations. Run viz/extractors/extract_real.py first.")
    coords, emotions, clusters, texts = real
    n_layers, n_pts, _ = coords.shape

    unique_emotions = sorted(set(emotions))
    emotion_palette = _emotion_palette(EMOTIONS_23)
    emotions_arr = np.array(emotions)

    def short(t: str, n: int = 70) -> str:
        if not t:
            return ""
        t = t.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        return t if len(t) <= n else t[: n - 1] + "…"

    # Per-emotion centroid trajectories across the 13 layers
    centroid_traj = np.zeros((n_layers, len(unique_emotions), dim))
    for L in range(n_layers):
        for ei, emo in enumerate(unique_emotions):
            centroid_traj[L, ei] = coords[L, emotions_arr == emo].mean(axis=0)

    cluster_for_emo = {e: EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad")
                       for e in unique_emotions}

    def _coords_at(t_layer: float) -> np.ndarray:
        """Linearly interpolate point coords at fractional layer t_layer ∈ [0, n_layers-1]."""
        L0 = int(np.floor(t_layer))
        L1 = min(L0 + 1, n_layers - 1)
        alpha = t_layer - L0
        return coords[L0] * (1 - alpha) + coords[L1] * alpha

    def _centroids_at(t_layer: float) -> np.ndarray:
        L0 = int(np.floor(t_layer))
        L1 = min(L0 + 1, n_layers - 1)
        alpha = t_layer - L0
        return centroid_traj[L0] * (1 - alpha) + centroid_traj[L1] * alpha

    def make_traces_at(t_layer: float, layer_label: str,
                       with_hover: bool = True) -> list:
        """Build traces for a real or fractional layer index.

        ``with_hover=False`` skips the per-point hovertext, dropping the
        per-frame payload by an order of magnitude — used for the
        intermediate (interpolation) frames where the user isn't hovering
        anyway because they're mid-animation.
        """
        pts_all = _coords_at(t_layer)
        cur     = _centroids_at(t_layer)
        traces = []
        for emo in unique_emotions:
            mask = emotions_arr == emo
            pts = pts_all[mask]
            cluster = cluster_for_emo[emo]
            color = emotion_palette[emo]
            if with_hover:
                sample_hover = [
                    (f"<b>{emo}</b><br>cluster: {cluster}<br>"
                     + (f"<i>{short(texts[i])}</i>" if texts and texts[i] else ""))
                    for i in np.where(mask)[0]
                ]
                hover_kwargs = dict(hovertext=sample_hover, hoverinfo="text")
            else:
                hover_kwargs = dict(hoverinfo="skip")
            if dim == 3:
                traces.append(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=3.6, color=color, opacity=0.7,
                                line=dict(color="white", width=0)),
                    name=emo, legendgroup=cluster,
                    legendgrouptitle=dict(text=_L["clusters"].get(cluster, cluster)),
                    showlegend=True,
                    **hover_kwargs,
                ))
            else:
                traces.append(go.Scatter(
                    x=pts[:, 0], y=pts[:, 1],
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=0.7,
                                line=dict(color="white", width=0)),
                    name=emo, legendgroup=cluster,
                    legendgrouptitle=dict(text=_L["clusters"].get(cluster, cluster)),
                    showlegend=True,
                    **hover_kwargs,
                ))

        if dim == 3:
            traces.append(go.Scatter3d(
                x=cur[:, 0], y=cur[:, 1], z=cur[:, 2],
                mode="markers+text",
                marker=dict(size=14,
                            color=[emotion_palette[e] for e in unique_emotions],
                            symbol="diamond",
                            line=dict(color=st.INK, width=1.5), opacity=0.95),
                text=unique_emotions, textposition="top center",
                textfont=dict(size=10, color=st.INK, family="serif"),
                hovertext=[f"<b>{e}</b><br>{_L['centroid_h']} {layer_label}"
                           for e in unique_emotions],
                hoverinfo="text",
                name=_L["centroid"], showlegend=False,
            ))
        else:
            traces.append(go.Scatter(
                x=cur[:, 0], y=cur[:, 1],
                mode="markers+text",
                marker=dict(size=18,
                            color=[emotion_palette[e] for e in unique_emotions],
                            symbol="diamond",
                            line=dict(color=st.INK, width=1.5), opacity=0.95),
                text=unique_emotions, textposition="top center",
                textfont=dict(size=10, color=st.INK, family="serif"),
                hovertext=[f"<b>{e}</b><br>{_L['centroid_h']} {layer_label}"
                           for e in unique_emotions],
                hoverinfo="text",
                name=_L["centroid"], showlegend=False,
            ))
        return traces

    # Build animation frames with N_INTERP intermediates between each pair
    # of real layers — gives a smooth "morph" instead of a hard jump.
    # Intermediate frames have hover stripped to keep the page payload
    # manageable (full hover lists explode the file size otherwise).
    N_INTERP = 4
    frames = []
    real_frame_names: list[str] = []
    for layer in range(n_layers):
        label = _layer_label(layer)
        frames.append(go.Frame(
            data=make_traces_at(float(layer), label, with_hover=True),
            name=label,
        ))
        real_frame_names.append(label)
        if layer < n_layers - 1:
            for k in range(1, N_INTERP + 1):
                t = layer + k / (N_INTERP + 1)
                frames.append(go.Frame(
                    data=make_traces_at(t, label, with_hover=False),
                    name=f"{label}__{k}",   # unnamed in slider
                ))

    fig = go.Figure(data=make_traces_at(0.0, _layer_label(0), with_hover=True),
                    frames=frames)

    # Slider only steps through the real-layer frames; intermediates animate
    # automatically when Play runs.
    steps = [dict(
        method="animate",
        args=[[name],
              dict(mode="immediate", frame=dict(duration=0, redraw=True),
                   transition=dict(duration=400, easing="cubic-in-out"))],
        label=name,
    ) for name in real_frame_names]

    layout_kwargs = dict(
        **st.thesis_layout(
            title=(f"Galaxy formation · 23 emociones cristalizando en {dim}D"
                   "<br><sub>LDA supervisada sobre el CLS pooled de L12. "
                   "Diamantes: centroide. Puntos pequeños: frases individuales. "
                   "Click en la leyenda para aislar emociones.</sub>"),
            height=820, width=1380,
        ),
        sliders=[dict(
            active=0, x=0.04, y=-0.04, len=0.85,
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
            x=0.04, y=-0.18, xanchor="left", yanchor="top",
            buttons=[
                dict(label=_L["play"], method="animate",
                     args=[None, dict(frame=dict(duration=90, redraw=True),
                                      transition=dict(duration=60, easing="linear"),
                                      fromcurrent=True, mode="immediate")]),
                dict(label=_L["pause"], method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
                dict(label=_L["reset"], method="animate",
                     args=[[_layer_label(0)], dict(mode="immediate",
                                                    frame=dict(duration=0, redraw=True))]),
            ],
            font=dict(size=11, color=st.INK_2),
            bgcolor="white", bordercolor=st.SPINE,
        )],
        legend=dict(x=1.02, y=0.5, xanchor="left", yanchor="middle",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                    borderwidth=0.5, font=dict(size=10)),
    )

    if dim == 3:
        layout_kwargs["scene"] = dict(
            xaxis=dict(title="LD1", showticklabels=False, backgroundcolor=st.BG,
                       gridcolor=st.GRID, zerolinecolor=st.SPINE),
            yaxis=dict(title="LD2", showticklabels=False, backgroundcolor=st.BG,
                       gridcolor=st.GRID, zerolinecolor=st.SPINE),
            zaxis=dict(title="LD3", showticklabels=False, backgroundcolor=st.BG,
                       gridcolor=st.GRID, zerolinecolor=st.SPINE),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.0)),
            aspectmode="cube",
        )
    else:
        # 2D — fix the axis ranges across all frames so they don't jump per layer
        all_x = coords[..., 0]
        all_y = coords[..., 1]
        xpad = (all_x.max() - all_x.min()) * 0.08
        ypad = (all_y.max() - all_y.min()) * 0.08
        layout_kwargs["xaxis"] = dict(
            title=dict(text="LD1", font=dict(size=12, color=st.INK_2)),
            showticklabels=False, gridcolor=st.GRID, zerolinecolor=st.SPINE,
            range=[all_x.min() - xpad, all_x.max() + xpad],
            scaleanchor="y", scaleratio=1,
        )
        layout_kwargs["yaxis"] = dict(
            title=dict(text="LD2", font=dict(size=12, color=st.INK_2)),
            showticklabels=False, gridcolor=st.GRID, zerolinecolor=st.SPINE,
            range=[all_y.min() - ypad, all_y.max() + ypad],
        )

    fig.update_layout(**layout_kwargs)
    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_galaxy_figure()
    out = out_dir / "07_galaxy_formation.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
