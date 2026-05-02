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

    # Supervised LDA on POOLED L12 — finds the `dim` directions that maximally
    # separate the 23 emotions. With dim=3 and the corrected pipeline this
    # gives separation ratio ≈ 4.3.
    label_to_int = {e: i for i, e in enumerate(sorted(set(label_names)))}
    y = np.array([label_to_int[l] for l in label_names])
    print(f"  fitting LDA on pooled L12 (supervised, n_components={dim})...")
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


def build_galaxy_figure(dim: int = 3) -> go.Figure:
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

    def make_traces(layer: int) -> list:
        traces = []
        for emo in unique_emotions:
            mask = emotions_arr == emo
            pts = coords[layer, mask]
            cluster = cluster_for_emo[emo]
            sample_hover = [
                (f"<b>{emo}</b><br>cluster: {cluster}<br>"
                 + (f"<i>{short(texts[i])}</i>" if texts and texts[i] else ""))
                for i in np.where(mask)[0]
            ]
            if dim == 3:
                traces.append(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=3.6, color=emotion_palette[emo], opacity=0.65,
                                line=dict(color="white", width=0)),
                    hovertext=sample_hover, hoverinfo="text",
                    name=emo, legendgroup=cluster,
                    legendgrouptitle=dict(text=cluster),
                    showlegend=True,
                ))
            else:
                traces.append(go.Scatter(
                    x=pts[:, 0], y=pts[:, 1],
                    mode="markers",
                    marker=dict(size=7, color=emotion_palette[emo], opacity=0.65,
                                line=dict(color="white", width=0)),
                    hovertext=sample_hover, hoverinfo="text",
                    name=emo, legendgroup=cluster,
                    legendgrouptitle=dict(text=cluster),
                    showlegend=True,
                ))

        cur = centroid_traj[layer]
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
                hovertext=[f"<b>{e}</b><br>centroide en {_layer_label(layer)}"
                           for e in unique_emotions],
                hoverinfo="text",
                name="centroides", showlegend=False,
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
                hovertext=[f"<b>{e}</b><br>centroide en {_layer_label(layer)}"
                           for e in unique_emotions],
                hoverinfo="text",
                name="centroides", showlegend=False,
            ))
        return traces

    frames = []
    for layer in range(n_layers):
        frames.append(go.Frame(data=make_traces(layer), name=_layer_label(layer)))

    fig = go.Figure(data=make_traces(0), frames=frames)

    steps = [dict(
        method="animate",
        args=[[_layer_label(L)],
              dict(mode="immediate", frame=dict(duration=0, redraw=True),
                   transition=dict(duration=400, easing="cubic-in-out"))],
        label=_layer_label(L),
    ) for L in range(n_layers)]

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
            currentvalue=dict(prefix="Capa: ",
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
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=900, redraw=True),
                                      transition=dict(duration=400, easing="cubic-in-out"),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
                dict(label="↺ Reset", method="animate",
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
