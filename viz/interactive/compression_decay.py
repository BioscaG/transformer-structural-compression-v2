"""Hero interactive — Compression galaxy decay.

Shows the same 588 sentences in 3D PCA space, but with a slider that swaps
between 6 compression ranks (768 / 512 / 384 / 256 / 128 / 64). At rank
768 the 6 emergent clusters are clean and separated. At rank 384 some
edges blur. At rank 256 the structure starts to dissolve. At rank 64 we're
back to chaos — the model has forgotten the geometry that took 12 layers
to build.

Companion panel: a "cluster tightness" line chart showing how the silhouette
score collapses as rank decreases. Same data, two angles. The viewer
WATCHES the transition de fase (§4.2) happen.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import silhouette_score

from viz.thesis_data import EXTENDED_CLUSTER_MAP
from viz import thesis_data as td
from viz import style as st


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


LANG = {
    "es": {
        "subplot_3d":   "Galaxia de embeddings · L12 a varios rangos de compresión",
        "subplot_line": "Métricas de degradación por rango",
        "silhouette":   "Silhouette score (L12)",
        "f1_ret":       "Retención F1 (Tabla 8)",
        "axis_rank":    "Rango SVD (uniforme)",
        "axis_f1":      "Retención F1",
        "rank_prefix":  "Rango SVD: ",
        "play":         "▶ Play decay",
        "pause":        "⏸ Pause",
        "reset":        "↺ Reset",
        "baseline":     "baseline",
        "sil_h":        "rango %{x}<br>silhouette: %{y:.3f}<extra></extra>",
        "f1_h":         "rango %{x}<br>F1 retenida: %{y:.1%}<extra></extra>",
        "captions":     {
            768: ("Modelo sin compresión. Los 6 clusters psicológicos están "
                  "limpios y separados. Silhouette ≈ {sil:.2f}."),
            512: ("Compresión leve (rango 512). Apenas se nota — la "
                  "geometría se mantiene. F1 retiene {f1:.0%}."),
            384: ("Break-even de la compresión attention. La estructura aún "
                  "es reconocible pero los clusters se rozan. F1 cae al {f1:.0%}."),
            256: ("Cliff. La estructura se difumina notablemente. "
                  "Silhouette baja a {sil:.2f}, F1 al {f1:.0%}. "
                  "Aquí empieza la transición de fase."),
            128: ("Modelo casi muerto. La galaxia es ya una nube amorfa. "
                  "F1 ≈ 0%. Los clusters dejan de existir."),
            64:  ("Colapso total. Los embeddings se han colapsado en un "
                  "blob central. El modelo no diferencia ninguna emoción."),
        },
        "default":      "Rango {r}. Silhouette {sil:.2f}.",
        "initial":      "<b>baseline</b> · Modelo sin compresión. Los 6 clusters psicológicos están limpios y separados. Silhouette ≈ {sil:.2f}.",
    },
    "en": {
        "subplot_3d":   "Embedding galaxy · L12 across compression ranks",
        "subplot_line": "Degradation metrics by rank",
        "silhouette":   "Silhouette score (L12)",
        "f1_ret":       "F1 retention (Table 8)",
        "axis_rank":    "SVD rank (uniform)",
        "axis_f1":      "F1 retention",
        "rank_prefix":  "SVD rank: ",
        "play":         "▶ Play decay",
        "pause":        "⏸ Pause",
        "reset":        "↺ Reset",
        "baseline":     "baseline",
        "sil_h":        "rank %{x}<br>silhouette: %{y:.3f}<extra></extra>",
        "f1_h":         "rank %{x}<br>F1 retained: %{y:.1%}<extra></extra>",
        "captions":     {
            768: ("Uncompressed model. The 6 psychological clusters are "
                  "clean and separated. Silhouette ≈ {sil:.2f}."),
            512: ("Mild compression (rank 512). Barely noticeable — the "
                  "geometry holds. F1 retains {f1:.0%}."),
            384: ("Attention compression break-even. Structure is still "
                  "recognisable but clusters start touching. F1 drops to {f1:.0%}."),
            256: ("Cliff. The structure visibly blurs. Silhouette down to "
                  "{sil:.2f}, F1 at {f1:.0%}. The phase transition starts here."),
            128: ("Model nearly dead. The galaxy is now an amorphous cloud. "
                  "F1 ≈ 0%. Clusters cease to exist."),
            64:  ("Total collapse. The embeddings have collapsed into a "
                  "central blob. The model can't tell any emotion apart."),
        },
        "default":      "Rank {r}. Silhouette {sil:.2f}.",
        "initial":      "<b>baseline</b> · Uncompressed model. The 6 psychological clusters are clean and separated. Silhouette ≈ {sil:.2f}.",
    },
}


def build_decay_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    data = np.load(CACHE_DIR / "compression_decay.npz", allow_pickle=True)
    coords = data["coords"]    # (n_ranks, 13, N, 3)
    ranks = data["ranks"].tolist()
    labels = [str(l) for l in data["labels"]]
    n_ranks, n_layers, n_pts, _ = coords.shape

    cluster_for = [EXTENDED_CLUSTER_MAP.get(l, "Baja especificidad") for l in labels]
    cluster_color = [td.CLUSTER_COLORS.get(c, "#A0A09A") for c in cluster_for]
    cluster_int = {c: i for i, c in enumerate(td.CLUSTER_DEFS.keys())}
    cluster_for.extend(["Baja especificidad" for _ in range(0)])  # noop
    int_labels = np.array([cluster_int.get(c, len(cluster_int)) for c in cluster_for])

    # Compute silhouette score for L12 at each rank — this is the explanatory
    # companion: how cluster-separable is the final layer at each compression?
    silhouettes = []
    for ri in range(n_ranks):
        try:
            s = float(silhouette_score(coords[ri, 12], int_labels, metric="euclidean"))
        except Exception:
            s = 0.0
        silhouettes.append(s)

    # F1 retention at each rank from thesis_data (uniform compression)
    rank_to_f1 = {
        s.name.replace("uniform_r", ""): s
        for s in td.STRATEGIES if s.family == "uniform"
    }
    rank_retentions = []
    for r in ranks:
        if str(r) in rank_to_f1:
            rank_retentions.append(rank_to_f1[str(r)].retention)
        elif r >= 768:
            rank_retentions.append(1.0)
        else:
            rank_retentions.append(None)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.66, 0.34],
        specs=[[{"type": "scene"}, {"type": "xy", "secondary_y": True}]],
        subplot_titles=(L["subplot_3d"], L["subplot_line"]),
        horizontal_spacing=0.06,
    )

    # ─── 3D scatter at L12, current rank initially r=768 ───
    hover = [
        f"<b>{labels[i]}</b><br>Cluster: {cluster_for[i]}"
        for i in range(n_pts)
    ]

    init = coords[0, 12]
    fig.add_trace(go.Scatter3d(
        x=init[:, 0], y=init[:, 1], z=init[:, 2],
        mode="markers",
        marker=dict(size=4.2, color=cluster_color, opacity=0.78,
                    line=dict(color="white", width=0.4)),
        hovertext=hover, hoverinfo="text",
        showlegend=False,
    ), row=1, col=1)

    # ─── Companion line chart: silhouette + F1 retention ───
    fig.add_trace(go.Scatter(
        x=ranks, y=silhouettes,
        mode="lines+markers",
        line=dict(color=st.BLUE, width=2.6, shape="spline", smoothing=0.5),
        marker=dict(size=10, color=st.BLUE, line=dict(color="white", width=1.5)),
        name=L["silhouette"],
        hovertemplate=L["sil_h"],
    ), row=1, col=2)

    # F1 retention overlaid (right axis)
    valid_f1 = [(r, ret) for r, ret in zip(ranks, rank_retentions) if ret is not None]
    if valid_f1:
        rs, rets = zip(*valid_f1)
        fig.add_trace(go.Scatter(
            x=list(rs), y=list(rets),
            mode="lines+markers",
            line=dict(color=st.TERRA, width=2.6, dash="dot"),
            marker=dict(size=9, color=st.TERRA, symbol="diamond",
                        line=dict(color="white", width=1.5)),
            name=L["f1_ret"],
            hovertemplate=L["f1_h"],
        ), row=1, col=2, secondary_y=True)

    # Frames per rank — only update the 3D trace (trace 0)
    frames = []
    for ri in range(n_ranks):
        L12 = coords[ri, 12]
        sil = silhouettes[ri]
        f1_ret = rank_retentions[ri] if rank_retentions[ri] is not None else 0
        rank_label = L["baseline"] if ranks[ri] >= 768 else f"r={ranks[ri]}"

        # Story-line caption per frame (bilingual via LANG dict)
        caption_template = L["captions"].get(ranks[ri], L["default"])
        caption = caption_template.format(r=ranks[ri], sil=sil, f1=f1_ret)

        frames.append(go.Frame(
            data=[
                go.Scatter3d(
                    x=L12[:, 0], y=L12[:, 1], z=L12[:, 2],
                    mode="markers",
                    marker=dict(size=4.2, color=cluster_color, opacity=0.78,
                                line=dict(color="white", width=0.4)),
                    hovertext=hover, hoverinfo="text",
                ),
            ],
            traces=[0],     # only update the 3D scatter; the line chart stays static
            name=str(ranks[ri]),
            layout=dict(
                annotations=[
                    dict(text=L["subplot_3d"],
                         x=0.33, y=1.03, xref="paper", yref="paper", showarrow=False,
                         font=dict(size=12.5, color=st.INK_2, family="serif")),
                    dict(text=L["subplot_line"],
                         x=0.85, y=1.03, xref="paper", yref="paper", showarrow=False,
                         font=dict(size=12.5, color=st.INK_2, family="serif")),
                    dict(text=f"<b>{rank_label}</b> · {caption}",
                         x=0.5, y=-0.16, xref="paper", yref="paper", showarrow=False,
                         align="center", xanchor="center",
                         font=dict(size=11.5, color=st.INK_2, family="serif"),
                         bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                         borderwidth=0.5, borderpad=10),
                ],
            ),
        ))
    fig.frames = frames

    # Slider
    steps = [dict(
        method="animate",
        args=[[str(ranks[ri])],
              dict(mode="immediate", frame=dict(duration=0, redraw=True),
                   transition=dict(duration=500, easing="cubic-in-out"))],
        label=(L["baseline"] if ranks[ri] >= 768 else f"r={ranks[ri]}"),
    ) for ri in range(n_ranks)]

    initial_caption = L["initial"].format(sil=silhouettes[0])

    fig.update_layout(
        **st.thesis_layout(
            title=("Compression galaxy decay · ¿Qué destruye la SVD?"
                   "<br><sub>Misma geometría (PCA fija en baseline L12), distintos rangos. "
                   "Mira cómo los clusters se disuelven al bajar el rango.</sub>"),
            height=720, width=1380,
        ),
        annotations=[
            dict(text=L["subplot_3d"],
                 x=0.33, y=1.03, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12.5, color=st.INK_2, family="serif")),
            dict(text=L["subplot_line"],
                 x=0.85, y=1.03, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12.5, color=st.INK_2, family="serif")),
            dict(text=initial_caption,
                 x=0.5, y=-0.16, xref="paper", yref="paper", showarrow=False,
                 align="center", xanchor="center",
                 font=dict(size=11.5, color=st.INK_2, family="serif"),
                 bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                 borderwidth=0.5, borderpad=10),
        ],
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
        sliders=[dict(
            active=0, x=0.05, y=-0.04, len=0.55,
            currentvalue=dict(prefix=L["rank_prefix"],
                              font=dict(size=13, color=st.INK, family="serif"),
                              xanchor="left"),
            steps=steps,
            pad=dict(t=20, b=10),
            tickcolor=st.INK_3,
            font=dict(size=10, color=st.INK_3),
            transition=dict(duration=500, easing="cubic-in-out"),
        )],
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.05, y=-0.18, xanchor="left", yanchor="top",
            buttons=[
                dict(label=L["play"], method="animate",
                     args=[None, dict(frame=dict(duration=1500, redraw=True),
                                      transition=dict(duration=600, easing="cubic-in-out"),
                                      fromcurrent=True, mode="immediate")]),
                dict(label=L["pause"], method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
                dict(label=L["reset"], method="animate",
                     args=[[str(ranks[0])], dict(mode="immediate",
                                                  frame=dict(duration=0, redraw=True))]),
            ],
            font=dict(size=11, color=st.INK_2),
            bgcolor="white", bordercolor=st.SPINE,
        )],
        legend=dict(font=dict(size=10), x=0.97, y=0.02,
                    xanchor="right", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.85)"),
    )

    fig.update_yaxes(
        title=dict(text=L["axis_f1"], font=dict(size=11, color=st.TERRA)),
        side="right",
        tickformat=".0%", range=[0, 1.05],
        tickfont=dict(size=10, color=st.TERRA),
        showgrid=False,
        row=1, col=2, secondary_y=True,
    )

    fig.update_xaxes(
        title=dict(text=L["axis_rank"], font=dict(size=12, color=st.INK_2)),
        type="category",
        gridcolor=st.GRID, linecolor=st.SPINE, showline=True,
        tickfont=dict(size=10, color=st.INK_3),
        row=1, col=2,
    )
    sil_min = min(silhouettes)
    sil_max = max(silhouettes)
    sil_pad = max(0.05, (sil_max - sil_min) * 0.20)
    fig.update_yaxes(
        title=dict(text=L["silhouette"], font=dict(size=11, color=st.BLUE)),
        gridcolor=st.GRID, linecolor=st.SPINE, showline=True,
        tickfont=dict(size=10, color=st.BLUE),
        range=[sil_min - sil_pad, sil_max + sil_pad],
        row=1, col=2, secondary_y=False,
    )

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_decay_figure()
    out = out_dir / "13_compression_decay.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
