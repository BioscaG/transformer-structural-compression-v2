"""Compresión interna · cómo el modelo se comprime a sí mismo capa a capa.

Two complementary metrics computed on the user's fine-tuned 23emo model
from the existing token_trajectories cache:

  1. Norma del estado oculto (||h[L, t]||₂) per layer. Grows roughly
     monotonically with depth — a known signature of transformer
     architectures (Kobayashi 2021).

  2. Rango efectivo del subespacio de tokens en cada capa. Compute the
     SVD of the (N_tokens × 768) matrix of all content-token
     representations at layer L, then measure:
       - effective rank = exp(spectral entropy of singular values)
       - k95 = number of singular values needed to capture 95% of energy

The story: vectors get LARGER but occupy FEWER directions as you go
deeper. This is **internal compression**: the model itself collapses
its representation onto a low-dimensional subspace by L11-L12. Externally
applying SVD to the late-layer matrices is just materializing what the
model already does.

Provides direct theoretical motivation for §4 (SVD compression): if
content-token reps in L12 live in ~22 effective dimensions out of 768,
then the late-layer weight matrices that produce them can safely be
low-rank approximated.

References: Kobayashi et al. "Incorporating Residual and Normalization
Layers into Analysis of Masked Language Models" (EMNLP 2021); Dong et
al. "Attention is Not All You Need: Pure Attention Loses Rank
Doubly Exponentially with Depth" (ICML 2021).
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import style as st


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"

SPECIAL_TOKENS = {"⟨CLS⟩", "⟨SEP⟩", "[CLS]", "[SEP]", "[PAD]", ""}


def _build_masks(tokens: list[list[str]], T: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (cls_mask, content_mask) of shape (N, T)."""
    n_sent = len(tokens)
    cls_m = np.zeros((n_sent, T), dtype=bool)
    cnt_m = np.zeros((n_sent, T), dtype=bool)
    for si, toks in enumerate(tokens):
        for ti, tok in enumerate(toks):
            if ti >= T:
                break
            if tok in ("⟨CLS⟩", "[CLS]"):
                cls_m[si, ti] = True
            elif tok not in SPECIAL_TOKENS:
                cnt_m[si, ti] = True
    return cls_m, cnt_m


def _compute_metrics():
    d = np.load(CACHE_DIR / "token_trajectories.npz")
    H = d["hidden"].astype(np.float32)            # (N, L, T, 768)
    meta = json.loads((CACHE_DIR / "token_trajectories_meta.json").read_text())
    tokens = meta["tokens"]
    n_sent, n_layers, T, dim = H.shape

    cls_m, cnt_m = _build_masks(tokens, T)
    cls_idx = cls_m.nonzero()
    cnt_idx = cnt_m.nonzero()

    norm_cls = np.zeros(n_layers)
    norm_cnt = np.zeros(n_layers)
    eff_rank = np.zeros(n_layers)
    k95 = np.zeros(n_layers, dtype=np.int32)
    # Energy curve per layer (cumulative variance up to k); store first 256 dims
    K_VIEW = 256
    cum_energy = np.zeros((n_layers, K_VIEW), dtype=np.float32)

    for L in range(n_layers):
        norm_cls[L] = float(np.linalg.norm(
            H[cls_idx[0], L, cls_idx[1]], axis=-1).mean())
        norm_cnt[L] = float(np.linalg.norm(
            H[cnt_idx[0], L, cnt_idx[1]], axis=-1).mean())

        # Pool all content tokens at this layer
        pooled = H[cnt_idx[0], L, cnt_idx[1]]  # (n_tokens, 768)
        # Center to remove mean (so anisotropy of cluster doesn't dominate)
        pooled_c = pooled - pooled.mean(axis=0, keepdims=True)
        s = np.linalg.svd(pooled_c, compute_uv=False)
        var = s ** 2
        p = var / var.sum()
        p_safe = np.clip(p, 1e-12, None)
        eff_rank[L] = float(np.exp(-np.sum(p_safe * np.log(p_safe))))
        cum = np.cumsum(p)
        k95[L] = int(np.argmax(cum >= 0.95)) + 1
        cum_energy[L, :K_VIEW] = cum[:K_VIEW]

    return {
        "n_layers": n_layers, "dim": dim,
        "norm_cls": norm_cls, "norm_cnt": norm_cnt,
        "eff_rank": eff_rank, "k95": k95,
        "cum_energy": cum_energy, "k_view": K_VIEW,
    }


def build_figure() -> go.Figure:
    m = _compute_metrics()
    n_layers = m["n_layers"]
    layer_x = list(range(n_layers))
    layer_labels = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        subplot_titles=(
            "norma ↑ vs rango efectivo ↓ · capa por capa",
            "energía acumulada del espectro · cuántas dimensiones bastan",
        ),
        vertical_spacing=0.16,
    )

    # ─── Top panel: dual-axis curves ──────────────────────────────────────
    # Norms (left axis)
    fig.add_trace(go.Scatter(
        x=layer_x, y=m["norm_cnt"],
        mode="lines+markers",
        name="norma media (tokens contenido)",
        line=dict(color=st.BLUE, width=3, shape="spline", smoothing=0.6),
        marker=dict(size=8, color=st.BLUE,
                    line=dict(color="white", width=1.5)),
        hovertemplate="%{x}: ‖h‖ = %{y:.2f}<extra>tokens contenido</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=layer_x, y=m["norm_cls"],
        mode="lines+markers",
        name="norma del [CLS]",
        line=dict(color=st.BLUE_L, width=2, shape="spline",
                  smoothing=0.6, dash="dot"),
        marker=dict(size=6, color=st.BLUE_L),
        hovertemplate="%{x}: ‖CLS‖ = %{y:.2f}<extra>[CLS]</extra>",
    ), row=1, col=1)

    # Effective rank on secondary y-axis
    fig.add_trace(go.Scatter(
        x=layer_x, y=m["eff_rank"],
        mode="lines+markers",
        name="rango efectivo (entropía espectral)",
        line=dict(color=st.TERRA, width=3, shape="spline", smoothing=0.6),
        marker=dict(size=9, color=st.TERRA, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        yaxis="y2",
        hovertemplate=("%{x}: rango efectivo = %{y:.1f} dim"
                       "<extra>de 768 posibles</extra>"),
    ), row=1, col=1)

    # k95 also on secondary axis (lighter)
    fig.add_trace(go.Scatter(
        x=layer_x, y=m["k95"],
        mode="lines+markers",
        name="k95 (95% energía)",
        line=dict(color=st.TERRA_L, width=2, shape="spline",
                  smoothing=0.6, dash="dash"),
        marker=dict(size=6, color=st.TERRA_L),
        yaxis="y2",
        hovertemplate=("%{x}: %{y} valores singulares cubren 95%"
                       "<extra>k95</extra>"),
    ), row=1, col=1)

    # ─── Bottom panel: cumulative energy heatmap ──────────────────────────
    K_VIEW = m["k_view"]
    fig.add_trace(go.Heatmap(
        z=m["cum_energy"],
        x=list(range(1, K_VIEW + 1)),
        y=layer_labels,
        colorscale=[
            [0.0, "#FFFFFF"],
            [0.50, st.SAND_L],
            [0.85, st.SAND],
            [0.95, st.TERRA_L],
            [1.0, st.TERRA],
        ],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            thickness=12, len=0.42, x=1.01, y=0.22,
            title=dict(text="energía<br>acumulada",
                       font=dict(size=10, color=st.INK_3)),
            tickfont=dict(size=10, color=st.INK_3),
            tickformat=".0%",
        ),
        hovertemplate=("capa <b>%{y}</b><br>top-%{x} valores singulares<br>"
                       "explican %{z:.1%} de la varianza<extra></extra>"),
    ), row=2, col=1)

    # Overlay k95 markers as a line on the heatmap
    fig.add_trace(go.Scatter(
        x=m["k95"].tolist(),
        y=layer_labels,
        mode="lines+markers",
        line=dict(color=st.INK, width=1.5, dash="dot"),
        marker=dict(size=7, color="white",
                    line=dict(color=st.INK, width=1.5)),
        showlegend=True,
        name="k95",
        hovertemplate="%{y}: k95 = %{x}<extra></extra>",
        xaxis="x2", yaxis="y3",
    ), row=2, col=1)

    # ─── Phase shading on top panel ───────────────────────────────────────
    shapes = [
        # Phase bands (top panel)
        dict(type="rect", xref="x", yref="paper",
             x0=-0.5, x1=4.5, y0=0.45, y1=1,
             fillcolor="rgba(58,110,165,0.06)", line=dict(width=0),
             layer="below"),
        dict(type="rect", xref="x", yref="paper",
             x0=4.5, x1=8.5, y0=0.45, y1=1,
             fillcolor="rgba(212,168,67,0.05)", line=dict(width=0),
             layer="below"),
        dict(type="rect", xref="x", yref="paper",
             x0=8.5, x1=12.5, y0=0.45, y1=1,
             fillcolor="rgba(193,85,58,0.07)", line=dict(width=0),
             layer="below"),
        # Hinge L8.5 (top panel)
        dict(type="line", xref="x", yref="paper",
             x0=8.5, x1=8.5, y0=0.45, y1=1,
             line=dict(color=st.INK_3, width=1, dash="dot")),
    ]

    annotations = [
        dict(x=2, y=0.97, xref="x", yref="paper", showarrow=False,
             text="<b>fase léxica</b><br><span style='font-size:10px'>"
                  "rango efectivo alto<br>(~130 dim)</span>",
             font=dict(size=11, color=st.BLUE), align="center"),
        dict(x=6.5, y=0.97, xref="x", yref="paper", showarrow=False,
             text="<b>fase de mezcla</b><br><span style='font-size:10px'>"
                  "rango se mantiene<br>norma crece</span>",
             font=dict(size=11, color=st.SAND), align="center"),
        dict(x=10.5, y=0.97, xref="x", yref="paper", showarrow=False,
             text="<b>colapso espectral</b><br><span style='font-size:10px'>"
                  "rango → 22 dim<br>k95 → 35 dim</span>",
             font=dict(size=11, color=st.TERRA), align="center"),
    ]

    fig.update_layout(
        **st.thesis_layout(
            title=("Compresión interna · el modelo se comprime a sí mismo "
                   "antes de que apliquemos SVD"
                   "<br><sub>Mismas activaciones que la viz anterior. "
                   "Las normas <b>crecen</b> con la profundidad pero el "
                   "subespacio efectivo <b>se estrecha</b> drásticamente. "
                   "Por L12, los vectores de tokens viven en ~22 de 768 "
                   "dimensiones. Esto motiva directamente §4 (SVD).</sub>"),
            height=920, width=1400,
        ),
        legend=dict(
            x=0.99, y=0.59, xanchor="right", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
            borderwidth=0.5, font=dict(size=11),
            orientation="v",
        ),
        shapes=shapes,
        annotations=list(fig.layout.annotations) + annotations,
        yaxis2=dict(
            title=dict(text="rango efectivo / k95 (dim)",
                       font=dict(size=12, color=st.TERRA)),
            overlaying="y", side="right",
            tickfont=dict(size=10, color=st.TERRA),
            range=[0, max(m["k95"].max(), m["eff_rank"].max()) * 1.1],
            showgrid=False, linecolor=st.SPINE,
        ),
    )

    fig.update_xaxes(
        title=dict(text="capa", font=dict(size=12, color=st.INK_2)),
        tickmode="array", tickvals=layer_x, ticktext=layer_labels,
        gridcolor=st.GRID, linecolor=st.SPINE, showline=True,
        tickfont=dict(size=10, color=st.INK_3),
        row=1, col=1,
    )
    fig.update_yaxes(
        title=dict(text="‖h‖₂ (norma euclídea)",
                   font=dict(size=12, color=st.BLUE)),
        gridcolor=st.GRID, linecolor=st.SPINE, showline=True,
        tickfont=dict(size=10, color=st.BLUE),
        row=1, col=1,
    )

    fig.update_xaxes(
        title=dict(text="índice del valor singular (1 = más fuerte)",
                   font=dict(size=12, color=st.INK_2)),
        gridcolor=st.GRID, linecolor=st.SPINE, showline=True,
        tickfont=dict(size=10, color=st.INK_3),
        range=[0, K_VIEW + 1],
        row=2, col=1,
    )
    fig.update_yaxes(
        autorange="reversed",
        showgrid=False, linecolor=st.SPINE, showline=True,
        tickfont=dict(size=10, color=st.INK_3),
        row=2, col=1,
    )

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    out = out_dir / "29_internal_compression.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
